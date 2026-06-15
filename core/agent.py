"""LangGraph agent for OmniPod.

State machine with three intent routes:
  - factual:    Standard RAG (retrieve → answer)
  - synthetic:  Map-reduce with sub-queries → consolidated summary
  - generative: Book agent (planner → loop [search + write] → compiler)
"""

import json
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from core.config import HYBRID_TOP_K, RERANKER_TOP_K, SUB_QUERIES_COUNT
from core.llm import SYSTEM_PROMPT, get_json_llm, get_llm
from core.vectorstore import get_qdrant_client, hybrid_search

# ── State Types ──────────────────────────────────────────────────


class AgentState(TypedDict):
    """Main state passed through the graph nodes."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    intent: str
    query: str
    contexts: list
    sub_queries: list[str]
    book_title: str
    book_plan: list[dict]  # [{"title": "...", "description": "..."}, ...]
    current_section_idx: int
    book_sections: dict  # title -> content
    final_answer: str


# ── Intent Router ────────────────────────────────────────────────

ROUTER_PROMPT = """You are an intent classifier for a podcast knowledge base.
Given a user query, classify it into one of these categories:

- "factual": Simple queries about what a specific guest said about a topic.
  Examples: "What did Andrej Karpathy say about AI?", "Does Huberman recommend cold exposure?"

- "synthetic": Multi-source comparison or summary queries.
  Examples: "Summarize opinions on AI safety across all guests", "Compare guests' views on meditation"

- "generative": Long-form content generation requests like books or essays.
  Examples: "Write a book about the future of AI", "Create a comprehensive essay on human consciousness"

Respond in JSON format: {{"intent": "factual" | "synthetic" | "generative", "reasoning": "short explanation"}}

User query: {query}"""


def router_node(state: AgentState) -> AgentState:
    query = state["messages"][-1].content if state["messages"] else ""
    state["query"] = query

    llm = get_json_llm()
    response = llm.invoke([HumanMessage(content=ROUTER_PROMPT.format(query=query))])

    try:
        result = json.loads(response.content)
        intent = result.get("intent", "factual")
    except (json.JSONDecodeError, AttributeError):
        intent = "factual"

    state["intent"] = intent
    return state


# ── Retrieval ────────────────────────────────────────────────────

_qdrant_client = None


def _get_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = get_qdrant_client()
    return _qdrant_client


def retrieve_context(
    query: str, top_k: int = HYBRID_TOP_K, guest_filter: str | None = None
) -> list[dict]:
    client = _get_client()
    return hybrid_search(client, query, top_k=top_k, guest_filter=guest_filter)


# ── Simple ranking (BM25 scores from Qdrant) ──────────────────────


def rerank(query: str, results: list[dict], top_k: int = RERANKER_TOP_K) -> list[dict]:
    """Results are already scored by BM25, just take top_k."""
    return results[:top_k]


def _format_context(ctx: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(ctx):
        parts.append(
            f"[Source {i + 1}] Guest: {chunk['guest']}\n"
            f"Title: {chunk['title']}\n"
            f"Content: {chunk['text']}\n"
        )
    return "\n---\n".join(parts)


# ── Factual Node ─────────────────────────────────────────────────


def _extract_guest(query: str) -> str | None:
    """Try to extract a guest name from the query."""
    import re

    # Patterns like "what did X say about...", "what does X think...", "X sobre..."
    patterns = [
        r"(?:did|does|would|is|are|was|were)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:say|think|mean|believe|recommend|talk)",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:said|mentioned|talked|thinks|believes|recommends)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            return match.group(1)
    return None


FACTUAL_PROMPT = """{system_prompt}

Context from podcast transcripts:
{context}

Question: {query}

Instructions:
- Answer based STRICTLY on the context above.
- If the context contains relevant information but NOT from the specific guest asked, mention what other guests said about the topic and note that the specific guest's views were not found.
- If the context has NO relevant information at all, say: "This topic has not been discussed in the provided transcripts."
- ALWAYS cite the source guest and title for each claim."""


def factual_node(state: AgentState) -> AgentState:
    query = state["query"]

    # Try to extract a guest name and search with filter
    guest = _extract_guest(query)
    if guest:
        # Search within this guest's content first
        results = retrieve_context(query, guest_filter=guest)
        if len(results) < 2:
            # Not enough from this guest, supplement with general search
            general = retrieve_context(query)
            # Merge, deduplicate by keeping guest-specific first
            seen = {r["text"][:100] for r in results}
            for r in general:
                if r["text"][:100] not in seen:
                    seen.add(r["text"][:100])
                    results.append(r)
            results = results[:HYBRID_TOP_K]
    else:
        results = retrieve_context(query)

    top = rerank(query, results)
    state["contexts"] = top

    context_text = _format_context(top)
    prompt = FACTUAL_PROMPT.format(
        system_prompt=SYSTEM_PROMPT,
        context=context_text,
        query=query,
    )

    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    state["final_answer"] = response.content
    state["messages"] = [AIMessage(content=response.content)]
    return state


# ── Synthetic Node (Map-Reduce) ──────────────────────────────────

SUB_QUERIES_PROMPT = """You are a research assistant helping to gather comprehensive information.
Given the user's query, generate {count} specific sub-questions that will help
cover different aspects of the topic. Each sub-question should be answerable
from podcast transcripts.

Respond in JSON format: {{"sub_queries": ["sub_query_1", "sub_query_2", "sub_query_3"]}}

User query: {query}"""

SYNTHESIS_PROMPT = """{system_prompt}

You have been provided with multiple excerpts from podcast transcripts covering different
aspects of the user's question. Synthesize the information into a coherent, well-structured
answer that covers all the points.

Context:
{context}

Question: {query}

Provide a comprehensive synthesis based strictly on the context above."""


def _deduplicate_contexts(contexts: list[list[dict]]) -> list[dict]:
    seen = set()
    unique = []
    for ctx_list in contexts:
        for item in ctx_list:
            text_hash = hash(item["text"][:200])
            if text_hash not in seen:
                seen.add(text_hash)
                unique.append(item)
    return unique


def synthetic_node(state: AgentState) -> AgentState:
    query = state["query"]

    llm = get_json_llm()
    sub_response = llm.invoke(
        [
            HumanMessage(
                content=SUB_QUERIES_PROMPT.format(count=SUB_QUERIES_COUNT, query=query)
            )
        ]
    )

    try:
        sub_data = json.loads(sub_response.content)
        sub_queries = sub_data.get("sub_queries", [query])
    except (json.JSONDecodeError, AttributeError):
        sub_queries = [query]

    state["sub_queries"] = sub_queries

    all_contexts = []
    for sq in sub_queries:
        results = retrieve_context(sq)
        top = rerank(sq, results)
        all_contexts.append(top)

    unique_contexts = _deduplicate_contexts(all_contexts)
    state["contexts"] = unique_contexts

    context_text = _format_context(unique_contexts)
    prompt = SYNTHESIS_PROMPT.format(
        system_prompt=SYSTEM_PROMPT,
        context=context_text,
        query=query,
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    state["final_answer"] = response.content
    state["messages"] = [AIMessage(content=response.content)]
    return state


# ── Generative Flow (Book Agent) ─────────────────────────────────

PLANNER_PROMPT = """{system_prompt}

You are a book planner. Based on the knowledge available in the podcast transcripts,
create a detailed table of contents for a book about the following topic.

Topic: {query}

Generate 4-8 chapters/sections. For each section, provide a title and a brief
description of what it should cover.

Respond in JSON format:
{{
    "title": "Book Title",
    "sections": [
        {{"title": "Section 1 Title", "description": "What this section covers"}},
        {{"title": "Section 2 Title", "description": "What this section covers"}}
    ]
}}"""

WRITER_PROMPT = """{system_prompt}

You are writing a section of a book based on podcast transcripts.

Book Title: {book_title}
Section Title: {section_title}
Section Description: {section_description}

Research context from transcripts:
{context}

Write a comprehensive section based ONLY on the provided context.
Include citations to the guest and title for each key claim.
If the context does not contain enough information about this section,
say so and write what you can based on what's available."""

COMPILER_PROMPT = """{system_prompt}

You are compiling a complete book from drafted sections.

Book Title: {book_title}

Sections:
{sections}

Write a complete book with:
1. A compelling introduction that sets the stage
2. All the sections seamlessly joined
3. A conclusion that ties everything together

Format it as a well-structured book with proper headings and paragraphs.
Base everything STRICTLY on the content provided in the sections above."""


def planner_node(state: AgentState) -> AgentState:
    """Generate a table of contents for the book."""
    query = state["query"]

    llm = get_json_llm()
    response = llm.invoke(
        [
            HumanMessage(
                content=PLANNER_PROMPT.format(system_prompt=SYSTEM_PROMPT, query=query)
            )
        ]
    )

    try:
        plan = json.loads(response.content)
        sections = plan.get("sections", [])
        state["book_plan"] = sections
        state["book_title"] = plan.get("title", "Untitled")
    except (json.JSONDecodeError, AttributeError):
        state["book_plan"] = [{"title": "Overview", "description": query}]
        state["book_title"] = "Untitled"

    state["current_section_idx"] = 0
    state["book_sections"] = {}
    return state


def section_writer_node(state: AgentState) -> AgentState:
    """Research and write the current section."""
    idx = state["current_section_idx"]
    plan = state["book_plan"]

    if idx >= len(plan):
        # All sections done → go to compiler
        return state

    section = plan[idx]
    section_title = section.get("title", f"Section {idx + 1}")
    section_description = section.get("description", "")

    # 1. Research: search for context related to this section
    search_query = f"{section_title}: {section_description}"
    results = retrieve_context(search_query)
    top = rerank(search_query, results)
    context_text = _format_context(top)

    # 2. Write the section
    prompt = WRITER_PROMPT.format(
        system_prompt=SYSTEM_PROMPT,
        book_title=state["book_title"],
        section_title=section_title,
        section_description=section_description,
        context=context_text,
    )

    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])

    # Store
    sections = dict(state["book_sections"])
    sections[section_title] = response.content
    state["book_sections"] = sections
    state["current_section_idx"] = idx + 1

    return state


def compile_node(state: AgentState) -> AgentState:
    """Compile all sections into a complete book."""
    sections = state.get("book_sections", {})
    book_title = state.get("book_title", "Untitled")

    sections_text = ""
    for title, content in sections.items():
        sections_text += f"\n\n## {title}\n\n{content}"

    if not sections_text.strip():
        state["final_answer"] = (
            "No sections were generated. Please try again with a more specific topic."
        )
        state["messages"] = [AIMessage(content=state["final_answer"])]
        return state

    llm = get_llm()
    prompt = COMPILER_PROMPT.format(
        system_prompt=SYSTEM_PROMPT,
        book_title=book_title,
        sections=sections_text,
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    state["final_answer"] = response.content
    state["messages"] = [AIMessage(content=response.content)]
    return state


# ── Graph Construction ───────────────────────────────────────────


def decide_route(
    state: AgentState,
) -> Literal["factual_node", "synthetic_node", "generative_planner"]:
    return {
        "factual": "factual_node",
        "synthetic": "synthetic_node",
        "generative": "generative_planner",
    }.get(state["intent"], "factual_node")


def should_continue_generative(
    state: AgentState,
) -> Literal["section_writer_node", "generative_compiler"]:
    """After writing a section, decide: more sections or compile?"""
    idx = state["current_section_idx"]
    plan = state["book_plan"]
    if idx < len(plan):
        return "section_writer_node"
    return "generative_compiler"


def build_graph():
    """Build the complete LangGraph state machine with generative loop."""
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("factual_node", factual_node)
    workflow.add_node("synthetic_node", synthetic_node)
    workflow.add_node("generative_planner", planner_node)
    workflow.add_node("section_writer_node", section_writer_node)
    workflow.add_node("generative_compiler", compile_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        decide_route,
        {
            "factual_node": "factual_node",
            "synthetic_node": "synthetic_node",
            "generative_planner": "generative_planner",
        },
    )

    workflow.add_edge("factual_node", END)
    workflow.add_edge("synthetic_node", END)
    workflow.add_edge("generative_planner", "section_writer_node")

    # Generative loop: write section → check if more → compile or write next
    workflow.add_conditional_edges(
        "section_writer_node",
        should_continue_generative,
        {
            "section_writer_node": "section_writer_node",
            "generative_compiler": "generative_compiler",
        },
    )

    workflow.add_edge("generative_compiler", END)

    return workflow.compile()


# ── Execution Helper ─────────────────────────────────────────────

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


async def run_agent(query: str) -> dict:
    """Run the agent with a user query and return the result."""
    graph = get_graph()
    initial_state = AgentState(
        messages=[HumanMessage(content=query)],
        intent="",
        query=query,
        contexts=[],
        sub_queries=[],
        book_title="",
        book_plan=[],
        current_section_idx=0,
        book_sections={},
        final_answer="",
    )

    result = await graph.ainvoke(initial_state)
    return result
