"""OmniPod agent — Intent router + RAG + Synthesis + Book generation.

Removed LangGraph overhead. Pure async functions with explicit routing.
"""

import asyncio
import functools
import json

from langchain_core.messages import HumanMessage

from core.config import COLLECTION_NAME, HYBRID_TOP_K, RERANKER_TOP_K, SUB_QUERIES_COUNT
from core.llm import SYSTEM_PROMPT, get_json_llm, get_llm, verify_groundedness
from core.vectorstore import get_qdrant_client, hybrid_search

_qdrant_client = None
_known_guests: list[str] | None = None
_llm_semaphore = asyncio.Semaphore(5)  # Max 5 concurrent LLM calls


def _get_client():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = get_qdrant_client()
    return _qdrant_client


# ── Retrieval ────────────────────────────────────────────────────


def retrieve_context(
    query: str, top_k: int = HYBRID_TOP_K, guest_filter: str | None = None
) -> list[dict]:
    client = _get_client()
    return hybrid_search(client, query, top_k=top_k, guest_filter=guest_filter)


def rerank(results: list[dict], top_k: int = RERANKER_TOP_K) -> list[dict]:
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


# ── Guest extraction ──────────────────────────────────────────

_guest_cache: list[str] | None = None


def get_known_guests() -> list[str]:
    """Fetch all unique guest names from the vector DB."""
    global _guest_cache
    if _guest_cache is not None:
        return _guest_cache
    try:
        client = _get_client()
        from qdrant_client import models

        # Scroll all points, collect unique guests
        guests = set()
        next_offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                with_payload=["guest"],
                offset=next_offset,
            )
            for p in points:
                g = p.payload.get("guest", "")
                if g:
                    guests.add(g)
            if next_offset is None:
                break
        _guest_cache = sorted(guests)
    except Exception:
        _guest_cache = []
    return _guest_cache


def extract_guest(query: str) -> str | None:
    """Find a known guest name mentioned in the query.
    Falls back to heuristic pattern matching if not found.
    """
    q_lower = query.lower()
    # 1. Check against known guest list from Qdrant
    for guest in get_known_guests():
        if guest.lower() in q_lower:
            return guest
    # 2. Heuristic regex fallback
    import re

    patterns = [
        r"(?:did|does|would|is|are|was|were)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:say|think|mean|believe|recommend|talk)",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:said|mentioned|talked|thinks|believes|recommends)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            return match.group(1)
    return None


async def _rate_limited_llm(llm_func, messages):
    """Call LLM with semaphore-based rate limiting."""
    async with _llm_semaphore:
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: llm_func(messages)
        )


# ── Intent router ───────────────────────────────────────────────

ROUTER_PROMPT = """You are an intent classifier for a podcast knowledge base.
Given a user query, classify it into one of these categories:

- "factual": Simple queries about what a specific guest said about a topic.
- "synthetic": Multi-source comparison or summary queries.
- "generative": Long-form content generation requests like books or essays.

Respond in JSON format: {{"intent": "factual" | "synthetic" | "generative"}}

User query: {query}"""


@functools.lru_cache(maxsize=128)
def _classify_intent_cached(query: str) -> str:
    """Synchronous cached intent classification."""
    llm = get_json_llm()
    response = llm.invoke([HumanMessage(content=ROUTER_PROMPT.format(query=query))])
    try:
        return json.loads(response.content).get("intent", "factual")
    except (json.JSONDecodeError, AttributeError):
        return "factual"


async def classify_intent(query: str) -> str:
    return _classify_intent_cached(query)


# ── Factual RAG ─────────────────────────────────────────────────

FACTUAL_PROMPT = """{system_prompt}

Context from podcast transcripts:
{context}

Question: {query}

Instructions:
- Answer based STRICTLY on the context above.
- If the context contains relevant information but NOT from the specific guest asked, say so.
- If the context has NO relevant information, say: "This topic has not been discussed in the provided transcripts."
- ALWAYS cite the source guest and title for every claim."""


async def answer_factual(query: str) -> dict:
    """Standard RAG: retrieve → answer → verify groundedness."""
    guest = extract_guest(query)
    if guest:
        results = retrieve_context(query, guest_filter=guest)
        if len(results) < 2:
            seen = {r["text"][:100] for r in results}
            general = retrieve_context(query)
            for r in general:
                if r["text"][:100] not in seen:
                    seen.add(r["text"][:100])
                    results.append(r)
            results = results[:HYBRID_TOP_K]
    else:
        results = retrieve_context(query)

    top = rerank(results)
    context_text = _format_context(top)
    prompt = FACTUAL_PROMPT.format(
        system_prompt=SYSTEM_PROMPT,
        context=context_text,
        query=query,
    )

    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    answer = response.content

    # Groundedness verification
    answer = verify_groundedness(answer, top)

    return {"intent": "factual", "final_answer": answer, "contexts": top}


# ── Synthetic (Map-Reduce) ──────────────────────────────────────

SUB_QUERIES_PROMPT = """You are a research assistant. Given a query, generate {count} specific
sub-questions that cover different aspects of the topic.

Respond in JSON: {{"sub_queries": ["q1", "q2", "q3"]}}

Query: {query}"""

SYNTHESIS_PROMPT = """{system_prompt}

Context from podcast transcripts:
{context}

Question: {query}

Synthesize the information into a coherent answer covering all points."""


async def answer_synthetic(query: str) -> dict:
    """Map-Reduce: sub-queries → retrieve → dedup → synthesize."""
    llm = get_json_llm()

    # Generate sub-queries
    resp = llm.invoke(
        [
            HumanMessage(
                content=SUB_QUERIES_PROMPT.format(count=SUB_QUERIES_COUNT, query=query)
            )
        ]
    )
    try:
        sub_queries = json.loads(resp.content).get("sub_queries", [query])
    except (json.JSONDecodeError, AttributeError):
        sub_queries = [query]

    # Retrieve for each
    all_ctx = []
    for sq in sub_queries:
        results = retrieve_context(sq)
        all_ctx.extend(rerank(results))

    # Dedup by text hash
    seen = set()
    unique = []
    for item in all_ctx:
        h = hash(item["text"][:200])
        if h not in seen:
            seen.add(h)
            unique.append(item)

    context_text = _format_context(unique)
    prompt = SYNTHESIS_PROMPT.format(
        system_prompt=SYSTEM_PROMPT,
        context=context_text,
        query=query,
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    answer = verify_groundedness(response.content, unique)

    return {"intent": "synthetic", "final_answer": answer, "contexts": unique}


# ── Generative (Book) ───────────────────────────────────────────

PLANNER_PROMPT = """{system_prompt}

You are a book planner. Create a table of contents for a book about: {query}

Respond in JSON format:
{{
    "title": "Book Title",
    "sections": [
        {{"title": "Section 1", "description": "What it covers"}},
        {{"title": "Section 2", "description": "What it covers"}}
    ]
}}"""

WRITER_PROMPT = """{system_prompt}

Book: {book_title}
Section: {section_title}
Description: {section_description}

Research context:
{context}

Write this section based ONLY on the provided context. Cite sources."""

COMPILER_PROMPT = """{system_prompt}

Compile a complete book from these sections.

Title: {book_title}

Sections:
{sections}

Write: introduction, all sections, conclusion. Base everything on the content provided."""


async def answer_generative(query: str) -> dict:
    """Book generation: planner → write sections → compile."""
    llm = get_json_llm()

    # 1. Plan
    resp = llm.invoke(
        [
            HumanMessage(
                content=PLANNER_PROMPT.format(system_prompt=SYSTEM_PROMPT, query=query)
            )
        ]
    )
    try:
        plan = json.loads(resp.content)
        book_title = plan.get("title", "Untitled")
        sections = plan.get("sections", [])
    except (json.JSONDecodeError, AttributeError):
        book_title = "Untitled"
        sections = [{"title": "Overview", "description": query}]

    if not sections:
        return {
            "intent": "generative",
            "final_answer": "No sections generated.",
            "contexts": [],
        }

    # 2. Write each section
    writer = get_llm()
    drafted = {}
    for section in sections:
        title = section.get("title", "Section")
        desc = section.get("description", "")
        search_q = f"{title}: {desc}"
        results = retrieve_context(search_q)
        top = rerank(results)
        ctx = _format_context(top)

        prompt = WRITER_PROMPT.format(
            system_prompt=SYSTEM_PROMPT,
            book_title=book_title,
            section_title=title,
            section_description=desc,
            context=ctx,
        )
        resp = writer.invoke([HumanMessage(content=prompt)])
        drafted[title] = resp.content

    # 3. Compile
    sections_text = "\n\n".join(f"## {t}\n\n{c}" for t, c in drafted.items())
    prompt = COMPILER_PROMPT.format(
        system_prompt=SYSTEM_PROMPT,
        book_title=book_title,
        sections=sections_text,
    )
    compiler = get_llm()
    resp = compiler.invoke([HumanMessage(content=prompt)])

    return {"intent": "generative", "final_answer": resp.content, "contexts": []}


# ── Public API ──────────────────────────────────────────────────


async def run_agent(query: str) -> dict:
    """Route query to appropriate handler based on intent."""
    intent = await classify_intent(query)

    if intent == "synthetic":
        return await answer_synthetic(query)
    elif intent == "generative":
        return await answer_generative(query)
    else:
        return await answer_factual(query)
