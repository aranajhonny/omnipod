"""DeepSeek V4 Flash LLM client via LangChain ChatOpenAI."""

from langchain_openai import ChatOpenAI

from core.config import LLM_MODEL, OPENCODE_API_KEY, OPENCODE_BASE_URL

# ── Anti-Hallucination System Prompt ─────────────────────────────
SYSTEM_PROMPT = """You are an expert assistant based on a database of podcast transcripts.
Your knowledge base is STRICTLY limited to the text fragments provided in the context.
RULES:
- Answer ONLY using the information from the provided context.
- If the answer is not found in the context, say exactly: "This topic has not been discussed in the provided transcripts."
- NEVER use your external general knowledge nor make up information.
- ALWAYS cite the source using the guest's name and title provided in the context metadata.
- For long-form tasks (such as writing books or essays), use the search tool to research each section before drafting it. Do not invent details."""


def get_llm(temperature: float = 0.0, **kwargs):
    """Return a configured ChatOpenAI instance pointed at DeepSeek V4 Flash."""
    return ChatOpenAI(
        openai_api_base=OPENCODE_BASE_URL,
        openai_api_key=OPENCODE_API_KEY,
        model=LLM_MODEL,
        streaming=True,
        temperature=temperature,
        **kwargs,
    )


def get_json_llm(temperature: float = 0.0):
    """Return an LLM configured for JSON output mode."""
    return get_llm(
        temperature=temperature,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
