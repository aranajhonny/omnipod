"""DeepSeek V4 Flash LLM client via LangChain ChatOpenAI."""

import json
import time
from functools import wraps

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from core.config import LLM_MODEL, OPENCODE_API_KEY, OPENCODE_BASE_URL

SYSTEM_PROMPT = """You are an expert assistant based on a database of podcast transcripts.
Your knowledge base is STRICTLY limited to the text fragments provided in the context.
RULES:
- Answer ONLY using the information from the provided context.
- If the answer is not found in the context, say exactly: "This topic has not been discussed in the provided transcripts."
- NEVER use your external general knowledge nor make up information.
- ALWAYS cite the source using the guest's name and title provided in the context metadata.
- For long-form tasks (such as writing books or essays), use the search tool to research each section before drafting it. Do not invent details."""

GROUNDEDNESS_PROMPT = """You are a fact-checker. Verify if the following ANSWER is fully supported by the provided CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

Respond in JSON format:
{{
    "is_grounded": true or false,
    "unsupported_claims": ["list of claims not found in context"],
    "corrected_answer": "the answer with unsupported claims removed, or null if fully grounded"
}}"""


def retry(max_retries: int = 3, delay: float = 1.0):
    """Retry decorator for transient API failures."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
            raise last_exc

        return wrapper

    return decorator


def get_llm(temperature: float = 0.0, **kwargs):
    return ChatOpenAI(
        openai_api_base=OPENCODE_BASE_URL,
        openai_api_key=OPENCODE_API_KEY,
        model=LLM_MODEL,
        streaming=True,
        temperature=temperature,
        **kwargs,
    )


def get_json_llm(temperature: float = 0.0):
    return get_llm(
        temperature=temperature,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def verify_groundedness(answer: str, contexts: list[dict]) -> str:
    """Verify answer against context using LLM fact-checker pass."""
    if not contexts or "has not been discussed" in answer.lower():
        return answer

    context_text = "\n---\n".join(
        f"[Source {i + 1}] Guest: {c['guest']}\nTitle: {c['title']}\nContent: {c['text']}"
        for i, c in enumerate(contexts[:5])
    )

    llm = get_json_llm()
    try:
        response = llm.invoke(
            [
                HumanMessage(
                    content=GROUNDEDNESS_PROMPT.format(
                        context=context_text, answer=answer
                    )
                )
            ]
        )
        result = json.loads(response.content)
        if not result.get("is_grounded", True):
            corrected = result.get("corrected_answer")
            if corrected:
                return corrected
    except Exception:
        pass
    return answer
