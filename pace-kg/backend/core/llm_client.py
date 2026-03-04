"""LLM client singletons — load once, reuse everywhere.
Uses ChatGroq exclusively (temperature=0 always).
If the primary LLM returns HTTP 429, call get_fallback_llm() instead.
Free tier: ~30 requests/minute — the pipeline fits within this comfortably.
"""
from langchain_groq import ChatGroq

from core.config import settings

_primary_llm: ChatGroq | None = None
_fallback_llm: ChatGroq | None = None


def get_llm() -> ChatGroq:
    global _primary_llm
    if _primary_llm is None:
        _primary_llm = ChatGroq(
            model=settings.llm_primary,         # llama-3.3-70b-versatile
            temperature=0,
            api_key=settings.groq_api_key,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    return _primary_llm


def get_fallback_llm() -> ChatGroq:
    """Use when Groq returns HTTP 429 (rate limit) on the primary model."""
    global _fallback_llm
    if _fallback_llm is None:
        _fallback_llm = ChatGroq(
            model=settings.llm_fallback,        # llama-3.1-8b-instant
            temperature=0,
            api_key=settings.groq_api_key,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    return _fallback_llm
