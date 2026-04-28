# utils/llm_clients.py
"""
LLM client factory for BargainBot.
Returns CrewAI-compatible LLM objects via LiteLLM routing.
All models use free-tier APIs (Groq + Gemini).
"""

import os
from crewai import LLM

GROQ_MODEL   = "groq/llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini/gemini-2.5-flash-lite"


def get_groq_llm(temperature: float = 0.3) -> LLM:
    """Return a Groq LLM instance (free tier, no rate limit issues)."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set in environment.")
    return LLM(model=GROQ_MODEL, temperature=temperature)


def get_gemini_llm(temperature: float = 0.2) -> LLM:
    """
    Return a Gemini LLM instance (free tier, 20 req/day limit).
    Falls back to Groq automatically if quota is exhausted.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set in environment.")
    try:
        llm = LLM(model=GEMINI_MODEL, temperature=temperature)
        return llm
    except Exception:
        # Fallback to Groq if Gemini quota is exhausted
        return get_groq_llm(temperature=temperature)