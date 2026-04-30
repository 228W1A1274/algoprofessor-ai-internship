"""
llm_client.py
─────────────
Centralised factory for the Groq LLM used by all agents.

Why a factory?
  - One place to swap models / tweak parameters for the whole project.
  - Keeps every agent file free of config boilerplate.
  - Easy to add retry / fallback logic later.

Groq LLM via CrewAI's LLM wrapper
  CrewAI accepts crewai.LLM(model=...) where the model string follows the
  LiteLLM convention: "groq/<model_name>".
  API key is read automatically from os.environ["GROQ_API_KEY"].
"""

import os
from crewai import LLM


def get_llm(
    model: str = "groq/llama-3.3-70b-versatile",
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> LLM:
    """
    Return a CrewAI LLM object backed by Groq.

    Parameters
    ----------
    model       : LiteLLM-format model string, e.g. "groq/llama-3.3-70b-versatile"
    temperature : 0.0 = deterministic, 1.0 = creative
    max_tokens  : upper bound on response length per agent call

    Returns
    -------
    crewai.LLM  : drop-in llm= argument for any Agent()
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )

    return LLM(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,           # explicit; LiteLLM also picks it from env
    )


# Convenience singleton — import and use directly
default_llm = get_llm()
