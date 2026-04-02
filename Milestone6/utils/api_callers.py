"""
api_callers.py
──────────────
LLM API caller functions for Gemini, Groq (Llama), and GPT-4o-mini.

Each function:
  - Accepts a prompt string
  - Returns the raw response string (unparsed)
  - Retries with exponential backoff on rate limit errors
  - Returns empty string "" on unrecoverable failure

Usage:
    from utils.api_callers import init_clients, predict_with_gemini, predict_with_groq

    clients = init_clients(
        gemini_api_key="...",
        groq_api_key="...",
        openai_api_key=""        # leave empty to skip GPT
    )

    raw = predict_with_gemini(clients["gemini"], prompt)
    raw = predict_with_groq(clients["groq"], prompt)
    raw = predict_with_gpt(clients["gpt"], prompt)    # returns "" if client is None
"""

import time
import logging

logger = logging.getLogger(__name__)

# ── Model names ────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash-preview-04-17"
GROQ_MODEL   = "llama-3.3-70b-versatile"
GPT_MODEL    = "gpt-4o-mini"

# System prompt used for Groq and GPT (chat-completion format)
SYSTEM_PROMPT = (
    'You are a product pricing expert. '
    'Return ONLY valid JSON in format: {"price": <number>}. '
    'No explanation, no markdown, no extra text.'
)


# ── Client Initializer ────────────────────────────────────────────

def init_clients(
    gemini_api_key: str = "",
    groq_api_key:   str = "",
    openai_api_key: str = ""
) -> dict:
    """
    Initialize all LLM API clients based on which keys are provided.

    Args:
        gemini_api_key (str): Google AI Studio API key.
        groq_api_key   (str): Groq console API key.
        openai_api_key (str): OpenAI platform API key (optional).

    Returns:
        dict: {
            "gemini": GenerativeModel | None,
            "groq":   Groq client    | None,
            "gpt":    OpenAI client  | None
        }
    """
    clients = {"gemini": None, "groq": None, "gpt": None}

    # Gemini
    if gemini_api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_api_key)
            clients["gemini"] = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=150,
                    response_mime_type="application/json"
                )
            )
            logger.info("Gemini client initialized (%s)", GEMINI_MODEL)
        except Exception as e:
            logger.error("Failed to initialize Gemini: %s", e)

    # Groq / Llama
    if groq_api_key:
        try:
            from groq import Groq
            clients["groq"] = Groq(api_key=groq_api_key)
            logger.info("Groq client initialized (%s)", GROQ_MODEL)
        except Exception as e:
            logger.error("Failed to initialize Groq: %s", e)

    # OpenAI / GPT (optional)
    if openai_api_key:
        try:
            from openai import OpenAI
            clients["gpt"] = OpenAI(api_key=openai_api_key)
            logger.info("OpenAI client initialized (%s)", GPT_MODEL)
        except Exception as e:
            logger.error("Failed to initialize OpenAI: %s", e)

    return clients


# ── Gemini Caller ─────────────────────────────────────────────────

def predict_with_gemini(gemini_model, prompt: str, retries: int = 3) -> str:
    """
    Send a prompt to Gemini 2.5 Flash and return the raw response string.

    Args:
        gemini_model: Initialized GenerativeModel instance (from init_clients).
        prompt  (str): Full prompt string.
        retries (int): Number of retry attempts on rate limit errors.

    Returns:
        str: Raw response text, or "" on failure.
    """
    if gemini_model is None:
        return ""

    for attempt in range(retries):
        try:
            response = gemini_model.generate_content(prompt)
            return response.text

        except Exception as e:
            err = str(e).lower()
            if "429" in err or "quota" in err or "rate" in err:
                wait = 2 ** attempt          # 1s → 2s → 4s
                logger.warning("[Gemini] Rate limit hit. Retrying in %ds (attempt %d/%d)", wait, attempt + 1, retries)
                time.sleep(wait)
            elif attempt == retries - 1:
                logger.error("[Gemini] Failed after %d attempts: %s", retries, str(e)[:120])
                return ""
            else:
                time.sleep(1)

    return ""


# ── Groq / Llama Caller ───────────────────────────────────────────

def predict_with_groq(groq_client, prompt: str, retries: int = 3) -> str:
    """
    Send a prompt to Llama 3.3 70B via Groq and return the raw response string.

    Args:
        groq_client: Initialized Groq client instance (from init_clients).
        prompt  (str): Full prompt string.
        retries (int): Number of retry attempts on rate limit errors.

    Returns:
        str: Raw response text, or "" on failure.
    """
    if groq_client is None:
        return ""

    for attempt in range(retries):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.1,
                max_tokens=150,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content

        except Exception as e:
            err = str(e).lower()
            if "429" in err or "rate" in err:
                wait = 2 ** attempt
                logger.warning("[Groq] Rate limit hit. Retrying in %ds (attempt %d/%d)", wait, attempt + 1, retries)
                time.sleep(wait)
            elif attempt == retries - 1:
                logger.error("[Groq] Failed after %d attempts: %s", retries, str(e)[:120])
                return ""
            else:
                time.sleep(1)

    return ""


# ── GPT-4o-mini Caller (Optional) ────────────────────────────────

def predict_with_gpt(gpt_client, prompt: str, retries: int = 3) -> str:
    """
    Send a prompt to GPT-4o-mini via OpenAI and return the raw response string.
    Returns "" silently if gpt_client is None (key not provided).

    Args:
        gpt_client: Initialized OpenAI client instance (from init_clients).
        prompt  (str): Full prompt string.
        retries (int): Number of retry attempts on rate limit errors.

    Returns:
        str: Raw response text, or "" on failure / not configured.
    """
    if gpt_client is None:
        return ""

    for attempt in range(retries):
        try:
            response = gpt_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ],
                temperature=0.1,
                max_tokens=150,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content

        except Exception as e:
            err = str(e).lower()
            if "429" in err or "rate" in err:
                wait = 2 ** attempt
                logger.warning("[GPT] Rate limit hit. Retrying in %ds (attempt %d/%d)", wait, attempt + 1, retries)
                time.sleep(wait)
            elif attempt == retries - 1:
                logger.error("[GPT] Failed after %d attempts: %s", retries, str(e)[:120])
                return ""
            else:
                time.sleep(1)

    return ""
