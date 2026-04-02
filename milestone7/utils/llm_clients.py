"""
utils/llm_clients.py
====================
LLM client wrappers with rate limiting and retry logic.
Supports Groq (Llama) and Google Gemini (FREE tiers).
"""

import os
import time
from typing import Optional


# ── Base class ───────────────────────────────────────────────

class BaseLLMClient:
    model_name: str = "base"

    def __init__(self):
        self._n_calls    = 0
        self._total_time = 0.0

    def predict(self, prompt: str, retries: int = 2) -> Optional[str]:
        raise NotImplementedError

    @property
    def avg_latency_ms(self):
        return (self._total_time / self._n_calls * 1000) if self._n_calls > 0 else 0.0


# ── Groq (Llama 3.3 70B — FREE) ─────────────────────────────

class GroqClient(BaseLLMClient):
    """Rate-limited Groq client for llama-3.3-70b-versatile."""

    def __init__(self):
        super().__init__()
        self.model_name   = "llama-3.3-70b-versatile"
        self.min_interval = 60.0 / 28   # 28 RPM free tier
        self._last_call   = 0.0
        self._client      = self._build_client()
        print(f"  GroqClient ready: {self.model_name}")

    def _build_client(self):
        from groq import Groq
        try:
            return Groq(api_key=os.environ["GROQ_API_KEY"])
        except TypeError:
            import httpx
            return Groq(
                api_key=os.environ["GROQ_API_KEY"],
                http_client=httpx.Client(),
            )

    def _call_api(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model       = self.model_name,
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.0,
            max_tokens  = 50,
        )
        return resp.choices[0].message.content.strip()

    def predict(self, prompt: str, retries: int = 2) -> Optional[str]:
        wait = self.min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

        for attempt in range(retries + 1):
            try:
                t0   = time.perf_counter()
                resp = self._call_api(prompt)
                self._total_time += time.perf_counter() - t0
                self._n_calls    += 1
                return resp
            except Exception as e:
                if attempt < retries:
                    w = 2 ** attempt
                    print(f"  Retry {attempt+1} ({self.model_name}): {e}. Waiting {w}s...")
                    time.sleep(w)
                else:
                    print(f"  FAIL ({self.model_name}): {e}")
                    return None


# ── Google Gemini (FREE) ─────────────────────────────────────

class GeminiClient(BaseLLMClient):
    """Rate-limited Google Gemini Flash client."""

    def __init__(self):
        super().__init__()
        self.model_name   = "gemini-2.0-flash"
        self.min_interval = 60.0 / 15   # 15 RPM free tier
        self._last_call   = 0.0
        self._client      = self._build_client()
        print(f"  GeminiClient ready: {self.model_name}")

    def _build_client(self):
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        return genai.GenerativeModel(self.model_name)

    def predict(self, prompt: str, retries: int = 2) -> Optional[str]:
        wait = self.min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

        for attempt in range(retries + 1):
            try:
                t0   = time.perf_counter()
                resp = self._client.generate_content(prompt)
                self._total_time += time.perf_counter() - t0
                self._n_calls    += 1
                return resp.text.strip()
            except Exception as e:
                if attempt < retries:
                    w = 2 ** attempt
                    print(f"  Retry {attempt+1} ({self.model_name}): {e}. Waiting {w}s...")
                    time.sleep(w)
                else:
                    print(f"  FAIL ({self.model_name}): {e}")
                    return None


# ── Factory ──────────────────────────────────────────────────

def build_clients() -> list:
    """Build all available LLM clients based on set env vars."""
    clients = []

    if os.environ.get("GROQ_API_KEY", ""):
        try:
            clients.append(GroqClient())
        except Exception as e:
            print(f"  ⚠️  Groq init failed: {e}")

    if os.environ.get("GOOGLE_API_KEY", ""):
        try:
            clients.append(GeminiClient())
        except Exception as e:
            print(f"  ⚠️  Gemini init failed: {e}")

    return clients
