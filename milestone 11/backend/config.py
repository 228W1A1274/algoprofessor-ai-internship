"""
Configuration — loads from .env via pydantic-settings
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── AI Model ──────────────────────────────────────────────────────────────
    # Best FREE option: Groq + llama-3.3-70b-versatile
    #   • ~300 tokens/s (fastest free inference)
    #   • Native tool/function calling support
    #   • 6,000 req/day free tier
    #   • Extremely stable uptime
    GROQ_API_KEY: str = ""

    # Fallback options (set only one)
    OPENROUTER_API_KEY: str = ""
    GEMINI_API_KEY: str = ""

    # ── Model selection ───────────────────────────────────────────────────────
    # Primary: Groq — llama-3.3-70b-versatile
    # Fallback: openrouter — meta-llama/llama-3-70b-instruct
    LLM_PROVIDER: str = "groq"           # "groq" | "openrouter" | "gemini"
    LLM_MODEL: str = "llama-3.3-70b-versatile"

    # ── Playwright ────────────────────────────────────────────────────────────
    PLAYWRIGHT_HEADLESS: bool = False    # Set True in production/cloud
    PLAYWRIGHT_SLOW_MO: int = 100        # ms delay (easier to observe)
    PLAYWRIGHT_TIMEOUT: int = 30000      # 30s page timeout

    # ── App ───────────────────────────────────────────────────────────────────
    MAX_AGENT_STEPS: int = 20            # Prevent infinite loops
    DEBUG: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
