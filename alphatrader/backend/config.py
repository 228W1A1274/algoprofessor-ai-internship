"""
config.py
---------
Centralised settings for AlphaTrader.
Reads from environment variables (loaded from .env by python-dotenv).
All future API keys, model names, and service URLs live here.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── App metadata ───────────────────────────────────────────────
    APP_NAME: str = "AlphaTrader"
    APP_VERSION: str = "0.1.0"
    ENV: str = "development"          # "development" | "production"
    DEBUG: bool = True

    # ── Future API keys (empty for now) ───────────────────────────
    ALPACA_API_KEY: str = ""
    ALPACA_SECRET_KEY: str = ""
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"

    # ── Future LLM settings ───────────────────────────────────────
    LLM_MODEL: str = ""              # e.g. "llama3-70b-8192" via Groq
    LLM_TEMPERATURE: float = 0.1

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"             # silently ignore unknown .env keys


# Singleton instance imported everywhere
settings = Settings()
