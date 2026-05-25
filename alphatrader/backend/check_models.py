"""
check_models.py
---------------
Run this ONCE to see which Groq models are available on your account.
You then paste the model name you want into your .env file.

Usage:
    python check_models.py

Requires:
    GROQ_API_KEY set in your .env file (or as an environment variable)
"""

import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


def fetch_groq_models() -> None:
    if not GROQ_API_KEY:
        print("❌  GROQ_API_KEY not set.")
        print("    Add it to your .env file:  GROQ_API_KEY=gsk_...")
        sys.exit(1)

    print("Fetching available Groq models...\n")

    try:
        r = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=10,
        )
        r.raise_for_status()
        models = r.json().get("data", [])
    except requests.exceptions.HTTPError as e:
        print(f"❌  HTTP error: {e}")
        print(f"    Response: {e.response.text if e.response else 'no response'}")
        sys.exit(1)
    except Exception as e:
        print(f"❌  Request failed: {e}")
        sys.exit(1)

    if not models:
        print("No models returned. Check your API key.")
        sys.exit(1)

    # Sort by model id
    models_sorted = sorted(models, key=lambda m: m.get("id", ""))

    print("=" * 60)
    print(f"  {'MODEL ID':<40} {'CONTEXT'}")
    print("=" * 60)
    for m in models_sorted:
        model_id = m.get("id", "unknown")
        context  = m.get("context_window", "?")
        print(f"  {model_id:<40} {context:,}" if isinstance(context, int) else f"  {model_id:<40} {context}")
    print("=" * 60)
    print(f"\nTotal: {len(models_sorted)} models\n")
    print("Pick a model and add it to your .env file:")
    print("  LLM_MODEL=<paste model id here>")
    print("\nRecommended for trading system (fast + capable):")
    recommended = [m["id"] for m in models_sorted if "llama" in m["id"].lower() and "70b" in m["id"].lower()]
    for r in recommended:
        print(f"  ✅  {r}")


if __name__ == "__main__":
    fetch_groq_models()
