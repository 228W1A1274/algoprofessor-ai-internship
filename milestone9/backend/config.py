"""
config.py — Configuration file for TalentTwin AI Agent
------------------------------------------------------
This file loads ALL secret keys and settings from a .env file.
We never write secret keys directly in code (that's a security rule).

The python-dotenv library reads the .env file and makes values available
via os.getenv("VARIABLE_NAME").
"""

import os
from dotenv import load_dotenv

# This line reads your .env file and loads all variables into memory
load_dotenv()

# ── Gemini AI (FREE — get key at https://aistudio.google.com/apikey) ──
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── GitHub API (FREE — get token at https://github.com/settings/tokens) ──
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "")
GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN", "")     # Optional but avoids rate limits

# ── Calendly (FREE — get link from https://calendly.com) ──
CALENDLY_URL = os.getenv("CALENDLY_URL", "https://calendly.com/your-name")

# ── App Settings ──
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")   # Listen on all network interfaces
APP_PORT = int(os.getenv("APP_PORT", "8000"))  # Port number for FastAPI server

# ── CORS Settings (which frontend URLs can talk to our backend) ──
# During development: allow everything ("*")
# In production: change to your actual Vercel URL
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
