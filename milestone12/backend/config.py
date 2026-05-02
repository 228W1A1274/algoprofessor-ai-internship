import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o")
DOCKER_BASE_IMAGE: str = os.getenv("DOCKER_BASE_IMAGE", "python:3.11-slim")
GENERATED_AGENTS_DIR: str = os.getenv("GENERATED_AGENTS_DIR", "../generated_agents")
AGENT_PORT_START: int = int(os.getenv("AGENT_PORT_START", "8100"))
MAX_FIX_RETRIES: int = int(os.getenv("MAX_FIX_RETRIES", "3"))
DOCKER_NETWORK: str = os.getenv("DOCKER_NETWORK", "agent-net")
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
BASE_URL: str = os.getenv("BASE_URL", "https://api.openai.com/v1")