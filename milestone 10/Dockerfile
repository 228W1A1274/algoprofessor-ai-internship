# Dockerfile
# ───────────
# Production image for DevSquad.
# Supports two modes via CMD override:
#
#   API server (default — used by Railway/Render):
#     docker run -e GROQ_API_KEY=... -p 8000:8000 devsquad
#
#   CLI one-shot run:
#     docker run -e GROQ_API_KEY=... devsquad \
#       python main.py --task "Build a weather CLI."
#
# Multi-stage build keeps the final image lean (~220 MB vs ~900 MB).

# ── Stage 1: builder — installs deps into a venv ─────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System packages needed to compile some Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies into an isolated venv
COPY requirements.txt .
RUN python -m venv /venv && \
    /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime — copy venv + source, no build tools ────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Only git is needed at runtime (for any git CLI calls; optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built venv from builder stage
COPY --from=builder /venv /venv

# Make venv binaries the default Python/pip
ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy application source
COPY . .

# Create outputs directory (persisted via volume in docker-compose)
RUN mkdir -p outputs

# Port that the FastAPI server listens on.
# Railway injects $PORT at runtime; uvicorn reads it via the CMD below.
EXPOSE 8000

# Default: start the FastAPI web server.
# Railway overrides $PORT automatically — we read it with shell variable expansion.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
