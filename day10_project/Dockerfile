FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- ⚡ SPEED OPTIMIZATION LAYER ⚡ ---
# We install the massive PyTorch library HERE, before copying requirements.
# This creates a "Cached Layer". As long as you don't touch this line,
# Docker will NEVER download PyTorch again, even if you change requirements.txt.
# (Note: Using CPU version to keep it small ~200MB instead of 900MB)
RUN pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# --- STANDARD REQUIREMENTS ---
COPY requirements.txt .

# Install Python dependencies with CACHE enabled
# 1. We removed '--no-cache-dir'
# 2. We added '--mount=type=cache' so Docker remembers downloaded files
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy application code (This changes often, so it goes last)
COPY backend/ ./backend/
COPY data/ ./data/
# COPY .env.example .env

# Create necessary directories
RUN mkdir -p logs outputs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]