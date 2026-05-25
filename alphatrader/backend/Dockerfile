# backend/Dockerfile
# ------------------
# Builds the AlphaTrader FastAPI backend container.

FROM python:3.11-slim

# Prevents .pyc files and buffers stdout/stderr immediately
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend source code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start the backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
