# mcp_servers/news/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY news_server.py .

EXPOSE 8002

CMD ["uvicorn", "news_server:app", "--host", "0.0.0.0", "--port", "8002"]