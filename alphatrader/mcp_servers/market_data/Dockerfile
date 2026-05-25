# mcp_servers/market_data/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY market_data_server.py .

EXPOSE 8001

CMD ["uvicorn", "market_data_server:app", "--host", "0.0.0.0", "--port", "8001"]