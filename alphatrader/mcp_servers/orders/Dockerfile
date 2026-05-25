# mcp_servers/orders/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY orders_server.py .

EXPOSE 8004

CMD ["uvicorn", "orders_server:app", "--host", "0.0.0.0", "--port", "8004"]