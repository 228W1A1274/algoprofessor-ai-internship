# 📚 API Documentation - HCL Knowledge Assistant

Complete API reference for the HCL Knowledge Assistant backend.

**Base URL**: `http://localhost:8000`  
**API Version**: v1  
**Authentication**: Bearer Token (JWT)

---

## 📋 Table of Contents

1. [Authentication](#authentication)
2. [Knowledge Queries](#knowledge-queries)
3. [User Management](#user-management)
4. [System Endpoints](#system-endpoints)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [Examples](#examples)

---

## 🔐 Authentication

All API endpoints (except `/auth/login` and `/health`) require authentication.

### POST /api/v1/auth/login

Login and receive JWT token.

**Request:**
```http
POST /api/v1/auth/login HTTP/1.1
Content-Type: application/x-www-form-urlencoded

username=demo&password=demo123
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**Status Codes:**
- `200 OK` - Success
- `401 Unauthorized` - Invalid credentials

**curl Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -d "username=demo&password=demo123"
```

---

## 💬 Knowledge Queries

### POST /api/v1/query

Submit a query to the knowledge base.

**Authentication**: Required

**Request:**
```http
POST /api/v1/query HTTP/1.1
Authorization: Bearer <token>
Content-Type: application/json

{
  "query": "What is the leave policy?",
  "mode": "rag",
  "max_results": 5
}
```

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| query | string | Yes | User question (1-1000 chars) |
| mode | string | No | Query mode: 'rag', 'agent', or 'hybrid'. Default: 'rag' |
| max_results | integer | No | Max documents to retrieve (1-20). Default: 5 |

**Response:**
```json
{
  "answer": "Based on HCL's knowledge base: Annual leave: 25 days per year...",
  "sources": [
    {
      "content": "Leave Policy\nAnnual leave: 25 days per year...",
      "score": 5
    }
  ],
  "confidence": 0.85,
  "mode_used": "rag",
  "processing_time": 1.23,
  "timestamp": "2025-02-12T10:30:00Z"
}
```

**Response Fields:**
| Field | Type | Description |
|-------|------|-------------|
| answer | string | Generated answer |
| sources | array | Retrieved source documents |
| confidence | float | Confidence score (0-1) |
| mode_used | string | Query mode used |
| processing_time | float | Processing time in seconds |
| timestamp | string | Response timestamp (ISO 8601) |

**Status Codes:**
- `200 OK` - Success
- `400 Bad Request` - Invalid request
- `401 Unauthorized` - Missing/invalid token
- `500 Internal Server Error` - Processing error

**curl Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the leave policy?",
    "mode": "rag",
    "max_results": 5
  }'
```

**Python Example:**
```python
import requests

# Login
response = requests.post(
    "http://localhost:8000/api/v1/auth/login",
    data={"username": "demo", "password": "demo123"}
)
token = response.json()["access_token"]

# Query
response = requests.post(
    "http://localhost:8000/api/v1/query",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "query": "What is the leave policy?",
        "mode": "rag",
        "max_results": 5
    }
)
result = response.json()
print(result["answer"])
```

**Query Modes:**

1. **RAG Mode** (`mode="rag"`)
   - Fast document retrieval
   - Best for factual queries
   - Returns source documents

2. **Agent Mode** (`mode="agent"`)
   - Intelligent reasoning
   - Classified by agent type
   - Best for complex queries

3. **Hybrid Mode** (`mode="hybrid"`)
   - Combines both approaches
   - Highest accuracy
   - Slightly slower

---

## 📊 System Endpoints

### GET /api/v1/health

Health check endpoint.

**Authentication**: Not required

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-02-12T10:30:00Z",
  "version": "1.0.0",
  "services": {
    "rag_system": "online",
    "agent_system": "online",
    "database": "online"
  }
}
```

**curl Example:**
```bash
curl "http://localhost:8000/api/v1/health"
```

### GET /api/v1/stats

Get system statistics.

**Authentication**: Required

**Response:**
```json
{
  "total_queries": 142,
  "knowledge_base_size": 45,
  "active_users": 5,
  "timestamp": "2025-02-12T10:30:00Z"
}
```

### GET /api/v1/history

Get user's query history.

**Authentication**: Required

**Response:**
```json
{
  "history": [
    {
      "user": "demo",
      "query": "What is the leave policy?",
      "mode": "rag",
      "timestamp": "2025-02-12T10:25:00Z"
    },
    {
      "user": "demo",
      "query": "Tell me about HCL DX",
      "mode": "agent",
      "timestamp": "2025-02-12T10:28:00Z"
    }
  ]
}
```

---

## 📄 Document Management

### POST /api/v1/documents

Upload a new document to the knowledge base.

**Authentication**: Required

**Request:**
```json
{
  "title": "New Product Guide",
  "content": "This guide covers...",
  "category": "product",
  "metadata": {
    "author": "John Doe",
    "version": "1.0"
  }
}
```

**Response:**
```json
{
  "message": "Document uploaded successfully",
  "id": "doc_123"
}
```

---

## ❌ Error Handling

### Error Response Format

```json
{
  "error": "Error description",
  "timestamp": "2025-02-12T10:30:00Z"
}
```

### Common Error Codes

| Code | Meaning | Solution |
|------|---------|----------|
| 400 | Bad Request | Check request format |
| 401 | Unauthorized | Login or refresh token |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Check endpoint URL |
| 429 | Too Many Requests | Slow down requests |
| 500 | Internal Server Error | Contact support |

### Example Error Responses

**401 Unauthorized:**
```json
{
  "detail": "Invalid authentication credentials",
  "timestamp": "2025-02-12T10:30:00Z"
}
```

**400 Bad Request:**
```json
{
  "detail": "Query must be between 1 and 1000 characters",
  "timestamp": "2025-02-12T10:30:00Z"
}
```

---

## ⚡ Rate Limiting

Default limits:
- **60 requests per minute** per user
- **1000 requests per hour** per user

Response headers:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1644667800
```

When limit exceeded:
```json
{
  "error": "Rate limit exceeded. Try again in 42 seconds.",
  "timestamp": "2025-02-12T10:30:00Z"
}
```

---

## 📝 Complete Examples

### Example 1: Basic Query Flow

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Login
login_response = requests.post(
    f"{BASE_URL}/api/v1/auth/login",
    data={"username": "demo", "password": "demo123"}
)
token = login_response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# 2. Query knowledge base
query_response = requests.post(
    f"{BASE_URL}/api/v1/query",
    headers=headers,
    json={
        "query": "What is HCL DX?",
        "mode": "rag"
    }
)
result = query_response.json()
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']*100}%")
print(f"Processing time: {result['processing_time']}s")

# 3. Get statistics
stats_response = requests.get(
    f"{BASE_URL}/api/v1/stats",
    headers=headers
)
stats = stats_response.json()
print(f"Total queries: {stats['total_queries']}")
```

### Example 2: Multiple Query Modes

```python
queries = [
    ("What is the leave policy?", "rag"),
    ("How do I request hardware?", "agent"),
    ("Tell me about HCL products", "hybrid")
]

for query_text, mode in queries:
    response = requests.post(
        f"{BASE_URL}/api/v1/query",
        headers=headers,
        json={"query": query_text, "mode": mode}
    )
    result = response.json()
    print(f"\nQuery: {query_text}")
    print(f"Mode: {mode}")
    print(f"Answer: {result['answer'][:200]}...")
```

### Example 3: Error Handling

```python
try:
    response = requests.post(
        f"{BASE_URL}/api/v1/query",
        headers=headers,
        json={"query": "Q" * 2000}  # Too long
    )
    response.raise_for_status()
except requests.exceptions.HTTPError as e:
    error = e.response.json()
    print(f"Error: {error['detail']}")
```

---

## 🧪 Testing with cURL

### Complete Test Suite

```bash
# 1. Health check
curl "http://localhost:8000/api/v1/health"

# 2. Login
TOKEN=$(curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -d "username=demo&password=demo123" | jq -r '.access_token')

# 3. Query (RAG mode)
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the leave policy?",
    "mode": "rag"
  }'

# 4. Query (Agent mode)
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tell me about HCL DX pricing",
    "mode": "agent"
  }'

# 5. Get statistics
curl "http://localhost:8000/api/v1/stats" \
  -H "Authorization: Bearer $TOKEN"

# 6. Get history
curl "http://localhost:8000/api/v1/history" \
  -H "Authorization: Bearer $TOKEN"
```

---

## 📖 OpenAPI/Swagger Documentation

Interactive API documentation is available at:

- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

These provide:
- Interactive API testing
- Complete schema documentation
- Request/response examples
- Authentication testing

---

## 🔗 Additional Resources

- [Deployment Guide](deployment_guide.md)
- [User Manual](user_manual.pdf)
- [README](../README.md)
- [GitHub Repository](#)

---

## 📞 Support

For API support:
- **Email**: api-support@hcl.com
- **Documentation**: http://localhost:8000/api/docs
- **Issues**: GitHub Issues

---

**API Version**: 1.0.0  
**Last Updated**: February 2025  
**Status**: Production Ready ✅
