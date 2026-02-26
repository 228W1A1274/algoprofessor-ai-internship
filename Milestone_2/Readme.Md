# Enterprise Knowledge Navigator

## Milestone Project 2 - AlgoProfessor Internship 2026

## Project Overview

AI knowledge worker using advanced RAG to answer
company questions from 50+ Tesla documents.

## Architecture

User Query → HyDE + Multi-Query → Hybrid Search (BM25+Vector)
→ Cross-Encoder Rerank → Groq Llama 3 → Cited Answer

## Tech Stack

- LLM: Groq Llama 3 70B (free)
- Embeddings: sentence-transformers all-MiniLM-L6-v2
- Vector DB: ChromaDB (persistent)
- Keyword Search: BM25Okapi
- Reranking: Cross-Encoder ms-marco-MiniLM-L-6-v2
- Backend: FastAPI
- Frontend: Streamlit
- Evaluation: RAGAS

## Features

- 50+ Tesla company documents ingested
- Graph RAG with Neo4j knowledge graph
- HyDE + Multi-query retrieval
- Hybrid BM25 + Vector search with RRF fusion
- Conversational memory: 10-turn window
- Live RAGAS evaluation dashboard

## How to Run

1. Open notebook in Google Colab
2. Run master restart cell
3. Add Groq API key in Step 2
4. Run all cells in order

## Results

- Total chunks indexed: 500+
- Embedding dimensions: 384
- RAGAS faithfulness score: (add after Day 4)

```

---

**requirements.txt:**
```

sentence-transformers
chromadb
rank-bm25
groq
pymupdf
pdfplumber
fastapi
uvicorn
streamlit
ragas
requests
beautifulsoup4
