"""
retrieval.py - All retrieval strategies for the RAG pipeline
Enterprise Knowledge Navigator - AlgoProfessor Internship 2026

Strategies implemented:
- Vector search (semantic similarity via ChromaDB)
- BM25 keyword search
- Hybrid search with RRF (Reciprocal Rank Fusion)
- HyDE (Hypothetical Document Embeddings)
- Multi-query retrieval
- Cross-encoder reranking
- Graph-enhanced search (uses KnowledgeGraph from pipeline.py)
"""

import re


def tokenize_bm25(text):
    """Tokenize text for BM25 keyword search."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 -]", " ", text)
    return [t for t in text.split() if len(t) > 1]


class Retriever:
    """
    Unified retriever combining all search strategies.
    
    Args:
        collection: ChromaDB collection
        bm25: BM25Okapi index
        all_chunks: list of all chunk dicts
        embedding_model: SentenceTransformer model
        reranker: CrossEncoder model
        groq_client: Groq client for HyDE and multi-query
    """

    def __init__(self, collection, bm25, all_chunks, embedding_model, reranker, groq_client):
        self.collection = collection
        self.bm25 = bm25
        self.all_chunks = all_chunks
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.groq_client = groq_client

    def embed(self, text):
        return self.embedding_model.encode(text, normalize_embeddings=True)

    def call_groq(self, prompt, max_tokens=200):
        response = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Stage 1: Candidate retrieval
    # ------------------------------------------------------------------

    def vector_search(self, query, n=10):
        """Semantic vector search against ChromaDB."""
        qe = self.embed(query).tolist()
        results = self.collection.query(
            query_embeddings=[qe],
            n_results=n,
            include=["documents", "metadatas", "distances"]
        )
        return [
            {"text": d, "metadata": m, "score": round(1 - dist, 4)}
            for d, m, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]

    def bm25_search(self, query, n=10):
        """BM25 keyword search."""
        tokens = tokenize_bm25(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [
            {
                "text": self.all_chunks[i]["text"],
                "metadata": self.all_chunks[i]["metadata"],
                "score": round(float(scores[i]), 4)
            }
            for i in top_idx if scores[i] > 0
        ]

    def hybrid_search(self, query, n_final=10, n_retrieve=20):
        """
        Combine BM25 and vector search with Reciprocal Rank Fusion.
        RRF score = sum of 1/(rank + 60) across result lists.
        """
        vec = self.vector_search(query, n=n_retrieve)
        bm = self.bm25_search(query, n=n_retrieve)
        rrf, data = {}, {}
        for rank, r in enumerate(vec):
            cid = r["metadata"]["chunk_id"]
            rrf[cid] = rrf.get(cid, 0) + 1.0 / (rank + 60)
            data[cid] = r
        for rank, r in enumerate(bm):
            cid = r["metadata"]["chunk_id"]
            rrf[cid] = rrf.get(cid, 0) + 1.0 / (rank + 60)
            data[cid] = r
        sorted_ids = sorted(rrf.keys(), key=lambda k: rrf[k], reverse=True)
        results = []
        for cid in sorted_ids[:n_final]:
            item = dict(data[cid])
            item["rrf_score"] = round(rrf[cid], 6)
            results.append(item)
        return results

    def hyde_search(self, query, n=10):
        """
        HyDE: generate a hypothetical answer, embed it, search with that embedding.
        Better retrieval because hypothetical answer text resembles real document text.
        """
        prompt = (
            "Write a short factual paragraph answering this question about Tesla.\n"
            "Question: " + query + "\n"
            "Write only the answer paragraph, no introduction."
        )
        hypothetical = self.call_groq(prompt, max_tokens=150)
        emb = self.embed(hypothetical).tolist()
        results = self.collection.query(
            query_embeddings=[emb],
            n_results=n,
            include=["documents", "metadatas", "distances"]
        )
        return [
            {"text": d, "metadata": m, "score": round(1 - dist, 4)}
            for d, m, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]

    def multi_query_search(self, query, n_per_query=8):
        """
        Generate 3 query variants and merge results.
        Increases recall by covering different phrasings of the same question.
        """
        prompt = (
            "Generate 3 different search queries to find information about this question.\n"
            "Original: " + query + "\n"
            "Return exactly 3 queries, one per line, no numbering."
        )
        variants_text = self.call_groq(prompt, max_tokens=100)
        variants = [q.strip() for q in variants_text.strip().split("\n") if len(q.strip()) > 5][:3]
        all_variants = [query] + variants
        seen = {}
        for variant in all_variants:
            for r in self.vector_search(variant, n=n_per_query):
                cid = r["metadata"]["chunk_id"]
                if cid not in seen or r["score"] > seen[cid]["score"]:
                    seen[cid] = r
        return sorted(seen.values(), key=lambda x: x["score"], reverse=True)

    # ------------------------------------------------------------------
    # Stage 2: Reranking
    # ------------------------------------------------------------------

    def rerank(self, query, chunks, top_n=5):
        """
        Cross-encoder reranking for precise relevance scoring.
        Looks at query + chunk together. Much more accurate than bi-encoder.
        """
        if not chunks:
            return []
        scores = self.reranker.predict([(query, c["text"]) for c in chunks])
        for i, c in enumerate(chunks):
            c["rerank_score"] = round(float(scores[i]), 4)
        return sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)[:top_n]

    # ------------------------------------------------------------------
    # Full retrieval pipeline
    # ------------------------------------------------------------------

    def retrieve(self, query, use_hyde=True, use_multiquery=True, top_n=5):
        """
        Full retrieval: HyDE + MultiQuery + Hybrid + Rerank.
        Returns top_n most relevant chunks.
        """
        all_candidates = {}

        if use_hyde:
            for r in self.hyde_search(query, n=10):
                cid = r["metadata"]["chunk_id"]
                if cid not in all_candidates:
                    all_candidates[cid] = r

        if use_multiquery:
            for r in self.multi_query_search(query, n_per_query=8):
                cid = r["metadata"]["chunk_id"]
                if cid not in all_candidates:
                    all_candidates[cid] = r

        for r in self.hybrid_search(query, n_final=15):
            cid = r["metadata"]["chunk_id"]
            if cid not in all_candidates:
                all_candidates[cid] = r

        candidates = list(all_candidates.values())
        return self.rerank(query, candidates, top_n=top_n)
