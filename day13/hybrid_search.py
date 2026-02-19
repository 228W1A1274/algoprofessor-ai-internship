# =============================================================================
# hybrid_search.py
# PURPOSE: Perform hybrid semantic search using:
#          1. Dense Retrieval  — vector similarity via ChromaDB (semantic meaning)
#          2. Sparse Retrieval — BM25 keyword matching (exact term overlap)
#          3. Reciprocal Rank Fusion (RRF) — merge the two ranked lists
#          4. Cross-Encoder Reranking — final precision pass with a heavier model
#
# WHY HYBRID?
#   - Dense-only: misses exact keyword matches ("BM25 algorithm" won't hit "BM25")
#   - Sparse-only: misses semantic meaning ("car" won't match "automobile")
#   - Hybrid: best of both worlds — the standard in production RAG systems.
# =============================================================================

import os
from dotenv import load_dotenv

# --- Embedding model (same one used to build the index) ---
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- BM25 for sparse retrieval ---
# BM25Okapi is a Python implementation of the BM25 ranking algorithm.
# It tokenizes documents and scores them by term frequency + inverse document frequency.
from rank_bm25 import BM25Okapi

# --- ChromaDB for dense retrieval ---
import chromadb

# --- Import corpus from embedding_gen.py ---
from embedding_gen import DOCUMENTS, DOC_IDS, MODEL_NAME

# Load .env for Pinecone key (kept here in case you want to add Pinecone search later)
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

CHROMA_PERSIST_PATH = "./chroma_db_storage"          # Must match vector_db_setup.py
CHROMA_COLLECTION_NAME = "semantic_search_collection" # Must match vector_db_setup.py

# Number of candidates each retriever fetches before fusion & reranking.
# We fetch more than we need (top-10) so the reranker has a good pool to work with.
TOP_K_RETRIEVAL = 10

# Final number of results to display after reranking.
TOP_K_FINAL = 3

# Cross-encoder model for reranking. This is a heavier model that takes a
# (query, document) pair and outputs a relevance score from -inf to +inf.
# It's more accurate than cosine similarity but too slow to run on all docs.
# "ms-marco-MiniLM-L-6-v2" is a well-tested, lightweight reranker.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# =============================================================================
# COMPONENT 1: Dense Retriever (ChromaDB)
# =============================================================================

class DenseRetriever:
    """
    Retrieves documents by semantic similarity using pre-computed embeddings.
    Uses ChromaDB's HNSW index for approximate nearest neighbor search.
    """
    
    def __init__(self, model: SentenceTransformer):
        # Load the same embedding model used during indexing.
        # CRITICAL: If you use a different model here, embeddings won't be comparable.
        self.model = model
        
        # Connect to the existing on-disk ChromaDB (must have run vector_db_setup.py first)
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
        self.collection = self.client.get_collection(name=CHROMA_COLLECTION_NAME)
        print(f"[Dense] Connected to ChromaDB. Collection has {self.collection.count()} documents.")
    
    def retrieve(self, query: str, top_k: int) -> list[dict]:
        """
        Embed the query and find the most similar documents.
        
        Args:
            query: The user's natural language question.
            top_k: Number of results to return.
        Returns:
            List of dicts with 'id', 'text', and 'score' keys.
        """
        # Embed the query using the same model that embedded the documents.
        # The resulting vector is then compared against all stored vectors.
        query_embedding = self.model.encode(query).tolist()
        
        # query() does the actual ANN (Approximate Nearest Neighbor) search.
        # n_results: how many closest vectors to return.
        # include: what data to return alongside the vectors.
        results = self.collection.query(
            query_embeddings=[query_embedding],  # Note: expects a list of queries
            n_results=top_k,
            include=["documents", "distances"]   # distances = 1 - cosine_similarity
        )
        
        # Unpack ChromaDB's nested result format.
        # results["ids"] = [["doc_2", "doc_7", ...]]  (outer list = one per query)
        ids = results["ids"][0]
        texts = results["documents"][0]
        distances = results["distances"][0]
        
        # Convert distance (lower=better) to similarity score (higher=better).
        # ChromaDB with cosine space returns distance = 1 - cosine_sim,
        # so similarity = 1 - distance.
        retrieved = []
        for doc_id, text, dist in zip(ids, texts, distances):
            retrieved.append({
                "id": doc_id,
                "text": text,
                "score": 1.0 - dist   # Convert to similarity (higher = more relevant)
            })
        
        return retrieved


# =============================================================================
# COMPONENT 2: Sparse Retriever (BM25)
# =============================================================================

class SparseRetriever:
    """
    Retrieves documents using BM25, a classic TF-IDF-based keyword ranking algorithm.
    BM25 excels at finding documents with strong exact term overlap with the query.
    """
    
    def __init__(self, documents: list[str], doc_ids: list[str]):
        self.documents = documents
        self.doc_ids = doc_ids
        
        # Tokenize each document into a list of lowercase words.
        # BM25Okapi expects a list of token lists (not raw strings).
        # Example: "HR department" → ["hr", "department"]
        tokenized_corpus = [doc.lower().split() for doc in documents]
        
        # Build the BM25 index from the tokenized corpus.
        # This computes term frequencies and inverse document frequencies internally.
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"[Sparse] BM25 index built over {len(documents)} documents.")
    
    def retrieve(self, query: str, top_k: int) -> list[dict]:
        """
        Score all documents against the query using BM25 and return top results.
        
        Args:
            query: The user's natural language question.
            top_k: Number of top results to return.
        Returns:
            List of dicts with 'id', 'text', and 'score' keys.
        """
        # Tokenize the query the same way we tokenized the corpus.
        tokenized_query = query.lower().split()
        
        # get_scores() returns a score for EVERY document in the corpus.
        # Higher score = more relevant. Zero = no keyword overlap.
        scores = self.bm25.get_scores(tokenized_query)
        
        # Pair each score with its index, sort descending, and take top_k.
        scored_docs = sorted(
            enumerate(scores),
            key=lambda x: x[1],  # Sort by score
            reverse=True          # Highest first
        )[:top_k]
        
        retrieved = []
        for idx, score in scored_docs:
            retrieved.append({
                "id": self.doc_ids[idx],
                "text": self.documents[idx],
                "score": float(score)
            })
        
        return retrieved


# =============================================================================
# COMPONENT 3: Reciprocal Rank Fusion (RRF)
# =============================================================================

def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60
) -> list[dict]:
    """
    Merge two ranked lists into one using Reciprocal Rank Fusion (RRF).
    
    HOW RRF WORKS:
    Instead of trying to normalize scores (hard because BM25 and cosine scales differ),
    RRF uses only the *rank position* of each document.
    
    For each document, its RRF score = sum of 1/(k + rank_in_each_list)
    
    - k=60 is the standard constant (proposed in the original RRF paper).
      It dampens the advantage of being ranked #1 over #2, making the fusion
      more robust to outliers.
    - A document ranked #1 in both lists gets score: 1/61 + 1/61 ≈ 0.033
    - A document ranked #1 in one and #5 in other: 1/61 + 1/65 ≈ 0.032
    
    Args:
        dense_results: Ranked list from dense retrieval.
        sparse_results: Ranked list from sparse retrieval.
        k: RRF constant (default 60 per the original paper).
    Returns:
        Combined, re-ranked list of unique documents with RRF scores.
    """
    # Track the cumulative RRF score for each doc ID.
    rrf_scores: dict[str, float] = {}
    # Track the text for each doc ID so we can reconstruct results.
    doc_texts: dict[str, str] = {}
    
    # Process dense results: enumerate gives (rank_0_indexed, result_dict)
    for rank, result in enumerate(dense_results):
        doc_id = result["id"]
        # RRF score contribution from this list: 1 / (k + rank+1)
        # +1 because ranks are 1-indexed in the formula (rank 0 → position 1)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        doc_texts[doc_id] = result["text"]
    
    # Process sparse results: same formula, scores accumulate.
    for rank, result in enumerate(sparse_results):
        doc_id = result["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        doc_texts[doc_id] = result["text"]
    
    # Sort by RRF score descending to get the fused ranking.
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [
        {"id": doc_id, "text": doc_texts[doc_id], "rrf_score": score}
        for doc_id, score in fused
    ]


# =============================================================================
# COMPONENT 4: Cross-Encoder Reranker
# =============================================================================

class Reranker:
    """
    Reranks the fused candidate list using a cross-encoder model.
    
    DIFFERENCE FROM BI-ENCODER (SBERT):
    - Bi-encoder (used for retrieval): embeds query and doc SEPARATELY, compares vectors.
      Fast, but less accurate — it can't attend to interactions between query & doc words.
    
    - Cross-encoder (used for reranking): processes query+doc TOGETHER in one forward pass.
      Slower (can't pre-compute), but far more accurate for relevance scoring.
    
    TYPICAL PIPELINE:
      Retrieve 100 candidates cheaply → Rerank top 100 → Return top 5 precisely.
    """
    
    def __init__(self, model_name: str):
        print(f"[Reranker] Loading cross-encoder model: '{model_name}'")
        # CrossEncoder from sentence-transformers loads a BERT-based classifier.
        # It outputs a single float score per (query, document) pair.
        self.model = CrossEncoder(model_name)
        print("[Reranker] Cross-encoder ready.")
    
    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """
        Score each candidate against the query and return the top_k highest scored.
        
        Args:
            query: The user's query string.
            candidates: List of candidate documents (output of RRF fusion).
            top_k: How many to return after reranking.
        Returns:
            Re-ranked list of top_k documents with cross-encoder scores.
        """
        if not candidates:
            return []
        
        # Create (query, document) pairs for the cross-encoder.
        # The model needs both together to perform joint attention.
        query_doc_pairs = [(query, candidate["text"]) for candidate in candidates]
        
        # predict() runs all pairs through the cross-encoder in one batch.
        # Returns a numpy array of raw logit scores (not probabilities).
        # Higher score = more relevant. No fixed scale.
        scores = self.model.predict(query_doc_pairs, show_progress_bar=False)
        
        # Attach scores back to the candidate dicts.
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores[i])
        
        # Sort by cross-encoder score descending and return top_k.
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]


# =============================================================================
# FULL HYBRID SEARCH PIPELINE
# =============================================================================

def hybrid_search(
    query: str,
    dense_retriever: DenseRetriever,
    sparse_retriever: SparseRetriever,
    reranker: Reranker,
    top_k_retrieval: int = TOP_K_RETRIEVAL,
    top_k_final: int = TOP_K_FINAL
) -> list[dict]:
    """
    End-to-end hybrid search: Dense + Sparse → RRF Fusion → Cross-Encoder Rerank.
    
    Args:
        query: The user's search query.
        dense_retriever: Initialized DenseRetriever object.
        sparse_retriever: Initialized SparseRetriever object.
        reranker: Initialized Reranker object.
        top_k_retrieval: How many candidates each retriever fetches.
        top_k_final: How many results to show after reranking.
    Returns:
        Final ranked list of the most relevant documents.
    """
    print(f"\n{'='*60}")
    print(f"QUERY: \"{query}\"")
    print(f"{'='*60}")
    
    # --- Stage 1: Dense Retrieval ---
    print(f"\n[Stage 1] Dense retrieval (top {top_k_retrieval})...")
    dense_results = dense_retriever.retrieve(query, top_k=top_k_retrieval)
    print(f"  Top dense result: '{dense_results[0]['text'][:60]}...' (score: {dense_results[0]['score']:.4f})")
    
    # --- Stage 2: Sparse BM25 Retrieval ---
    print(f"\n[Stage 2] Sparse BM25 retrieval (top {top_k_retrieval})...")
    sparse_results = sparse_retriever.retrieve(query, top_k=top_k_retrieval)
    print(f"  Top sparse result: '{sparse_results[0]['text'][:60]}...' (BM25: {sparse_results[0]['score']:.4f})")
    
    # --- Stage 3: Reciprocal Rank Fusion ---
    print(f"\n[Stage 3] Fusing results with Reciprocal Rank Fusion (RRF)...")
    fused_results = reciprocal_rank_fusion(dense_results, sparse_results)
    print(f"  {len(fused_results)} unique candidates after fusion.")
    
    # --- Stage 4: Cross-Encoder Reranking ---
    print(f"\n[Stage 4] Reranking top {len(fused_results)} candidates with cross-encoder...")
    final_results = reranker.rerank(query, fused_results, top_k=top_k_final)
    
    return final_results


def display_results(results: list[dict], query: str):
    """Pretty-print the final search results."""
    print(f"\n{'='*60}")
    print(f"FINAL TOP {len(results)} RESULTS for: \"{query}\"")
    print(f"{'='*60}")
    for i, result in enumerate(results, start=1):
        print(f"\n#{i} [Rerank Score: {result['rerank_score']:.4f}] [RRF Score: {result['rrf_score']:.4f}]")
        print(f"    ID: {result['id']}")
        print(f"    Text: {result['text']}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # --- Initialize all components once ---
    # (Loading models is expensive; do it once and reuse across queries)
    
    print("[INIT] Loading embedding model for dense retrieval...")
    embedding_model = SentenceTransformer(MODEL_NAME)
    
    print("[INIT] Setting up dense retriever (ChromaDB)...")
    dense_retriever = DenseRetriever(model=embedding_model)
    
    print("[INIT] Setting up sparse retriever (BM25)...")
    sparse_retriever = SparseRetriever(documents=DOCUMENTS, doc_ids=DOC_IDS)
    
    print("[INIT] Loading cross-encoder reranker...")
    reranker = Reranker(model_name=RERANKER_MODEL)
    
    # --- Run example queries ---
    # These test different aspects of hybrid search:
    
    # Query 1: Semantic — "leave policy" semantically relates to "remote work" even without exact word match
    query1 = "What is the company leave and remote work policy?"
    results1 = hybrid_search(query1, dense_retriever, sparse_retriever, reranker)
    display_results(results1, query1)
    
    # Query 2: Technical keyword — "BM25" and "HNSW" are exact terms that sparse retrieval handles well
    query2 = "How does HNSW algorithm work in vector databases?"
    results2 = hybrid_search(query2, dense_retriever, sparse_retriever, reranker)
    display_results(results2, query2)
    
    # Query 3: HR domain — tests whether the system blends HR semantic concepts correctly
    query3 = "employee health insurance and benefits enrollment"
    results3 = hybrid_search(query3, dense_retriever, sparse_retriever, reranker)
    display_results(results3, query3)
    
    print(f"\n\n✅ Hybrid search complete! All 3 queries processed.")