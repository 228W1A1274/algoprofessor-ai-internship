"""
advanced_rag.py — Day 15: HyDE, Self-Query & Multi-Query Retrieval
=================================================================
Author : AI/ML Mentor (Day 15 Curriculum)
Purpose: Demonstrates three advanced RAG retrieval strategies using LlamaIndex.

Techniques covered:
  1. HyDE  — Hypothetical Document Embeddings (zero-shot query expansion)
  2. Self-Query — LLM generates structured metadata filters on-the-fly
  3. Multi-Query — LLM expands one query into N variants; results are ensembled & deduplicated
"""

import os
from dotenv import load_dotenv

# ── LlamaIndex core ──────────────────────────────────────────────────────────
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine

# ── LLM & Embeddings ─────────────────────────────────────────────────────────
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# ── Self-Query (Metadata Filtering) ──────────────────────────────────────────
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.tools import FunctionTool

# ── Multi-Query ───────────────────────────────────────────────────────────────
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()  # Loads OPENAI_API_KEY from .env

# Global LlamaIndex settings — applied to every index/query engine automatically
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.chunk_size = 512          # Characters per chunk fed into the index
Settings.chunk_overlap = 64        # Overlap preserves context at chunk boundaries

DATA_DIR   = "./data"              # Put your .txt / .pdf files here
PERSIST_DIR = "./storage"          # LlamaIndex will cache the index here


# ─────────────────────────────────────────────────────────────────────────────
# 1. BUILD OR LOAD INDEX
# ─────────────────────────────────────────────────────────────────────────────

def get_index() -> VectorStoreIndex:
    """
    Build a VectorStoreIndex from ./data on first run, then reload from disk.
    LlamaIndex stores embeddings in ./storage so we don't re-embed every time.
    """
    if os.path.exists(PERSIST_DIR):
        log.info("Loading existing index from disk…")
        storage_ctx = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        return load_index_from_storage(storage_ctx)

    log.info("Building new index from documents in %s…", DATA_DIR)

    # SimpleDirectoryReader handles PDF, TXT, DOCX, HTML, etc. automatically
    documents = SimpleDirectoryReader(DATA_DIR).load_data()

    # Each document can carry metadata (filename, page_number, author, etc.)
    # We attach a simple 'category' tag here so Self-Query can filter on it.
    for doc in documents:
        doc.metadata["category"] = doc.metadata.get("file_name", "general")

    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    log.info("Index built and saved to %s", PERSIST_DIR)
    return index


# ─────────────────────────────────────────────────────────────────────────────
# 2. HyDE — Hypothetical Document Embeddings
# ─────────────────────────────────────────────────────────────────────────────

def hyde_query(index: VectorStoreIndex, question: str) -> str:
    """
    HyDE workflow:
      Step 1 — LLM generates a *hypothetical* answer to the question.
      Step 2 — That hypothetical answer is embedded (not the question itself).
      Step 3 — The embedding is used to search the vector store.

    WHY this works: A hypothetical answer lives in the same embedding space as
    real document chunks — much closer than a raw question does. This is
    especially powerful for zero-shot domains where question↔document similarity
    is naturally low (e.g., medical, legal, scientific corpora).
    """
    log.info("[HyDE] Processing question: %s", question)

    # HyDEQueryTransform intercepts the query, calls the LLM to hallucinate an
    # answer, then replaces the query embedding with that answer's embedding.
    hyde_transform = HyDEQueryTransform(
        include_original=True   # Also retrieve using the original query (safety net)
    )

    base_engine = index.as_query_engine(similarity_top_k=5)

    # TransformQueryEngine wraps any query engine with a pre-retrieval transform.
    # The transform runs BEFORE the vector search happens.
    hyde_engine = TransformQueryEngine(base_engine, hyde_transform)

    response = hyde_engine.query(question)

    log.info("[HyDE] Sources used: %d chunks", len(response.source_nodes))
    for i, node in enumerate(response.source_nodes):
        log.info("  [%d] score=%.3f | file=%s", i, node.score or 0, node.metadata.get("file_name", "?"))

    return str(response)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SELF-QUERY RETRIEVAL — LLM generates metadata filters
# ─────────────────────────────────────────────────────────────────────────────

def self_query(index: VectorStoreIndex, question: str) -> str:
    """
    Self-Query workflow:
      Step 1 — A structured LLM call parses the user question for filter intent.
      Step 2 — LLM outputs a JSON-like filter spec (e.g., category == "finance").
      Step 3 — Those filters are applied to the vector store BEFORE similarity search.

    WHY this matters: Pure vector search can't distinguish "only show me Q3 docs"
    from general similarity. Self-query gives you SQL-like precision on top of
    semantic search. Interview angle: "It's like adding a WHERE clause to ANN search."

    NOTE: LlamaIndex's full auto-filter extraction requires a vector store that
    supports metadata filtering (Qdrant, Weaviate, Pinecone, Chroma).
    Below we show manual filter construction for clarity, plus the LLM-driven approach.
    """
    log.info("[Self-Query] Processing question: %s", question)

    # ── Manual filter construction (for illustration / testing) ──────────────
    # In production, replace this with an LLM call that parses the question
    # and returns the filter spec dynamically.
    detected_filters = _llm_extract_filters(question)

    if detected_filters:
        log.info("[Self-Query] Detected filters: %s", detected_filters)
        metadata_filters = MetadataFilters(
            filters=[ExactMatchFilter(key=k, value=v) for k, v in detected_filters.items()]
        )
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=5,
            filters=metadata_filters,
        )
    else:
        log.info("[Self-Query] No filters detected — falling back to plain vector search")
        retriever = VectorIndexRetriever(index=index, similarity_top_k=5)

    query_engine = RetrieverQueryEngine.from_args(
        retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
    )
    response = query_engine.query(question)
    return str(response)


def _llm_extract_filters(question: str) -> dict:
    """
    Calls the LLM with a structured prompt to extract metadata filters from the question.
    Returns a dict like {"category": "finance"} or {} if no filters found.

    This is a simplified version; production code would use a Pydantic output parser.
    """
    prompt = f"""You are a metadata filter extractor. Given a user question, extract any 
explicit metadata constraints. Return ONLY valid JSON with keys matching document 
metadata fields (category, author, year, page_number). If no filters, return {{}}.

Question: {question}
JSON:"""

    llm = Settings.llm
    raw = llm.complete(prompt).text.strip()

    import json, re
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# 4. MULTI-QUERY RETRIEVAL — Query variants + ensemble + deduplication
# ─────────────────────────────────────────────────────────────────────────────

def multi_query(index: VectorStoreIndex, question: str, num_variants: int = 4) -> str:
    """
    Multi-Query workflow:
      Step 1 — LLM rewrites the original query into N distinct variants.
      Step 2 — Each variant is independently sent to the vector store.
      Step 3 — Results are pooled; duplicates are removed via reciprocal rank fusion (RRF).
      Step 4 — The merged, deduplicated nodes are passed to the LLM for synthesis.

    WHY this works: A single query may miss relevant documents due to vocabulary
    mismatch. Multiple phrasings cast a wider semantic net. RRF rewards nodes that
    appear across multiple variant result sets, boosting recall without sacrificing
    precision. Interview angle: "It's ensemble learning applied to information retrieval."
    """
    log.info("[Multi-Query] Expanding query into %d variants…", num_variants)

    # QueryFusionRetriever is LlamaIndex's built-in multi-query engine.
    # mode="reciprocal_rerank" → implements RRF fusion (Cormack et al., 2009)
    # mode="simple" → plain union with score averaging
    fusion_retriever = QueryFusionRetriever(
        retrievers=[VectorIndexRetriever(index=index, similarity_top_k=5)],
        llm=Settings.llm,
        num_queries=num_variants,          # Number of query variants to generate
        mode="reciprocal_rerank",          # RRF deduplication & reranking
        use_async=False,                   # Set True in async FastAPI/Streamlit contexts
        verbose=True,
    )

    query_engine = RetrieverQueryEngine.from_args(fusion_retriever)
    response = query_engine.query(question)

    log.info("[Multi-Query] Total unique source nodes after dedup: %d", len(response.source_nodes))
    return str(response)


# ─────────────────────────────────────────────────────────────────────────────
# 5. DEMO RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Make sure ./data has at least one document before running
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.listdir(DATA_DIR):
        # Create a tiny demo document so the script runs without real files
        with open(f"{DATA_DIR}/demo.txt", "w") as f:
            f.write(
                "LlamaIndex is a framework for building LLM-powered applications. "
                "It supports advanced RAG techniques including HyDE, multi-query retrieval, "
                "and graph-based retrieval. OpenAI GPT-4 is commonly used as the backbone LLM. "
                "Financial documents from Q3 2024 show strong performance in the AI sector. "
                "The category of this document is technology."
            )

    index = get_index()
    question = "What advanced retrieval techniques does LlamaIndex support?"

    print("\n" + "═" * 70)
    print("1️⃣  HyDE RESULT")
    print("═" * 70)
    print(hyde_query(index, question))

    print("\n" + "═" * 70)
    print("2️⃣  SELF-QUERY RESULT")
    print("═" * 70)
    print(self_query(index, "Show me technology documents about retrieval"))

    print("\n" + "═" * 70)
    print("3️⃣  MULTI-QUERY RESULT")
    print("═" * 70)
    print(multi_query(index, question))


if __name__ == "__main__":
    main()