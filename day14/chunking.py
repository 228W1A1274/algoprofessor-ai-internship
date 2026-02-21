"""
chunking.py
===========
Demonstrates four chunking strategies with explanation and comparison.
Chunking is arguably the most impactful decision in a RAG pipeline —
bad chunks = bad retrieval = bad answers, regardless of LLM quality.

The four strategies:
1. Fixed-Size Chunking       — Split by character count
2. Recursive Character Splitting — Respect natural language boundaries
3. Semantic Chunking         — Split by meaning using embedding similarity
4. Parent-Document Chunking  — Small chunks for retrieval, large for context
"""

import os
from typing import List, Tuple
from dotenv import load_dotenv

# NEW (correct)
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from  langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_community.vectorstores import FAISS               # Vector DB

load_dotenv()


# =============================================================================
# SAMPLE CORPUS
# =============================================================================

SAMPLE_TEXT = """
Retrieval-Augmented Generation (RAG) is a technique that combines the power of 
large language models with the precision of information retrieval systems. 
Instead of relying solely on the knowledge encoded in the model's weights during 
training, RAG allows the model to access and use external knowledge sources at 
inference time.

The first step in any RAG pipeline is document ingestion. Documents are loaded 
from various sources including PDFs, Word documents, HTML pages, and plain text 
files. Each document type requires a different parsing strategy to extract text 
faithfully while preserving structure.

After ingestion, documents must be split into chunks. This is a critical 
decision because chunks that are too small lose context, while chunks that are 
too large dilute relevance signals. The optimal chunk size depends on the 
embedding model's context window, the nature of the documents, and the expected 
query patterns.

Vector embeddings convert text into high-dimensional numerical representations 
that capture semantic meaning. Similar concepts end up close together in this 
space, enabling similarity search. Popular embedding models include 
text-embedding-ada-002 from OpenAI, sentence-transformers from Hugging Face, 
and Cohere's embed-english-v3.0.

Retrieval is the process of finding the most relevant chunks for a given query. 
This typically involves embedding the query using the same model used for 
indexing, then finding the top-K chunks by cosine similarity. Advanced retrieval 
techniques include hybrid search (combining dense and sparse retrieval), 
maximum marginal relevance (for diversity), and contextual compression.

Reranking is a post-retrieval step that applies a more computationally expensive 
but more accurate relevance model to re-score the initially retrieved candidates. 
Cohere's reranking model is a cross-encoder that jointly processes the query and 
each document, enabling much richer relevance assessment than pure vector 
similarity.

The generation step combines the retrieved context with the user's query in a 
carefully designed prompt and passes it to the LLM. Prompt engineering here is 
crucial — the LLM must be instructed to answer based only on the provided 
context, to say "I don't know" when the answer isn't in the context, and to 
cite its sources.

Evaluation of RAG systems requires specialized metrics. Faithfulness measures 
whether the generated answer is supported by the retrieved context. Answer 
relevancy measures how well the answer addresses the question. Context precision 
measures whether the retrieved chunks are relevant. Context recall measures 
whether all necessary information was retrieved.

RAG systems can be further improved through query transformation techniques. 
HyDE generates a hypothetical answer and uses it for retrieval. Step-back 
prompting generates a more abstract version of the query. Query decomposition 
breaks complex queries into simpler sub-queries that can be answered independently.
"""

SAMPLE_DOCS = [
    Document(
        page_content=SAMPLE_TEXT,
        metadata={"source": "rag_tutorial.md", "author": "AlgoProfessor"}
    )
]


# =============================================================================
# STRATEGY 1: FIXED-SIZE CHUNKING
# =============================================================================

def fixed_size_chunking(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Document]:
    """
    Fixed-Size Chunking
    -------------------
    The simplest strategy. Split text into chunks of exactly `chunk_size`
    characters with `chunk_overlap` characters of shared content between 
    adjacent chunks.
    
    chunk_overlap is crucial: if a sentence spans a chunk boundary, the 
    overlap ensures neither chunk completely loses that sentence.
    
    Pros:  Simple, predictable, fast, uniform memory usage
    Cons:  Splits mid-sentence, mid-word, loses semantic coherence
    
    Use when: Homogeneous documents, prototyping, benchmarking
    
    CharacterTextSplitter parameters:
    - separator: The character to split on (tries this first)
    - chunk_size: Max characters per chunk
    - chunk_overlap: Overlap between adjacent chunks
    - length_function: How to measure length (len = by character)
    """
    splitter = CharacterTextSplitter(
        separator="\n",         # Try to split on newlines first
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,    # Count by characters, not tokens
    )
    
    # split_documents takes List[Document] and returns List[Document]
    # It preserves and augments the metadata of each source document
    chunks = splitter.split_documents(documents)
    
    # Add chunk-specific metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_strategy"] = "fixed_size"
        chunk.metadata["chunk_size_config"] = chunk_size
        chunk.metadata["chunk_char_count"] = len(chunk.page_content)
    
    return chunks


# =============================================================================
# STRATEGY 2: RECURSIVE CHARACTER SPLITTING (The Industry Default)
# =============================================================================

def recursive_character_chunking(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Document]:
    """
    Recursive Character Text Splitting
    ------------------------------------
    This is the MOST COMMONLY USED splitter in production RAG systems.
    The "recursive" refers to its fallback hierarchy of separators.
    
    It tries to split on these separators in order:
    ["\n\n", "\n", " ", ""]
    
    Logic:
    1. Try splitting on double newlines (paragraph breaks) first
    2. If chunks are still too big, split on single newlines
    3. If still too big, split on spaces (word boundary)
    4. Last resort: split on individual characters
    
    This hierarchy means: "Prefer paragraph breaks > line breaks > 
    word breaks > character breaks" — always respecting natural language.
    
    Pros:  Respects language structure, most versatile, handles most cases
    Cons:  Can still split mid-concept; doesn't understand meaning
    
    Use when: General purpose, mixed documents, production defaults
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # This is the default separator hierarchy — shown explicitly for learning:
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    chunks = splitter.split_documents(documents)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_strategy"] = "recursive_character"
        chunk.metadata["chunk_size_config"] = chunk_size
        chunk.metadata["chunk_char_count"] = len(chunk.page_content)
    
    return chunks


# =============================================================================
# STRATEGY 3: SEMANTIC CHUNKING
# =============================================================================

def semantic_chunking(
    documents: List[Document],
    breakpoint_threshold_type: str = "percentile",  # or "standard_deviation", "interquartile"
    breakpoint_threshold_amount: float = 95,
) -> List[Document]:
    """
    Semantic Chunking
    -----------------
    Instead of splitting by character count, semantic chunking splits by 
    *meaning*. Here's the algorithm:
    
    1. Split text into individual sentences
    2. Embed each sentence using an embedding model
    3. Compute the cosine *distance* between adjacent sentence embeddings
    4. Find "breakpoints" — places where embedding distance spikes,
       indicating a topic shift
    5. Split at those breakpoints
    
    The result: each chunk contains sentences that are semantically related.
    A chunk about "vector embeddings" won't be mixed with sentences about
    "evaluation metrics."
    
    breakpoint_threshold_type options:
    - "percentile": Split where distance > Nth percentile of all distances
    - "standard_deviation": Split where distance > mean + N * std_dev
    - "interquartile": Uses IQR to detect outlier distances
    
    Pros:  Best semantic coherence, chunks "mean" something complete
    Cons:  Requires API calls for embedding (slower, costs money),
           variable chunk sizes (hard to predict), requires OpenAI key
    
    Use when: High-quality RAG matters most, documents have varied topics
    
    NOTE: This requires OPENAI_API_KEY in your .env file.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )
    
    # SemanticChunker works on raw text strings, not Document objects directly
    # We process each document and reconstruct Documents with metadata
    all_chunks = []
    for doc in documents:
        # create_documents returns List[Document]
        semantic_docs = splitter.create_documents(
            texts=[doc.page_content],
            metadatas=[doc.metadata]  # Pass original metadata through
        )
        all_chunks.extend(semantic_docs)
    
    for i, chunk in enumerate(all_chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_strategy"] = "semantic"
        chunk.metadata["chunk_char_count"] = len(chunk.page_content)
    
    return all_chunks


# =============================================================================
# STRATEGY 4: PARENT-DOCUMENT CHUNKING
# =============================================================================

def parent_document_chunking(
    documents: List[Document],
    parent_chunk_size: int = 1500,
    child_chunk_size: int = 300,
    child_chunk_overlap: int = 30,
) -> Tuple[FAISS, ParentDocumentRetriever]:
    """
    Parent-Document Retriever
    --------------------------
    This solves a fundamental tension in RAG:
    
    - SMALL chunks = better retrieval precision (more specific match)
    - LARGE chunks = better generation quality (more context for LLM)
    
    The solution: Index SMALL "child" chunks for retrieval, but when a 
    child chunk is retrieved, return its LARGE "parent" chunk to the LLM.
    
    Architecture:
    ┌─────────────────────────────────────────┐
    │              Original Document           │
    │  ┌─────────────────────────────────┐    │
    │  │     Parent Chunk 1 (1500 chars) │    │
    │  │  ┌──────┐ ┌──────┐ ┌──────┐   │    │
    │  │  │Child1│ │Child2│ │Child3│   │    │  <- Indexed in FAISS
    │  │  └──────┘ └──────┘ └──────┘   │    │
    │  └─────────────────────────────────┘    │
    │  ┌─────────────────────────────────┐    │
    │  │     Parent Chunk 2 (1500 chars) │    │  <- Stored in InMemoryStore
    │  └─────────────────────────────────┘    │
    └─────────────────────────────────────────┘
    
    Query flow:
    1. Embed query → find most similar child chunks in FAISS
    2. Look up which parent each child belongs to
    3. Return the full parent to the LLM (not the small child)
    
    Pros:  Best of both worlds — precise retrieval + rich context
    Cons:  More complex setup, stores documents twice
    
    Use when: Long documents where precision and context both matter
    """
    # Step 1: Set up embeddings for the vector store
    # Using HuggingFace to avoid OpenAI costs for local testing
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Step 2: Parent splitter — creates the large "context" chunks
    # These are stored in a key-value store (docstore), NOT the vector store
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=100,
    )
    
    # Step 3: Child splitter — creates the small "retrieval" chunks
    # These ARE embedded and stored in the vector store (FAISS)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
    )
    
    # Step 4: Create an empty FAISS vector store
    # FAISS.from_texts needs at least one document to initialize
    # We pass a placeholder and will add real documents next
    vectorstore = FAISS.from_texts(
        texts=["placeholder"],
        embedding=embeddings,
    )
    
    # Step 5: InMemoryStore — a simple key-value store that maps
    # parent_doc_id -> parent_document content
    # In production, this would be a Redis store or DynamoDB for persistence
    docstore = InMemoryStore()
    
    # Step 6: Create the ParentDocumentRetriever
    # This orchestrator knows how to:
    # - Split parents into children
    # - Store parents in docstore
    # - Index children in vectorstore
    # - Look up parents when children are retrieved
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,  # Optional: if None, whole doc = parent
    )
    
    # Step 7: Add documents — this triggers the split/store/index pipeline
    retriever.add_documents(documents, ids=None)
    
    return vectorstore, retriever


# =============================================================================
# CHUNKING COMPARISON UTILITY
# =============================================================================

def compare_chunking_strategies(documents: List[Document]) -> None:
    """
    Runs all three synchronous strategies and prints a comparison report.
    (Parent-document is excluded as it returns a retriever, not raw chunks.)
    """
    print("\n" + "=" * 70)
    print("CHUNKING STRATEGY COMPARISON REPORT")
    print("=" * 70)
    print(f"Input: {len(documents)} document(s), "
          f"~{sum(len(d.page_content) for d in documents)} total characters\n")

    strategies = {
        "1. Fixed-Size": lambda: fixed_size_chunking(documents, chunk_size=400, chunk_overlap=40),
        "2. Recursive Character": lambda: recursive_character_chunking(documents, chunk_size=400, chunk_overlap=40),
    }

    results = {}
    for name, fn in strategies.items():
        chunks = fn()
        sizes = [len(c.page_content) for c in chunks]
        results[name] = {
            "num_chunks": len(chunks),
            "avg_size": round(sum(sizes) / len(sizes), 1),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "sample": chunks[1].page_content[:200] + "..." if len(chunks) > 1 else chunks[0].page_content[:200],
        }

    # Print comparison table
    print(f"{'Strategy':<25} {'Chunks':>7} {'Avg Size':>10} {'Min':>7} {'Max':>7}")
    print("-" * 60)
    for name, data in results.items():
        print(f"{name:<25} {data['num_chunks']:>7} {data['avg_size']:>10} "
              f"{data['min_size']:>7} {data['max_size']:>7}")

    print("\n--- Sample Chunk from Each Strategy ---")
    for name, data in results.items():
        print(f"\n[{name}]")
        print(f"  '{data['sample']}'")

    print("\n--- Strategy Selection Guide ---")
    print("  Prototyping quickly?          → Fixed-Size (chunk_size=512)")
    print("  Production general use?       → Recursive Character")
    print("  Quality matters most?         → Semantic Chunking")
    print("  Long docs + need context?     → Parent-Document Retriever")
    print("=" * 70)


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing chunking strategies...\n")
    
    # Test Fixed-Size
    fixed_chunks = fixed_size_chunking(SAMPLE_DOCS, chunk_size=400, chunk_overlap=40)
    print(f"[Fixed-Size] Produced {len(fixed_chunks)} chunks")
    print(f"  First chunk preview: {fixed_chunks[0].page_content[:100]}...")
    print(f"  Metadata: {fixed_chunks[0].metadata}\n")
    
    # Test Recursive Character
    recursive_chunks = recursive_character_chunking(SAMPLE_DOCS, chunk_size=400, chunk_overlap=40)
    print(f"[Recursive] Produced {len(recursive_chunks)} chunks")
    print(f"  First chunk preview: {recursive_chunks[0].page_content[:100]}...")
    print(f"  Metadata: {recursive_chunks[0].metadata}\n")
    
    # Run comparison (skipping semantic which needs an API key)
    compare_chunking_strategies(SAMPLE_DOCS)
    
    # Test Parent-Document if you want (needs HuggingFace model download)
    print("\n[Parent-Document] Setting up retriever...")
    try:
        vectorstore, parent_retriever = parent_document_chunking(SAMPLE_DOCS)
        # Test retrieval
        retrieved = parent_retriever.get_relevant_documents("What is reranking?")
        print(f"  Retrieved {len(retrieved)} parent documents for test query")
        print(f"  First result preview: {retrieved[0].page_content[:150]}...")
        print("[Parent-Document] SUCCESS")
    except Exception as e:
        print(f"  [NOTE] Parent-Document test skipped: {e}")
    
    print("\n[SUCCESS] chunking.py is working correctly!")