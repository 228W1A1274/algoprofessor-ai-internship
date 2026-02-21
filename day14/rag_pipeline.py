"""
rag_pipeline.py
===============
A complete Advanced RAG pipeline implementing:
  retriever → Cohere reranker → prompt template → LLM → response

Also includes a note section explaining how LlamaIndex would be used.

Pipeline Architecture:
─────────────────────────────────────────────────────────────
User Query
    │
    ▼
[1] Query Preprocessing (optional)
    │
    ▼
[2] FAISS Dense Retriever  ← top-K candidates (e.g., K=10)
    │
    ▼
[3] Cohere Reranker        ← re-scores with cross-encoder, keeps top-N (e.g., N=3)
    │
    ▼
[4] Context Compression    ← optional: extract only relevant sentences
    │
    ▼
[5] Prompt Template        ← system + human messages with context injected
    │
    ▼
[6] GPT-4o / GPT-3.5      ← LLM generates the answer
    │
    ▼
[7] Response               ← with source citations
─────────────────────────────────────────────────────────────
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# LangChain Core
# NEW (correct)
from langchain_core.documents import Document
# NEW
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# LangChain Components
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# Our custom modules
from doc_processor import load_document, load_directory
from chunking import recursive_character_chunking

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

class RAGConfig:
    """Central configuration class — change values here, not in code."""
    
    # Retrieval settings
    INITIAL_RETRIEVAL_K: int = 10    # How many docs to fetch from FAISS initially
    RERANKED_TOP_N: int = 3          # How many to keep after Cohere reranking
    
    # Chunking settings
    CHUNK_SIZE: int = 512            # Characters per chunk
    CHUNK_OVERLAP: int = 64          # Overlap between chunks
    
    # LLM settings
    LLM_MODEL: str = "gpt-4o-mini"  # Cheaper than gpt-4o, still very capable
    LLM_TEMPERATURE: float = 0.0    # 0 = deterministic, important for RAG
    MAX_TOKENS: int = 1024
    
    # Embedding settings
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # Cheaper than ada-002
    
    # Cohere settings
    COHERE_MODEL: str = "rerank-english-v3.0"


# =============================================================================
# STEP 1: BUILD THE VECTOR STORE (Ingestion + Indexing)
# =============================================================================

def build_vectorstore(
    documents: List[Document],
    config: RAGConfig = None
) -> FAISS:
    """
    Takes raw Document objects (from doc_processor.py), chunks them using
    recursive character splitting, generates embeddings, and indexes them
    in a FAISS vector store.
    
    FAISS (Facebook AI Similarity Search):
    - An open-source library for efficient similarity search
    - Stores vectors in memory (fast, no server needed)
    - For production: replace with Pinecone, Weaviate, or Qdrant
    
    OpenAIEmbeddings:
    - Calls the OpenAI API to convert text to 1536-dimensional vectors
    - text-embedding-3-small is fast and cheap; ada-002 is slightly better
    - For local/free: use HuggingFaceEmbeddings("all-MiniLM-L6-v2")
    
    Returns: A FAISS object that can be used as a retriever.
    """
    config = config or RAGConfig()
    
    print(f"[VectorStore] Chunking {len(documents)} documents...")
    chunks = recursive_character_chunking(
        documents,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    print(f"[VectorStore] Generated {len(chunks)} chunks")
    
    print("[VectorStore] Generating embeddings (this calls OpenAI API)...")
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    
    # FAISS.from_documents():
    # 1. Extracts text from each Document
    # 2. Sends batches to the embedding API
    # 3. Stores (vector, metadata) pairs in the FAISS index
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    
    print(f"[VectorStore] Index built with {vectorstore.index.ntotal} vectors")
    return vectorstore


def save_vectorstore(vectorstore: FAISS, path: str = "vectorstore") -> None:
    """Persist the FAISS index to disk so you don't re-embed on every run."""
    vectorstore.save_local(path)
    print(f"[VectorStore] Saved to '{path}/'")


def load_vectorstore(path: str = "vectorstore") -> FAISS:
    """Load a previously saved FAISS index from disk."""
    embeddings = OpenAIEmbeddings(model=RAGConfig.EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True  # Required flag in newer LangChain
    )
    print(f"[VectorStore] Loaded from '{path}/' with {vectorstore.index.ntotal} vectors")
    return vectorstore


# =============================================================================
# STEP 2: BUILD THE RERANKING RETRIEVER
# =============================================================================

def build_retriever(
    vectorstore: FAISS,
    config: RAGConfig = None
) -> ContextualCompressionRetriever:
    """
    Creates a two-stage retriever:
    Stage 1 — Dense Retrieval (FAISS):
        Uses cosine similarity between query embedding and chunk embeddings.
        Fast but imprecise — retrieves the top-K candidates (e.g., 10).
    
    Stage 2 — Cross-Encoder Reranking (Cohere):
        Takes the K candidates and re-scores each by jointly encoding
        the (query, document) pair. Much more accurate but slower.
        Cohere's reranker outputs a relevance_score in [0, 1].
        We keep only the top-N after reranking (e.g., 3).
    
    Why this matters:
        Vector similarity = "are these texts statistically similar?"
        Cross-encoder = "does this document answer this specific question?"
        These are not the same! Reranking bridges that gap.
    
    ContextualCompressionRetriever:
        LangChain's wrapper that chains a base retriever (FAISS) with
        a "document compressor" (Cohere reranker) transparently.
        When you call .get_relevant_documents(query), it automatically
        runs both stages.
    """
    config = config or RAGConfig()
    
    # Stage 1: Base FAISS retriever
    # search_type="similarity" = standard cosine similarity
    # search_kwargs={"k": N} = return top N results
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.INITIAL_RETRIEVAL_K}
    )
    
    # Stage 2: Cohere Reranker
    # CohereRerank wraps Cohere's reranking API as a LangChain compressor
    # top_n: how many documents to keep after reranking
    # model: which Cohere model to use for reranking
    reranker = CohereRerank(
        top_n=config.RERANKED_TOP_N,
        model=config.COHERE_MODEL,
    )
    
    # Combine: ContextualCompressionRetriever runs base_retriever first,
    # then passes results through the reranker (base_compressor)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )
    
    print(f"[Retriever] Built: FAISS(k={config.INITIAL_RETRIEVAL_K}) → "
          f"Cohere Rerank(top_n={config.RERANKED_TOP_N})")
    return compression_retriever


# =============================================================================
# STEP 3: BUILD THE PROMPT TEMPLATE
# =============================================================================

def build_prompt() -> ChatPromptTemplate:
    """
    Creates the RAG prompt template.
    
    Prompt engineering for RAG has specific requirements:
    
    1. GROUNDING INSTRUCTION: Tell the LLM to use ONLY the provided context.
       Without this, the LLM will blend retrieved context with its training
       data, making it impossible to attribute answers to sources.
    
    2. UNCERTAINTY HANDLING: Tell the LLM what to do when context is 
       insufficient. "I don't know" is better than a confident wrong answer.
    
    3. SOURCE CITATION: Encourage the LLM to reference which part of the
       context it used — enables faithfulness verification.
    
    ChatPromptTemplate uses a list of message templates:
    - SystemMessagePromptTemplate: The "instruction manual" for the LLM
    - HumanMessagePromptTemplate: The actual user query
    
    Variables in {curly_braces} are filled in at runtime.
    """
    
    system_template = """You are an expert assistant for question-answering tasks.

Your responsibilities:
1. Answer the user's question using ONLY the information provided in the context below.
2. If the answer is not contained in the context, respond with: 
   "I cannot answer this question based on the provided context."
3. Always be precise, concise, and factual.
4. When possible, reference which part of the context supports your answer.
5. Do not speculate, assume, or add information from outside the context.

CONTEXT:
{context}

Remember: Base your answer ONLY on the context above. Do not use prior knowledge."""

    human_template = """Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ])
    
    return prompt


# =============================================================================
# STEP 4: CONTEXT FORMATTER
# =============================================================================

def format_context(docs: List[Document]) -> str:
    """
    Converts a list of retrieved Document objects into a single formatted
    string that gets injected into the prompt's {context} variable.
    
    We number each document and include its source metadata so the LLM
    can potentially cite sources. This is called "grounded generation."
    """
    formatted_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", f"Document {i}")
        page = doc.metadata.get("page_number", "")
        section = doc.metadata.get("section_heading", "")
        
        # Build a source citation string
        source_info = f"Source {i}: {source}"
        if page:
            source_info += f", Page {page}"
        if section:
            source_info += f", Section: {section}"
        
        formatted_parts.append(f"[{source_info}]\n{doc.page_content}")
    
    return "\n\n---\n\n".join(formatted_parts)


# =============================================================================
# STEP 5: BUILD THE COMPLETE LCEL CHAIN
# =============================================================================

def build_rag_chain(
    retriever: ContextualCompressionRetriever,
    config: RAGConfig = None
):
    """
    Assembles the complete RAG chain using LangChain Expression Language (LCEL).
    
    LCEL uses the pipe operator (|) to chain components together.
    Each component receives the output of the previous one.
    
    The chain we're building:
    
    {"context": retriever | format_context, "question": passthrough}
                ↓
           prompt_template
                ↓
               llm
                ↓
         output_parser (→ plain string)
    
    RunnableParallel runs multiple branches in parallel:
    - Branch 1: Retrieves context docs, formats them into a string
    - Branch 2: Passes the question through unchanged
    Both results are merged into a dict for the prompt template.
    
    RunnablePassthrough: Simply forwards its input unchanged.
    
    StrOutputParser: Converts the AIMessage object from the LLM
    into a plain Python string.
    """
    config = config or RAGConfig()
    
    # Initialize the LLM
    # temperature=0 is important for RAG — we want deterministic, factual answers
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
    )
    
    prompt = build_prompt()
    
    # The LCEL chain construction
    chain = (
        # Step A: Parallel branches — run retrieval and question passthrough simultaneously
        RunnableParallel(
            # Retriever → format_context: gets docs, converts to string
            context=retriever | format_context,
            # RunnablePassthrough: just forwards the question as-is
            question=RunnablePassthrough(),
        )
        # Step B: Fill in the prompt template with {context} and {question}
        | prompt
        # Step C: Send filled prompt to the LLM
        | llm
        # Step D: Extract the string content from the LLM's response object
        | StrOutputParser()
    )
    
    print(f"[Chain] Built LCEL chain with {config.LLM_MODEL}")
    return chain


# =============================================================================
# STEP 6: RESPONSE WITH SOURCES
# =============================================================================

def query_with_sources(
    question: str,
    retriever: ContextualCompressionRetriever,
    chain,
) -> Dict[str, Any]:
    """
    Runs the full RAG pipeline and returns both the answer AND the
    source documents that were used to generate it.
    
    This is critical for:
    - Debugging (which docs was the answer based on?)
    - Faithfulness verification (is the answer actually in the docs?)
    - User trust (showing sources increases user confidence)
    """
    # Get the retrieved (and reranked) documents
    # We call the retriever separately so we can inspect what was retrieved
    retrieved_docs = retriever.get_relevant_documents(question)
    
    # Run the full chain
    answer = chain.invoke(question)
    
    # Package the result
    return {
        "question": question,
        "answer": answer,
        "source_documents": retrieved_docs,
        "num_sources_used": len(retrieved_docs),
        "sources": [
            {
                "content_preview": doc.page_content[:200] + "...",
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page_number", "N/A"),
            }
            for doc in retrieved_docs
        ]
    }


# =============================================================================
# LLAMA INDEX COMPARISON NOTE
# =============================================================================

LLAMAINDEX_EQUIVALENT_CODE = '''
"""
How This Pipeline Would Look in LlamaIndex
==========================================
LlamaIndex is a "Modular RAG" framework — think of it as having
higher-level abstractions and better multi-index support.

Key conceptual differences:
- LangChain: "Here are components, you wire them together"
- LlamaIndex: "Here is the complete RAG system, configure what you need"

LlamaIndex Core Concepts:
- SimpleDirectoryReader: Equivalent to our doc_processor.py
- VectorStoreIndex: Equivalent to FAISS + embeddings
- QueryEngine: Equivalent to our LCEL chain
- RetrieverQueryEngine: The equivalent of our compression retriever + chain

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import CohereRerank
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Step 1: Configure global settings (equivalent to our RAGConfig)
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.chunk_size = 512
Settings.chunk_overlap = 64

# Step 2: Load documents (equivalent to our doc_processor.py)
reader = SimpleDirectoryReader("sample_docs")
documents = reader.load_data()

# Step 3: Build index (equivalent to our build_vectorstore)
index = VectorStoreIndex.from_documents(documents)

# Step 4: Create retriever with reranking (equivalent to build_retriever)
retriever = index.as_retriever(similarity_top_k=10)
reranker = CohereRerank(top_n=3)

# Step 5: Create query engine (equivalent to build_rag_chain)
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[reranker],  # Reranker is a "post-processor" in LlamaIndex
)

# Step 6: Query (equivalent to query_with_sources)
response = query_engine.query("What is reranking?")
print(response.response)
print(response.source_nodes)  # Retrieved documents

# KEY ADVANTAGE of LlamaIndex: Multi-index routing
# You can combine different index types and route queries:

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import RouterQueryEngine

vector_tool = QueryEngineTool(query_engine=query_engine, metadata=ToolMetadata(
    name="vector_search", description="Searches the main document corpus"
))

# Add a SQL index for structured data:
# sql_engine = NLSQLTableQueryEngine(...)
# sql_tool = QueryEngineTool(query_engine=sql_engine, ...)

# RouterQueryEngine automatically picks the right index for each query
# This is the "Modular RAG" paradigm in action
router = RouterQueryEngine.from_defaults(
    query_engine_tools=[vector_tool],
    verbose=True,  # Shows which engine was selected for each query
)
"""
'''


# =============================================================================
# MAIN RAG PIPELINE CLASS
# =============================================================================

class RAGPipeline:
    """
    Wraps all components into a clean class interface.
    Manages vectorstore persistence so embeddings aren't regenerated
    on every run.
    """
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.vectorstore: Optional[FAISS] = None
        self.retriever = None
        self.chain = None
        self._is_built = False
    
    def build_from_documents(self, documents: List[Document]) -> None:
        """Build the pipeline from a list of Document objects."""
        print("\n[Pipeline] Building RAG pipeline...")
        self.vectorstore = build_vectorstore(documents, self.config)
        self.retriever = build_retriever(self.vectorstore, self.config)
        self.chain = build_rag_chain(self.retriever, self.config)
        self._is_built = True
        print("[Pipeline] ✓ Pipeline ready")
    
    def build_from_files(self, file_paths: List[str]) -> None:
        """Build the pipeline from a list of file paths."""
        all_docs = []
        for path in file_paths:
            docs = load_document(path)
            all_docs.extend(docs)
        self.build_from_documents(all_docs)
    
    def build_from_directory(self, dir_path: str) -> None:
        """Build the pipeline from a directory of documents."""
        docs = load_directory(dir_path)
        self.build_from_documents(docs)
    
    def query(self, question: str) -> Dict[str, Any]:
        """Run a query through the full pipeline."""
        if not self._is_built:
            raise RuntimeError("Pipeline not built. Call build_from_* first.")
        return query_with_sources(question, self.retriever, self.chain)
    
    def save(self, path: str = "vectorstore") -> None:
        """Persist the vector store to disk."""
        if self.vectorstore:
            save_vectorstore(self.vectorstore, path)
    
    def load(self, path: str = "vectorstore") -> None:
        """Load a previously saved vector store and rebuild pipeline components."""
        self.vectorstore = load_vectorstore(path)
        self.retriever = build_retriever(self.vectorstore, self.config)
        self.chain = build_rag_chain(self.retriever, self.config)
        self._is_built = True
    
    def interactive_session(self) -> None:
        """Start an interactive Q&A session in the terminal."""
        if not self._is_built:
            raise RuntimeError("Pipeline not built.")
        
        print("\n" + "=" * 60)
        print("RAG Pipeline — Interactive Session")
        print("Type 'quit' or 'exit' to stop")
        print("=" * 60)
        
        while True:
            try:
                question = input("\nYour question: ").strip()
                if question.lower() in ["quit", "exit", "q"]:
                    print("Exiting session.")
                    break
                if not question:
                    continue
                
                result = self.query(question)
                
                print(f"\n{'─' * 60}")
                print(f"Answer: {result['answer']}")
                print(f"\nSources used ({result['num_sources_used']} documents):")
                for i, src in enumerate(result['sources'], 1):
                    print(f"  [{i}] {src['source']} (Page: {src['page']})")
                    print(f"      Preview: {src['content_preview'][:100]}...")
                print(f"{'─' * 60}")
                
            except KeyboardInterrupt:
                print("\nSession interrupted.")
                break


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    from chunking import SAMPLE_DOCS
    
    print("=" * 60)
    print("Testing rag_pipeline.py")
    print("=" * 60)
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not found in .env file")
        exit(1)
    if not os.getenv("COHERE_API_KEY"):
        print("[ERROR] COHERE_API_KEY not found in .env file")
        exit(1)
    
    # Build the pipeline using our sample corpus from chunking.py
    pipeline = RAGPipeline()
    pipeline.build_from_documents(SAMPLE_DOCS)
    
    # Test queries
    test_questions = [
        "What is the difference between reranking and retrieval?",
        "How does semantic chunking work?",
        "What metrics does RAGAS measure?",
    ]
    
    print("\n[Testing 3 sample queries]\n")
    for question in test_questions:
        print(f"Q: {question}")
        result = pipeline.query(question)
        print(f"A: {result['answer']}")
        print(f"   (Based on {result['num_sources_used']} documents)")
        print()
    
    print("[SUCCESS] rag_pipeline.py is working correctly!")
    
    # Optionally start interactive session
    start_interactive = input("\nStart interactive session? (y/n): ").strip().lower()
    if start_interactive == 'y':
        pipeline.interactive_session()