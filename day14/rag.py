import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# LangChain Core
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# LangChain Components - UPDATED FOR FREE TIER
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Local Embeddings
from langchain_groq import ChatGroq                     # Free API LLM
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# Our custom modules (Assuming these exist in your path)
try:
    from doc_processor import load_document, load_directory
    from chunking import recursive_character_chunking
except ImportError:
    print("[Warning] Custom modules not found. Ensure doc_processor.py and chunking.py are in the folder.")

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

class RAGConfig:
    """Central configuration class — Optimized for Free Tier."""
    
    # Retrieval settings
    INITIAL_RETRIEVAL_K: int = 10
    RERANKED_TOP_N: int = 3
    
    # Chunking settings
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    
    # LLM settings - Using Groq (Llama 3) for FREE high-speed inference
    # Get key at: https://console.groq.com/
    LLM_MODEL: str = "llama-3.3-70b-versatile" 
    LLM_TEMPERATURE: float = 0.0
    MAX_TOKENS: int = 1024
    
    # Embedding settings - Local HuggingFace model (Free, no API needed)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Cohere settings (Optional: disable in build_retriever if no key)
    COHERE_MODEL: str = "rerank-english-v3.0"


# =============================================================================
# STEP 1: BUILD THE VECTOR STORE (Local HuggingFace)
# =============================================================================

def build_vectorstore(
    documents: List[Document],
    config: RAGConfig = None
) -> FAISS:
    config = config or RAGConfig()
    
    print(f"[VectorStore] Chunking {len(documents)} documents...")
    chunks = recursive_character_chunking(
        documents,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    
    print(f"[VectorStore] Generating local embeddings (HuggingFace)...")
    # This runs on your CPU, no API key required
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    
    print(f"[VectorStore] Index built with {vectorstore.index.ntotal} vectors")
    return vectorstore


def save_vectorstore(vectorstore: FAISS, path: str = "vectorstore") -> None:
    vectorstore.save_local(path)
    print(f"[VectorStore] Saved to '{path}/'")


def load_vectorstore(path: str = "vectorstore") -> FAISS:
    # Use same local embeddings for loading
    embeddings = HuggingFaceEmbeddings(model_name=RAGConfig.EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


# =============================================================================
# STEP 2: BUILD THE RERANKING RETRIEVER
# =============================================================================

def build_retriever(
    vectorstore: FAISS,
    config: RAGConfig = None
) -> Any:
    config = config or RAGConfig()
    
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.INITIAL_RETRIEVAL_K}
    )
    
    # Check if Cohere key exists, otherwise fallback to base retriever
    if os.getenv("COHERE_API_KEY"):
        reranker = CohereRerank(
            top_n=config.RERANKED_TOP_N,
            model=config.COHERE_MODEL,
        )
        
        retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever,
        )
        print(f"[Retriever] Built: FAISS → Cohere Rerank")
    else:
        print("[Retriever] No Cohere Key found. Using base FAISS retriever.")
        retriever = base_retriever
        
    return retriever

# =============================================================================
# STEP 3 & 4: PROMPT & FORMATTER (Unchanged)
# =============================================================================

def build_prompt() -> ChatPromptTemplate:
    system_template = """You are an expert assistant for question-answering tasks.
Answer using ONLY the context below. If unknown, say you don't know.
CONTEXT:
{context}"""
    human_template = """Question: {question}\nAnswer:"""
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ])

def format_context(docs: List[Document]) -> str:
    return "\n\n".join([f"[Source: {d.metadata.get('source', 'NA')}]\n{d.page_content}" for d in docs])

# =============================================================================
# STEP 5: BUILD THE LCEL CHAIN (Groq Update)
# =============================================================================

def build_rag_chain(
    retriever: Any,
    config: RAGConfig = None
):
    config = config or RAGConfig()
    
    # Using ChatGroq instead of OpenAI
    llm = ChatGroq(
        model_name=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    prompt = build_prompt()
    
    chain = (
        RunnableParallel(
            context=retriever | format_context,
            question=RunnablePassthrough(),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print(f"[Chain] Built LCEL chain with {config.LLM_MODEL} (Groq)")
    return chain

# =============================================================================
# STEP 6: RESPONSE WITH SOURCES (Unchanged logic)
# =============================================================================

def query_with_sources(question: str, retriever: Any, chain: Any) -> Dict[str, Any]:
    # Use invoke for modern LCEL compatibility
    retrieved_docs = retriever.invoke(question)
    answer = chain.invoke(question)
    
    return {
        "question": question,
        "answer": answer,
        "num_sources_used": len(retrieved_docs),
        "sources": [{"source": d.metadata.get("source", "Unknown")} for d in retrieved_docs]
    }

# =============================================================================
# MAIN RAG PIPELINE CLASS
# =============================================================================

class RAGPipeline:
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.vectorstore: Optional[FAISS] = None
        self.retriever = None
        self.chain = None
        self._is_built = False

    def build_from_documents(self, documents: List[Document]) -> None:
        self.vectorstore = build_vectorstore(documents, self.config)
        self.retriever = build_retriever(self.vectorstore, self.config)
        self.chain = build_rag_chain(self.retriever, self.config)
        self._is_built = True

    def query(self, question: str) -> Dict[str, Any]:
        if not self._is_built:
            raise RuntimeError("Pipeline not built.")
        return query_with_sources(question, self.retriever, self.chain)

if __name__ == "__main__":
    # For testing, ensure SAMPLE_DOCS is imported or defined
    from chunking import SAMPLE_DOCS
    
    # Check for Groq Key instead of OpenAI
    if not os.getenv("GROQ_API_KEY"):
        print("[ERROR] GROQ_API_KEY not found in .env. Get one free at console.groq.com")
        exit(1)
        
    pipeline = RAGPipeline()
    pipeline.build_from_documents(SAMPLE_DOCS)
    
    res = pipeline.query("What is the difference between reranking and retrieval?")
    print(f"\nResult: {res['answer']}")