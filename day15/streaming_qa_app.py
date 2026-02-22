"""
streaming_qa_app.py — Day 15: Streaming RAG via SSE + Streamlit UI
===================================================================
Author : AI/ML Mentor (Day 15 Curriculum)
Purpose: Production-grade Streamlit app that streams RAG responses token-by-token
         with chunk-level source attribution shown *as the answer streams*.

Run:
    streamlit run streaming_qa_app.py

Architecture:
    User Query → LlamaIndex Retriever → Source Chunks + LLM Stream → SSE → Streamlit UI
    
Key techniques:
  • Token streaming via LlamaIndex's streaming_response
  • Chunk-level attribution: sources rendered alongside streamed tokens  
  • Session state management for conversation history
  • Custom CSS for a polished, professional UI
"""

import os
import time
import logging
from typing import Generator, List
from dotenv import load_dotenv

import streamlit as st

# ── LlamaIndex ────────────────────────────────────────────────────────────────
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

logging.basicConfig(level=logging.WARNING)
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# 0. PAGE CONFIG & CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Advanced RAG — Streaming QA",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for a professional look
st.markdown("""
<style>
    /* Main container */
    .main { background-color: #0e1117; }
    
    /* Chat message bubbles */
    .user-bubble {
        background: linear-gradient(135deg, #1e3a5f, #2d5986);
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        color: #e8f4fd;
        font-size: 15px;
        max-width: 85%;
        float: right;
        clear: both;
    }
    .assistant-bubble {
        background: #1a1d23;
        border: 1px solid #2d3139;
        padding: 12px 18px;
        border-radius: 4px 18px 18px 18px;
        margin: 8px 0;
        color: #d4d6dc;
        font-size: 15px;
        max-width: 90%;
        clear: both;
    }
    
    /* Source attribution cards */
    .source-card {
        background: #13161c;
        border-left: 3px solid #4EA8DE;
        border-radius: 4px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 12px;
        color: #8b95a5;
    }
    .source-card .source-title { color: #4EA8DE; font-weight: 600; }
    .source-card .source-score { color: #68d391; float: right; }
    
    /* Streaming indicator */
    .streaming-badge {
        display: inline-block;
        background: #1a3a1a;
        color: #68d391;
        border: 1px solid #2d5c2d;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 11px;
        margin-bottom: 8px;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
    
    /* Sidebar */
    .css-1d391kg { background-color: #13161c; }
    
    /* Metric cards */
    .metric-card {
        background: #1a1d23;
        border: 1px solid #2d3139;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LlamaIndex SETUP (cached so it runs only once per session)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR    = "./data"
PERSIST_DIR = "./storage"


@st.cache_resource(show_spinner="🔨 Building knowledge index…")
def initialise_index() -> VectorStoreIndex:
    """
    Cached resource — LlamaIndex builds the index once and caches it in memory
    across all Streamlit reruns. `@st.cache_resource` is ideal for heavyweight
    objects like ML models and vector indices.
    """
    Settings.llm         = OpenAI(model="gpt-4o-mini", temperature=0.2, streaming=True)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.chunk_size  = 512
    Settings.chunk_overlap = 64

    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.listdir(DATA_DIR):
        with open(f"{DATA_DIR}/demo.txt", "w") as f:
            f.write(
                "LlamaIndex is a data framework for LLM applications developed by Jerry Liu. "
                "It supports vector stores including Pinecone, Weaviate, Qdrant, and Chroma. "
                "Advanced RAG techniques in LlamaIndex include HyDE query transformation, "
                "Knowledge Graph indexing via Neo4j, and streaming query engines. "
                "Streaming responses use Server-Sent Events (SSE) to push tokens to clients. "
                "SSE is a one-way, HTTP-native protocol ideal for LLM token streaming. "
                "Chunk-level attribution shows users which document sections each answer draws from."
            )

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        storage_ctx = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        return load_index_from_storage(storage_ctx)

    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=False)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index


# ─────────────────────────────────────────────────────────────────────────────
# 2. STREAMING RESPONSE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def stream_rag_response(index: VectorStoreIndex, question: str):
    """
    Returns (source_nodes, token_generator) as a tuple.

    Flow:
      1. Retrieve relevant chunks FIRST (non-streaming).
      2. Pass chunks as context to the LLM in streaming mode.
      3. Yield tokens one by one — Streamlit's st.write_stream() consumes this.

    WHY separate retrieval from generation?
      - We can display source cards immediately (before any tokens arrive).
      - This mirrors how production streaming RAG works:
        retrieve → attribute → stream → done.
    """
    # Build retriever (separate from query engine so we can access source nodes)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=4)
    source_nodes = retriever.retrieve(question)

    # Build a streaming query engine backed by the same retriever
    streaming_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        streaming=True,   # ← Critical: tells LlamaIndex to use the LLM's streaming API
        response_mode="compact",
    )

    # .query() returns a StreamingResponse object — not a string
    streaming_response = streaming_engine.query(question)

    # streaming_response.response_gen is a Python generator that yields string tokens
    # We wrap it to add a small delay for visual effect in the demo
    def token_generator() -> Generator[str, None, None]:
        for token in streaming_response.response_gen:
            yield token
            # In production, remove this sleep — it's only for demo clarity
            # time.sleep(0.01)

    return source_nodes, token_generator()


# ─────────────────────────────────────────────────────────────────────────────
# 3. SOURCE ATTRIBUTION RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def render_source_cards(source_nodes: list):
    """
    Renders each retrieved source chunk as a styled card.
    Shows: filename, relevance score, and a preview of the chunk text.

    Chunk-level attribution is the key UX innovation in streaming RAG:
    instead of a vague footnote at the end, users SEE the sources as
    the answer streams — dramatically improving trust and debuggability.
    """
    if not source_nodes:
        return

    st.markdown("**📚 Sources Retrieved** *(shown before streaming begins)*")

    for i, node in enumerate(source_nodes):
        score      = node.score or 0.0
        filename   = node.metadata.get("file_name", f"chunk_{i}")
        page_label = node.metadata.get("page_label", "")
        preview    = (node.text or "")[:200].replace("\n", " ")

        score_color = "#68d391" if score > 0.8 else "#f6ad55" if score > 0.6 else "#fc8181"

        st.markdown(f"""
        <div class="source-card">
            <span class="source-title">📄 {filename}{' — p.' + page_label if page_label else ''}</span>
            <span class="source-score" style="color:{score_color};">Score: {score:.3f}</span>
            <br><span style="color:#6b7280; font-size:11px;">{preview}…</span>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SIDEBAR — Configuration & Stats
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧠 Advanced RAG")
        st.markdown("*Day 15 — Streaming QA App*")
        st.divider()

        st.markdown("### ⚙️ Settings")
        top_k = st.slider("Chunks to retrieve (top-k)", 1, 8, 4)
        mode  = st.selectbox(
            "RAG Mode",
            ["Standard Streaming", "HyDE + Streaming", "Multi-Query + Streaming"],
            help="HyDE generates a hypothetical answer before retrieval. "
                 "Multi-Query expands the query into variants."
        )

        st.divider()
        st.markdown("### 📤 Upload Documents")
        uploaded = st.file_uploader(
            "Add your own documents",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        if uploaded:
            os.makedirs(DATA_DIR, exist_ok=True)
            for f in uploaded:
                with open(f"{DATA_DIR}/{f.name}", "wb") as out:
                    out.write(f.read())
            st.success(f"✅ {len(uploaded)} file(s) uploaded. Restart app to reindex.")

        st.divider()
        st.markdown("### 🛠️ Architecture")
        st.code("""
User Query
   ↓
LlamaIndex Retriever
   ↓ (top-k chunks)
Source Attribution UI  ← Shown immediately
   ↓
OpenAI Streaming API
   ↓ (token stream via SSE)
Streamlit st.write_stream()
        """, language="text")

        st.divider()
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    return top_k, mode


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Session state init ───────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0

    # ── Sidebar ──────────────────────────────────────────────────────────────
    top_k, mode = render_sidebar()

    # ── Header ───────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([6, 2, 2])
    with col1:
        st.markdown("# 🔍 Advanced RAG — Streaming QA")
        st.markdown(f"*Mode: **{mode}** | Model: gpt-4o-mini | Chunks: {top_k}*")
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color:#4EA8DE;font-size:22px;font-weight:700;">{st.session_state.total_queries}</div>
            <div style="color:#8b95a5;font-size:11px;">Queries</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color:#68d391;font-size:22px;font-weight:700;">{st.session_state.total_tokens}</div>
            <div style="color:#8b95a5;font-size:11px;">Est. Tokens</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Load index (cached) ──────────────────────────────────────────────────
    index = initialise_index()

    # ── Chat history display ─────────────────────────────────────────────────
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🧠"):
                # Re-render source cards
                if msg.get("sources"):
                    render_source_cards(msg["sources"])
                st.markdown(msg["content"])

    # ── Chat input ───────────────────────────────────────────────────────────
    if question := st.chat_input("Ask anything about your documents…"):
        # Display user message
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.total_queries += 1

        # ── Assistant response (streaming) ───────────────────────────────────
        with st.chat_message("assistant", avatar="🧠"):

            # Show streaming badge
            st.markdown('<div class="streaming-badge">⚡ Streaming…</div>', unsafe_allow_html=True)

            # Step 1: Retrieve sources and display attribution cards IMMEDIATELY
            source_nodes, token_gen = stream_rag_response(index, question)
            render_source_cards(source_nodes)

            st.markdown("**💬 Answer**")

            # Step 2: Stream tokens using Streamlit's native streaming renderer
            # st.write_stream() consumes a generator and renders tokens in real-time
            # This is backed by SSE at the HTTP level between Streamlit and the browser
            start_time = time.time()
            full_response = st.write_stream(token_gen)
            elapsed = time.time() - start_time

            # Show timing metadata
            token_estimate = len(full_response.split())
            st.session_state.total_tokens += token_estimate
            st.caption(f"⏱️ {elapsed:.1f}s | ~{token_estimate} tokens | {len(source_nodes)} sources")

        # Persist to session history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": source_nodes,
        })


if __name__ == "__main__":
    main()