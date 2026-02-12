"""
WHAT: Interactive Streamlit web app for stock Q&A
WHY: User-friendly interface to interact with RAG system
HOW: Streamlit UI + RAG pipeline backend
"""

import streamlit as st
import os
from dotenv import load_dotenv
from vector_db_setup import VectorDBManager
from embedding_generator import EmbeddingGenerator
from rag_pipeline import RAGPipeline

load_dotenv()

# Page config
st.set_page_config(
    page_title="Stock Market RAG Q&A",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .realtime-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    with st.spinner("🔧 Initializing RAG system..."):
        try:
            embedder = EmbeddingGenerator()
            vector_db = VectorDBManager()
            st.session_state.rag_system = RAGPipeline(vector_db, embedder)
            st.session_state.db_stats = vector_db.get_stats()
            st.success("✅ RAG system ready!")
        except Exception as e:
            st.error(f"❌ Error initializing system: {e}")
            st.stop()

# Header
st.markdown('<h1 class="main-header">📈 Stock Market RAG Q&A System</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Database stats
    st.subheader("📊 Database Info")
    st.metric("Total Chunks", st.session_state.db_stats['total_chunks'])
    st.metric("Collection", st.session_state.db_stats['collection_name'])
    
    st.divider()
    
    # Query settings
    st.subheader("🔧 Query Settings")
    top_k = st.slider("Number of sources to retrieve", 1, 10, 5)
    include_ticker = st.checkbox("Include real-time stock data", value=True)
    
    st.divider()
    
    # Example queries
    st.subheader("💡 Example Questions")
    example_queries = [
        "What was the company's revenue?",
        "What are the main risk factors?",
        "How is the company performing financially?",
        "What products does the company sell?",
        "What are the recent developments?"
    ]
    
    for query in example_queries:
        if st.button(query, key=query):
            st.session_state.current_query = query

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("❓ Ask a Question")
    
    # Query input
    query = st.text_input(
        "Enter your question:",
        value=st.session_state.get('current_query', ''),
        placeholder="e.g., What was Apple's revenue in Q3?"
    )
    
    # Ticker input (optional)
    ticker = None
    if include_ticker:
        ticker = st.text_input(
            "Stock ticker (optional, for real-time data):",
            placeholder="e.g., AAPL"
        ).upper()

with col2:
    st.subheader("🚀 Actions")
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    clear_button = st.button("🗑️ Clear", use_container_width=True)

if clear_button:
    st.session_state.current_query = ""
    st.rerun()

# Process query
if search_button and query:
    with st.spinner("🔍 Searching and generating answer..."):
        try:
            # Get response from RAG
            response = st.session_state.rag_system.query(
                query,
                ticker=ticker if ticker else None,
                top_k=top_k
            )
            
            # Display answer
            st.markdown("---")
            st.subheader("💡 Answer")
            st.markdown(response['answer'])
            
            # Display real-time data if available
            if response['realtime_data']:
                st.markdown("---")
                st.subheader("📊 Real-Time Stock Data")
                
                rt_data = response['realtime_data']
                
                cols = st.columns(4)
                cols[0].metric("Company", rt_data['company_name'])
                cols[1].metric("Price", f"${rt_data['current_price']}")
                cols[2].metric("Market Cap", f"${rt_data['market_cap']:,}" if rt_data['market_cap'] != 'N/A' else 'N/A')
                cols[3].metric("P/E Ratio", rt_data['pe_ratio'])
                
                st.caption(f"Data as of: {rt_data['timestamp']}")
            
            # Display sources
            st.markdown("---")
            st.subheader("📚 Sources")
            
            for i, source in enumerate(response['sources']):
                with st.expander(f"Source {i+1}: {source['filename']} ({source['doc_type']})"):
                    st.text(source['text_preview'])
            
        except Exception as e:
            st.error(f"❌ Error: {e}")

elif search_button:
    st.warning("⚠️ Please enter a question first!")

# Footer
st.markdown("---")
st.caption("Powered by Groq (Llama 3.1), HuggingFace Embeddings, and ChromaDB")