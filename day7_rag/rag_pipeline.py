"""
WHAT: Main RAG pipeline that orchestrates query -> retrieval -> generation
WHY: Combines all components into a working Q&A system
HOW: Query embedding -> Vector search -> Context + LLM -> Answer
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class RAGPipeline:
    def __init__(self, vector_db, embedder, model_name="llama-3.3-70b-versatile"):
        """
        WHAT: Initialize RAG pipeline with all components
        
        PARAMS:
        - vector_db: VectorDBManager instance
        - embedder: EmbeddingGenerator instance
        - model_name: Groq model to use for generation
        
        SUPPORTED MODELS (as of Feb 2025):
        - llama-3.3-70b-versatile (recommended - most capable)
        - llama-3.1-8b-instant (fastest)
        - mixtral-8x7b-32768 (large context window)
        - gemma2-9b-it (Google's model)
        """
        print("🔧 Initializing RAG Pipeline...")
        
        self.vector_db = vector_db
        self.embedder = embedder
        self.model_name = model_name
        
        # Initialize Groq client
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("❌ GROQ_API_KEY not found in .env file!")
        
        self.llm_client = Groq(api_key=api_key)
        
        print(f"✅ RAG Pipeline ready!")
        print(f"   Model: {model_name}")
        print(f"   Vector DB chunks: {vector_db.get_stats()['total_chunks']}")
    
    def retrieve_context(self, query: str, top_k: int = 5, 
                        doc_type: Optional[str] = None) -> List[Dict]:
        """
        WHAT: Find relevant document chunks for the query
        WHY: This is the "Retrieval" part of RAG
        
        RETURNS: List of relevant chunks with metadata
        """
        print(f"\n🔍 Retrieving context for: '{query}'")
        
        # 1. Convert query to embedding
        query_embedding = self.embedder.generate_embedding(query)
        
        # 2. Search vector database
        if doc_type:
            results = self.vector_db.search_with_metadata_filter(
                query_embedding, 
                doc_type=doc_type, 
                top_k=top_k
            )
        else:
            results = self.vector_db.search(query_embedding, top_k=top_k)
        
        # 3. Format results
        context_chunks = []
        for doc, distance, metadata in zip(
            results['documents'],
            results['distances'],
            results['metadatas']
        ):
            context_chunks.append({
                'text': doc,
                'similarity': 1 - distance,  # Convert distance to similarity
                'source': metadata.get('source', 'unknown'),
                'filename': metadata.get('filename', 'unknown'),
                'doc_type': metadata.get('doc_type', 'unknown')
            })
        
        print(f"✅ Retrieved {len(context_chunks)} relevant chunks")
        for i, chunk in enumerate(context_chunks):
            print(f"   {i+1}. {chunk['filename']} (similarity: {chunk['similarity']:.3f})")
        
        return context_chunks
    
    def generate_answer(self, query: str, context_chunks: List[Dict], 
                       realtime_data: Optional[Dict] = None) -> str:
        """
        WHAT: Generate answer using LLM with retrieved context
        WHY: This is the "Generation" part of RAG
        
        RETURNS: Natural language answer
        """
        print(f"\n💬 Generating answer with Groq ({self.model_name})...")
        
        # Build context from chunks
        context_text = "\n\n---\n\n".join([
            f"Source: {chunk['filename']}\n{chunk['text']}"
            for chunk in context_chunks
        ])
        
        # Add real-time data if available
        realtime_context = ""
        if realtime_data:
            realtime_context = f"""
REAL-TIME STOCK DATA:
Company: {realtime_data['company_name']}
Current Price: ${realtime_data['current_price']}
Market Cap: ${realtime_data['market_cap']}
P/E Ratio: {realtime_data['pe_ratio']}
52-Week High: ${realtime_data['52_week_high']}
52-Week Low: ${realtime_data['52_week_low']}
Sector: {realtime_data['sector']}
Last Updated: {realtime_data['timestamp']}
"""
        
        # Create prompt
        system_prompt = """You are a helpful financial analyst assistant. 
Answer questions based on the provided context from SEC filings, news articles, and real-time stock data.

INSTRUCTIONS:
- Use ONLY information from the provided context
- If you don't know something, say so
- Cite specific sources when making claims
- Be concise but thorough
- For financial metrics, include numbers when available"""

        user_prompt = f"""Context from documents:
{context_text}

{realtime_context}

Question: {query}

Please provide a clear, accurate answer based on the context above."""

        try:
            # Call Groq API
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.3  # Lower temperature for factual responses
            )
            
            answer = response.choices[0].message.content
            
            print(f"✅ Answer generated ({response.usage.total_tokens} tokens)")
            
            return answer
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return f"Error: {str(e)}"
    
    def query(self, question: str, ticker: Optional[str] = None, 
              top_k: int = 5) -> Dict:
        """
        WHAT: Main RAG query method - full pipeline
        WHY: Single method to handle entire RAG workflow
        
        PARAMS:
        - question: User's question
        - ticker: Stock ticker for real-time data (optional)
        - top_k: Number of chunks to retrieve
        
        RETURNS: Dict with answer, sources, and metadata
        """
        print("\n" + "="*60)
        print(f"🎯 Processing query: {question}")
        print("="*60)
        
        # 1. Retrieve relevant context
        context_chunks = self.retrieve_context(question, top_k=top_k)
        
        # 2. Get real-time stock data if ticker provided
        realtime_data = None
        if ticker:
            try:
                from data_fetcher import StockDataFetcher
                fetcher = StockDataFetcher()
                realtime_data = fetcher.fetch_realtime_stock_data(ticker)
            except Exception as e:
                print(f"⚠️ Could not fetch real-time data: {e}")
        
        # 3. Generate answer
        answer = self.generate_answer(question, context_chunks, realtime_data)
        
        # 4. Format response
        response = {
            'question': question,
            'answer': answer,
            'sources': [
                {
                    'filename': chunk['filename'],
                    'doc_type': chunk['doc_type'],
                    'similarity': chunk['similarity'],
                    'text_preview': chunk['text'][:300] + "..."
                }
                for chunk in context_chunks
            ],
            'num_sources': len(context_chunks),
            'realtime_data': realtime_data,
            'ticker': ticker
        }
        
        return response


# USAGE EXAMPLE
if __name__ == "__main__":
    from vector_db_setup import VectorDBManager
    from embedding_generator import EmbeddingGenerator
    
    # Initialize components
    print("Initializing RAG system...")
    embedder = EmbeddingGenerator()
    vector_db = VectorDBManager()
    
    # Check if database has data
    stats = vector_db.get_stats()
    if stats['total_chunks'] == 0:
        print("\n⚠️ Vector database is empty!")
        print("Please run: python run_pipeline.py")
        exit(1)
    
    rag = RAGPipeline(vector_db, embedder)
    
    # Example queries
    test_queries = [
        ("What was Apple's revenue in the most recent quarter?", "AAPL"),
        ("What are the main risk factors mentioned?", None),
        ("How is the company performing financially?", "AAPL"),
    ]
    
    for query, ticker in test_queries:
        response = rag.query(query, ticker=ticker)
        
        print("\n" + "="*60)
        print(f"Q: {response['question']}")
        print(f"\nA: {response['answer']}")
        print(f"\nSources ({response['num_sources']}):")
        for i, source in enumerate(response['sources']):
            print(f"  {i+1}. {source['filename']} ({source['doc_type']}) - similarity: {source['similarity']:.3f}")
        
        if response['realtime_data']:
            print(f"\nReal-time: {response['realtime_data']['company_name']} @ ${response['realtime_data']['current_price']}")
        
        print("\n" + "-"*60)