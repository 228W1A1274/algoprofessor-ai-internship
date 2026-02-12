# run_pipeline.py
from data_fetcher import StockDataFetcher
from document_processor import DocumentProcessor
from chunking_strategy import SmartChunker
from embedding_generator import EmbeddingGenerator
from vector_db_setup import VectorDBManager

def main():
    print("="*60)
    print("🚀 BUILDING RAG SYSTEM")
    print("="*60)
    
    # Step 1: Fetch data
    print("\n📥 STEP 1: Fetching stock data...")
    fetcher = StockDataFetcher()
    ticker = "AAPL"
    
    # Get real-time data
    realtime = fetcher.fetch_realtime_stock_data(ticker)
    print(f"   Current price: ${realtime['current_price']}")
    
    # Download SEC filings
    sec_files = fetcher.download_sec_filings(ticker, "10-K", num_filings=2)
    print(f"   Downloaded {len(sec_files)} SEC filings")
    
    # Fetch news
    news_files = fetcher.fetch_financial_news(ticker, days=7)
    print(f"   Downloaded {len(news_files)} news articles")
    
    # Step 2: Process documents
    print("\n📄 STEP 2: Processing documents...")
    processor = DocumentProcessor()
    
    sec_docs = processor.process_directory("data/sec_filings")
    news_docs = processor.process_directory("data/news_articles")
    
    all_docs = sec_docs + news_docs
    print(f"   Total documents: {len(all_docs)}")
    
    # Step 3: Chunk documents
    print("\n✂️ STEP 3: Chunking documents...")
    chunker = SmartChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.chunk_documents(all_docs)
    print(f"   Created {len(chunks)} chunks")
    
    # Step 4: Generate embeddings
    print("\n🔢 STEP 4: Generating embeddings...")
    embedder = EmbeddingGenerator()
    embedded_chunks = embedder.embed_chunks(chunks)
    print(f"   Embedded {len(embedded_chunks)} chunks")
    
    # Step 5: Store in database
    print("\n💾 STEP 5: Storing in vector database...")
    db = VectorDBManager()
    db.add_chunks(embedded_chunks)
    
    # Show final stats
    stats = db.get_stats()
    print("\n" + "="*60)
    print("✅ RAG SYSTEM BUILT SUCCESSFULLY!")
    print("="*60)
    print(f"Total chunks in database: {stats['total_chunks']}")
    print(f"Collection name: {stats['collection_name']}")
    print("\nNext step: Run the web app:")
    print("  streamlit run qa_app.py")  # ← Updated name
    print("="*60)

if __name__ == "__main__":
    main()