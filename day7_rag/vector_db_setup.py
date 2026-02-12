"""
WHAT: Sets up ChromaDB vector database for semantic search
WHY: Enables fast similarity search across thousands of chunks
HOW: Stores embeddings + metadata, uses cosine similarity
"""

import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict

class VectorDBManager:
    def __init__(self, persist_directory="data/vector_db", collection_name="stock_documents"):
        """
        WHAT: Initialize ChromaDB client
        WHY: ChromaDB is fast, free, and works locally
        
        persist_directory: Where to save the database
        collection_name: Name for this dataset (like a table in SQL)
        """
        print(f"🗄️ Initializing ChromaDB...")
        print(f"   Persist directory: {persist_directory}")
        print(f"   Collection name: {collection_name}")
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Stock market documents with RAG"}
        )
        
        print(f"✅ Collection ready: {self.collection.name}")
        print(f"   Current document count: {self.collection.count()}")
    
    def add_chunks(self, chunks: List[Dict]):
        """
        WHAT: Add embedded chunks to vector database
        WHY: Makes them searchable by semantic similarity
        HOW: Stores embeddings, text, and metadata
        """
        print(f"\n📥 Adding {len(chunks)} chunks to database...")
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            # Generate unique ID
            chunk_id = f"chunk_{i}_{hash(chunk['text'][:50])}"
            ids.append(chunk_id)
            
            # Extract embedding
            embeddings.append(chunk['embedding'])
            
            # Store original text
            documents.append(chunk['text'])
            
            # Store metadata (ChromaDB requires all values to be simple types)
            metadata = {
                'source': chunk['metadata'].get('source', 'unknown'),
                'filename': chunk['metadata'].get('filename', 'unknown'),
                'doc_type': chunk['metadata'].get('doc_type', 'unknown'),
                'chunk_index': chunk['metadata'].get('chunk_index', 0),
                'token_count': chunk.get('token_count', 0)
            }
            metadatas.append(metadata)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"✅ Added {len(chunks)} chunks successfully!")
        print(f"   Total documents in DB: {self.collection.count()}")
    
    def search(self, query_embedding: List[float], top_k=5, filter_metadata=None):
        """
        WHAT: Search for similar chunks using vector similarity
        WHY: Core of RAG - finds relevant context for user query
        HOW: Uses cosine similarity between query and stored embeddings
        
        PARAMS:
        - query_embedding: Vector representation of user query
        - top_k: How many results to return
        - filter_metadata: Dict to filter results (e.g., {'doc_type': 'SEC_FILING'})
        
        RETURNS: Dict with documents, distances, and metadata
        """
        print(f"\n🔍 Searching for top {top_k} similar chunks...")
        
        # Prepare where clause for filtering
        where = filter_metadata if filter_metadata else None
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )
        
        print(f"✅ Found {len(results['documents'][0])} results")
        
        return {
            'documents': results['documents'][0],
            'distances': results['distances'][0],
            'metadatas': results['metadatas'][0],
            'ids': results['ids'][0]
        }
    
    def search_with_metadata_filter(self, query_embedding: List[float], 
                                   doc_type=None, filename=None, top_k=5):
        """
        WHAT: Search with filtering by document type or filename
        WHY: Useful for queries like "What did Apple's 10-K say about revenue?"
        EXAMPLE: Only search SEC filings, not news articles
        """
        filter_dict = {}
        
        if doc_type:
            filter_dict['doc_type'] = doc_type
        if filename:
            filter_dict['filename'] = filename
        
        return self.search(query_embedding, top_k, filter_metadata=filter_dict if filter_dict else None)
    
    def reset_database(self):
        """
        WHAT: Delete all data and start fresh
        WHY: Useful during development/testing
        WARNING: This deletes everything!
        """
        print("⚠️ Resetting database...")
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(name=self.collection_name)
        print("✅ Database reset complete")
    
    def get_stats(self):
        """WHAT: Get database statistics"""
        count = self.collection.count()
        return {
            'total_chunks': count,
            'collection_name': self.collection_name
        }


# USAGE EXAMPLE
if __name__ == "__main__":
    from document_processor import DocumentProcessor
    from chunking_strategy import SmartChunker
    from embedding_generator import EmbeddingGenerator
    
    # Full pipeline test
    print("="*60)
    print("FULL INGESTION PIPELINE TEST")
    print("="*60)
    
    # 1. Process documents
    processor = DocumentProcessor()
    documents = processor.process_directory("data/sec_filings")[:2]
    
    # 2. Chunk
    chunker = SmartChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_documents(documents)
    
    # 3. Generate embeddings
    embedder = EmbeddingGenerator()
    embedded_chunks = embedder.embed_chunks(chunks)
    
    # 4. Store in vector DB
    print("\n" + "="*60)
    db = VectorDBManager()
    db.add_chunks(embedded_chunks)
    
    # 5. Test search
    print("\n" + "="*60)
    print("TESTING SEARCH")
    test_query = "What was the company's revenue?"
    query_embedding = embedder.generate_embedding(test_query)
    
    results = db.search(query_embedding, top_k=3)
    
    print(f"\nQuery: {test_query}")
    print(f"\nTop 3 Results:")
    for i, (doc, distance, metadata) in enumerate(zip(
        results['documents'], 
        results['distances'], 
        results['metadatas']
    )):
        print(f"\n--- Result {i+1} (similarity: {1-distance:.3f}) ---")
        print(f"Source: {metadata['filename']}")
        print(f"Text: {doc[:200]}...")
    
    # Show stats
    print("\n" + "="*60)
    stats = db.get_stats()
    print("DATABASE STATS:")
    for key, value in stats.items():
        print(f"  {key}: {value}")