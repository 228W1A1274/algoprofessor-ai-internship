"""
WHAT: Generates vector embeddings from text chunks
WHY: Vector embeddings enable semantic search
HOW: Uses HuggingFace sentence-transformers (FREE, local)
"""

import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class EmbeddingGenerator:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print(f"🔧 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"✅ Model loaded! Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def generate_embedding(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def generate_embeddings_batch(self, texts: List[str], batch_size=32) -> List[List[float]]:
        print(f"\n🔄 Generating embeddings for {len(texts)} texts...")
        print(f"   Batch size: {batch_size}")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"   Embedding shape: {embeddings.shape}")
        
        return embeddings.tolist()
    
    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        print(f"\n📊 Embedding {len(chunks)} chunks...")
        
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.generate_embeddings_batch(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
        
        print(f"✅ All chunks embedded successfully!")
        
        return chunks


# USAGE EXAMPLE
if __name__ == "__main__":
    # Import WITHOUT src prefix
    from document_processor import DocumentProcessor
    from chunking_strategy import SmartChunker
    
    print("="*60)
    processor = DocumentProcessor()
    documents = processor.process_directory("data/sec_filings")[:2]
    
    chunker = SmartChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_documents(documents)
    
    print("\n" + "="*60)
    embedder = EmbeddingGenerator()
    embedded_chunks = embedder.embed_chunks(chunks)
    
    print("\n" + "="*60)
    print("SAMPLE EMBEDDED CHUNK:")
    sample = embedded_chunks[0]
    print(f"Text: {sample['text'][:200]}...")
    print(f"Embedding dimensions: {len(sample['embedding'])}")
    print(f"First 10 values: {sample['embedding'][:10]}")
    print(f"Metadata: {sample['metadata']}")