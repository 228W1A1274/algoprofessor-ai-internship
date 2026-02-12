"""
WHAT: Splits documents into optimal chunks for RAG
WHY: LLMs have token limits, chunks must be semantically meaningful
HOW: Uses tiktoken for accurate token counting, adds overlap
"""

import tiktoken
from typing import List, Dict

class SmartChunker:
    def __init__(self, 
                 chunk_size=1000,      # Target tokens per chunk
                 chunk_overlap=100,     # Overlap between chunks
                 model_name="gpt-3.5-turbo"):
        """
        PARAMS:
        - chunk_size: Target size in tokens (not characters!)
        - chunk_overlap: Tokens to overlap (preserves context)
        - model_name: For tiktoken encoding
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize tokenizer (same as OpenAI uses)
        self.encoding = tiktoken.encoding_for_model(model_name)
        
        print(f"🔧 Chunker initialized:")
        print(f"   Chunk size: {chunk_size} tokens")
        print(f"   Overlap: {chunk_overlap} tokens")
    
    def count_tokens(self, text: str) -> int:
        """
        WHAT: Counts exact tokens in text
        WHY: Characters != tokens (critical for embeddings)
        EXAMPLE: "Hello world" = 2 tokens, not 11 characters
        """
        return len(self.encoding.encode(text))
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        WHAT: Splits text into sentences
        WHY: Better than arbitrary character splits
        HOW: Simple approach (can be improved with spaCy/NLTK)
        """
        # Simple sentence splitting (you can use spaCy for better results)
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_chunks(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        WHAT: Main chunking logic
        WHY: Creates semantically meaningful chunks with overlap
        RETURNS: List of chunk dicts with text + metadata
        """
        print(f"\n✂️ Chunking document...")
        print(f"   Total tokens: {self.count_tokens(text)}")
        
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_token_count = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence exceeds chunk_size, save current chunk
            if current_token_count + sentence_tokens > self.chunk_size and current_chunk:
                # Save chunk
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'token_count': current_token_count,
                    'metadata': metadata or {}
                })
                
                # Start new chunk with overlap
                # Keep last few sentences for context
                overlap_text = " ".join(current_chunk[-3:])  # Last 3 sentences
                overlap_tokens = self.count_tokens(overlap_text)
                
                if overlap_tokens <= self.chunk_overlap:
                    current_chunk = current_chunk[-3:]
                    current_token_count = overlap_tokens
                else:
                    current_chunk = []
                    current_token_count = 0
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_token_count += sentence_tokens
        
        # Don't forget the last chunk!
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'token_count': current_token_count,
                'metadata': metadata or {}
            })
        
        print(f"✅ Created {len(chunks)} chunks")
        print(f"   Avg tokens per chunk: {sum(c['token_count'] for c in chunks) / len(chunks):.0f}")
        
        return chunks
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        WHAT: Chunks multiple documents
        WHY: Batch processing with metadata preservation
        RETURNS: All chunks from all documents
        """
        print(f"\n📦 Chunking {len(documents)} documents...")
        
        all_chunks = []
        
        for i, doc in enumerate(documents):
            print(f"\nDocument {i+1}/{len(documents)}: {doc['filename']}")
            
            # Prepare metadata
            metadata = {
                'source': doc['source'],
                'filename': doc['filename'],
                'doc_type': doc['doc_type']
            }
            
            # Create chunks
            chunks = self.create_chunks(doc['text'], metadata)
            
            # Add chunk index
            for j, chunk in enumerate(chunks):
                chunk['metadata']['chunk_index'] = j
                chunk['metadata']['total_chunks'] = len(chunks)
            
            all_chunks.extend(chunks)
        
        print(f"\n✅ Total chunks created: {len(all_chunks)}")
        return all_chunks


# USAGE EXAMPLE
if __name__ == "__main__":
    # Import WITHOUT src prefix
    from document_processor import DocumentProcessor
    
    # Process documents first
    processor = DocumentProcessor()
    documents = processor.process_directory("data/sec_filings")
    
    # Chunk them
    chunker = SmartChunker(chunk_size=800, chunk_overlap=100)
    chunks = chunker.chunk_documents(documents[:2])  # Test with 2 docs
    
    # Show sample chunk
    print("\n" + "="*50)
    print("SAMPLE CHUNK:")
    print(f"Text: {chunks[0]['text'][:300]}...")
    print(f"Tokens: {chunks[0]['token_count']}")
    print(f"Metadata: {chunks[0]['metadata']}")



        
    """ **EXPLANATION:**
    - **Token-based chunking**: Uses tiktoken (same as OpenAI) for accurate counting
    - **Sentence boundaries**: Doesn't split mid-sentence
    - **Overlap**: Last 3 sentences of previous chunk start next chunk (preserves context)
    - **Metadata**: Each chunk knows its source document, type, position

    **Why overlap matters:**
    ```
    Chunk 1: "...Apple reported revenue of $90B. This was a 10% increase..."
    Chunk 2: "This was a 10% increase. The growth was driven by iPhone sales..."
    """