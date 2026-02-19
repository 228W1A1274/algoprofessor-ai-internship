# =============================================================================
# embedding_gen.py
# PURPOSE: Load sample documents and generate dense vector embeddings using
#          a local SBERT model (no API key or cost required).
# =============================================================================

# SentenceTransformer is the main class from the sentence-transformers library.
# It wraps HuggingFace models and gives us a clean .encode() method.
from sentence_transformers import SentenceTransformer

# 'all-MiniLM-L6-v2' is a lightweight but powerful SBERT model.
# - Output dimension: 384 floats per sentence
# - It's fast, free, runs fully locally, and is great for semantic similarity.
# - On first run, it downloads ~90MB from HuggingFace automatically.
MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# SAMPLE CORPUS
# These are the documents we want to store and search through.
# In a real project, you'd load these from a database, CSV, or text files.
# Each string is one "document" or "chunk" that will get its own embedding.
# ---------------------------------------------------------------------------
DOCUMENTS = [
    "The HR department handles employee onboarding and benefits administration.",
    "Python is a versatile programming language used in data science and web development.",
    "Vector databases store high-dimensional embeddings for fast similarity search.",
    "Machine learning models require large datasets for effective training.",
    "The annual performance review process begins every December for all employees.",
    "Neural networks are inspired by the structure of biological neurons in the brain.",
    "Remote work policies allow employees to work from home up to three days a week.",
    "Transformer models revolutionized natural language processing after the 2017 paper.",
    "Health insurance enrollment opens during the first two weeks of January.",
    "HNSW (Hierarchical Navigable Small World) is a graph-based ANN algorithm used in vector DBs.",
]

# Each document gets a unique string ID so we can reference it later in the DB.
# IDs must be strings for both ChromaDB and Pinecone.
DOC_IDS = [f"doc_{i}" for i in range(len(DOCUMENTS))]
# Result: ["doc_0", "doc_1", "doc_2", ..., "doc_9"]


def load_model(model_name: str) -> SentenceTransformer:
    """
    Load the SBERT model from HuggingFace (downloads on first run, cached after).
    
    Args:
        model_name: The HuggingFace model identifier string.
    Returns:
        A loaded SentenceTransformer model ready for encoding.
    """
    print(f"[INFO] Loading embedding model: '{model_name}'")
    print("[INFO] (First run will download ~90MB — subsequent runs use cache)")
    
    # SentenceTransformer downloads and caches the model in ~/.cache/huggingface/
    model = SentenceTransformer(model_name)
    
    print(f"[INFO] Model loaded successfully.")
    print(f"[INFO] Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def generate_embeddings(model: SentenceTransformer, documents: list[str]) -> list[list[float]]:
    """
    Convert a list of text strings into their vector embeddings.
    
    Args:
        model: A loaded SentenceTransformer model.
        documents: List of text strings to embed.
    Returns:
        List of embedding vectors (each is a list of 384 floats).
    """
    print(f"\n[INFO] Generating embeddings for {len(documents)} documents...")
    
    # .encode() processes all documents in one efficient batch.
    # - show_progress_bar=True prints a tqdm progress bar during encoding.
    # - Returns a numpy array of shape (num_docs, embedding_dim).
    embeddings_numpy = model.encode(documents, show_progress_bar=True)
    
    # Convert from numpy array to a plain Python list of lists.
    # We do this because ChromaDB and Pinecone both expect standard Python lists,
    # not numpy arrays. .tolist() handles the conversion cleanly.
    embeddings = embeddings_numpy.tolist()
    
    print(f"[INFO] Done! Generated {len(embeddings)} embeddings.")
    print(f"[INFO] Each embedding has {len(embeddings[0])} dimensions.")
    return embeddings


# This block only runs when you execute this file directly (python embedding_gen.py),
# NOT when it's imported by another script like vector_db_setup.py.
if __name__ == "__main__":
    # Step 1: Load the model
    model = load_model(MODEL_NAME)
    
    # Step 2: Generate embeddings
    embeddings = generate_embeddings(model, DOCUMENTS)
    
    # Step 3: Show a preview so you can verify things are working
    print("\n--- PREVIEW ---")
    print(f"Document: '{DOCUMENTS[0]}'")
    print(f"Embedding (first 8 values): {embeddings[0][:8]}")
    print(f"Total documents embedded: {len(embeddings)}")
    print("\n[SUCCESS] embedding_gen.py completed. Ready for vector_db_setup.py")