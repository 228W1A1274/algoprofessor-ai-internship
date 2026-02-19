# =============================================================================
# vector_db_setup.py
# PURPOSE: Take the embeddings generated in embedding_gen.py and store them in:
#          1. ChromaDB  — local, on-disk vector database (great for prototyping)
#          2. Pinecone  — managed cloud vector database (production-ready)
# =============================================================================

import os
import time

# python-dotenv loads variables from your .env file into os.environ.
# This is the safest way to handle secrets — never hardcode API keys in code.
from dotenv import load_dotenv

# --- ChromaDB imports ---
# chromadb is the local vector DB library. It runs entirely on your machine.
import chromadb
from chromadb.config import Settings

# --- Pinecone imports ---
# The modern Pinecone Python SDK (v3+) uses this import style.
from pinecone import Pinecone, ServerlessSpec

# --- Import our embedding logic from the sibling script ---
# Since embedding_gen.py is in the same folder, Python can import it directly.
# We import the functions and constants we defined there.
from embedding_gen import (
    load_model,
    generate_embeddings,
    DOCUMENTS,
    DOC_IDS,
    MODEL_NAME,
)

# Load the .env file so PINECONE_API_KEY becomes available via os.getenv()
load_dotenv()


# =============================================================================
# CONFIGURATION — change these values to match your preferences
# =============================================================================

# ChromaDB will save its data in this local folder (created automatically).
CHROMA_PERSIST_PATH = "./chroma_db_storage"

# Name for the collection inside ChromaDB (like a table in SQL).
CHROMA_COLLECTION_NAME = "semantic_search_collection"

# Name for your NEW Pinecone index.
# IMPORTANT: This is a different index from your existing "HR index".
# You can have multiple indexes on the free Pinecone tier (check your quota).
PINECONE_INDEX_NAME = "semantic-search-index"

# Must match the output dimension of your embedding model.
# all-MiniLM-L6-v2 produces 384-dimensional vectors.
EMBEDDING_DIMENSION = 384

# Cosine similarity is the standard metric for text semantic search.
# It measures the angle between vectors, ignoring magnitude.
PINECONE_METRIC = "cosine"


# =============================================================================
# PART 1: ChromaDB Setup
# =============================================================================

def setup_chromadb(documents: list[str], embeddings: list[list[float]], ids: list[str]):
    """
    Initialize a persistent ChromaDB client, create a collection, and upsert data.
    
    Args:
        documents: The raw text strings (stored as metadata for retrieval).
        embeddings: The dense vector embeddings (384-dim each).
        ids: Unique string IDs for each document.
    """
    print("\n" + "="*60)
    print("SETTING UP CHROMADB (Local)")
    print("="*60)

    # PersistentClient saves the database to disk at the given path.
    # Unlike the old EphemeralClient (in-memory), this survives script restarts.
    # The folder is created automatically if it doesn't exist.
    print(f"[ChromaDB] Initializing persistent client at: '{CHROMA_PERSIST_PATH}'")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)

    # get_or_create_collection: If the collection already exists, retrieve it.
    # If not, create a new one. This makes the script safe to re-run.
    # metadata={"hnsw:space": "cosine"} tells ChromaDB to use cosine similarity
    # for its internal HNSW index (matches our Pinecone config for consistency).
    print(f"[ChromaDB] Getting or creating collection: '{CHROMA_COLLECTION_NAME}'")
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # upsert() = "update if exists, insert if not".
    # This is safer than add() because re-running won't cause duplicate errors.
    # Parameters:
    #   - ids: list of unique string IDs
    #   - embeddings: the pre-computed vectors we pass in (ChromaDB won't re-compute them)
    #   - documents: the original text, stored alongside the vector as metadata
    print(f"[ChromaDB] Upserting {len(documents)} documents...")
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
    )

    # Verify the upsert worked by checking the count.
    count = collection.count()
    print(f"[ChromaDB] ✅ Success! Collection '{CHROMA_COLLECTION_NAME}' now has {count} vectors.")
    
    return collection


# =============================================================================
# PART 2: Pinecone Setup
# =============================================================================

def setup_pinecone(documents: list[str], embeddings: list[list[float]], ids: list[str]):
    """
    Initialize Pinecone, create a serverless index if needed, and upsert data.
    
    Args:
        documents: The raw text strings (stored as metadata in Pinecone).
        embeddings: The dense vector embeddings (384-dim each).
        ids: Unique string IDs for each document.
    """
    print("\n" + "="*60)
    print("SETTING UP PINECONE (Cloud)")
    print("="*60)

    # Retrieve the API key from environment variables loaded from .env
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError(
            "[Pinecone] ERROR: PINECONE_API_KEY not found. "
            "Make sure your .env file exists and contains PINECONE_API_KEY=your_key"
        )

    # Create a Pinecone client instance using your API key.
    # This replaces the old `pinecone.init()` pattern from SDK v2.
    print("[Pinecone] Authenticating with API key...")
    pc = Pinecone(api_key=api_key)

    # Check if our target index already exists (to avoid creating it twice).
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    print(f"[Pinecone] Existing indexes in your account: {existing_indexes}")

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"[Pinecone] Index '{PINECONE_INDEX_NAME}' not found. Creating it...")
        
        # create_index() provisions a new vector index in the cloud.
        # ServerlessSpec(cloud="aws", region="us-east-1") uses the free-tier
        # serverless option (no pods needed, pay-per-query, scales to zero).
        # "us-east-1" on AWS is the recommended default for the free tier.
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,   # MUST match your embedding model output
            metric=PINECONE_METRIC,          # cosine, euclidean, or dotproduct
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        
        # Pinecone index creation is asynchronous — we wait until it's ready.
        # Polling every 3 seconds until the index status becomes "Ready".
        print("[Pinecone] Waiting for index to become ready", end="", flush=True)
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(3)
            print(".", end="", flush=True)
        print(" Ready!")
    else:
        print(f"[Pinecone] Index '{PINECONE_INDEX_NAME}' already exists. Reusing it.")

    # Get a handle to the index for data operations (upsert, query, etc.)
    index = pc.Index(PINECONE_INDEX_NAME)

    # Pinecone upsert expects a list of tuples: (id, vector, metadata_dict)
    # The metadata dict can hold any key-value pairs you want returned at query time.
    # Here we store the original text so we can display it in search results.
    print(f"[Pinecone] Preparing {len(documents)} vectors for upsert...")
    vectors_to_upsert = [
        (
            ids[i],            # Unique string ID
            embeddings[i],     # The 384-dim float vector
            {"text": documents[i]}  # Metadata: store original text for display
        )
        for i in range(len(documents))
    ]

    # upsert() sends vectors to the cloud index.
    # namespace="" uses the default namespace (no separation needed for this project).
    index.upsert(vectors=vectors_to_upsert, namespace="")
    
    # Brief pause to let Pinecone index the vectors before we query stats.
    time.sleep(2)
    
    # describe_index_stats() returns live info about the index contents.
    stats = index.describe_index_stats()
    print(f"[Pinecone] ✅ Success! Index stats: {stats}")
    
    return index


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # --- Step 1: Generate embeddings (reuses embedding_gen.py logic) ---
    print("[STEP 1/3] Loading model and generating embeddings...")
    model = load_model(MODEL_NAME)
    embeddings = generate_embeddings(model, DOCUMENTS)

    # --- Step 2: Set up ChromaDB ---
    print("\n[STEP 2/3] Setting up ChromaDB...")
    chroma_collection = setup_chromadb(DOCUMENTS, embeddings, DOC_IDS)

    # --- Step 3: Set up Pinecone ---
    print("\n[STEP 3/3] Setting up Pinecone...")
    pinecone_index = setup_pinecone(DOCUMENTS, embeddings, DOC_IDS)

    print("\n" + "="*60)
    print("✅ ALL DONE! Both databases are populated and ready.")
    print("   Run hybrid_search.py next to perform searches.")
    print("="*60)