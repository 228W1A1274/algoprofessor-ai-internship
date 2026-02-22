"""
graph_rag.py — Day 15: Knowledge Graph RAG with Neo4j + NetworkX
================================================================
Author : AI/ML Mentor (Day 15 Curriculum)
Purpose: Build a knowledge graph from raw documents via entity/relation extraction,
         store it in Neo4j, visualise with NetworkX, and query it with LlamaIndex's
         KnowledgeGraphIndex for graph-aware RAG.

Architecture:
  Documents → LLM Entity/Relation Extraction → NetworkX (in-memory graph)
           → Neo4j (persistent graph DB) → LlamaIndex KnowledgeGraphIndex → RAG
"""

import os
import json
import logging
from typing import List, Tuple
from dotenv import load_dotenv

# ── Graph libraries ──────────────────────────────────────────────────────────
import networkx as nx
import matplotlib.pyplot as plt

# ── Neo4j driver ─────────────────────────────────────────────────────────────
from neo4j import GraphDatabase

# ── LlamaIndex ───────────────────────────────────────────────────────────────
from llama_index.core import (
    KnowledgeGraphIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
)
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.chunk_size = 256  # Smaller chunks = more precise entity extraction

DATA_DIR = "./data"

# Neo4j connection — set these in your .env file
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME",  "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD",  "your_password")


# ─────────────────────────────────────────────────────────────────────────────
# 1. ENTITY & RELATION EXTRACTION (LLM-powered)
# ─────────────────────────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are a knowledge graph extractor. Given a text passage, 
extract all (subject, predicate, object) triples that describe factual relationships.

Rules:
- Subject and Object should be concise noun phrases (1-4 words).
- Predicate should be a verb phrase (1-3 words), e.g., "founded", "works at", "is part of".
- Only extract clearly stated facts, no inferences.
- Return ONLY a JSON array of triples. No markdown, no explanation.

Format: [["subject", "predicate", "object"], ...]

Text:
{text}

JSON:"""


def extract_triples(text: str) -> List[Tuple[str, str, str]]:
    """
    Uses the LLM to extract (subject, predicate, object) triples from a text chunk.
    Returns a list of 3-tuples.

    This is the core of Graph RAG construction — turning unstructured prose into
    a structured knowledge graph.
    """
    prompt = EXTRACTION_PROMPT.format(text=text[:2000])  # Truncate to avoid token limit
    llm = Settings.llm
    raw = llm.complete(prompt).text.strip()

    # Strip markdown fences if the LLM wraps output in ```json
    import re
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        data = json.loads(raw)
        triples = [(t[0].strip(), t[1].strip(), t[2].strip()) for t in data if len(t) == 3]
        log.info("  Extracted %d triples", len(triples))
        return triples
    except (json.JSONDecodeError, IndexError) as e:
        log.warning("  Triple extraction failed: %s | Raw: %s", e, raw[:200])
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 2. NETWORKX IN-MEMORY GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_networkx_graph(all_triples: List[Tuple[str, str, str]]) -> nx.DiGraph:
    """
    Builds a directed graph where:
      - Nodes  = entities (subjects & objects)
      - Edges  = predicates (relations between entities)

    NetworkX is used for in-memory graph analytics:
      - Degree centrality: which entities are most connected?
      - Shortest path: how are two entities related?
      - Community detection: which entity clusters exist?
    """
    G = nx.DiGraph()  # Directed — "A founded B" ≠ "B founded A"

    for subj, pred, obj in all_triples:
        # Add nodes (nx ignores duplicate add calls)
        G.add_node(subj, type="entity")
        G.add_node(obj,  type="entity")
        # Add edge with predicate as attribute
        G.add_edge(subj, obj, relation=pred)

    log.info("NetworkX graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def visualise_graph(G: nx.DiGraph, output_path: str = "knowledge_graph.png"):
    """
    Renders the knowledge graph as a matplotlib figure.
    Saves to disk — attach to your Streamlit UI or eval notebook.
    """
    plt.figure(figsize=(16, 10))

    # Spring layout positions nodes based on edge forces (like a force-directed graph)
    pos = nx.spring_layout(G, k=2.0, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color="#4EA8DE", node_size=800, alpha=0.9)

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="#888", arrows=True,
                           arrowsize=20, connectionstyle="arc3,rad=0.1")

    # Node labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="white", font_weight="bold")

    # Edge labels (predicates)
    edge_labels = {(u, v): d["relation"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color="#333")

    plt.title("Knowledge Graph — Entity-Relation Map", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Graph visualisation saved to %s", output_path)


def analyse_graph(G: nx.DiGraph):
    """Prints key graph statistics useful for understanding the knowledge structure."""
    print("\n📊 Graph Analytics:")
    print(f"  Nodes (entities): {G.number_of_nodes()}")
    print(f"  Edges (relations): {G.number_of_edges()}")

    if G.number_of_nodes() > 0:
        # Top entities by in-degree (most referenced objects)
        in_deg = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n  Top entities by in-degree (most cited):")
        for node, deg in in_deg:
            print(f"    {node}: {deg}")

        # Degree centrality — nodes that act as "hubs"
        centrality = nx.degree_centrality(G)
        top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n  Top entities by degree centrality (hubs):")
        for node, score in top_central:
            print(f"    {node}: {score:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. NEO4J GRAPH STORE
# ─────────────────────────────────────────────────────────────────────────────

class Neo4jGraphStore:
    """
    Wrapper around the Neo4j Python driver for bulk triple insertion and querying.

    Neo4j stores the graph persistently with native graph traversal (Cypher queries),
    enabling production-scale graph RAG where:
      - NetworkX = analytics & visualisation (in-memory)
      - Neo4j = persistence & traversal at scale (disk-backed)
    """

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        log.info("Connected to Neo4j at %s", uri)

    def close(self):
        self.driver.close()

    def clear_graph(self):
        """Wipes all nodes and edges — useful for a fresh import."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        log.info("Neo4j graph cleared")

    def insert_triples(self, triples: List[Tuple[str, str, str]]):
        """
        Bulk-inserts (subject, predicate, object) triples into Neo4j using Cypher.

        MERGE creates a node/edge only if it doesn't already exist — idempotent.
        The predicate becomes the relationship type (edge label).

        Cypher pattern:
          MERGE (a:Entity {name: $subj})
          MERGE (b:Entity {name: $obj})
          MERGE (a)-[:PREDICATE]->(b)
        """
        with self.driver.session() as session:
            for subj, pred, obj in triples:
                # Cypher doesn't allow dynamic relationship types with MERGE directly,
                # so we use APOC or a parameterised workaround via apocMergeRelationship.
                # For simplicity here we use SET on a property instead.
                cypher = """
                MERGE (a:Entity {name: $subj})
                MERGE (b:Entity {name: $obj})
                MERGE (a)-[r:RELATES_TO {predicate: $pred}]->(b)
                """
                session.run(cypher, subj=subj, obj=obj, pred=pred)

        log.info("Inserted %d triples into Neo4j", len(triples))

    def query_neighbours(self, entity: str, depth: int = 2) -> List[dict]:
        """
        Returns all neighbours of an entity up to `depth` hops away.
        This is the core Graph RAG retrieval operation — instead of vector search,
        we traverse the graph to find related context.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH path = (start:Entity {name: $entity})-[*1..$depth]-(neighbour)
                RETURN 
                  neighbour.name AS neighbour,
                  [rel in relationships(path) | rel.predicate] AS predicates,
                  length(path) AS hops
                ORDER BY hops
                LIMIT 50
                """,
                entity=entity,
                depth=depth,
            )
            return [dict(record) for record in result]

    def full_text_search(self, keyword: str) -> List[str]:
        """Find all entities whose names contain a keyword."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($kw) RETURN e.name",
                kw=keyword,
            )
            return [r["e.name"] for r in result]


# ─────────────────────────────────────────────────────────────────────────────
# 4. LLAMAINDEX KNOWLEDGE GRAPH INDEX
# ─────────────────────────────────────────────────────────────────────────────

def build_llamaindex_kg(documents) -> KnowledgeGraphIndex:
    """
    LlamaIndex's KnowledgeGraphIndex handles the full pipeline:
      1. Chunks documents (respecting Settings.chunk_size)
      2. Calls the LLM to extract triples from each chunk
      3. Stores the graph in a SimpleGraphStore (in-memory; swap with Neo4jGraphStore for prod)
      4. Builds a hybrid retriever that combines graph traversal + vector search

    The resulting index supports both:
      - include_text=True  → attaches source text to graph nodes (richer context)
      - retriever_mode="keyword" | "embedding" | "hybrid"
    """
    log.info("Building LlamaIndex KnowledgeGraphIndex…")

    # SimpleGraphStore is LlamaIndex's built-in in-memory graph store.
    # Replace with Neo4jGraphStore adapter for production persistence.
    graph_store = SimpleGraphStore()
    storage_ctx = StorageContext.from_defaults(graph_store=graph_store)

    kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_ctx,
        max_triplets_per_chunk=10,   # LLM extracts up to 10 triples per chunk
        include_text=True,           # Source text is stored alongside graph nodes
        show_progress=True,
    )

    log.info("KnowledgeGraphIndex built successfully")
    return kg_index


def graph_rag_query(kg_index: KnowledgeGraphIndex, question: str) -> str:
    """
    Queries the knowledge graph index.

    retriever_mode options:
      "keyword"   → entity keyword match → graph traversal → context assembly
      "embedding" → embed question → find nearest graph nodes → traversal
      "hybrid"    → both of the above, results merged

    Graph traversal depth (graph_store_query_depth) controls how many hops
    away from the matched entity the retriever will walk.
    """
    query_engine = kg_index.as_query_engine(
        include_text=True,
        retriever_mode="hybrid",
        response_mode="tree_summarize",  # Hierarchical summarisation for long context
        graph_store_query_depth=2,       # Walk up to 2 hops from matched entity
        similarity_top_k=3,
        verbose=True,
    )

    log.info("[Graph RAG] Running query: %s", question)
    response = query_engine.query(question)
    return str(response)


# ─────────────────────────────────────────────────────────────────────────────
# 5. DEMO RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Create demo data if none exists ──────────────────────────────────────
    if not os.listdir(DATA_DIR):
        with open(f"{DATA_DIR}/demo.txt", "w") as f:
            f.write(
                "Anthropic was founded by Dario Amodei and Daniela Amodei. "
                "Anthropic developed Claude, which is an AI assistant. "
                "Claude uses Constitutional AI, a technique pioneered by Anthropic. "
                "OpenAI created GPT-4 and also developed DALL-E. "
                "Sam Altman is the CEO of OpenAI. "
                "LlamaIndex is maintained by Jerry Liu and supports Neo4j integration. "
                "Neo4j is a graph database developed by Neo Technology. "
                "NetworkX is a Python library for graph analysis developed at Los Alamos."
            )

    documents = SimpleDirectoryReader(DATA_DIR).load_data()

    # ── Step 1: Extract triples from all document chunks ─────────────────────
    all_triples = []
    for doc in documents:
        # In production you'd chunk docs first; here we use the full doc text
        triples = extract_triples(doc.text)
        all_triples.extend(triples)

    print(f"\n✅ Total triples extracted: {len(all_triples)}")
    for t in all_triples[:10]:
        print(f"  ({t[0]}) --[{t[1]}]--> ({t[2]})")

    # ── Step 2: Build NetworkX graph ─────────────────────────────────────────
    G = build_networkx_graph(all_triples)
    analyse_graph(G)
    visualise_graph(G, "knowledge_graph.png")

    # ── Step 3: Persist to Neo4j (optional — requires Docker) ────────────────
    try:
        neo4j_store = Neo4jGraphStore(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        neo4j_store.clear_graph()
        neo4j_store.insert_triples(all_triples)

        # Example: find what's related to "Anthropic" within 2 hops
        neighbours = neo4j_store.query_neighbours("Anthropic", depth=2)
        print("\n🔗 Neo4j neighbours of 'Anthropic':")
        for n in neighbours[:5]:
            print(f"  {n}")

        neo4j_store.close()
    except Exception as e:
        log.warning("Neo4j unavailable (is Docker running?): %s", e)
        log.warning("Skipping Neo4j step — continuing with in-memory graph only.")

    # ── Step 4: LlamaIndex KG-based RAG query ────────────────────────────────
    kg_index = build_llamaindex_kg(documents)
    question = "Who founded Anthropic and what did they create?"
    answer = graph_rag_query(kg_index, question)

    print("\n" + "═" * 70)
    print("🧠  GRAPH RAG ANSWER")
    print("═" * 70)
    print(answer)


if __name__ == "__main__":
    main()