# ═══════════════════════════════════════════════════════════════════════
# Day 15 — Advanced RAG Setup Guide
# ═══════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────
# PHASE 2: ENVIRONMENT SETUP
# ─────────────────────────────────────────────────────────────────────

# ── STEP 1: Python dependencies ──────────────────────────────────────
pip install llama-index \
            llama-index-llms-openai \
            llama-index-embeddings-openai \
            llama-index-graph-stores-neo4j \
            llama-index-readers-file \
            openai \
            streamlit \
            neo4j \
            networkx \
            matplotlib \
            ragas \
            datasets \
            pandas \
            seaborn \
            tqdm \
            nest_asyncio \
            python-dotenv

# ── STEP 2: Create your .env file ────────────────────────────────────
# Create a file called .env in your project root with these values:

OPENAI_API_KEY=sk-your-openai-key-here
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password_here    # You set this in Step 3

# ── STEP 3: Neo4j via Docker ──────────────────────────────────────────
# Option A — Docker (recommended, no installation needed):
docker run \
  --name neo4j-day15 \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_neo4j_password_here \
  -e NEO4J_PLUGINS='["apoc"]' \
  --detach \
  neo4j:5.18

# Wait ~30 seconds for Neo4j to start, then verify:
# Open http://localhost:7474 in your browser
# Login: neo4j / your_neo4j_password_here

# Option B — Docker Compose (add to your project):
# See docker-compose.yml below

# To stop Neo4j:      docker stop neo4j-day15
# To restart Neo4j:   docker start neo4j-day15
# To remove Neo4j:    docker rm neo4j-day15

# ── STEP 4: Project structure ─────────────────────────────────────────
mkdir -p day15_project/data
cd day15_project

# Copy all 4 deliverable files here:
# day15_project/
#   ├── advanced_rag.py
#   ├── graph_rag.py
#   ├── streaming_qa_app.py
#   ├── day15_eval.ipynb
#   ├── .env
#   └── data/          ← Put your .pdf or .txt files here

# ─────────────────────────────────────────────────────────────────────
# PHASE 4: EXECUTION — Step-by-step
# ─────────────────────────────────────────────────────────────────────

# ── Terminal 1: Start Neo4j ──────────────────────────────────────────
docker start neo4j-day15
# Verify at http://localhost:7474

# ── Terminal 2: Run advanced_rag.py ─────────────────────────────────
cd day15_project
python advanced_rag.py
# Expected output:
#   ══════ 1️⃣  HyDE RESULT ══════
#   [LlamaIndex answer here]
#   ══════ 2️⃣  SELF-QUERY RESULT ══════
#   [LlamaIndex answer here]
#   ══════ 3️⃣  MULTI-QUERY RESULT ══════
#   [LlamaIndex answer here]

# ── Terminal 3: Run graph_rag.py ─────────────────────────────────────
python graph_rag.py
# Expected output:
#   ✅ Total triples extracted: N
#   📊 Graph Analytics: ...
#   🧠 GRAPH RAG ANSWER: ...
#   knowledge_graph.png saved

# ── Terminal 4: Launch Streamlit app ─────────────────────────────────
streamlit run streaming_qa_app.py
# Opens http://localhost:8501 in your browser automatically
# Type any question → see sources appear → watch tokens stream in real-time

# ── Terminal 5: Open evaluation notebook ─────────────────────────────
jupyter notebook day15_eval.ipynb
# Run cells top-to-bottom
# Final cell prints interview-ready strategy comparison

# ─────────────────────────────────────────────────────────────────────
# DOCKER COMPOSE (optional — runs Neo4j + app together)
# ─────────────────────────────────────────────────────────────────────

# docker-compose.yml:
# version: '3.8'
# services:
#   neo4j:
#     image: neo4j:5.18
#     ports:
#       - "7474:7474"
#       - "7687:7687"
#     environment:
#       - NEO4J_AUTH=neo4j/your_password
#       - NEO4J_PLUGINS=["apoc"]
#     volumes:
#       - ./neo4j_data:/data
#   streamlit:
#     build: .
#     ports:
#       - "8501:8501"
#     command: streamlit run streaming_qa_app.py
#     depends_on:
#       - neo4j
#     env_file:
#       - .env

# ─────────────────────────────────────────────────────────────────────
# TROUBLESHOOTING
# ─────────────────────────────────────────────────────────────────────

# Neo4j connection refused?
#   → Run: docker ps   (check container is running)
#   → Run: docker logs neo4j-day15   (check for startup errors)

# OpenAI rate limit?
#   → Add time.sleep(1) between eval loop iterations
#   → Switch to gpt-3.5-turbo for faster/cheaper evals

# LlamaIndex index not building?
#   → Delete ./storage folder and re-run (forces rebuild)
#   → Check that ./data/ has at least one .txt or .pdf file

# Streamlit not streaming?
#   → Ensure Settings.llm has streaming=True
#   → Check that query engine uses streaming=True parameter
