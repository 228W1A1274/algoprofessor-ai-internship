# 🤖 AlgoProfessor Agent Creator
### Day 12 Milestone — AutoGen + Docker + FastAPI + GPT-4o + GitHub Actions

> Takes a natural language prompt → generates agent code → builds & tests in Docker → deploys as a live FastAPI service. Fully autonomous.

---

## 🏗️ Architecture

```
User Prompt
    │
    ▼
┌──────────────────────────────────────┐
│         AutoGen Pipeline             │
│  PlannerAgent → CoderAgent →         │
│  ReviewerAgent → OrchestratorAgent   │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│       Docker Deploy Pipeline         │
│  Write Code → Build Image →          │
│  Run Container → Test → Auto-Fix     │
└──────────────────────────────────────┘
    │
    ▼
Live Agent running at http://localhost:PORT/run
```

---

## 📁 Project Structure

```
agent-creator/
├── backend/
│   ├── main.py              # FastAPI app — all endpoints
│   ├── agent_engine.py      # AutoGen multi-agent pipeline
│   ├── code_generator.py    # GPT-4o code generation & fix
│   ├── docker_manager.py    # Docker build/run/test helpers
│   ├── deployer.py          # Full deploy pipeline + registry
│   ├── config.py            # Environment config
│   ├── requirements.txt
│   └── .env.example
├── docker/
│   └── Dockerfile           # Backend service image
├── docker-compose.yml
├── generated_agents/        # Auto-created agents live here
│   ├── registry.json        # Agent registry
│   └── calculator_agent/
│       └── agent.py
├── .github/
│   └── workflows/
│       └── deploy.yml       # CI/CD pipeline
└── README.md
```

---

## ⚡ Local Setup (5 Steps)

### Prerequisites
- Python 3.11+
- Docker Desktop running
- OpenAI API Key

### 1. Clone & Enter
```bash
git clone <your-repo>
cd agent-creator
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 4. Set Environment Variables
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 5. Run Backend
```bash
uvicorn main:app --reload
# → http://localhost:8000
# → Docs: http://localhost:8000/docs
```

---

## 🐳 Docker Compose (Full Stack)

```bash
# Copy env file
cp backend/.env.example backend/.env
# Edit backend/.env with your OPENAI_API_KEY

# Build & start
docker-compose up --build

# Stop
docker-compose down
```

---

## 🔥 End-to-End Test: "Create a Calculator Agent"

### Step 1 — Create Agent
```bash
curl -X POST http://localhost:8000/create-agent \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Create a calculator agent that solves math problems", "auto_deploy": true}'
```

**Response:**
```json
{
  "success": true,
  "agent_name": "calculator_agent",
  "description": "Solves arithmetic and word math problems using GPT-4o",
  "endpoint": "http://localhost:8100",
  "port": 8100,
  "test_output": "The answer is 42.",
  "attempts": 1,
  "container_id": "a3f7c9d1b2e4"
}
```

### Step 2 — Run Agent Directly
```bash
curl -X POST http://localhost:8100/run \
  -H "Content-Type: application/json" \
  -d '{"input": "What is 123 * 456?"}'
```

**Response:**
```json
{"output": "123 × 456 = 56,088"}
```

### Step 3 — Run via Proxy
```bash
curl -X POST http://localhost:8000/agents/calculator_agent/run \
  -H "Content-Type: application/json" \
  -d '{"agent_name": "calculator_agent", "input": "What is 25 squared?"}'
```

### Step 4 — List All Agents
```bash
curl http://localhost:8000/agents
```

---

## 🌐 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Service health check |
| `POST` | `/create-agent` | Create & deploy a new agent |
| `GET`  | `/agents` | List all deployed agents |
| `GET`  | `/agents/{name}` | Get agent details |
| `POST` | `/agents/{name}/run` | Proxy call to agent |

### POST /create-agent
```json
{
  "prompt": "Create a sentiment analysis agent",
  "auto_deploy": true
}
```

---

## 🛠️ Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `OPENAI_API_KEY not set` | Add key to `backend/.env` |
| `docker: command not found` | Install Docker Desktop, ensure it's running |
| `Port already in use` | `docker rm -f agent-<name>` or change `AGENT_PORT_START` |
| `Build failed: pip not found` | Ensure `python:3.11-slim` image is available (`docker pull python:3.11-slim`) |
| `Cannot connect to Docker daemon` | Start Docker Desktop or run `sudo systemctl start docker` |
| `ModuleNotFoundError in container` | Add missing lib to `requirements` list in the prompt |
| `openai.AuthenticationError` | Check your `OPENAI_API_KEY` is valid and has credits |

---

## ⚙️ GitHub Actions CI/CD Setup

Add these secrets in **GitHub → Settings → Secrets → Actions**:

| Secret | Value |
|--------|-------|
| `OPENAI_API_KEY` | Your OpenAI key |
| `DEPLOY_HOST` | Your server IP |
| `DEPLOY_USER` | SSH username |
| `DEPLOY_SSH_KEY` | Private SSH key |

Pipeline on `git push main`:
1. ✅ Lint (ruff) + Tests
2. 🐳 Build & push image to GitHub Container Registry
3. 🚀 SSH deploy to production server

---

## 📦 Manual ZIP Steps

```
1. Create folder:  agent-creator/
2. Paste all files into correct subfolders
3. Right-click → Compress / zip agent-creator/
4. Extract in VS Code:  File → Open Folder → agent-creator/
```

---

## 🧠 AutoGen Pipeline Explained

```
PlannerAgent    → Reads prompt → Outputs structured JSON spec
CoderAgent      → Reads spec  → Writes complete FastAPI agent.py
ReviewerAgent   → Reviews code → Flags issues → Returns fixed code
OrchestratorAgent → Runs the loop, decides when code is ready
```

Up to 2 review cycles. If Docker tests fail: GPT-4o auto-fixes code (up to 3 retries).

---

*Built for AlgoProfessor AI R&D Internship — Day 12 Milestone* 🎓
