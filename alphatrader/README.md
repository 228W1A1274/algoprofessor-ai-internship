# AlphaTrader — Autonomous AI Trading Floor

> **Prompt 1 / Base System** — skeleton only. No trading logic yet.

---

## Project Structure

```
alpha-trader/
├── backend/
│   ├── main.py                 # FastAPI app + endpoints
│   ├── agent_orchestrator.py   # Agent lifecycle manager
│   ├── config.py               # Settings (reads .env)
│   ├── requirements.txt        # Python dependencies
│   └── .env.example            # Template for secrets
└── README.md
```

---

## Quick Start

### 1. Python version
Requires **Python 3.11+**.

```bash
python --version   # should show 3.11.x or 3.12.x
```

### 2. Create and activate a virtual environment

```bash
# From the repo root
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Leave all values empty for now — the base system needs none.
```

### 5. Run the server

```bash
# From inside backend/
uvicorn main:app --reload
```

Expected terminal output:
```
09:00:00 | INFO     | __main__ | Starting AlphaTrader v0.1.0 [development]
09:00:00 | INFO     | agent_orchestrator | [Orchestrator] Registered agent: MarketAnalyst
09:00:00 | INFO     | agent_orchestrator | [Orchestrator] Registered agent: QuantStrategist
09:00:00 | INFO     | agent_orchestrator | [Orchestrator] Registered agent: RiskGuardian
09:00:00 | INFO     | agent_orchestrator | [Orchestrator] Registered agent: ExecutionEngine
09:00:00 | INFO     | agent_orchestrator | [Orchestrator] All agents registered. Awaiting workflow trigger.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

## Testing

### Interactive docs (Swagger UI)
Open **http://127.0.0.1:8000/docs** in your browser.

### Endpoints

| Method | Path      | What it does                          |
|--------|-----------|---------------------------------------|
| GET    | `/`       | Health check                          |
| POST   | `/start`  | Trigger dummy workflow over all agents|
| GET    | `/status` | Report per-agent status               |

### curl examples

```bash
# Health check
curl http://localhost:8000/

# Trigger workflow
curl -X POST http://localhost:8000/start

# Agent status
curl http://localhost:8000/status
```

---

## Architecture Notes

| File | Role |
|------|------|
| `config.py` | Single `Settings` object (Pydantic). Add every new env var here. |
| `agent_orchestrator.py` | `AgentOrchestrator` class. Future prompts inject AutoGen/LangGraph here. |
| `main.py` | FastAPI app. Future prompts add routers — nothing else changes. |

---

## Roadmap (future prompts)

- [ ] Prompt 2 — MarketAnalyst agent + market data MCP server
- [ ] Prompt 3 — QuantStrategist + statistical tools
- [ ] Prompt 4 — RiskGuardian + compliance rules
- [ ] Prompt 5 — ExecutionEngine + Alpaca paper trading
- [ ] Prompt 6 — Docker Compose full stack
- [ ] Prompt 7 — Grafana dashboard
- [ ] Prompt 8 — Kubernetes deployment
