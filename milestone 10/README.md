# DevSquad — 4-Agent Autonomous Engineering Team

**Milestone 10 | AlgoProfessor AI R&D Internship 2026 | Phase 3: Agentic AI**

---

## What DevSquad Does

DevSquad accepts a plain-English software requirement and autonomously produces:

1. A structured engineering specification (Project Manager agent)
2. Complete, runnable Python source code (Developer agent)
3. A detailed code review report (Code Reviewer agent)
4. A full pytest suite with QA verdict (QA Tester agent)

Optionally it runs the generated tests inside an isolated Docker container and pushes all output files to a new GitHub repository — without any human in the loop.

---

## System Architecture

```
User (VS Code terminal)
         │
         │  python main.py --task "..."
         ▼
┌─────────────────────────────────────────┐
│         CrewAI Orchestrator             │
│         Process: Sequential             │
│         LLM: Groq Llama 3.3 70B        │
└─────────────────────────────────────────┘
         │
         ├──► [1] Project Manager Agent
         │         Input : task description (string)
         │         Output: 8-section engineering spec
         │
         ├──► [2] Developer Agent
         │         Input : spec (from PM)
         │         Output: complete Python source + requirements.txt
         │
         ├──► [3] Code Reviewer Agent
         │         Input : spec + code
         │         Output: APPROVED / NEEDS_REVISION + issue list
         │
         └──► [4] QA Tester Agent
                   Input : spec + code + review
                   Output: test_suite.py + OVERALL QA VERDICT
                         │
                         ├──► outputs/  (all files saved locally)
                         ├──► Docker container (--docker flag)
                         └──► GitHub repo  (--push flag)
```

---

## Project Structure

```
devsquad/
│
├── main.py                  Entry point. Parses args, validates env, runs crew.
├── crew.py                  DevSquadCrew class — wires agents, tasks, post-processing.
├── llm_client.py            Factory that builds the Groq LLM object used by all agents.
├── requirements.txt         Pinned dependencies for the whole project.
├── conftest.py              Adds project root to sys.path so pytest resolves imports.
├── docker-compose.yml       Defines runner + sandbox services.
├── .env.example             Template for environment variables (copy to .env).
├── .gitignore               Excludes .env, __pycache__, outputs/, venv/.
│
├── agents/
│   ├── __init__.py          Exports all four build_* functions.
│   ├── project_manager.py   PM agent — converts task → structured spec.
│   ├── developer.py         Dev agent — writes source code from spec.
│   ├── code_reviewer.py     Reviewer agent — audits code against spec.
│   └── qa_tester.py         QA agent — writes pytest suite + reports verdict.
│
├── tasks/
│   ├── __init__.py          Exports build_tasks().
│   └── task_definitions.py  Defines all 4 CrewAI Task objects with context chains.
│
├── tools/
│   ├── __init__.py          Exports docker_runner and github_pusher utilities.
│   ├── docker_runner.py     run_in_docker() — executes code in python:3.11-slim.
│   └── github_pusher.py     push_to_github() — creates repo + uploads files via API.
│
├── docker/
│   ├── Dockerfile.runner    Image for the main DevSquad process.
│   └── Dockerfile.sandbox   Minimal image for isolated QA test execution.
│
├── tests/
│   └── test_devsquad.py     9 unit tests covering all system components.
│
└── outputs/                 Auto-created. Stores all crew output and extracted files.
    └── .gitkeep
```

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.10 or 3.11 | https://python.org/downloads |
| Git | any recent | https://git-scm.com |
| Docker Desktop | latest | https://docker.com/products/docker-desktop (optional) |
| VS Code | latest | https://code.visualstudio.com |

---

## Step-by-Step Setup in VS Code

### Step 1 — Clone or create the project folder

Open VS Code, then open the integrated terminal (`Ctrl+`` ` ``).

```bash
# If you have the project as a zip, extract it and open the folder:
cd ~/Desktop
# Then: File → Open Folder → select devsquad/

# Or create from scratch:
mkdir devsquad && cd devsquad
```

### Step 2 — Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell blocks execution:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\Activate.ps1
```

**macOS / Linux (bash/zsh):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the start of your terminal prompt.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

Expected output (abbreviated):
```
Successfully installed crewai-0.80.0 crewai-tools-0.14.0 groq-0.11.0
litellm-1.48.7 tenacity-8.2.3 PyGithub-2.1.1 python-dotenv-1.0.1 ...
```

### Step 4 — Create your .env file

In the project root, copy the example and fill it in:

```bash
cp .env.example .env
```

Open `.env` in VS Code and set your keys:

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

- **GROQ_API_KEY** — free at https://console.groq.com → API Keys → Create API Key
- **GITHUB_TOKEN** — at https://github.com/settings/tokens → Generate new token (classic) → tick **repo** scope

> ⚠️ Never commit `.env` to Git. It is already listed in `.gitignore`.

### Step 5 — Run the system (local, no Docker)

```bash
python main.py --task "Build a Python CLI tool that fetches weather data from the Open-Meteo API for a given city and prints current temperature and wind speed."
```

Expected terminal output:
```
============================================================
  DevSquad — Autonomous Engineering Team
============================================================
  Task   : Build a Python CLI tool that fetches weather data...
  Docker : disabled (mock mode)
  GitHub : local only
============================================================

[2026-XX-XX] Starting Crew execution...
[Senior Project Manager] Working on: Transform the user task...
...
[Senior Python Developer] Working on: Implement COMPLETE...
...
[Senior Code Reviewer] Working on: Perform a thorough...
...
[Senior QA Engineer] Working on: Write a complete pytest...
...
[DevSquad] Crew finished in ~45.0s
[DevSquad] Raw output saved → outputs/devsquad_output_YYYYMMDD_HHMMSS.txt
[DevSquad] 3 source files extracted → outputs/files_YYYYMMDD_HHMMSS/

============================================================
  FINAL SQUAD OUTPUT
============================================================
[Full agent output printed here]
```

### Step 6 — Run with Docker QA (optional)

Requires Docker Desktop running.

```bash
python main.py --task "Build a Python CLI calculator." --docker
```

DevSquad will:
1. Run all 4 agents (same as above)
2. Extract `test_suite.py` from QA output
3. Pull `python:3.11-slim` if not cached
4. Install test deps inside the container
5. Execute pytest inside the isolated container
6. Print container stdout/stderr

### Step 7 — Run with GitHub push (optional)

```bash
python main.py --task "Build a Python CLI calculator." --push
```

DevSquad will create a new public repo named `devsquad-output` in your GitHub account and push all generated files to it.

### Step 8 — Run the unit tests

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_devsquad.py::test_get_llm_raises_when_api_key_missing PASSED
tests/test_devsquad.py::test_project_manager_agent_role PASSED
tests/test_devsquad.py::test_developer_agent_no_delegation PASSED
tests/test_devsquad.py::test_task_context_chain_is_correct PASSED
tests/test_devsquad.py::test_docker_runner_returns_structured_dict_on_success PASSED
tests/test_devsquad.py::test_docker_runner_reports_error_on_nonzero_exit PASSED
tests/test_devsquad.py::test_extract_files_parses_crew_output_correctly PASSED
tests/test_devsquad.py::test_push_to_github_returns_error_when_token_missing PASSED
tests/test_devsquad.py::test_devsquad_crew_raises_without_groq_key PASSED

9 passed in X.XXs
```

---

## Running with Docker Compose (full containerised mode)

```bash
# Build and run the entire system inside Docker
docker compose up --build

# Override the task at runtime
docker compose run devsquad-runner python main.py \
  --task "Build a REST API with FastAPI that returns a random quote."
```

---

## Free Deployment on Render

1. Push the project to GitHub (without `.env` — keys go in Render dashboard).
2. Go to https://render.com → **New** → **Background Worker**
3. Connect your GitHub account → select the `devsquad` repo.
4. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python main.py --task "Your default task here"`
5. Under **Environment**, add:
   - `GROQ_API_KEY` = your key
   - `GITHUB_TOKEN` = your token (if using push)
6. Click **Deploy**. Render streams logs live — you will see each agent's output.

---

## Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `EnvironmentError: GROQ_API_KEY is not set` | `.env` file missing or not loaded | Run `cp .env.example .env` and fill in the key |
| `ModuleNotFoundError: No module named 'crewai'` | venv not activated | Run `source .venv/bin/activate` (Linux/Mac) or `.venv\Scripts\Activate.ps1` (Windows) |
| `ValidationError` on `Agent(llm=...)` | Passing raw `ChatGroq` object instead of `crewai.LLM` | `llm_client.py` already uses `crewai.LLM` — do not change it |
| `litellm.RateLimitError` (HTTP 429) | Groq free-tier rate limit hit | Tenacity retries automatically (up to 3×). If still failing, wait 60s |
| `groq.AuthenticationError` | Invalid or expired API key | Regenerate at console.groq.com → API Keys |
| `GithubException: 401` | Invalid GITHUB_TOKEN | Regenerate at github.com/settings/tokens with `repo` scope |
| `docker: command not found` | Docker not installed | Install Docker Desktop; start the daemon before using `--docker` |
| `pytest: command not found` | pytest not installed or venv not active | `pip install pytest` inside the active venv |
| `ImportError: cannot import name 'X' from 'crewai'` | Version mismatch | `pip install crewai==0.80.0 --force-reinstall` |

---

## Agent Communication Map

```
Task 1 output (spec)
    └──► Task 2 context   (Developer reads spec)
             └──► Task 3 context   (Reviewer reads spec + code)
                      └──► Task 4 context   (QA reads spec + code + review)
```

Every agent sees only what it needs — causality is preserved, no future task
leaks into a past agent's context.

---

## GitHub Submission Checklist

- [ ] `main.py` runs without errors on a fresh `.env`
- [ ] `pytest tests/ -v` shows 9 passed
- [ ] `outputs/` contains at least one `devsquad_output_*.txt`
- [ ] `outputs/files_*/` contains extracted source files
- [ ] `README.md` present with this content
- [ ] `requirements.txt` matches installed packages
- [ ] `.env` is NOT committed (check with `git status`)
- [ ] Commit message: `[Day-70] Milestone 10 — DevSquad 4-Agent Engineering Team`

---

*Stack: CrewAI 0.80 · Groq Llama 3.3 70B · Docker · PyGithub · tenacity · pytest*
