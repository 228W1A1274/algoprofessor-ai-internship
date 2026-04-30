"""
api.py
───────
FastAPI web server that exposes DevSquad as a deployable HTTP service.

Endpoints
─────────
GET  /             → health check (used by Railway/Render uptime monitor)
POST /run          → submit a task; returns full crew output as JSON
POST /run/stream   → submit a task; streams agent output line-by-line (SSE)
GET  /outputs      → list all saved output files
GET  /outputs/{filename} → download a specific output file

Why FastAPI?
  - Railway and Render both detect a running web process on $PORT.
  - The /run/stream endpoint lets you watch agents think in real time
    from the browser or curl — no need to wait for the full run.
  - CORS is open so a frontend (React, plain HTML) can call it directly.

Run locally:
  uvicorn api:app --reload --port 8000

Then open:
  http://localhost:8000          → health check
  http://localhost:8000/docs     → Swagger UI (interactive)
"""

import os
import asyncio
from pathlib import Path
from datetime import datetime
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DevSquad API",
    description="4-Agent Autonomous Engineering Team — CrewAI + Groq",
    version="1.0.0",
)

# Allow all origins so a hosted frontend can call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Request / Response models ─────────────────────────────────────────────────

class TaskRequest(BaseModel):
    task: str = Field(
        ...,
        min_length=10,
        description="Plain-English description of the software to build.",
        example="Build a Python CLI tool that fetches weather from Open-Meteo API.",
    )
    push_to_github: bool = Field(
        False,
        description="If True, push generated files to a GitHub repo (requires GITHUB_TOKEN).",
    )
    use_docker: bool = Field(
        False,
        description="If True, run QA tests inside an isolated Docker container.",
    )


class TaskResponse(BaseModel):
    status: str
    output: str
    files_saved: int
    timestamp: str


# ── Helper — run crew in a thread (CrewAI is sync) ───────────────────────────

def _run_crew_sync(task: str, push: bool, docker: bool) -> str:
    """
    Runs the DevSquadCrew synchronously.
    Called via asyncio.to_thread() so it doesn't block the event loop.
    """
    from crew import DevSquadCrew
    crew = DevSquadCrew(
        task_description=task,
        use_docker=docker,
        push_to_github=push,
    )
    return crew.run()


def _count_output_files() -> int:
    """Count all files inside outputs/ recursively."""
    return sum(1 for f in OUTPUT_DIR.rglob("*") if f.is_file() and f.name != ".gitkeep")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Railway and Render ping this to verify the service is alive.
    Returns 200 OK with a simple JSON body.
    """
    return {
        "status": "ok",
        "service": "DevSquad API",
        "groq_key_set": bool(os.environ.get("GROQ_API_KEY")),
    }


@app.post("/run", response_model=TaskResponse, tags=["DevSquad"])
async def run_task(request: TaskRequest):
    """
    Submit a task to the 4-agent DevSquad crew.

    Runs all agents sequentially (PM → Dev → Reviewer → QA) and returns
    the full output once all agents have finished.

    For long tasks this may take 60–120 seconds.
    Use /run/stream for real-time output.
    """
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured on this server.",
        )

    try:
        # Run crew in a background thread so FastAPI event loop stays responsive
        output = await asyncio.to_thread(
            _run_crew_sync,
            request.task,
            request.push_to_github,
            request.use_docker,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return TaskResponse(
        status="success",
        output=output,
        files_saved=_count_output_files(),
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.post("/run/stream", tags=["DevSquad"])
async def run_task_stream(request: TaskRequest):
    """
    Submit a task and stream agent output line-by-line via Server-Sent Events.

    Each line is prefixed with `data: ` so any SSE client can consume it.
    The stream ends with `data: [DONE]`.

    Example (curl):
      curl -X POST http://localhost:8000/run/stream \\
           -H "Content-Type: application/json" \\
           -d '{"task": "Build a hello world CLI tool."}'
    """
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured.")

    async def event_generator() -> AsyncIterator[str]:
        """
        Runs the crew in a thread and yields output lines as SSE events.
        Since CrewAI is synchronous, we run it in a thread and buffer output
        by capturing stdout via a queue (simple line-based streaming).
        """
        import io
        import sys
        import queue
        import threading

        line_queue: queue.Queue = queue.Queue()
        done_sentinel = object()

        # Custom stdout that pushes lines into a queue
        class QueueWriter(io.TextIOBase):
            def write(self, text: str) -> int:
                if text.strip():
                    line_queue.put(text.rstrip())
                return len(text)

        def crew_thread():
            old_stdout = sys.stdout
            sys.stdout = QueueWriter()
            try:
                _run_crew_sync(
                    request.task,
                    request.push_to_github,
                    request.use_docker,
                )
            except Exception as exc:
                line_queue.put(f"[ERROR] {exc}")
            finally:
                sys.stdout = old_stdout
                line_queue.put(done_sentinel)

        thread = threading.Thread(target=crew_thread, daemon=True)
        thread.start()

        while True:
            try:
                item = line_queue.get(timeout=120)
            except queue.Empty:
                yield "data: [TIMEOUT]\n\n"
                break

            if item is done_sentinel:
                yield "data: [DONE]\n\n"
                break
            yield f"data: {item}\n\n"
            await asyncio.sleep(0)   # yield control to event loop

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disables Nginx buffering on Railway
        },
    )


@app.get("/outputs", tags=["Files"])
async def list_outputs():
    """List all files saved to the outputs/ directory."""
    files = [
        {
            "name": f.name,
            "size_bytes": f.stat().st_size,
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
        }
        for f in OUTPUT_DIR.rglob("*")
        if f.is_file() and f.name != ".gitkeep"
    ]
    return {"count": len(files), "files": sorted(files, key=lambda x: x["modified"], reverse=True)}


@app.get("/outputs/{filename}", tags=["Files"])
async def download_output(filename: str):
    """Download a specific output file by name."""
    # Security: prevent path traversal
    safe_name = Path(filename).name
    target = OUTPUT_DIR / safe_name
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"File '{safe_name}' not found.")
    return FileResponse(path=target, filename=safe_name)
