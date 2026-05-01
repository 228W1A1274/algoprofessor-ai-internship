"""
AI Browser Sidekick — FastAPI Backend
"""

import sys
import asyncio

# ── Windows fix ───────────────────────────────────────────────────────────────
# Must be done BEFORE any other imports so uvicorn never overrides it.
# ProactorEventLoop is required for Playwright's subprocess-based browser launch.
# SelectorEventLoop (the default on Windows) raises NotImplementedError on subprocess_exec.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    _loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(_loop)
# ─────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import logging
from agent import run_agent, stream_agent
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Browser Sidekick API",
    description="OpenAI Operator-style browser agent powered by LangGraph + Playwright",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TaskRequest(BaseModel):
    instruction: str
    url: str | None = None
    context: dict | None = None


class TaskResponse(BaseModel):
    status: str
    result: str
    steps: list[str]


@app.get("/")
async def health():
    return {"status": "ok", "service": "AI Browser Sidekick"}


@app.post("/task", response_model=TaskResponse)
async def run_task(req: TaskRequest):
    """Run a browser automation task synchronously."""
    logger.info(f"Task received: {req.instruction[:80]}...")
    try:
        result = await run_agent(
            instruction=req.instruction,
            url=req.url,
            context=req.context or {},
        )
        return TaskResponse(
            status="success",
            result=result["output"],
            steps=result["steps"],
        )
    except Exception as e:
        logger.error(f"Task failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/task/stream")
async def run_task_stream(req: TaskRequest):
    """
    Run a browser automation task with Server-Sent Events streaming.
    SSE event types:
        step      - agent action taken
        thinking  - agent reasoning text
        result    - final task result (carries the actual output text)
        complete  - task finished signal
        error     - something went wrong
        done      - stream closing
    """
    logger.info(f"Streaming task: {req.instruction[:80]}...")

    async def event_generator():
        try:
            async for event in stream_agent(
                instruction=req.instruction,
                url=req.url,
                context=req.context or {},
            ):
                yield f"data: {json.dumps(event)}\n\n"
                await asyncio.sleep(0)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'done', 'content': 'Stream closed'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    if sys.platform == "win32":
        # loop="none" tells uvicorn NOT to touch our ProactorEventLoop.
        # reload=False because the reloader spawns child processes that reset the loop policy.
        # If you need live reload, restart the server manually after changes.
        config = uvicorn.Config(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            loop="none",
        )
        server = uvicorn.Server(config)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(server.serve())
    else:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
