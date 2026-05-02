from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from config import HOST, PORT, DEBUG, MAX_FIX_RETRIES
from agent_engine import run_agent_pipeline
from deployer import deploy_agent, list_agents, get_agent

app = FastAPI(
    title="AlgoProfessor Agent Creator",
    description="Dynamically create, test and deploy AI agents from natural language prompts.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ───────────────────────────────────────────────

class CreateAgentRequest(BaseModel):
    prompt: str
    auto_deploy: bool = True


class RunAgentRequest(BaseModel):
    agent_name: str
    input: str


class CreateAgentResponse(BaseModel):
    success: bool
    agent_name: Optional[str] = None
    description: Optional[str] = None
    endpoint: Optional[str] = None
    port: Optional[int] = None
    test_output: Optional[str] = None
    attempts: Optional[int] = None
    container_id: Optional[str] = None
    agent_code: Optional[str] = None
    error: Optional[str] = None
    pipeline_info: Optional[dict] = None


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "agent-creator"}


@app.post("/create-agent", response_model=CreateAgentResponse)
def create_agent(req: CreateAgentRequest):
    """
    Main endpoint: takes a natural language prompt, runs the full
    AutoGen pipeline, deploys the agent as a Docker container.
    """
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    # Run multi-agent pipeline (Planner → Coder → Reviewer)
    try:
        pipeline_result = run_agent_pipeline(req.prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent pipeline failed: {str(e)}")

    agent_code = pipeline_result["agent_code"]
    agent_name = pipeline_result["agent_name"]
    requirements = pipeline_result["requirements"]
    description = pipeline_result["description"]
    dockerfile_extra = pipeline_result.get("dockerfile_extra", "")

    # If auto_deploy is False, just return the generated code
    if not req.auto_deploy:
        return CreateAgentResponse(
            success=True,
            agent_name=agent_name,
            description=description,
            agent_code=agent_code,
            pipeline_info={
                "spec": pipeline_result.get("spec"),
                "review_cycles": pipeline_result.get("review_cycles"),
                "requirements": requirements,
            },
        )

    # Deploy: build image, run container, test it, auto-fix if needed
    deploy_result = deploy_agent(
        agent_name=agent_name,
        agent_code=agent_code,
        requirements=requirements,
        dockerfile_extra=dockerfile_extra,
        description=description,
        max_retries=MAX_FIX_RETRIES,
    )

    return CreateAgentResponse(
        success=deploy_result["success"],
        agent_name=deploy_result.get("agent_name"),
        description=deploy_result.get("description"),
        endpoint=deploy_result.get("endpoint"),
        port=deploy_result.get("port"),
        test_output=deploy_result.get("test_output"),
        attempts=deploy_result.get("attempts"),
        container_id=deploy_result.get("container_id"),
        agent_code=agent_code,
        error=deploy_result.get("error"),
        pipeline_info={
            "spec": pipeline_result.get("spec"),
            "review_cycles": pipeline_result.get("review_cycles"),
        },
    )


@app.get("/agents")
def list_all_agents():
    """List all deployed agents."""
    return {"agents": list_agents()}


@app.get("/agents/{agent_name}")
def get_agent_info(agent_name: str):
    """Get info about a specific deployed agent."""
    agent = get_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")
    return agent


@app.post("/agents/{agent_name}/run")
def run_agent(agent_name: str, req: RunAgentRequest):
    """
    Proxy a request to a deployed agent's /run endpoint.
    """
    import requests as req_lib
    agent = get_agent(agent_name)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")

    endpoint = agent.get("endpoint")
    try:
        r = req_lib.post(f"{endpoint}/run", json={"input": req.input}, timeout=30)
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to reach agent: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=DEBUG)
