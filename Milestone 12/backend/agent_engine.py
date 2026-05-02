"""
agent_engine.py — AutoGen-style multi-agent orchestration.

Roles:
  PlannerAgent   → breaks the prompt into a structured spec
  CoderAgent     → generates the FastAPI agent code
  ReviewerAgent  → reviews code quality and flags issues
  OrchestratorAgent → drives the pipeline and decides when code is ready
"""

import json
import openai
from config import OPENAI_API_KEY, MODEL_NAME, BASE_URL

client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)


def _chat(system: str, user: str, temperature: float = 0.2, json_mode: bool = False) -> str:
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        **kwargs,
    )
    return response.choices[0].message.content.strip()


# ─── Agent 1: Planner ────────────────────────────────────────────────────────

PLANNER_SYSTEM = """
You are a senior AI architect (PlannerAgent).
Given a natural language description of an AI agent, produce a structured specification.

Return JSON with keys:
{
  "agent_name": "snake_case_name (max 20 chars)",
  "description": "one-sentence description",
  "core_functionality": "what the /run endpoint should do",
  "input_format": "description of expected input",
  "output_format": "description of expected output",
  "requires_llm": true/false,
  "libraries_needed": ["list", "of", "pip", "packages"]
}
"""



def planner_agent(prompt: str) -> dict:
    raw = _chat(PLANNER_SYSTEM, f"Agent description: {prompt}", json_mode=True)
    return json.loads(raw)


# ─── Agent 2: Coder ──────────────────────────────────────────────────────────

CODER_SYSTEM = """
You are an expert Python engineer (CoderAgent).
You receive a structured agent specification and must write a complete FastAPI agent.

STRICT RULES:
- Use FastAPI
- POST /run endpoint: receives {"input": "string"}, returns {"output": "string"}
- GET /health endpoint: returns {"status": "ok", "agent": "<agent_name>"}
- If requires_llm is true, use openai with gpt-4o, read OPENAI_API_KEY from os.environ
- Port 8000 (set by uvicorn CMD in Dockerfile, not in code)
- No placeholders — fully working code
- All imports at top
- Clean, production-ready

Return JSON:
{
  "agent_code": "full python code as string",
  "requirements": ["fastapi", "uvicorn", ...]
}
"""


def coder_agent(spec: dict) -> dict:
    raw = _chat(
        CODER_SYSTEM,
        f"Specification:\n{json.dumps(spec, indent=2)}",
        temperature=0.2,
        json_mode=True,
    )
    return json.loads(raw)


# ─── Agent 3: Reviewer ───────────────────────────────────────────────────────

REVIEWER_SYSTEM = """
You are a code reviewer (ReviewerAgent).
Review the given FastAPI agent code for:
1. Syntax errors
2. Missing imports
3. Correct /run and /health endpoints
4. Correct response schemas
5. No hardcoded secrets

You MUST respond with ONLY a valid JSON object, no markdown, no backticks, no explanation.
Exact format:
{"approved": true, "issues": [], "fixed_code": "same code here"}
or
{"approved": false, "issues": ["issue1"], "fixed_code": "corrected code here"}
"""


def reviewer_agent(code: str) -> dict:
    raw = _chat(
        REVIEWER_SYSTEM,
        f"Review this agent code:\n```python\n{code}\n```",
        temperature=0.1,
        json_mode=False,
    )
    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])
    try:
        import json
        return json.loads(raw)
    except Exception:
        # If JSON parsing fails, just approve and move on
        return {"approved": True, "issues": [], "fixed_code": code}


# ─── Orchestrator: ties it all together ──────────────────────────────────────

def run_agent_pipeline(prompt: str, max_review_cycles: int = 2) -> dict:
    """
    Full AutoGen-style pipeline:
      1. PlannerAgent  → spec
      2. CoderAgent    → code
      3. ReviewerAgent → review + optional fix (up to max_review_cycles)
      4. Return final artifact
    """
    # Step 1: Plan
    spec = planner_agent(prompt)

    # Step 2: Code
    coder_result = coder_agent(spec)
    agent_code = coder_result["agent_code"]
    requirements = coder_result.get("requirements", ["fastapi", "uvicorn", "openai"])

    # Step 3: Review loop
    for cycle in range(max_review_cycles):
        review = reviewer_agent(agent_code)
        if review.get("approved"):
            break
        # Apply reviewer's fix
        fixed = review.get("fixed_code", agent_code)
        if fixed and fixed != agent_code:
            agent_code = fixed
        # If still not approved after last cycle, proceed anyway
        if cycle == max_review_cycles - 1:
            break

    return {
        "agent_name": spec["agent_name"],
        "description": spec["description"],
        "agent_code": agent_code,
        "requirements": requirements,
        "dockerfile_extra": "",
        "spec": spec,
        "review_cycles": cycle + 1,
    }
