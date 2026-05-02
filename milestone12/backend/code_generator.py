import openai
from config import OPENAI_API_KEY, MODEL_NAME, BASE_URL

client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)

AGENT_SYSTEM_PROMPT = """
You are an expert Python AI agent code generator.
Generate a complete, runnable FastAPI agent service based on the user's description.

REQUIREMENTS:
- Use FastAPI for the HTTP service
- Expose a POST /run endpoint that accepts {"input": "..."} and returns {"output": "..."}
- Expose a GET /health endpoint returning {"status": "ok", "agent": "<agent_name>"}
- Use openai library with GPT-4o for AI capabilities if needed
- Read OPENAI_API_KEY from environment variable
- Must be production-ready, no placeholders, fully functional
- Include all necessary imports
- The agent should genuinely solve the described task
- Port must be 8000 inside the container

OUTPUT FORMAT:
Return ONLY a JSON object with these exact keys:
{
  "agent_name": "snake_case_name",
  "description": "one line description",
  "agent_code": "full python code as string",
  "requirements": ["fastapi", "uvicorn", "openai", ...],
  "dockerfile_extra": "any extra RUN commands needed (or empty string)"
}
"""

FIX_SYSTEM_PROMPT = """
You are an expert Python debugger.
You will receive agent code that failed with an error.
Fix the code and return ONLY the corrected full Python file content.
No explanations, no markdown, just raw Python code.
"""


def generate_agent_code(prompt: str) -> dict:
    """Call GPT-4o to generate agent code from a natural language prompt."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Create an AI agent that: {prompt}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    import json
    content = response.choices[0].message.content
    return json.loads(content)


def fix_agent_code(original_code: str, error_output: str) -> str:
    """Ask GPT-4o to fix broken agent code given the error output."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": FIX_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"ORIGINAL CODE:\n```python\n{original_code}\n```\n\n"
                    f"ERROR OUTPUT:\n{error_output}\n\n"
                    "Return only the fixed Python code."
                ),
            },
        ],
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if model adds them
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])
    return raw
