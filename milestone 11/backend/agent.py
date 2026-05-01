"""
LangGraph Agent — ReAct-style browser automation agent.
"""

import json
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from config import settings
from tools import ALL_TOOLS, close_browser
import logging

logger = logging.getLogger(__name__)

VALID_TOOL_NAMES = {t.name for t in ALL_TOOLS}


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    steps: list[str]
    step_count: int


# ─────────────────────────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────────────────────────

def _make_llm(tool_choice=None):
    if settings.LLM_PROVIDER == "groq":
        base = ChatGroq(
            model_name=settings.LLM_MODEL,
            groq_api_key=settings.GROQ_API_KEY,
            temperature=0,
            max_tokens=2048,
        )
        return base.bind_tools(ALL_TOOLS) if tool_choice is None \
               else base.bind_tools(ALL_TOOLS, tool_choice=tool_choice)

    elif settings.LLM_PROVIDER == "openrouter":
        from langchain_openai import ChatOpenAI
        base = ChatOpenAI(
            model="meta-llama/llama-3-70b-instruct",
            openai_api_key=settings.OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0,
        )
        return base.bind_tools(ALL_TOOLS)

    elif settings.LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        base = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0,
        )
        return base.bind_tools(ALL_TOOLS)

    raise ValueError(f"Unknown LLM_PROVIDER: {settings.LLM_PROVIDER}")


llm       = _make_llm()
llm_force = _make_llm("auto")   # forces a tool call on retry
tool_node = ToolNode(ALL_TOOLS)


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────

TOOL_NAMES_STR = ", ".join(sorted(VALID_TOOL_NAMES))

SYSTEM_PROMPT = f"""You are an AI Browser Sidekick — a browser automation agent.
You control a real Chromium browser using Playwright tools.

## YOUR ONLY TOOLS
{TOOL_NAMES_STR}

## CRITICAL RULES
1. NEVER stop after just one tool call. Always continue until the task is fully complete.
2. After navigate_to, ALWAYS call get_page_info or extract_text to read the page.
3. Only give your final text summary when you have actually completed the task and have the answer.
4. Use the structured API tool-call format ONLY. Never write tool calls as plain text.
5. Each message: call ONE tool. Wait for result. Then decide next step.
6. Maximum {settings.MAX_AGENT_STEPS} steps total.

## Workflow example for "find most starred Python repo on GitHub":
  Step 1: navigate_to("https://github.com/search?q=python&s=stars&type=Repositories")
  Step 2: extract_text(".search-results")   <- READ the page after navigating
  Step 3: (if needed) scroll_page or click_element to get more info
  Step 4: Write final summary with the actual answer

## Final response format (ONLY when task is complete):
  Task completed: [what was done]
  Result: [the actual answer with specific details]
"""

CORRECTION_MSG = (
    "Your last response used the wrong format — do NOT write tool calls as plain text. "
    f"Use the API function-call format to call ONE of: {TOOL_NAMES_STR}. "
    "Call ONE tool now using the proper structured format."
)

CONTINUE_MSG = (
    "You stopped too early. The task is NOT complete yet. "
    "You navigated to the page but did not read it or complete the task. "
    "Call extract_text or get_page_info NOW to read the page content, "
    "then continue until the task is fully done."
)


# ─────────────────────────────────────────────────────────────────────────────
# Graph Nodes
# ─────────────────────────────────────────────────────────────────────────────

def _last_tool_was(messages: list, tool_name: str) -> bool:
    """Check if the most recent tool call was a specific tool."""
    for msg in reversed(messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            return msg.tool_calls[-1]["name"] == tool_name
        if isinstance(msg, ToolMessage):
            continue
        break
    return False


async def agent_node(state: AgentState) -> AgentState:
    """LLM reasoning node with format-error recovery and early-stop prevention."""
    messages  = state["messages"]
    step      = state["step_count"]
    steps     = list(state.get("steps", []))

    if step >= settings.MAX_AGENT_STEPS:
        return {
            **state,
            "messages": [AIMessage(content="Maximum steps reached. Stopping.")],
            "step_count": step + 1,
        }

    # ── Attempt LLM call ──────────────────────────────────────────────────────
    try:
        response = await llm.ainvoke(messages)
    except Exception as e:
        err = str(e)
        if "400" in err or "tool_use_failed" in err or "tool call" in err.lower():
            logger.warning(f"Step {step}: Bad tool format (400), correcting and retrying")
            corrected = list(messages) + [HumanMessage(content=CORRECTION_MSG)]
            try:
                response = await llm_force.ainvoke(corrected)
                steps.append(f"Step {step + 1}: [corrected tool format]")
            except Exception as e2:
                err_msg = AIMessage(content=f"Agent failed after correction: {e2}")
                return {**state, "messages": [err_msg], "steps": steps, "step_count": step + 1}
        else:
            err_msg = AIMessage(content=f"LLM error: {e}")
            return {**state, "messages": [err_msg], "steps": steps, "step_count": step + 1}

    # ── Detect early stop: LLM gave text but last action was navigate_to ──────
    # This is the "browser closes after step 0" bug — LLM navigates then stops.
    if (
        not response.tool_calls          # gave a text response (no tool call)
        and response.content             # has text content
        and _last_tool_was(messages, "navigate_to")  # just navigated
        and step < settings.MAX_AGENT_STEPS - 1
    ):
        logger.warning(f"Step {step}: LLM stopped after navigate_to — forcing continuation")
        forced_messages = list(messages) + [
            response,
            HumanMessage(content=CONTINUE_MSG),
        ]
        try:
            response = await llm_force.ainvoke(forced_messages)
            steps.append(f"Step {step + 1}: [forced continuation after navigate_to]")
        except Exception as e:
            # If forcing fails just let it end naturally
            logger.warning(f"Step {step}: Forced continuation failed: {e}")

    # ── Log ───────────────────────────────────────────────────────────────────
    content_preview = response.content[:80] if response.content else "[tool call]"
    logger.info(f"Step {step}: {content_preview}")

    if response.tool_calls:
        for tc in response.tool_calls:
            steps.append(f"Step {step + 1}: {tc['name']}({json.dumps(tc['args'])[:80]})")
    elif response.content:
        steps.append(f"Step {step + 1}: {response.content[:120]}")

    return {**state, "messages": [response], "steps": steps, "step_count": step + 1}


def should_continue(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END


# ─────────────────────────────────────────────────────────────────────────────
# Build Graph
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


GRAPH = build_graph()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_initial_state(instruction: str, url: str | None, context: dict) -> AgentState:
    user_content = f"Task: {instruction}"
    if url:
        user_content += f"\nStarting URL: {url}"
    if context:
        user_content += f"\nContext: {json.dumps(context)}"
    return {
        "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_content)],
        "steps": [],
        "step_count": 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public Interface
# ─────────────────────────────────────────────────────────────────────────────

async def run_agent(instruction: str, url: str | None, context: dict) -> dict:
    initial_state = _build_initial_state(instruction, url, context)
    try:
        final_state = await GRAPH.ainvoke(initial_state)
        last_msg = final_state["messages"][-1]
        return {
            "output": last_msg.content if hasattr(last_msg, "content") else str(last_msg),
            "steps": final_state["steps"],
        }
    finally:
        await close_browser()


async def stream_agent(instruction: str, url: str | None, context: dict):
    """Async generator — yields SSE event dicts as the agent works."""
    initial_state = _build_initial_state(instruction, url, context)
    last_agent_content = ""

    try:
        async for event in GRAPH.astream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():

                steps = node_output.get("steps", [])
                if steps:
                    yield {"type": "step", "node": node_name, "content": steps[-1]}

                for msg in node_output.get("messages", []):
                    if not hasattr(msg, "content") or not msg.content:
                        continue
                    content_str = str(msg.content)
                    if node_name == "agent":
                        last_agent_content = content_str
                        yield {"type": "thinking", "content": content_str[:300]}
                    elif node_name == "tools":
                        yield {"type": "tool_out", "content": f"Tool result: {content_str[:250]}"}

    except Exception as e:
        logger.error(f"stream_agent error: {e}")
        yield {"type": "error", "content": str(e)}

    finally:
        await close_browser()
        yield {"type": "complete", "content": "Task finished"}
        if last_agent_content:
            yield {"type": "result", "content": last_agent_content}