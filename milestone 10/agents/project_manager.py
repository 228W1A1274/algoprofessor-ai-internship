"""
agents/project_manager.py
──────────────────────────
Defines the Project Manager agent.

Role  : Translate the raw user task into a precise engineering specification.
Input : Free-text task description from the user.
Output: Structured spec (objective, stack, file list, acceptance criteria).

Why this agent?
  Without a clear spec, the Developer produces vague code and the Reviewer
  has no benchmark to judge against.  The PM converts ambiguity into
  structure that every downstream agent can act on.
"""

from crewai import Agent
from llm_client import get_llm


def build_project_manager() -> Agent:
    """
    Construct and return the Project Manager Agent.

    backstory  : provides persona context that shapes the agent's tone.
    goal       : the single most important outcome this agent must achieve.
    verbose    : prints each agent's reasoning chain during execution.
    allow_delegation: False — the PM does not sub-delegate; it owns its task.
    """
    return Agent(
        role="Senior Project Manager",
        goal=(
            "Transform the user's raw task description into a crystal-clear "
            "engineering specification that the Developer can implement without "
            "any further clarification. "
            "The spec MUST include: "
            "(1) objective in one sentence, "
            "(2) exact Python file names and their responsibility, "
            "(3) all external libraries with pinned versions, "
            "(4) function/class signatures the Developer must implement, "
            "(5) at least three acceptance criteria that QA will verify."
        ),
        backstory=(
            "You are a Senior Project Manager with 15 years of experience "
            "shipping Python backend systems at scale. "
            "You are famous for writing specs so precise that junior developers "
            "can deliver production-quality code on the first attempt. "
            "You never leave ambiguity in a spec — if something is unclear you "
            "make a reasonable assumption and document it explicitly."
        ),
        llm=get_llm(temperature=0.2),   # low temp → consistent, structured output
        verbose=True,
        allow_delegation=False,
    )
