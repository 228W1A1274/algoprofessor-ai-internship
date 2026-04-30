"""
agents/developer.py
────────────────────
Defines the Developer agent.

Role  : Write complete, runnable Python code that satisfies the PM spec.
Input : Engineering specification produced by the Project Manager.
Output: Full source code for every file listed in the spec, as a single
        markdown-fenced block per file, plus a requirements.txt.

Design notes
  - temperature=0.1 keeps code deterministic and avoids hallucinated APIs.
  - The Developer is the only agent that actually produces code artefacts.
  - It outputs a structured format so the Reviewer and QA can parse files
    without ambiguity.
"""

from crewai import Agent
from llm_client import get_llm


def build_developer() -> Agent:
    """
    Construct and return the Developer Agent.
    """
    return Agent(
        role="Senior Python Developer",
        goal=(
            "Implement COMPLETE, immediately runnable Python code for every "
            "file mentioned in the engineering specification. "
            "Rules you must follow: "
            "(1) Wrap each file in a markdown code fence with the filename as "
            "the header, e.g.  ## File: weather.py  followed by ```python ... ```. "
            "(2) Include a requirements.txt file listing every third-party "
            "package with pinned versions. "
            "(3) Add docstrings to every function and class. "
            "(4) Handle errors explicitly — never let exceptions propagate "
            "silently. "
            "(5) All code must be PEP-8 compliant and type-annotated."
        ),
        backstory=(
            "You are a Senior Python Engineer with deep expertise in writing "
            "production-grade, well-documented code. "
            "You have a pathological hatred of half-finished implementations — "
            "every function you write has a body, every import is used, and "
            "every edge case is handled. "
            "You treat the PM's spec as a contract: you implement exactly what "
            "was specified, nothing more, nothing less."
        ),
        llm=get_llm(temperature=0.1),   # very low temp → deterministic code
        verbose=True,
        allow_delegation=False,
    )
