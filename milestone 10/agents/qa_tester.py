"""
agents/qa_tester.py
────────────────────
Defines the QA Tester agent.

Role  : Write and (optionally) execute a comprehensive pytest suite for the
        Developer's code, then produce a final QA report.
Input : Source code files + the Reviewer's report.
Output: Complete pytest file (test_<module>.py) with at least 5 test cases,
        a QA report with PASS/FAIL per test, and an overall PASS/FAIL verdict.

Why this agent?
  The Reviewer reasons about code statically; the QA Tester validates behaviour
  dynamically. Together they form a two-layer quality gate before GitHub push.
"""

from crewai import Agent
from llm_client import get_llm


def build_qa_tester() -> Agent:
    """
    Construct and return the QA Tester Agent.
    """
    return Agent(
        role="Senior QA Engineer",
        goal=(
            "Write a complete pytest test suite for the developed code. "
            "Requirements: "
            "(1) At least 5 test functions covering: happy path, edge cases, "
            "and expected error conditions. "
            "(2) Use pytest fixtures for reusable setup. "
            "(3) Mock all external HTTP calls with pytest-mock or responses library "
            "so tests run offline. "
            "(4) Each test has a descriptive name following: "
            "test_<what>_<condition>_<expected_outcome>. "
            "(5) After writing tests, simulate running them and report each as "
            "PASSED or FAILED with a brief reason. "
            "Conclude with: OVERALL QA VERDICT: PASS or FAIL."
        ),
        backstory=(
            "You are a Senior QA Engineer with a reputation for finding bugs "
            "that developers thought were impossible. "
            "You have a personal mantra: 'If it isn't tested, it's broken.' "
            "You specialise in boundary conditions, network failure scenarios, "
            "and API contract violations that surface only in production. "
            "You write tests that are self-documenting — another engineer should "
            "understand exactly what is being tested just from the function name."
        ),
        llm=get_llm(temperature=0.2),
        verbose=True,
        allow_delegation=False,
    )
