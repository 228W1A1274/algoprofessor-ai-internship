"""
tasks/task_definitions.py
──────────────────────────
Defines the four CrewAI Task objects — one per agent.

Task flow (sequential):
  1. spec_task      → Project Manager   → structured spec
  2. dev_task       → Developer         → full source code  (reads spec)
  3. review_task    → Code Reviewer     → review report     (reads code)
  4. qa_task        → QA Tester         → test suite + QA report (reads code+review)

context=[...]
  Each task receives the outputs of prior tasks as context, so every agent
  has full visibility of what came before — no repeated work, no lost info.
"""

from crewai import Task
from crewai import Agent


def build_tasks(
    pm: Agent,
    dev: Agent,
    reviewer: Agent,
    qa: Agent,
    task_description: str,
) -> tuple[Task, Task, Task, Task]:
    """
    Build and return the four tasks in execution order.

    Parameters
    ----------
    pm, dev, reviewer, qa : the four Agent objects
    task_description      : raw user requirement string

    Returns
    -------
    (spec_task, dev_task, review_task, qa_task)
    """

    # ──────────────────────────────────────────────────────────────────────
    # Task 1 — Project Manager writes the engineering spec
    # ──────────────────────────────────────────────────────────────────────
    spec_task = Task(
        description=(
            f"A client has requested the following software:\n\n"
            f"  \"{task_description}\"\n\n"
            "Produce a complete engineering specification document. "
            "The spec must contain ALL of the following sections:\n"
            "  1. PROJECT TITLE\n"
            "  2. OBJECTIVE (one sentence)\n"
            "  3. SCOPE (what is included, what is explicitly excluded)\n"
            "  4. FILE LIST (every Python file the Developer must create, "
            "with a one-line responsibility statement for each)\n"
            "  5. FUNCTION SIGNATURES (exact def lines with type hints "
            "for every public function or class)\n"
            "  6. DEPENDENCIES (package name and pinned version for "
            "requirements.txt)\n"
            "  7. ACCEPTANCE CRITERIA (at least three numbered, testable "
            "statements that QA will verify)\n"
            "  8. ASSUMPTIONS (any decisions you made for undefined "
            "requirements)\n\n"
            "Be precise. The Developer will implement exactly what you write."
        ),
        expected_output=(
            "A structured engineering specification document in plain text, "
            "with clearly labelled sections matching the eight required areas. "
            "No bullet-point abbreviations — every item must be a complete, "
            "unambiguous statement."
        ),
        agent=pm,
    )

    # ──────────────────────────────────────────────────────────────────────
    # Task 2 — Developer writes all source code
    # ──────────────────────────────────────────────────────────────────────
    dev_task = Task(
        description=(
            "You have received the engineering specification from the Project "
            "Manager (available in your context). "
            "Implement EVERY file listed in the FILE LIST section of the spec.\n\n"
            "Output format rules (STRICT):\n"
            "  - For each file, write:  ## File: <filename>  on its own line\n"
            "  - Then immediately a fenced code block: ```python ... ```\n"
            "  - After all Python files, write:  ## File: requirements.txt\n"
            "    followed by a plain text block with one package==version per line.\n\n"
            "Code quality rules:\n"
            "  - Every function must have a docstring explaining parameters "
            "and return value.\n"
            "  - Use type hints on all function signatures.\n"
            "  - Wrap all I/O operations in try/except with informative error "
            "messages printed to stderr.\n"
            "  - No placeholder functions — every function must be fully "
            "implemented.\n"
            "  - Use only the packages listed in the spec's DEPENDENCIES section."
        ),
        expected_output=(
            "Complete source code for every file in the spec, each preceded by "
            "a '## File: <filename>' header and wrapped in a fenced code block. "
            "Followed by requirements.txt content. Zero placeholder functions."
        ),
        agent=dev,
        context=[spec_task],   # Developer reads PM's spec
    )

    # ──────────────────────────────────────────────────────────────────────
    # Task 3 — Reviewer audits the code
    # ──────────────────────────────────────────────────────────────────────
    review_task = Task(
        description=(
            "You have received both the engineering specification (Task 1) "
            "and the Developer's full source code (Task 2) in your context.\n\n"
            "Perform a thorough code review. Structure your output using these "
            "exact headings:\n\n"
            "## SUMMARY\n"
            "  Brief overview of what was implemented.\n\n"
            "## SPEC COMPLIANCE\n"
            "  For each acceptance criterion in the spec, write:\n"
            "    [MET / NOT MET] Criterion text — explanation\n\n"
            "## ISSUES\n"
            "  List every problem found. Format each as:\n"
            "    [CRITICAL/MAJOR/MINOR] File: <name> | Line: <approx> | "
            "Description | Suggested Fix\n\n"
            "## SUGGESTIONS\n"
            "  Non-blocking improvements (style, performance, readability).\n\n"
            "## VERDICT\n"
            "  Either:  APPROVED  or  NEEDS_REVISION\n"
            "  If NEEDS_REVISION, list exactly what must change before approval."
        ),
        expected_output=(
            "A structured code review report with sections: SUMMARY, "
            "SPEC COMPLIANCE, ISSUES, SUGGESTIONS, and a final VERDICT of "
            "either APPROVED or NEEDS_REVISION."
        ),
        agent=reviewer,
        context=[spec_task, dev_task],   # Reviewer reads spec + code
    )

    # ──────────────────────────────────────────────────────────────────────
    # Task 4 — QA Tester writes and reports tests
    # ──────────────────────────────────────────────────────────────────────
    qa_task = Task(
        description=(
            "You have received the spec, the Developer's code, and the "
            "Reviewer's report in your context.\n\n"
            "Your job:\n"
            "  1. Write a complete pytest test file named test_suite.py.\n"
            "     - At least 5 test functions.\n"
            "     - Cover: normal usage, edge cases, and error conditions.\n"
            "     - Mock all external HTTP/network calls so tests run offline.\n"
            "     - Use descriptive names: test_<what>_<condition>_<expected>.\n\n"
            "  2. After the test file, write a QA EXECUTION REPORT:\n"
            "     - For each test, write:  [PASS/FAIL] test_name — reason\n"
            "     - Simulate running the tests against the Developer's code "
            "and report the realistic expected outcome.\n\n"
            "  3. End with:\n"
            "     OVERALL QA VERDICT: PASS or FAIL\n\n"
            "Output format:\n"
            "  ## File: test_suite.py\n"
            "  ```python ... ```\n"
            "  ## QA EXECUTION REPORT\n"
            "  [results per test]\n"
            "  OVERALL QA VERDICT: ..."
        ),
        expected_output=(
            "A complete test_suite.py file in a fenced code block, followed by "
            "a QA EXECUTION REPORT showing PASS/FAIL per test and ending with "
            "OVERALL QA VERDICT: PASS or FAIL."
        ),
        agent=qa,
        context=[spec_task, dev_task, review_task],   # QA reads everything
    )

    return spec_task, dev_task, review_task, qa_task
