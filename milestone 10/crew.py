"""
crew.py
────────
DevSquadCrew — the central orchestrator.

This class:
  1. Instantiates all four agents.
  2. Builds all four tasks with correct context chains.
  3. Assembles and kicks off a CrewAI Crew in sequential process.
  4. Post-processes output: saves files, optionally runs Docker QA,
     optionally pushes to GitHub.

CrewAI sequential process
  Agents execute one after another in the order given to Crew(tasks=[...]).
  Each task's output is available as context to subsequent tasks via
  task.context=[...].  No LLM sees a future task's output — causality is
  preserved.

Retry logic
  Groq's free tier has rate limits (≈30 req/min).  We use tenacity to retry
  individual Crew runs on RateLimitError with exponential backoff.
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

from crewai import Crew, Process
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from agents import (
    build_project_manager,
    build_developer,
    build_code_reviewer,
    build_qa_tester,
)
from tasks import build_tasks
from tools.docker_runner import run_in_docker, is_docker_available
from tools.github_pusher import push_to_github, extract_files_from_crew_output


OUTPUT_DIR = Path("outputs")


class DevSquadCrew:
    """
    Orchestrates the four-agent DevSquad pipeline.

    Parameters
    ----------
    task_description : plain-English description of what to build
    use_docker       : if True, run QA tests inside a Docker container
    push_to_github   : if True, push extracted files to a new GitHub repo
    """

    def __init__(
        self,
        task_description: str,
        use_docker: bool = False,
        push_to_github: bool = False,
    ) -> None:
        self.task_description = task_description
        self.use_docker = use_docker
        self._push_to_github = push_to_github

        # Instantiate agents
        self.pm = build_project_manager()
        self.dev = build_developer()
        self.reviewer = build_code_reviewer()
        self.qa = build_qa_tester()

        # Build tasks with dependency chain
        (
            self.spec_task,
            self.dev_task,
            self.review_task,
            self.qa_task,
        ) = build_tasks(
            pm=self.pm,
            dev=self.dev,
            reviewer=self.reviewer,
            qa=self.qa,
            task_description=task_description,
        )

        # Assemble crew
        self.crew = Crew(
            agents=[self.pm, self.dev, self.reviewer, self.qa],
            tasks=[self.spec_task, self.dev_task, self.review_task, self.qa_task],
            process=Process.sequential,   # one agent at a time, in order
            verbose=True,
            memory=False,                 # stateless — context flows via task.context
            max_rpm=20,                   # stay within Groq free-tier rate limits
        )

    # ------------------------------------------------------------------
    # Retry wrapper around crew.kickoff()
    # Groq returns HTTP 429 on rate limit; tenacity catches it and waits.
    # ------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=10, max=60),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _kickoff_with_retry(self) -> str:
        """Call crew.kickoff() with automatic retry on transient errors."""
        result = self.crew.kickoff()
        # CrewAI ≥0.70 returns a CrewOutput object; get the string form
        return str(result) if not isinstance(result, str) else result

    def run(self) -> str:
        """
        Execute the full DevSquad pipeline.

        Returns
        -------
        str : final combined output (spec + code + review + QA report)
        """
        print("[DevSquad] Starting crew kickoff...")
        start = time.time()

        raw_output = self._kickoff_with_retry()

        elapsed = time.time() - start
        print(f"[DevSquad] Crew finished in {elapsed:.1f}s")

        # Save raw output
        self._save_output(raw_output)

        # Optionally run code in Docker
        if self.use_docker:
            self._run_docker_qa(raw_output)

        # Optionally push to GitHub
        if self._push_to_github:
            self._push_files(raw_output)

        return raw_output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_output(self, raw_output: str) -> None:
        """Save the full crew output and extracted files to outputs/."""
        OUTPUT_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw text
        raw_path = OUTPUT_DIR / f"devsquad_output_{timestamp}.txt"
        raw_path.write_text(raw_output, encoding="utf-8")
        print(f"[DevSquad] Raw output saved → {raw_path}")

        # Extract and save individual files
        files = extract_files_from_crew_output(raw_output)
        if files:
            files_dir = OUTPUT_DIR / f"files_{timestamp}"
            files_dir.mkdir(exist_ok=True)
            for filename, content in files.items():
                (files_dir / filename).write_text(content, encoding="utf-8")
            print(f"[DevSquad] {len(files)} source files extracted → {files_dir}/")

    def _run_docker_qa(self, raw_output: str) -> None:
        """Extract test_suite.py and run it inside Docker."""
        if not is_docker_available():
            print(
                "[DevSquad][Docker] Docker daemon not found — skipping container QA.\n"
                "  Install Docker Desktop and start the daemon, then re-run with --docker."
            )
            return

        files = extract_files_from_crew_output(raw_output)
        test_code = files.get("test_suite.py", "")
        requirements = files.get("requirements.txt", "")

        if not test_code:
            print("[DevSquad][Docker] test_suite.py not found in output — skipping.")
            return

        print("[DevSquad][Docker] Running test_suite.py in isolated container...")
        result = run_in_docker(
            code=test_code,
            requirements=f"pytest==7.4.4\npytest-mock==3.12.0\nresponses==0.25.0\n{requirements}",
            timeout_seconds=120,
        )
        print(f"[DevSquad][Docker] Exit code: {result['returncode']}")
        if result["stdout"]:
            print("[DevSquad][Docker] STDOUT:\n" + result["stdout"])
        if result["stderr"]:
            print("[DevSquad][Docker] STDERR:\n" + result["stderr"])

    def _push_files(self, raw_output: str) -> None:
        """Extract all files and push them to a new GitHub repository."""
        files = extract_files_from_crew_output(raw_output)
        if not files:
            print("[DevSquad][GitHub] No files extracted — skipping push.")
            return

        # Add a README
        files["README.md"] = (
            "# DevSquad Output\n\n"
            "Generated by [DevSquad](https://github.com/devsquad) — "
            "a 4-agent autonomous engineering team built with CrewAI + Groq.\n\n"
            f"**Task:** {self.task_description}\n"
        )

        print(f"[DevSquad][GitHub] Pushing {len(files)} files...")
        result = push_to_github(
            files=files,
            repo_name="devsquad-output",
            description=f"DevSquad output: {self.task_description[:80]}",
        )
        print(f"[DevSquad][GitHub] {result['message']}")
