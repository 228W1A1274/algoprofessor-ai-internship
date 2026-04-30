"""
tools/docker_runner.py
───────────────────────
Provides run_in_docker() — executes arbitrary Python code in an isolated
Docker container and returns stdout + stderr.

How it works
  1. Write the code string to a temp file on the host.
  2. Run `docker run --rm -v <tmpdir>:/app python:3.11-slim python /app/run.py`
  3. Capture output, clean up, and return.

Why Docker for QA?
  The QA agent's tests run in an environment with only the packages listed in
  requirements.txt — no accidental dependency on the host machine.  This
  catches missing imports before the code ever reaches a real server.

Note: If Docker is not installed or the daemon is not running, the function
raises RuntimeError with a helpful message.  The caller (crew.py) falls back
to a mock execution when use_docker=False.
"""

import subprocess
import tempfile
import textwrap
import os
import sys
from pathlib import Path


def run_in_docker(
    code: str,
    requirements: str = "",
    timeout_seconds: int = 60,
) -> dict[str, str]:
    """
    Execute Python code inside a Docker container.

    Parameters
    ----------
    code             : Python source code to run.
    requirements     : pip-installable packages, one per line (e.g. "requests==2.31.0").
    timeout_seconds  : kill the container after this many seconds.

    Returns
    -------
    dict with keys:
      "stdout"     : captured standard output
      "stderr"     : captured standard error
      "returncode" : process exit code as string ("0" = success)
      "status"     : "success" or "error"
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write the Python source file
        run_file = Path(tmpdir) / "run.py"
        run_file.write_text(code, encoding="utf-8")

        # Build the docker command
        if requirements.strip():
            # Write a minimal Dockerfile that installs deps then runs code
            req_file = Path(tmpdir) / "requirements.txt"
            req_file.write_text(requirements, encoding="utf-8")

            dockerfile = textwrap.dedent(f"""\
                FROM python:3.11-slim
                WORKDIR /app
                COPY requirements.txt .
                RUN pip install --no-cache-dir -r requirements.txt
                COPY run.py .
                CMD ["python", "run.py"]
            """)
            dockerfile_path = Path(tmpdir) / "Dockerfile"
            dockerfile_path.write_text(dockerfile, encoding="utf-8")

            image_tag = "devsquad-runner:latest"

            # Build image
            build_result = subprocess.run(
                ["docker", "build", "-t", image_tag, tmpdir],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if build_result.returncode != 0:
                return {
                    "stdout": "",
                    "stderr": build_result.stderr,
                    "returncode": str(build_result.returncode),
                    "status": "error",
                }

            # Run image
            run_result = subprocess.run(
                ["docker", "run", "--rm", image_tag],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        else:
            # No extra deps — mount the file directly into official python image
            run_result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "-v", f"{tmpdir}:/app",
                    "python:3.11-slim",
                    "python", "/app/run.py",
                ],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )

    status = "success" if run_result.returncode == 0 else "error"
    return {
        "stdout": run_result.stdout,
        "stderr": run_result.stderr,
        "returncode": str(run_result.returncode),
        "status": status,
    }


def is_docker_available() -> bool:
    """Return True if docker CLI is on PATH and the daemon responds."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
