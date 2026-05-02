import subprocess
import os
import time
import requests
from pathlib import Path
from config import DOCKER_BASE_IMAGE, AGENT_PORT_START, DOCKER_NETWORK


def _run(cmd: list[str], capture: bool = True, cwd: str = None) -> tuple[int, str, str]:
    result = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        cwd=cwd,
        encoding="utf-8",
        errors="replace",
    )
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    return result.returncode, stdout, stderr


def ensure_network():
    """Create the Docker network if it doesn't exist."""
    code, out, _ = _run(["docker", "network", "inspect", DOCKER_NETWORK])
    if code != 0:
        _run(["docker", "network", "create", DOCKER_NETWORK])


def write_dockerfile(agent_dir: Path, requirements: list[str], extra_run: str = "") -> None:
    """Generate a Dockerfile for the agent."""
    req_install = " ".join(requirements) if requirements else "fastapi uvicorn openai"
    extra_block = f"\nRUN {extra_run}" if extra_run.strip() else ""
    dockerfile = f"""FROM {DOCKER_BASE_IMAGE}

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir {req_install}{extra_block}

# Copy agent code
COPY agent.py .

EXPOSE 8000

CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    (agent_dir / "Dockerfile").write_text(dockerfile)


def build_image(agent_name: str, agent_dir: Path) -> tuple[bool, str]:
    """Build Docker image for the agent. Returns (success, output)."""
    image_tag = f"agent-{agent_name}:latest"
    code, out, err = _run(
        ["docker", "build", "-t", image_tag, "."],
        cwd=str(agent_dir),
    )
    combined = out + err
    return code == 0, combined


def run_container(agent_name: str, host_port: int, env_vars: dict = None) -> tuple[bool, str, str]:
    """
    Start the agent container. Returns (success, container_id, error).
    """
    ensure_network()
    image_tag = f"agent-{agent_name}:latest"
    container_name = f"agent-{agent_name}"

    # Remove existing container with same name
    _run(["docker", "rm", "-f", container_name])

    cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        "--network", DOCKER_NETWORK,
        "-p", f"{host_port}:8000",
    ]

    # Pass environment variables
    env = env_vars or {}
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]

    cmd.append(image_tag)

    code, out, err = _run(cmd)
    container_id = out.strip()
    return code == 0, container_id, err


def test_container(host_port: int, test_input: str = "hello", timeout: int = 30) -> tuple[bool, str]:
    """
    Wait for container to be ready, then call /run.
    Returns (success, response_output).
    """
    url_health = f"http://localhost:{host_port}/health"
    url_run = f"http://localhost:{host_port}/run"

    # Wait for health
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url_health, timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        return False, "Container did not become healthy in time."

    # Call /run
    try:
        r = requests.post(url_run, json={"input": test_input}, timeout=15)
        if r.status_code == 200:
            return True, r.json().get("output", str(r.json()))
        return False, f"HTTP {r.status_code}: {r.text}"
    except Exception as e:
        return False, str(e)


def get_container_logs(agent_name: str) -> str:
    """Fetch logs from a running container."""
    _, out, err = _run(["docker", "logs", f"agent-{agent_name}"])
    return (out + err).strip()


def stop_container(agent_name: str) -> None:
    """Stop and remove a container."""
    _run(["docker", "rm", "-f", f"agent-{agent_name}"])


def get_next_port(used_ports: set[int]) -> int:
    """Find the next available port starting from AGENT_PORT_START."""
    port = AGENT_PORT_START
    while port in used_ports:
        port += 1
    return port
