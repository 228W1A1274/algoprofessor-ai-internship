import os
import json
from pathlib import Path
from datetime import datetime
from config import GENERATED_AGENTS_DIR, OPENAI_API_KEY
from docker_manager import (
    write_dockerfile,
    build_image,
    run_container,
    test_container,
    get_container_logs,
    get_next_port,
)
from code_generator import fix_agent_code

REGISTRY_FILE = Path(GENERATED_AGENTS_DIR) / "registry.json"


def _load_registry() -> dict:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text())
    return {}


def _save_registry(registry: dict) -> None:
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2))


def _get_used_ports(registry: dict) -> set:
    return {v["port"] for v in registry.values() if "port" in v}


def deploy_agent(
    agent_name: str,
    agent_code: str,
    requirements: list[str],
    dockerfile_extra: str,
    description: str,
    max_retries: int = 3,
) -> dict:
    """
    Full deploy pipeline:
    1. Write code + Dockerfile
    2. Build image
    3. Run container
    4. Test it
    5. Auto-fix on failure (up to max_retries)
    6. Register in registry
    """
    base_dir = Path(GENERATED_AGENTS_DIR)
    agent_dir = base_dir / agent_name
    agent_dir.mkdir(parents=True, exist_ok=True)

    current_code = agent_code
    last_error = ""

    registry = _load_registry()
    used_ports = _get_used_ports(registry)
    host_port = get_next_port(used_ports)

    for attempt in range(1, max_retries + 1):
        # Write files
        agent_file = agent_dir / "agent.py"
        agent_file.write_text(current_code)
        write_dockerfile(agent_dir, requirements, dockerfile_extra)

        # Build
        build_ok, build_output = build_image(agent_name, agent_dir)
        if not build_ok:
            last_error = f"Build failed:\n{build_output}"
            if attempt < max_retries:
                current_code = fix_agent_code(current_code, last_error)
                continue
            else:
                return {
                    "success": False,
                    "agent_name": agent_name,
                    "error": last_error,
                    "attempts": attempt,
                }

        # Run container
        run_ok, container_id, run_err = run_container(
            agent_name,
            host_port,
            env_vars={"OPENAI_API_KEY": OPENAI_API_KEY},
        )
        if not run_ok:
            last_error = f"Container start failed:\n{run_err}"
            if attempt < max_retries:
                current_code = fix_agent_code(current_code, last_error)
                continue
            else:
                return {
                    "success": False,
                    "agent_name": agent_name,
                    "error": last_error,
                    "attempts": attempt,
                }

        # Test
        test_ok, test_output = test_container(host_port)
        if not test_ok:
            logs = get_container_logs(agent_name)
            last_error = f"Test failed: {test_output}\nContainer logs:\n{logs}"
            if attempt < max_retries:
                current_code = fix_agent_code(current_code, last_error)
                continue
            else:
                return {
                    "success": False,
                    "agent_name": agent_name,
                    "error": last_error,
                    "attempts": attempt,
                }

        # Success — register agent
        registry[agent_name] = {
            "agent_name": agent_name,
            "description": description,
            "port": host_port,
            "container_id": container_id[:12],
            "endpoint": f"http://localhost:{host_port}",
            "deployed_at": datetime.utcnow().isoformat() + "Z",
            "attempts": attempt,
            "status": "running",
        }
        _save_registry(registry)

        return {
            "success": True,
            "agent_name": agent_name,
            "description": description,
            "port": host_port,
            "endpoint": f"http://localhost:{host_port}",
            "test_output": test_output,
            "attempts": attempt,
            "container_id": container_id[:12],
        }

    return {
        "success": False,
        "agent_name": agent_name,
        "error": f"Failed after {max_retries} attempts. Last error: {last_error}",
        "attempts": max_retries,
    }


def list_agents() -> list:
    registry = _load_registry()
    return list(registry.values())


def get_agent(agent_name: str) -> dict | None:
    registry = _load_registry()
    return registry.get(agent_name)
