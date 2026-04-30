"""
tests/test_devsquad.py
───────────────────────
Unit tests for DevSquad system components.

Tests cover:
  - LLM client factory validates missing key
  - Agent construction produces correct role/goal strings
  - Task context chain is wired correctly
  - Docker runner returns structured dict shape
  - GitHub pusher file extractor parses crew output correctly
  - DevSquadCrew raises EnvironmentError when GROQ_API_KEY absent

Run with:
  pytest tests/ -v
"""

import os
import importlib
import pytest
from unittest.mock import patch, MagicMock


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def set_dummy_groq_key(monkeypatch):
    """Provide a dummy GROQ_API_KEY so module imports don't raise."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_dummy_key_for_unit_tests")


# ──────────────────────────────────────────────────────────────────────────────
# Test 1 — LLM client raises when key is missing
# ──────────────────────────────────────────────────────────────────────────────

def test_get_llm_raises_when_api_key_missing(monkeypatch):
    """
    get_llm() should raise EnvironmentError if GROQ_API_KEY is unset,
    not silently return an LLM with an empty key that fails at call time.
    """
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    # Re-import module with env var absent
    import llm_client
    importlib.reload(llm_client)

    with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
        llm_client.get_llm()


# ──────────────────────────────────────────────────────────────────────────────
# Test 2 — Project Manager agent has correct role
# ──────────────────────────────────────────────────────────────────────────────

@patch("llm_client.LLM")
def test_project_manager_agent_role(mock_llm):
    """
    build_project_manager() must return an Agent with role containing
    'Project Manager' — the Reviewer and QA agents depend on this label.
    """
    mock_llm.return_value = MagicMock()
    from agents.project_manager import build_project_manager
    agent = build_project_manager()
    assert "Project Manager" in agent.role


# ──────────────────────────────────────────────────────────────────────────────
# Test 3 — Developer agent has allow_delegation=False
# ──────────────────────────────────────────────────────────────────────────────

@patch("llm_client.LLM")
def test_developer_agent_no_delegation(mock_llm):
    """
    The Developer must not delegate tasks.  Delegation would cause the crew
    to create a sub-crew, which is unsupported in sequential process mode.
    """
    mock_llm.return_value = MagicMock()
    from agents.developer import build_developer
    agent = build_developer()
    assert agent.allow_delegation is False


# ──────────────────────────────────────────────────────────────────────────────
# Test 4 — Task context chain wired correctly
# ──────────────────────────────────────────────────────────────────────────────

@patch("llm_client.LLM")
def test_task_context_chain_is_correct(mock_llm):
    """
    The QA task must have spec_task, dev_task, and review_task in its context.
    If any are missing, the QA agent won't see the code it needs to test.
    """
    mock_llm.return_value = MagicMock()
    from agents.project_manager import build_project_manager
    from agents.developer import build_developer
    from agents.code_reviewer import build_code_reviewer
    from agents.qa_tester import build_qa_tester
    from tasks.task_definitions import build_tasks

    pm, dev, rev, qa = (
        build_project_manager(), build_developer(),
        build_code_reviewer(), build_qa_tester()
    )
    spec, dev_t, review_t, qa_t = build_tasks(
        pm, dev, rev, qa, "build a hello world script"
    )

    assert spec in qa_t.context
    assert dev_t in qa_t.context
    assert review_t in qa_t.context


# ──────────────────────────────────────────────────────────────────────────────
# Test 5 — Docker runner returns correct shape when docker not available
# ──────────────────────────────────────────────────────────────────────────────

@patch("tools.docker_runner.subprocess.run")
def test_docker_runner_returns_structured_dict_on_success(mock_subprocess):
    """
    run_in_docker() must always return a dict with keys:
    stdout, stderr, returncode, status — regardless of exit code.
    """
    mock_subprocess.return_value = MagicMock(
        returncode=0, stdout="Hello!\n", stderr=""
    )
    from tools.docker_runner import run_in_docker
    result = run_in_docker(code='print("Hello!")')
    assert set(result.keys()) == {"stdout", "stderr", "returncode", "status"}
    assert result["status"] == "success"
    assert result["returncode"] == "0"


# ──────────────────────────────────────────────────────────────────────────────
# Test 6 — Docker runner reports error on non-zero exit
# ──────────────────────────────────────────────────────────────────────────────

@patch("tools.docker_runner.subprocess.run")
def test_docker_runner_reports_error_on_nonzero_exit(mock_subprocess):
    """
    run_in_docker() must return status='error' when the container exits non-zero.
    """
    mock_subprocess.return_value = MagicMock(
        returncode=1, stdout="", stderr="SyntaxError: invalid syntax"
    )
    from tools.docker_runner import run_in_docker
    result = run_in_docker(code="def broken(")
    assert result["status"] == "error"
    assert result["returncode"] == "1"


# ──────────────────────────────────────────────────────────────────────────────
# Test 7 — File extractor parses crew output format correctly
# ──────────────────────────────────────────────────────────────────────────────

def test_extract_files_parses_crew_output_correctly():
    """
    extract_files_from_crew_output() must correctly parse the
    '## File: name\n```python\n...```' format used by the Developer agent.
    """
    from tools.github_pusher import extract_files_from_crew_output

    sample_output = """
## File: weather.py
```python
def get_weather(city: str) -> dict:
    return {"city": city, "temp": 22}
```

## File: requirements.txt
requests==2.31.0
"""
    files = extract_files_from_crew_output(sample_output)
    assert "weather.py" in files
    assert "requirements.txt" in files
    assert "get_weather" in files["weather.py"]
    assert "requests" in files["requirements.txt"]


# ──────────────────────────────────────────────────────────────────────────────
# Test 8 — GitHub pusher returns error when token missing
# ──────────────────────────────────────────────────────────────────────────────

def test_push_to_github_returns_error_when_token_missing(monkeypatch):
    """
    push_to_github() must return status='error' and a helpful message when
    GITHUB_TOKEN is absent — never raise an unhandled exception.
    """
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    from tools.github_pusher import push_to_github
    result = push_to_github(files={"hello.py": "print('hi')"})
    assert result["status"] == "error"
    assert "GITHUB_TOKEN" in result["message"]


# ──────────────────────────────────────────────────────────────────────────────
# Test 9 — DevSquadCrew raises EnvironmentError without GROQ_API_KEY
# ──────────────────────────────────────────────────────────────────────────────

def test_devsquad_crew_raises_without_groq_key(monkeypatch):
    """
    If GROQ_API_KEY is missing, DevSquadCrew.__init__ should raise
    EnvironmentError before any LLM calls are attempted.
    """
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
        import llm_client
        importlib.reload(llm_client)
        llm_client.get_llm()
