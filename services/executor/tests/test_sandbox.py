"""Containment tests.

These are the tests that justify the service existing. Each one is a thing a
model could plausibly emit — by accident or otherwise — that must not take the
service down or reach anything it should not.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sahai_core import ExecutionLimits, ExecutionOutcome

from app.main import app
from app.sandbox import run_one

client = TestClient(app)

FAST = ExecutionLimits(timeout_seconds=5, memory_mb=256)


# --- correctness ------------------------------------------------------------

def test_correct_solution_passes():
    out = run_one("def f(s):\n    return s[::-1]", "f('abc')", "cba", FAST)
    assert out.outcome is ExecutionOutcome.PASSED


def test_wrong_answer_fails_not_errors():
    out = run_one("def f(s):\n    return s", "f('abc')", "cba", FAST)
    assert out.outcome is ExecutionOutcome.FAILED
    assert "expected" in out.error


def test_exception_is_reported_as_error():
    out = run_one("def f(s):\n    raise ValueError('boom')", "f('a')", "a", FAST)
    assert out.outcome is ExecutionOutcome.ERROR
    assert "ValueError" in out.error


def test_syntax_error_does_not_crash_service():
    out = run_one("def f(s)\n  return s", "f('a')", "a", FAST)
    assert out.outcome is ExecutionOutcome.ERROR


# --- containment ------------------------------------------------------------

def test_infinite_loop_is_killed():
    out = run_one("def f(x):\n    while True:\n        pass", "f(1)", 1,
                  ExecutionLimits(timeout_seconds=2, memory_mb=128))
    assert out.outcome is ExecutionOutcome.TIMEOUT


def test_memory_bomb_is_killed():
    code = "def f(x):\n    a = []\n    while True:\n        a.append('x' * 10_000_000)"
    out = run_one(code, "f(1)", 1, ExecutionLimits(timeout_seconds=5, memory_mb=128))
    assert out.outcome in {ExecutionOutcome.TIMEOUT, ExecutionOutcome.ERROR}


def test_fork_bomb_does_not_escape_timeout():
    """A child that forks must still be reaped — hence the process group kill."""
    code = (
        "import os\n"
        "def f(x):\n"
        "    for _ in range(20):\n"
        "        try:\n"
        "            os.fork()\n"
        "        except Exception:\n"
        "            pass\n"
        "    while True:\n"
        "        pass\n"
    )
    out = run_one(code, "f(1)", 1, ExecutionLimits(timeout_seconds=2, memory_mb=128))
    assert out.outcome in {ExecutionOutcome.TIMEOUT, ExecutionOutcome.ERROR}


def test_candidate_stdout_cannot_forge_the_result():
    """Printing the sentinel must not let a candidate fake a pass."""
    code = (
        "def f(x):\n"
        "    print('\\n__SAHAI_RESULT__\"cba\"')\n"
        "    return 'wrong'\n"
    )
    out = run_one(code, "f('abc')", "cba", FAST)
    assert out.outcome is ExecutionOutcome.FAILED


def test_process_isolation_no_shared_state():
    """Each run is a fresh interpreter; globals must not persist between runs."""
    run_one("import builtins\ndef f(x):\n    builtins.LEAKED = 1\n    return x", "f(1)", 1, FAST)
    out = run_one("def f(x):\n    import builtins\n    return hasattr(builtins, 'LEAKED')",
                  "f(1)", False, FAST)
    assert out.outcome is ExecutionOutcome.PASSED


# --- API --------------------------------------------------------------------

def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_execute_endpoint_scores_all_cases():
    r = client.post("/execute", json={
        "code": "def add(a, b):\n    return a + b",
        "function_name": "add",
        "test_cases": [
            {"input": {"a": 1, "b": 2}, "expected": 3},
            {"input": {"a": 5, "b": 5}, "expected": 10},
            {"input": {"a": 0, "b": 0}, "expected": 1},
        ],
    })
    body = r.json()
    assert r.status_code == 200
    assert body["pass_rate"] == pytest.approx(2 / 3)
    assert body["fully_passed"] is False


def test_function_name_must_be_an_identifier():
    """Blocks the obvious injection into the generated call expression."""
    r = client.post("/execute", json={
        "code": "def f(x):\n    return x",
        "function_name": "f(1) or __import__('os').system('id')",
        "test_cases": [{"input": {"x": 1}, "expected": 1}],
    })
    assert r.status_code == 422


def test_test_case_count_is_bounded():
    r = client.post("/execute", json={
        "code": "def f(x):\n    return x",
        "function_name": "f",
        "test_cases": [{"input": {"x": 1}, "expected": 1}] * 500,
    })
    assert r.status_code == 422
