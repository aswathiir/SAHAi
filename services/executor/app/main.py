"""sahai-executor — sandboxed execution of model-generated code.

Deliberately has no database, no outbound network, and no knowledge of users,
sessions, or rewards. It takes code and test cases, runs them, and reports what
happened. Everything else in the system depends on that narrowness.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sahai_core import ExecutionLimits, ExecutionOutcome

from .sandbox import run_one

logger = logging.getLogger(__name__)

MAX_TEST_CASES = 50
MAX_CODE_BYTES = 100_000

app = FastAPI(
    title="sahai-executor",
    version="0.1.0",
    description="Sandboxed execution of untrusted candidate solutions.",
)

# Test cases within one request are independent; run them concurrently but
# bounded, so a large request cannot exhaust the host.
_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sandbox")


class TestCaseIn(BaseModel):
    input: dict[str, Any]
    expected: Any = None


class ExecuteRequest(BaseModel):
    code: str = Field(max_length=MAX_CODE_BYTES)
    function_name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    test_cases: list[TestCaseIn] = Field(max_length=MAX_TEST_CASES)
    limits: ExecutionLimits = Field(default_factory=ExecutionLimits)


class TestResultOut(BaseModel):
    outcome: ExecutionOutcome
    stdout: str | None = None
    error: str | None = None


class ExecuteResponse(BaseModel):
    results: list[TestResultOut]
    pass_rate: float
    fully_passed: bool


def _format_call(function_name: str, inputs: dict[str, Any]) -> str:
    args = ", ".join(f"{v!r}" for v in inputs.values())
    return f"{function_name}({args})"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "sahai-executor"}


@app.post("/execute", response_model=ExecuteResponse)
def execute(req: ExecuteRequest) -> ExecuteResponse:
    if not req.test_cases:
        return ExecuteResponse(results=[], pass_rate=0.0, fully_passed=False)

    def run(tc: TestCaseIn):
        return run_one(
            code=req.code,
            call=_format_call(req.function_name, tc.input),
            expected=tc.expected,
            limits=req.limits,
        )

    outcomes = list(_pool.map(run, req.test_cases))
    results = [
        TestResultOut(outcome=o.outcome, stdout=o.stdout, error=o.error) for o in outcomes
    ]
    passed = sum(r.outcome is ExecutionOutcome.PASSED for r in results)

    return ExecuteResponse(
        results=results,
        pass_rate=passed / len(results),
        fully_passed=passed == len(results),
    )
