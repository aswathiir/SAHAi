from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sahai.core.data import Problem, TestCase


@dataclass
class ExecutionResult:
    passed: bool
    output: str | None = None
    error: str | None = None
    timed_out: bool = False


class CodeVerifier:
    def __init__(self, timeout: int = 10, memory_mb: int = 256):
        self.timeout = timeout
        self.memory_mb = memory_mb

    def execute(self, code: str, test_case: TestCase, function_name: str) -> ExecutionResult:
        call = test_case.format_call(function_name)
        lines = ["import json, sys"]
        if platform.system() == "Linux":
            limit = self.memory_mb * 1024 * 1024
            lines.append(
                f"try:\n    import resource; resource.setrlimit(resource.RLIMIT_AS, ({limit}, -1))\nexcept Exception:\n    pass"
            )
        lines.append(code)
        lines.append(f"result = {call}")
        lines.append("print(json.dumps(result))")
        script = "\n".join(lines)
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(passed=False, timed_out=True)

        if proc.returncode != 0:
            return ExecutionResult(passed=False, error=proc.stderr.strip())

        try:
            import json

            output = json.loads(proc.stdout.strip())
            passed = output == test_case.expected
            return ExecutionResult(passed=passed, output=str(output))
        except (json.JSONDecodeError, ValueError):
            return ExecutionResult(passed=False, output=proc.stdout.strip())

    def verify(self, code: str, problem: Problem) -> float:
        if not problem.test_cases:
            return 0.0
        passed = sum(
            self.execute(code, tc, problem.function_name).passed for tc in problem.test_cases
        )
        return passed / len(problem.test_cases)


class SolveReward:
    def __init__(self, verifier: CodeVerifier, num_samples: int = 8):
        self.verifier = verifier
        self.num_samples = num_samples

    def compute(
        self,
        student,
        dialogue,
        problem: Problem,
    ) -> float:
        passed = 0
        for _ in range(self.num_samples):
            code = student.attempt_solution(dialogue, problem)
            score = self.verifier.verify(code, problem)
            if score == 1.0:
                passed += 1
        return passed / self.num_samples
