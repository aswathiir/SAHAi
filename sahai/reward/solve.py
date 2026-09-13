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


# Both the student's attempted solution and (since the leakage fix) the
# tutor's extracted code block run here verbatim. Models routinely append
# "# Example usage\nprint(...)" after a function definition, which writes to
# stdout *before* our own result line. Reading raw stdout as JSON then fails
# to parse and silently reports passed=False with no error — this was
# deflating r_sol in every run to date, not just the new leakage check, and
# it went unnoticed because the failure mode has no error message.
#
# Fixed the same way services/executor/app/sandbox.py already does it: read
# only the line after a sentinel, so anything the candidate itself prints
# cannot interfere with (or forge) the result.
_SENTINEL = "__SAHAI_RESULT__"

# `StudentSimulator.attempt_solution` decodes greedily, so every draw for a
# given dialogue is the same string. Flip this with that decoder, not on its
# own — they have to agree or `num_samples` silently pays for duplicates.
_SAMPLES_VARY = False


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
        lines.append(f"sys.stdout.write('\\n{_SENTINEL}' + json.dumps(result))")
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

        if _SENTINEL not in proc.stdout:
            return ExecutionResult(
                passed=False,
                output=proc.stdout.strip(),
                error="no result sentinel in output (candidate produced no return value)",
            )

        payload = proc.stdout.rsplit(_SENTINEL, 1)[1]
        try:
            import json

            output = json.loads(payload)
            passed = output == test_case.expected
            return ExecutionResult(passed=passed, output=str(output))
        except (json.JSONDecodeError, ValueError):
            return ExecutionResult(
                passed=False, output=payload.strip(), error="result was not JSON-serialisable"
            )

    def verify(self, code: str, problem: Problem) -> float:
        if not problem.test_cases:
            return 0.0
        passed = sum(
            self.execute(code, tc, problem.function_name).passed for tc in problem.test_cases
        )
        return passed / len(problem.test_cases)


class SolveOutcome(float):
    """Two readings of the same attempt, kept apart on purpose.

    `score` is the reward signal and `solved` is the headline number.
    Conflating them would either blunt the gradient or break comparability
    with every run recorded so far.

    It is a `float` subclass, not a dataclass, and that is load-bearing. The
    Kaggle kernel and the `sahai/` package are separate uploads and only the
    package is pushed before a run (see docs/07-operations-kaggle.md §2b), so
    the notebook that will execute this code is whatever was last pushed — and
    a cell in it does:

        r_sol = solve_rw.compute(student, dialogue, test_problem)
        print(f"r_sol={r_sol:.2f} ...")

    A dataclass would raise `unsupported format string` there and kill the run
    in the smoke-check cell, before training starts. Behaving as its own score
    means every existing caller keeps working while `.solved` is available to
    the ones that know to ask.
    """

    solved: float

    def __new__(cls, score: float, solved: float) -> SolveOutcome:
        outcome = super().__new__(cls, score)
        outcome.solved = solved
        return outcome

    @property
    def score(self) -> float:
        """Mean fraction of test cases passed — continuous, and what GRPO sees."""
        return float(self)

    def __repr__(self) -> str:
        return f"SolveOutcome(score={float(self):.4f}, solved={self.solved:.4f})"



class SolveReward:
    def __init__(self, verifier: CodeVerifier, num_samples: int = 8):
        self.verifier = verifier
        # `attempt_solution` decodes greedily, so repeated draws are identical
        # and any n > 1 buys nothing but wall-clock. Clamped rather than
        # removed so `settings.reward.num_solve_samples` keeps its meaning if
        # sampling is ever restored.
        self.num_samples = max(1, num_samples) if _SAMPLES_VARY else 1
        self.requested_samples = num_samples

    def compute(
        self,
        student,
        dialogue,
        problem: Problem,
    ) -> SolveOutcome:
        """Score the student's post-tutoring attempt on `problem`.

        Partial credit is the change that matters here. This used to count a
        sample only when `verify()` returned exactly 1.0, so an attempt passing
        4 of 5 tests scored the same zero as one that did not parse — the
        verifier already returns the fraction and the reward discarded it. On
        the term that most needs resolution, that was the largest avoidable
        loss of it: 22 of 24 rollouts in the dumps on disk scored exactly 0.00.

        The all-or-nothing reading is not thrown away, it is returned
        alongside as `solved`, because that is the number every previous run
        reported and the one a held-out comparison has to be made on.
        """
        scores = []
        for _ in range(self.num_samples):
            code = student.attempt_solution(dialogue, problem)
            scores.append(self.verifier.verify(code, problem))
        n = len(scores)
        return SolveOutcome(
            score=sum(scores) / n,
            solved=sum(1 for s in scores if s == 1.0) / n,
        )


def tutor_code_solves(dialogue, problem: Problem, verifier: CodeVerifier, estimator) -> bool:
    """Did the tutor write code that actually solves the problem?

    Execution is the only reliable test. Token overlap scored 0.000 for a
    tutor that wrote a complete working solution, because it chose a
    different algorithm than the reference — so it was measuring plagiarism,
    not leakage. Code that passes the problem's own tests IS the answer,
    however it is written.

    Lives here rather than on `LeakageEstimator` because that module is
    deliberately execution-free — running untrusted code is a privileged
    operation belonging to the caller. It is a shared function rather than a
    method on the trainer because both the trainer and the evaluator need it,
    and the evaluator silently not calling it was a real bug: held-out
    leakage fell back to token overlap and under-detected exactly the
    complete code leaks this check exists to catch.
    """
    for block in estimator.extract_tutor_code(dialogue):
        for variant in estimator.runnable_variants(block, problem.function_name):
            try:
                if verifier.verify(variant, problem) == 1.0:
                    return True
            except Exception:  # noqa: BLE001 - a broken block is not a leak
                continue
    return False
