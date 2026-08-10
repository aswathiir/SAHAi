"""Solve-rate scoring.

Deliberately contains **no execution**. Running model-generated code is a
privileged, isolated operation that belongs to `services/executor`; this module
only turns execution *results* into a reward. Keeping the two apart is what
allows every other service to depend on `sahai-core` without inheriting the
ability to execute arbitrary code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExecutionOutcome(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"
    REJECTED = "rejected"  # refused before running (e.g. static guard)


@dataclass(frozen=True)
class TestResult:
    """Outcome of one test case, as reported by the executor service."""

    outcome: ExecutionOutcome
    stdout: str | None = None
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.outcome is ExecutionOutcome.PASSED


@dataclass(frozen=True)
class AttemptResult:
    """One student attempt evaluated against every test case of a problem."""

    results: list[TestResult]

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.passed for r in self.results) / len(self.results)

    @property
    def fully_passed(self) -> bool:
        return bool(self.results) and all(r.passed for r in self.results)


class SolveScorer:
    """r_sol — fraction of sampled attempts that pass every test.

    The literature estimates the post-dialogue solve rate by sampling K student
    answers and taking the empirical mean of an all-or-nothing correctness
    indicator, which is why partial passes do not count.
    """

    @staticmethod
    def score(attempts: list[AttemptResult]) -> float:
        if not attempts:
            return 0.0
        return sum(a.fully_passed for a in attempts) / len(attempts)

    @staticmethod
    def partial_credit(attempts: list[AttemptResult]) -> float:
        """Mean pass rate. Diagnostic only — never feed this to the reward."""
        if not attempts:
            return 0.0
        return sum(a.pass_rate for a in attempts) / len(attempts)
