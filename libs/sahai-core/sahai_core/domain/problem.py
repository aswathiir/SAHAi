from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TestCase:
    input: dict[str, Any]
    expected: Any

    def format_call(self, function_name: str) -> str:
        args = ", ".join(f"{v!r}" for v in self.input.values())
        return f"{function_name}({args})"


@dataclass
class Problem:
    id: str
    title: str
    description: str
    difficulty: int
    skills: list[str]
    function_name: str
    function_signature: str
    test_cases: list[TestCase]
    solution: str

    @classmethod
    def from_dict(cls, d: dict) -> Problem:
        cases = [TestCase(input=tc["input"], expected=tc["expected"]) for tc in d["test_cases"]]
        return cls(
            id=d["id"],
            title=d["title"],
            description=d["description"],
            difficulty=d["difficulty"],
            skills=d["skills"],
            function_name=d["function_name"],
            function_signature=d["function_signature"],
            test_cases=cases,
            solution=d["solution"],
        )


@dataclass
class ProblemBank:
    problems: list[Problem] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> ProblemBank:
        path = Path(path)
        problems = []
        for f in sorted(path.glob("*.json")):
            with open(f) as fh:
                data = json.load(fh)
            items = data if isinstance(data, list) else [data]
            problems.extend(Problem.from_dict(d) for d in items)
        return cls(problems=problems)

    def filter_difficulty(self, low: int, high: int) -> list[Problem]:
        return [p for p in self.problems if low <= p.difficulty <= high]

    def filter_skills(self, skills: set[str]) -> list[Problem]:
        return [p for p in self.problems if skills & set(p.skills)]

    def max_difficulty(self) -> int:
        """The top of this bank's own difficulty scale, for ability mapping."""
        return max((p.difficulty for p in self.problems), default=1)

    def zpd_candidates(self, tracer) -> list[Problem]:
        """Problems sitting one step beyond what the learner can already do.

        Exposed separately so a caller can log how many there were. Under the
        old mastery-only band this was the interesting event, because the band
        emptied permanently after three epochs; with difficulty as the axis it
        should stay populated, and a warning now means something has gone
        wrong rather than something routine.
        """
        top = self.max_difficulty()
        return [p for p in self.problems if tracer.in_zpd(p.difficulty, p.skills, top)]

    def zpd_sample(self, tracer, n: int) -> list[Problem]:
        """`n` problems, nearest the learner's target difficulty.

        This is a *ranking*, not a filter with a fallback. The previous version
        filtered on a band that could be — and for seven of ten epochs in the
        2026-09-12 run was — completely empty, then topped the batch up with
        whatever sorted nearest. The top-up kept the batch at full size but the
        ordering it fell back on was distance from a band nothing was in, so
        the "curriculum" was decided by how rarely a skill happened to be
        tagged. Ranking directly on difficulty cannot degenerate that way: it
        returns the same `n` whether or not anything clears the band.

        Draws are random within the window, not ranked into it. See the body
        for why: ranking reintroduced the same freeze on a different axis.
        """
        if not self.problems:
            return []

        # Sample *within* the window rather than sorting to the single nearest
        # difficulty. Sorting looked right and simulated badly: at an ability
        # that puts the target near 2.0, `abs(d - target)` is 0.0 for every
        # difficulty-2 problem and 1.0 for every difficulty-1 one, so all ten
        # epochs drew difficulty 2 and the 186 easier problems were never seen.
        # That is the same failure as before wearing different clothes — one
        # frozen slice of the bank — and GRPO needs reward spread, which comes
        # from a mix.
        band = self.zpd_candidates(tracer)
        if len(band) >= n:
            return random.sample(band, n)

        # Only if the window really is short: fill from the nearest problems
        # outside it, closest difficulty first, then best mastery fit.
        chosen = list(band)
        rest = [p for p in self.problems if p not in chosen]
        target = tracer.target_difficulty(self.max_difficulty())
        random.shuffle(rest)
        rest.sort(
            key=lambda p: (
                abs(p.difficulty - target),
                tracer.mastery_gap(p.skills),
            )
        )
        return chosen + rest[: n - len(chosen)]
