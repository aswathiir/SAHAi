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

    def zpd_candidates(self, tracer) -> list[Problem]:
        """Problems whose average skill mastery sits inside the ZPD band.

        Exposed separately so a caller can log how many there were. The band
        running short is the interesting event, and `zpd_sample` deliberately
        hides it by topping the batch up.
        """
        return [p for p in self.problems if tracer.in_zpd(p.difficulty, p.skills)]

    def zpd_sample(self, tracer, n: int) -> list[Problem]:
        """`n` problems, preferring the ZPD band, topped up when it is short.

        This used to return `min(n, len(candidates))`, and the "band is empty"
        fallback only fired when it was *completely* empty. With one to three
        problems in the band and n=4 it returned a short batch, silently: epoch
        5 of three separate runs drew 2 problems instead of 4, which is half
        that epoch's rollouts and half its gradient, with nothing in the logs
        saying so beyond a line nobody reads as an error.

        The top-up takes the problems nearest the band rather than random ones,
        because "just outside the ZPD" is the best available substitute for
        "inside it" — the alternative is padding a thin epoch with work the
        learner has either mastered or cannot touch.
        """
        in_band = self.zpd_candidates(tracer)
        if len(in_band) >= n:
            return random.sample(in_band, n)

        chosen = list(in_band)
        rest = [p for p in self.problems if p not in chosen]
        if not rest:
            return chosen

        def distance_from_band(problem: Problem) -> float:
            if not problem.skills:
                return 0.0
            avg = sum(tracer.get_mastery(s) for s in problem.skills) / len(problem.skills)
            if avg < tracer.settings.zpd_low:
                return tracer.settings.zpd_low - avg
            if avg > tracer.settings.zpd_high:
                return avg - tracer.settings.zpd_high
            return 0.0

        # Shuffle first so ties are broken randomly rather than by bank order,
        # which would hand every short epoch the same problems.
        random.shuffle(rest)
        rest.sort(key=distance_from_band)
        return chosen + rest[: n - len(chosen)]
