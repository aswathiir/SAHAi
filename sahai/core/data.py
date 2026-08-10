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

    def zpd_sample(self, tracer, n: int) -> list[Problem]:
        candidates = [p for p in self.problems if tracer.in_zpd(p.difficulty, p.skills)]
        if not candidates:
            candidates = self.problems
        return random.sample(candidates, min(n, len(candidates)))
