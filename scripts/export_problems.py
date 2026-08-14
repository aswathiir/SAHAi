"""Curate a small, JSON-safe problem set for the student assessment interface.

Pulls from the same bank training uses (`load_mbpp(split="train")`), but keeps
only problems whose expected values survive a JSON round-trip unchanged (a
tuple becomes a list over JSON, and the executor compares with `==`, so a
tuple-returning problem would fail every correct submission). Deliberately
does not include the `solution` field — the exported file is served straight
to the frontend, and the tutor's whole job is to never reveal it.

Usage:
    python scripts/export_problems.py [--count 24] [--out services/gateway/app/data/problems.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sahai.core.dataset import load_mbpp


def _json_safe(value: object) -> bool:
    """True if value round-trips through JSON unchanged (so `==` still holds)."""
    try:
        return json.loads(json.dumps(value)) == value
    except (TypeError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--out", default="services/gateway/app/data/problems.json")
    args = parser.parse_args()

    bank = load_mbpp(split="train", max_problems=300)

    seen_skills: dict[str, int] = {}
    curated = []
    for p in bank.problems:
        if not all(_json_safe(tc.input) and _json_safe(tc.expected) for tc in p.test_cases):
            continue
        if len(p.test_cases) < 1:
            continue
        primary_skill = p.skills[0]
        # Spread across skills instead of taking the first N problems in order,
        # so the picker isn't dominated by whichever skill MBPP front-loads.
        if seen_skills.get(primary_skill, 0) >= 3:
            continue
        seen_skills[primary_skill] = seen_skills.get(primary_skill, 0) + 1

        curated.append({
            "id": p.id,
            "title": p.title,
            "description": p.description,
            "difficulty": p.difficulty,
            "skills": p.skills,
            "function_name": p.function_name,
            "function_signature": p.function_signature,
            "test_cases": [
                {"input": tc.input, "expected": tc.expected} for tc in p.test_cases
            ],
        })
        if len(curated) >= args.count:
            break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(curated, indent=2))
    print(f"wrote {len(curated)} problems -> {out_path}")


if __name__ == "__main__":
    main()
