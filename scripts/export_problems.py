"""Curate a small, JSON-safe problem set for the student assessment interface.

Pulls from the same bank training uses (`load_mbpp(split="train")`), but keeps
only problems whose expected values survive a JSON round-trip unchanged (a
tuple becomes a list over JSON, and the executor compares with `==`, so a
tuple-returning problem would fail every correct submission). Deliberately
does not include the `solution` field — the exported file is served straight
to the frontend, and the tutor's whole job is to never reveal it.

Usage:
    python scripts/export_problems.py [--count 120] [--per-skill 14] [--out ...]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
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
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument(
        "--per-skill",
        type=int,
        default=14,
        help="cap per primary skill, so one skill cannot dominate the picker",
    )
    parser.add_argument("--out", default="services/gateway/app/data/problems.json")
    args = parser.parse_args()

    # The whole train split. Deliberately never `split="test"`: those 20
    # problems are the held-out eval set, and serving them would make the one
    # metric that is comparable across every run meaningless.
    bank = load_mbpp(split="train", max_problems=1000)

    usable = [
        p
        for p in bank.problems
        if p.test_cases
        and all(_json_safe(tc.input) and _json_safe(tc.expected) for tc in p.test_cases)
        and p.skills
    ]

    # Hardest first within each skill. MBPP is overwhelmingly difficulty-1, so
    # taking problems in dataset order fills every skill with its easiest
    # examples and the bank ends up with no range to select within — the same
    # flatness the difficulty estimator used to have. Sorting by difficulty
    # before capping keeps the few genuinely harder problems instead of
    # discarding them at the cap.
    usable.sort(key=lambda p: -p.difficulty)

    seen_skills: dict[str, int] = {}
    curated = []
    for p in usable:
        primary_skill = p.skills[0]
        # Spread across skills instead of taking the first N problems in order,
        # so the picker isn't dominated by whichever skill MBPP front-loads.
        if seen_skills.get(primary_skill, 0) >= args.per_skill:
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

    # Stable order for the picker; the difficulty sort above was only to
    # decide *which* problems survive the per-skill cap, not how they list.
    curated.sort(key=lambda p: (p["difficulty"], p["id"]))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(curated, indent=2))

    by_difficulty = Counter(p["difficulty"] for p in curated)
    by_skill = Counter(s for p in curated for s in p["skills"])
    print(f"wrote {len(curated)} problems -> {out_path}")
    print(f"  difficulty: {dict(sorted(by_difficulty.items()))}")
    print(f"  skills:     {dict(by_skill.most_common())}")


if __name__ == "__main__":
    main()
