"""How many of these 60 problems can tutoring possibly change?

Every experiment this project has run reports a difference in solve rate
between arms. None of them has reported how many problems were *available* to
differ. A problem the student solves with no help and a problem it fails with
the answer in front of it are both outside the reach of any reward function,
and if those two classes account for most of the bank then the effect size the
experiment is powered to detect is irrelevant: there is nothing there to find.

This partitions the bank by the joint outcome across arms and counts the
partition that tutoring can actually move.
"""
from __future__ import annotations

import ast
import json
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "libs/sahai-core")

d = json.load(open("kaggle_upload/dialogues/ab_benchmark.json"))
arms = {a: {r["problem_id"]: r["solved"] for r in d["arms"][a]["dialogues"]}
        for a in ("unaided", "base", "trained")}
ids = list(arms["unaided"])

from sahai.core.dataset import load_mbpp
bank = load_mbpp(split="test", max_problems=60)
by_id = {p.id: p for p in bank.problems}


def nodes(pid):
    try:
        return sum(1 for _ in ast.walk(ast.parse(by_id[pid].solution)))
    except (SyntaxError, KeyError):
        return -1


cells: dict[tuple[int, int, int], list[str]] = {}
for pid in ids:
    key = (arms["unaided"][pid], arms["base"][pid], arms["trained"][pid])
    cells.setdefault(key, []).append(pid)

print("Joint outcome across the three arms, 60 problems")
print(f"  {'unaided':>8s} {'base':>5s} {'trained':>8s}   {'n':>3s}   what it means")
MEANING = {
    (1, 1, 1): "solved regardless; tutoring cannot help",
    (0, 0, 0): "failed regardless; tutoring did not help",
    (1, 0, 0): "LOST by both tutors",
    (1, 1, 0): "lost by the trained tutor only",
    (1, 0, 1): "lost by the untrained tutor only",
    (0, 1, 1): "GAINED by both tutors",
    (0, 0, 1): "gained by the trained tutor only",
    (0, 1, 0): "gained by the untrained tutor only",
}
for key in sorted(cells, reverse=True):
    print(f"  {key[0]:>8d} {key[1]:>5d} {key[2]:>8d}   {len(cells[key]):>3d}   {MEANING[key]}")

inert = len(cells.get((1, 1, 1), [])) + len(cells.get((0, 0, 0), []))
movable = 60 - inert
print(f"\n  inert   {inert}/60 = {inert/60:.1%}  (same outcome in all three arms)")
print(f"  movable {movable}/60 = {movable/60:.1%}")

print("\nWhat the trained tutor actually did, relative to no tutoring")
gained = [p for p in ids if arms["unaided"][p] == 0 and arms["trained"][p] == 1]
lost = [p for p in ids if arms["unaided"][p] == 1 and arms["trained"][p] == 0]
print(f"  gained {len(gained)}   lost {len(lost)}   net {len(gained) - len(lost):+d}")

print("\nDifficulty of each partition (AST nodes in the reference solution)")
def mean(xs):
    xs = [x for x in xs if x >= 0]
    return sum(xs) / len(xs) if xs else float("nan")
for key, label in (((1, 1, 1), "solved regardless"), ((0, 0, 0), "failed regardless")):
    grp = cells.get(key, [])
    print(f"  {label:20s} n={len(grp):3d}  mean nodes {mean(nodes(p) for p in grp):6.1f}")
mv = [p for p in ids if (arms['unaided'][p], arms['base'][p], arms['trained'][p])
      not in ((1, 1, 1), (0, 0, 0))]
print(f"  {'movable':20s} n={len(mv):3d}  mean nodes {mean(nodes(p) for p in mv):6.1f}")
