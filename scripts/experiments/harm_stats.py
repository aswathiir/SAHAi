"""Statistics for the retention split, plus the difficulty confound.

The split is: among problems the student solved unaided, does the tutored
dialogue containing the student's own "I am lost" predict losing the problem?

The obvious confound is difficulty. A problem the student sounds confused about
is plausibly a harder problem that it solved unaided by pattern-matching, and
"harder problems are lost more often" would not be a finding about framing. The
reference solution's length and AST size stand in for difficulty here, measured
over the same two groups.
"""
from __future__ import annotations

import ast
import json
import sys
from math import comb

sys.path.insert(0, ".")
sys.path.insert(0, "libs/sahai-core")
from sahai_core.turn_signals import STUCK_PHRASES

d = json.load(open("kaggle_upload/dialogues/ab_benchmark.json"))
unaided = {r["problem_id"]: r["solved"] for r in d["arms"]["unaided"]["dialogues"]}

from sahai.core.dataset import load_mbpp
bank = load_mbpp(split="test", max_problems=60)
by_id = {p.id: p for p in bank.problems}


def fisher_2x2(a, b, c, e):
    """Two-sided Fisher exact. Rows: stuck / not. Cols: kept / lost."""
    n = a + b + c + e
    def p_tab(x):
        return comb(a + b, x) * comb(c + e, (a + c) - x) / comb(n, a + c)
    obs = p_tab(a)
    lo = max(0, (a + c) - (c + e))
    hi = min(a + b, a + c)
    return sum(p_tab(x) for x in range(lo, hi + 1) if p_tab(x) <= obs + 1e-12)


rows = []
for arm in ("base", "trained"):
    for r in d["arms"][arm]["dialogues"]:
        if unaided.get(r["problem_id"], 0) != 1:
            continue
        stu = [t["content"] for t in r["turns"] if t["role"] == "student"]
        tut = [t["content"] for t in r["turns"] if t["role"] == "tutor"]
        p = by_id[r["problem_id"]]
        sol = p.solution.strip()
        try:
            nodes = sum(1 for _ in ast.walk(ast.parse(sol)))
        except SyntaxError:
            nodes = -1
        rows.append({
            "arm": arm,
            "problem_id": r["problem_id"],
            "solved": r["solved"],
            "any_stuck": int(any(any(ph in s.lower() for ph in STUCK_PHRASES) for s in stu)),
            "student_chars": sum(len(s) for s in stu),
            "tutor_chars": sum(len(s) for s in tut),
            "sol_lines": len(sol.split("\n")),
            "sol_nodes": nodes,
        })

st = [r for r in rows if r["any_stuck"]]
no = [r for r in rows if not r["any_stuck"]]
a, b = sum(r["solved"] for r in st), len(st) - sum(r["solved"] for r in st)
c, e = sum(r["solved"] for r in no), len(no) - sum(r["solved"] for r in no)

print("Pooled over both tutored arms, restricted to problems solved unaided")
print(f"  n = {len(rows)} dialogues ({len(rows)//2} problems x 2 arms)\n")
print(f"  {'':34s} {'kept':>5s} {'lost':>5s}  retention")
print(f"  student said 'I am lost' in dialogue {a:5d} {b:5d}  {a/(a+b):.3f}")
print(f"  it did not                           {c:5d} {e:5d}  {c/(c+e):.3f}")
print(f"\n  Fisher exact, two-sided: p = {fisher_2x2(a, b, c, e):.4f}")

def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float('nan')

print("\nDifficulty confound: are the 'lost' problems simply harder?")
print(f"  {'':24s} {'stuck':>8s} {'not stuck':>10s}")
for k in ("sol_lines", "sol_nodes", "tutor_chars", "student_chars"):
    print(f"    {k:22s} {mean(r[k] for r in st):8.1f} {mean(r[k] for r in no):10.1f}")

lost = [r for r in rows if r["solved"] == 0]
kept = [r for r in rows if r["solved"] == 1]
print("\nSame two columns, split by outcome rather than by what the student said")
print(f"  {'':24s} {'LOST':>8s} {'kept':>10s}")
for k in ("sol_lines", "sol_nodes", "tutor_chars", "student_chars"):
    print(f"    {k:22s} {mean(r[k] for r in lost):8.1f} {mean(r[k] for r in kept):10.1f}")
