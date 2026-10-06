"""Does the tutoring deficit come from the tutoring, or from how it is replayed?

`SAHAI_SOLVE_CONTEXT` has had two settings since the 2026-10-02 benchmark and
only one has ever been run. Its own docstring states the gap:

    `hints` exists so the two can be compared on the same problems instead of
    argued about, and the default is unchanged so existing results stay
    comparable until that comparison is run.

That comparison is this script. Under `full` the transcript is replayed with
the student's own turns as `assistant`, so the model writes its solution
immediately after a history in which it said it was confused -- measured at 20%
of 1002 student turns expressing confusion and 28% claiming an understanding
the solve rate does not support. Under `hints` only the tutor's turns reach the
solve step.

The two framings are compared **on the saved dialogues**, not on fresh ones.
Re-running the benchmark would resample the tutor and confound the framing with
new dialogue content; replaying one fixed set of transcripts under two framings
changes exactly one thing. It also costs 120 generations instead of a full
three-arm benchmark.

`attempt_solution` is greedy, so replaying `full` must reproduce the solved
flags recorded in the dump. That reproduction is checked rather than assumed:
if it fails, the framing comparison is not the finding, the nondeterminism is.
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, ".")
sys.path.insert(0, "libs/sahai-core")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sahai.agents.student import StudentPersona, StudentSimulator
from sahai.core.dataset import load_mbpp
from sahai.core.dialogue import Dialogue
from sahai.reward.solve import CodeVerifier

DUMP = sys.argv[1] if len(sys.argv) > 1 else "kaggle_upload/dialogues/ab_benchmark.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "framing_replay.json"
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 60

saved = json.load(open(DUMP))
bank = load_mbpp(split="test", max_problems=60)
by_id = {p.id: p for p in bank.problems}

name = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("mps").eval()
student = StudentSimulator(
    model, tok,
    StudentPersona(ability_level=2, code_mixing_ratio=0.3, language="hinglish", persistence=0.7),
    max_new_tokens=512,
)
verifier = CodeVerifier(timeout=10, memory_mb=256)


def replay(turns: list[dict], problem, mode: str) -> int:
    d = Dialogue(problem_id=problem.id)
    for t in turns:
        d.add(t["role"], t["content"])
    os.environ["SAHAI_SOLVE_CONTEXT"] = mode
    code = student.attempt_solution(d, problem)
    return int(verifier.verify(code, problem) >= 1.0)


results: dict[str, list[dict]] = {}
t0 = time.time()
for arm in ("base", "trained"):
    rows = saved["arms"][arm]["dialogues"][:LIMIT]
    out = []
    for i, r in enumerate(rows, 1):
        p = by_id.get(r["problem_id"])
        if p is None:
            continue
        full = replay(r["turns"], p, "full")
        hints = replay(r["turns"], p, "hints")
        out.append({
            "problem_id": r["problem_id"],
            "saved_full": r["solved"],
            "replay_full": full,
            "replay_hints": hints,
            "n_turns": len(r["turns"]),
        })
        mark = ""
        if full != r["solved"]:
            mark += "  REPLAY DISAGREES WITH DUMP"
        if hints != full:
            mark += "  FRAMING FLIPS IT"
        print(f"[{arm:7s} {i:3d}/{len(rows)}] {r['problem_id']:12s} "
              f"dump={r['solved']} full={full} hints={hints}{mark}", flush=True)
    results[arm] = out
    json.dump(results, open(OUT, "w"), indent=1)


def mcnemar(pairs: list[tuple[int, int]]) -> tuple[int, int, float]:
    """Exact two-sided McNemar on the discordant pairs."""
    from math import comb
    b = sum(1 for a, c in pairs if a == 1 and c == 0)
    c_ = sum(1 for a, c in pairs if a == 0 and c == 1)
    n = b + c_
    if n == 0:
        return b, c_, 1.0
    k = min(b, c_)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return b, c_, min(1.0, 2 * tail)


print("\n" + "=" * 72)
print(f"SOLVE-CONTEXT FRAMING, REPLAYED ON FIXED DIALOGUES   {(time.time()-t0)/60:.1f} min")
print("=" * 72)
for arm, rows in results.items():
    n = len(rows)
    sf = sum(r["saved_full"] for r in rows)
    rf = sum(r["replay_full"] for r in rows)
    rh = sum(r["replay_hints"] for r in rows)
    agree = sum(r["saved_full"] == r["replay_full"] for r in rows)
    b, c_, p = mcnemar([(r["replay_full"], r["replay_hints"]) for r in rows])
    print(f"\n{arm}  n={n}")
    print(f"  solved, dump (full)      {sf}/{n} = {sf/n:.3f}")
    print(f"  solved, replay full      {rf}/{n} = {rf/n:.3f}   "
          f"(reproduces the dump on {agree}/{n})")
    print(f"  solved, replay hints     {rh}/{n} = {rh/n:.3f}")
    print(f"  full->hints discordant   {b} lost, {c_} gained, McNemar exact p={p:.4f}")
print(f"\nwritten to {OUT}")
