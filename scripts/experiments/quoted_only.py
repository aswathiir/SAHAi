"""Does the role matter, or only the volume?

`hints` removes the student's turns from the context of its final attempt and
recovers a deficit of 0.184. It removes two things at once: the student's
content, and the fact that content sat in the model's own assistant history.

The hedging explanation for that deficit is already dead. Recovery does not
track whether the student said it was lost (Fisher p=0.44 pooled, p=1.00 in the
base arm), while it does track how much the student wrote: 843 characters among
recovered problems against 534 among those that stayed failed.

Two mechanisms are left and they imply different corrections:

  role    the model is derailed by reading its own voice, whatever it says.
          Then `quoted`, which keeps every character but puts the whole
          transcript in one user message, should score like `hints`, and
          `hints` is discarding usable context for no reason.

  volume  the model is derailed by the quantity of text before the question,
          regardless of whose voice it is in. Then `quoted` should score like
          `full`, self-conditioning is not involved, and `hints` is right for a
          reason nobody has yet stated correctly.

Only `quoted` is generated here. `full` and `hints` are read from the completed
three-arm replay, so this costs 60 generations rather than 360.
"""
from __future__ import annotations

import json
import os
import sys
import time
from math import comb

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, ".")
sys.path.insert(0, "libs/sahai-core")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sahai.agents.student import StudentPersona, StudentSimulator
from sahai.core.dataset import load_mbpp
from sahai.core.dialogue import Dialogue
from sahai.reward.solve import CodeVerifier

ARMS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["base"]
OUT = sys.argv[2] if len(sys.argv) > 2 else "artifacts/phase2/quoted.json"
# "discordant" restricts to the problems where `full` and `hints` disagree.
# Those are the only ones that can discriminate: a problem both framings fail,
# or both solve, says nothing about which mechanism is operating, and 89 of the
# 120 are in that category. Running them costs two hours and answers nothing.
SCOPE = sys.argv[3] if len(sys.argv) > 3 else "discordant"

prior = json.load(open("artifacts/phase2/framing.json"))
dump = json.load(open("kaggle_upload/dialogues/ab_benchmark.json"))
by_id = {p.id: p for p in load_mbpp(split="test", max_problems=60).problems}

name = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("mps").eval()
student = StudentSimulator(
    model, tok,
    StudentPersona(ability_level=2, code_mixing_ratio=0.3, language="hinglish", persistence=0.7),
    max_new_tokens=512,
)
verifier = CodeVerifier(timeout=10, memory_mb=256)


def mcnemar(pairs):
    b = sum(1 for a, c in pairs if a == 1 and c == 0)
    c_ = sum(1 for a, c in pairs if a == 0 and c == 1)
    n = b + c_
    if n == 0:
        return b, c_, 1.0
    k = min(b, c_)
    return b, c_, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


results = {}
t0 = time.time()
for arm in ARMS:
    turns_by = {r["problem_id"]: r["turns"] for r in dump["arms"][arm]["dialogues"]}
    prior_by = {r["problem_id"]: r for r in prior[arm]}
    if SCOPE == "discordant":
        prior_by = {
            k: v for k, v in prior_by.items()
            if v["replay_full"] != v["replay_hints"]
        }
    out = []
    for i, pid in enumerate(prior_by, 1):
        p = by_id[pid]
        d = Dialogue(problem_id=pid)
        for t in turns_by[pid]:
            d.add(t["role"], t["content"])
        os.environ["SAHAI_SOLVE_CONTEXT"] = "quoted"
        q = int(verifier.verify(student.attempt_solution(d, p), p) >= 1.0)
        row = dict(prior_by[pid])
        row["replay_quoted"] = q
        out.append(row)
        print(f"[{arm:7s} {i:3d}/{len(prior_by)}] {pid:12s} "
              f"full={row['replay_full']} hints={row['replay_hints']} quoted={q}", flush=True)
    results[arm] = out
    json.dump(results, open(OUT, "w"), indent=1)

print("\n" + "=" * 70)
print(f"ROLE OR VOLUME   {(time.time() - t0) / 60:.1f} min")
print("=" * 70)
for arm, rows in results.items():
    n = len(rows)
    tot = {m: sum(r[f"replay_{m}"] for r in rows) for m in ("full", "hints", "quoted")}
    print(f"\n{arm}  n={n}  (scope={SCOPE})")
    for m in ("full", "hints", "quoted"):
        print(f"  {m:7s} {tot[m]:2d}/{n} = {tot[m]/n:.3f}")
    # On the discordant set `full` and `hints` are by construction opposites,
    # so the only question is which of them `quoted` tracks.
    like_full = sum(1 for r in rows if r["replay_quoted"] == r["replay_full"])
    like_hints = sum(1 for r in rows if r["replay_quoted"] == r["replay_hints"])
    print(f"  quoted agrees with full  on {like_full}/{n}")
    print(f"  quoted agrees with hints on {like_hints}/{n}")
    for x, y in (("full", "quoted"), ("hints", "quoted")):
        b, c_, pv = mcnemar([(r[f"replay_{x}"], r[f"replay_{y}"]) for r in rows])
        print(f"  {x:6s} vs {y:6s}  {b} lost / {c_} gained  p={pv:.4f}")
    print("  reading: quoted close to hints -> role; close to full -> volume")
