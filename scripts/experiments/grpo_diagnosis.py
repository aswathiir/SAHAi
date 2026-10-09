"""Why has twenty runs of this not moved held-out solve rate?

Every previous investigation in this project went after the reward terms, the
prompts, the judge or the measurement. None of them asked whether the training
loop, as configured, can deliver a usable gradient at all. This asks that, from
the rollout dumps, which carry the advantage GRPO actually applied.

Four quantities decide it.

1. Problem coverage. The advantage in GRPO is a within-group z-score, so the
   unit of learning is a problem, not a rollout. 4 problems per epoch over 8
   epochs is at most 32 problem-instances out of a 194-problem bank.

2. Dead groups. A group whose rewards are all equal has sigma = 0, every
   advantage in it is zero, and it contributes nothing while consuming a full
   generation budget. This counts them.

3. Where the surviving variance comes from. If within-group reward variance is
   carried by r_ped and leakage rather than r_sol, the policy is being taught
   to write text that scores well on terms it controls, which is the documented
   failure mode. This decomposes it per group.

4. Curriculum drift. Rollout solve rate falling while a fixed-split probe rises
   is not a contradiction if the curriculum moved to harder problems. This
   measures the drift directly instead of inferring it.
"""
from __future__ import annotations

import ast
import glob
import json
import statistics as st
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "libs/sahai-core")

files = sorted(glob.glob("artifacts/retrain/rollouts/epoch-*.json"),
               key=lambda p: int(p.rsplit("-", 1)[1].split(".")[0]))
epochs = {int(f.rsplit("-", 1)[1].split(".")[0]): json.load(open(f)) for f in files}
allr = [r for rs in epochs.values() for r in rs]

print("=" * 72)
print(f"GRPO DIAGNOSIS   {len(epochs)} epochs, {len(allr)} rollouts")
print("=" * 72)

# ---- 1. coverage -----------------------------------------------------
ids = [r["problem_id"] for r in allr]
uniq = set(ids)
try:
    from sahai.core.dataset import load_mbpp
    bank = len(load_mbpp(split="train", max_problems=200).problems)
except Exception:
    bank = 198
print(f"\n1. PROBLEM COVERAGE")
print(f"   distinct problems seen in training : {len(uniq)}")
print(f"   problems in the bank               : {bank}")
print(f"   coverage                           : {len(uniq)/bank:.1%}")
reps = {p: ids.count(p) // 8 for p in uniq}
multi = {p: n for p, n in reps.items() if n > 1}
print(f"   problems drawn more than once      : {len(multi)}")
print(f"   groups (problem-epoch pairs)       : {len(allr)//8}")

# ---- 2. dead groups --------------------------------------------------
groups: dict[tuple, list] = {}
for r in allr:
    groups.setdefault((r["epoch"], r["problem_id"]), []).append(r)

dead_reward = dead_sol = 0
adv_abs = []
for g in groups.values():
    rw = [x["reward"] for x in g]
    sol = [x["r_sol"] for x in g]
    if max(rw) - min(rw) < 1e-9:
        dead_reward += 1
    if max(sol) - min(sol) < 1e-9:
        dead_sol += 1
    adv_abs.extend(abs(x["advantage"]) for x in g)

ng = len(groups)
print(f"\n2. DEAD GROUPS  (no variance means no gradient)")
print(f"   groups total                       : {ng}")
print(f"   groups with zero REWARD variance   : {dead_reward} = {dead_reward/ng:.1%}")
print(f"   groups with zero r_sol variance    : {dead_sol} = {dead_sol/ng:.1%}")
print(f"   mean |advantage|                   : {st.mean(adv_abs):.4f}")
print(f"   median |advantage|                 : {st.median(adv_abs):.4f}")
zero_adv = sum(1 for a in adv_abs if a < 1e-6)
print(f"   rollouts with |advantage| < 1e-6   : {zero_adv}/{len(adv_abs)} = {zero_adv/len(adv_abs):.1%}")

# ---- 3. where the variance lives -------------------------------------
print(f"\n3. WITHIN-GROUP VARIANCE BY TERM  (only this becomes gradient)")
terms = ("r_sol", "r_ped", "leakage", "correctness")
means = {}
for t in terms:
    vs = []
    for g in groups.values():
        xs = [x[t] for x in g]
        if len(xs) > 1:
            vs.append(st.pvariance(xs))
    means[t] = st.mean(vs) if vs else 0.0
    print(f"   {t:14s} mean within-group variance : {means[t]:.4f}")
tot = sum(means.values()) or 1.0
print("   share of usable variance:")
for t in terms:
    print(f"     {t:14s} {means[t]/tot:>6.1%}")

# ---- 4. curriculum drift --------------------------------------------
print(f"\n4. CURRICULUM DRIFT")
try:
    from sahai.core.dataset import load_mbpp
    by = {p.id: p for p in load_mbpp(split="train", max_problems=200).problems}
except Exception:
    by = {}


def nodes(pid):
    p = by.get(pid)
    if p is None:
        return None
    try:
        return sum(1 for _ in ast.walk(ast.parse(p.solution)))
    except SyntaxError:
        return None


print(f"   {'ep':>3} {'solved':>7} {'r_sol':>7} {'diff':>6} {'nodes':>6} {'probs':>5}")
for ep in sorted(epochs):
    rs = epochs[ep]
    pids = sorted({r["problem_id"] for r in rs})
    ns = [n for n in (nodes(p) for p in pids) if n]
    d = [by[p].difficulty for p in pids if p in by]
    print(f"   {ep:>3} {st.mean([r['solved'] for r in rs]):>7.3f} "
          f"{st.mean([r['r_sol'] for r in rs]):>7.3f} "
          f"{(st.mean(d) if d else float('nan')):>6.2f} "
          f"{(st.mean(ns) if ns else float('nan')):>6.1f} {len(pids):>5}")

first = [r for ep in sorted(epochs)[:2] for r in epochs[ep]]
last = [r for ep in sorted(epochs)[-2:] for r in epochs[ep]]
def md(rs):
    d = [by[r["problem_id"]].difficulty for r in rs if r["problem_id"] in by]
    return st.mean(d) if d else float("nan")
print(f"\n   mean difficulty, first 2 epochs    : {md(first):.2f}")
print(f"   mean difficulty, last 2 epochs     : {md(last):.2f}")
print(f"   strict solved, first 2 / last 2    : "
      f"{st.mean([r['solved'] for r in first]):.3f} / {st.mean([r['solved'] for r in last]):.3f}")

# ---- 5. optimiser budget --------------------------------------------
print(f"\n5. OPTIMISER BUDGET")
steps_per_epoch = 16
print(f"   optimiser steps                    : {steps_per_epoch * len(epochs)}")
print(f"   trainable parameters               : 2,179,072")
print(f"   rollouts generated                 : {len(allr)}")
print(f"   non-dead groups                    : {ng - dead_reward}")
