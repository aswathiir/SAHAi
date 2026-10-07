"""If the student's own hedging is the mechanism, recovery should track it.

Finding #25 showed that omitting the student's turns from the context of its
final attempt removes a deficit of 0.184. The explanation offered was that a
chat model conditions on its own prior assistant turns and 20% of student turns
express confusion. That is a mechanism, and it carries a prediction the
aggregate result does not test:

    the problems recovered by the framing change should be the ones whose
    dialogues contain the most student hedging.

If recovery is unrelated to how much the student hedged, the mechanism is wrong
even though the effect is real, and `hints` is then just a framing that happens
to score higher. That is the caveat this measures.

Within-problem and paired: each problem contributes its own `full` and `hints`
outcome on the same dialogue, so problem difficulty cannot drive the contrast.
"""
from __future__ import annotations

import json
import sys
from math import comb

sys.path.insert(0, "libs/sahai-core")
from sahai_core.turn_signals import STUCK_PHRASES

fr = json.load(open("artifacts/phase2/framing.json"))
dump = json.load(open("kaggle_upload/dialogues/ab_benchmark.json"))


def feats(turns):
    stu = [t["content"] for t in turns if t["role"] == "student"]
    stuck = sum(1 for s in stu if any(p in s.lower() for p in STUCK_PHRASES))
    return {
        "stuck_turns": stuck,
        "any_stuck": int(stuck > 0),
        "student_chars": sum(len(s) for s in stu),
        "n_student": len(stu),
    }


def fisher(a, b, c, e):
    n = a + b + c + e
    f = lambda x: comb(a + b, x) * comb(c + e, (a + c) - x) / comb(n, a + c)
    obs = f(a)
    lo, hi = max(0, (a + c) - (c + e)), min(a + b, a + c)
    return sum(f(x) for x in range(lo, hi + 1) if f(x) <= obs + 1e-12)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


rows = []
for arm in ("base", "trained"):
    turns_by = {r["problem_id"]: r["turns"] for r in dump["arms"][arm]["dialogues"]}
    for r in fr[arm]:
        f = feats(turns_by[r["problem_id"]])
        f.update(arm=arm, problem_id=r["problem_id"],
                 full=r["replay_full"], hints=r["replay_hints"])
        f["recovered"] = int(r["replay_full"] == 0 and r["replay_hints"] == 1)
        f["broken"] = int(r["replay_full"] == 1 and r["replay_hints"] == 0)
        rows.append(f)

print("=" * 72)
print("DOSE RESPONSE: does framing recovery track the student's own hedging?")
print("=" * 72)

for arm in ("base", "trained", "both"):
    sub = rows if arm == "both" else [r for r in rows if r["arm"] == arm]
    rec = [r for r in sub if r["recovered"]]
    # The comparison group is problems full also failed and hints did not fix,
    # i.e. the other candidates for recovery. Problems full already solved were
    # never available to be recovered and would bias the contrast.
    cand = [r for r in sub if r["full"] == 0 and not r["recovered"]]
    print(f"\n{arm}  recovered by hints: {len(rec)}   failed under both: {len(cand)}")
    if rec and cand:
        print(f"  {'':18s} {'recovered':>10s} {'still failed':>13s}")
        for k in ("stuck_turns", "any_stuck", "student_chars", "n_student"):
            print(f"    {k:16s} {mean(r[k] for r in rec):10.2f} {mean(r[k] for r in cand):13.2f}")
        a = sum(r["any_stuck"] for r in rec)
        b = len(rec) - a
        c = sum(r["any_stuck"] for r in cand)
        e = len(cand) - c
        print(f"  share whose dialogue contains student 'I am lost':")
        print(f"    recovered    {a}/{len(rec)} = {a/len(rec):.3f}")
        print(f"    still failed {c}/{len(cand)} = {c/len(cand):.3f}")
        print(f"    Fisher exact two-sided p = {fisher(a, b, c, e):.4f}")

print("\n" + "=" * 72)
print("The reverse direction: problems the framing change BROKE")
print("=" * 72)
brk = [r for r in rows if r["broken"]]
keep = [r for r in rows if r["full"] == 1 and not r["broken"]]
print(f"  broken by hints: {len(brk)}   solved under both: {len(keep)}")
if brk and keep:
    for k in ("stuck_turns", "any_stuck", "student_chars"):
        print(f"    {k:16s} broken {mean(r[k] for r in brk):8.2f}   kept {mean(r[k] for r in keep):8.2f}")
print("\nIf recovery tracks hedging and breakage does not, the mechanism holds.")
