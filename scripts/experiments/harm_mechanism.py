"""Which problems does tutoring LOSE, and what is in those dialogues?

Tutoring costs the learner. That is established. What is not established is
why, and three candidate mechanisms have been named without being separated:

  1. the tutor's hints are actively misleading
  2. the dialogue is long, so the solve prompt is crowded
  3. the `full` solve framing replays the student's own hedging as assistant
     history, so the model writes code straight after telling itself it is lost

(3) is the only one that implies the deficit is an artefact of the measurement
rather than a property of the tutoring. It is testable on the saved benchmark
dump with no GPU, by conditioning on the problems the student solved *unaided*
and asking what distinguishes the tutored dialogues that lost them.

Pairing on the unaided outcome is what makes this more than a correlation: a
dialogue in which the student sounds confused is probably a harder problem, and
"harder problems are failed more" is not a finding. Restricting to problems the
same student solved alone removes problem difficulty from the comparison.
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "libs/sahai-core")
from sahai_core.turn_signals import STUCK_PHRASES

DUMP = sys.argv[1] if len(sys.argv) > 1 else "kaggle_upload/dialogues/ab_benchmark.json"
d = json.load(open(DUMP))

unaided = {r["problem_id"]: r["solved"] for r in d["arms"]["unaided"]["dialogues"]}


def feats(turns):
    stu = [t["content"] for t in turns if t["role"] == "student"]
    tut = [t["content"] for t in turns if t["role"] == "tutor"]
    stuck = sum(1 for s in stu if any(p in s.lower() for p in STUCK_PHRASES))
    return {
        "n_turns": len(turns),
        "n_student": len(stu),
        "stuck_turns": stuck,
        "any_stuck": int(stuck > 0),
        "student_chars": sum(len(s) for s in stu),
        "tutor_chars": sum(len(s) for s in tut),
    }


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


for arm in ("base", "trained"):
    rows = []
    for r in d["arms"][arm]["dialogues"]:
        f = feats(r["turns"])
        f["problem_id"] = r["problem_id"]
        f["solved"] = r["solved"]
        f["unaided"] = unaided.get(r["problem_id"], 0)
        rows.append(f)

    print("=" * 72)
    print(f"ARM: {arm}    n={len(rows)}")
    print("=" * 72)

    # The population that matters: solvable without help, so any failure here
    # is something the dialogue did.
    lost = [r for r in rows if r["unaided"] == 1 and r["solved"] == 0]
    kept = [r for r in rows if r["unaided"] == 1 and r["solved"] == 1]
    gained = [r for r in rows if r["unaided"] == 0 and r["solved"] == 1]
    print(f"\n  solvable unaided: {len(lost) + len(kept)}")
    print(f"    LOST after tutoring   {len(lost)}")
    print(f"    kept                  {len(kept)}")
    print(f"  not solvable unaided, GAINED after tutoring  {len(gained)}")

    if lost and kept:
        print(f"\n  among problems solvable unaided:")
        print(f"  {'':22s} {'LOST':>8s} {'kept':>8s}")
        for k in ("n_turns", "stuck_turns", "any_stuck", "student_chars", "tutor_chars"):
            print(f"    {k:20s} {mean(r[k] for r in lost):8.2f} {mean(r[k] for r in kept):8.2f}")

    # Does the student having said it was lost track the outcome, within the
    # unaided-solvable set?
    pool = [r for r in rows if r["unaided"] == 1]
    if pool:
        with_s = [r for r in pool if r["any_stuck"]]
        without = [r for r in pool if not r["any_stuck"]]
        print("\n  retention of unaided-solvable problems:")
        for label, grp in (("dialogue has student 'I am lost'", with_s),
                           ("it does not", without)):
            k = sum(r["solved"] for r in grp)
            rate = f"{k / len(grp):.3f}" if grp else "n/a"
            print(f"    {label:38s} {len(grp):3d} problems, kept {k}/{len(grp)} = {rate}")
    print()
