"""Measure a hint judge before any policy trains against it.

This exists because measuring first has twice now changed the decision. The
technique-vocabulary detector looked reasonable and fired on 10% of dialogues,
which is no gradient. The first judge prompt looked reasonable and returned
MISLEADING for 109 of 109 turns, which is also no gradient. Neither would have
been caught by reading the code.

Three numbers decide whether a judge is worth training against, and they are
not interchangeable:

* **Coverage** -- the share of dialogues where the judge expresses any opinion.
  Below roughly half, most rollouts carry a 0.0 and the term is inert however
  it is weighted.
* **Variance** -- the spread of per-dialogue scores. GRPO's advantage is a
  z-score *within a group*, so a term with no within-group spread contributes
  literally nothing, and a constant has none regardless of its value.
* **Agreement with the outcome** -- whether the judge scores the dialogues that
  helped above the ones that hurt. A judge that supplies gradient without this
  is a target to be optimised rather than a measurement, which is the v16
  failure in a more sophisticated form.

The last is the one previous judges have failed, so it is reported last and
loudest: the LOST/WON contrast, over the problems where tutoring turned a
solved problem unsolved or the reverse.

Usage:
    python -m sahai.eval.judge_validation --judge Qwen/Qwen2.5-7B-Instruct \\
        --dialogues ab_benchmark.json --quantize-4bit
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics as st
import time

logger = logging.getLogger(__name__)


def _corr(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
    return num / den if den else 0.0


def validate(judge, dialogues: dict, problems: dict, arms: list[str]) -> dict:
    """Score every saved dialogue and report the three deciding numbers."""
    from sahai.core.dialogue import Dialogue

    report: dict = {"arms": {}}
    scored: dict[str, dict[str, float]] = {}

    for arm in arms:
        if arm not in dialogues["arms"]:
            continue
        rows = []
        t0 = time.time()
        for d in dialogues["arms"][arm]["dialogues"]:
            problem = problems.get(d["problem_id"])
            if problem is None:
                continue
            dlg = Dialogue(problem_id=d["problem_id"])
            for t in d["turns"]:
                dlg.add(t["role"], t["content"])
            out = judge.evaluate(dlg, problem)
            rows.append({"id": d["problem_id"], "score": out.score,
                         "judged": out.judged, "verdicts": out.verdicts,
                         "solved": d["solved"]})
        scored[arm] = {r["id"]: r["score"] for r in rows}

        verdicts = [v for r in rows for v in r["verdicts"]]
        x = [r["score"] for r in rows]
        y = [r["solved"] for r in rows]
        covered = sum(1 for r in rows if r["judged"] > 0)
        n = max(len(rows), 1)
        a = {
            "n": len(rows),
            "seconds": time.time() - t0,
            "turns_judged": len(verdicts),
            # Counted from what actually came back. A hardcoded label list
            # reported all zeros after the labels were renamed, which hid a
            # judge that had collapsed to a single verdict.
            "verdict_counts": {v: verdicts.count(v) for v in sorted(set(verdicts))},
            "coverage": covered / n,
            "mean_score": st.mean(x) if x else 0.0,
            "variance": st.pvariance(x) if len(x) > 1 else 0.0,
            "corr_with_solved": _corr(x, y),
            "solved_when_positive": st.mean([r["solved"] for r in rows if r["score"] > 0])
            if any(r["score"] > 0 for r in rows) else None,
            "solved_when_negative": st.mean([r["solved"] for r in rows if r["score"] < 0])
            if any(r["score"] < 0 for r in rows) else None,
            "rows": rows,
        }
        report["arms"][arm] = a

        print(f"\n=== {arm} (n={a['n']}, {a['seconds']:.0f}s) ===")
        print(f"  turns judged   : {a['turns_judged']}")
        for v, c in a["verdict_counts"].items():
            print(f"    {v:<11} {c:>4} ({100*c/max(a['turns_judged'],1):>4.0f}%)")
        print(f"  coverage       : {a['coverage']:.0%}   (needs > ~0.5 to be a gradient)")
        print(f"  variance       : {a['variance']:.4f}   (0 = constant = no gradient)")
        print(f"  corr(score,solved): {a['corr_with_solved']:+.3f}")
        if a["solved_when_positive"] is not None:
            print(f"    judge > 0 -> solved {a['solved_when_positive']:.3f}")
        if a["solved_when_negative"] is not None:
            print(f"    judge < 0 -> solved {a['solved_when_negative']:.3f}")

    # The test the 1.5B judge failed: does it see what the outcome sees?
    if "unaided" in dialogues["arms"] and "trained" in scored:
        unaided = {d["problem_id"]: d["solved"]
                   for d in dialogues["arms"]["unaided"]["dialogues"]}
        trained = {d["problem_id"]: d["solved"]
                   for d in dialogues["arms"]["trained"]["dialogues"]}
        lost = [p for p in unaided if unaided[p] == 1 and trained.get(p) == 0]
        won = [p for p in unaided if unaided[p] == 0 and trained.get(p) == 1]
        ls = [scored["trained"][p] for p in lost if p in scored["trained"]]
        ws = [scored["trained"][p] for p in won if p in scored["trained"]]
        report["lost_mean"] = st.mean(ls) if ls else None
        report["won_mean"] = st.mean(ws) if ws else None
        print("\n=== THE DECIDING TEST: does the judge see what the outcome sees? ===")
        if ls:
            print(f"  tutoring LOST the problem (n={len(ls)}): judge {st.mean(ls):+.3f}")
        if ws:
            print(f"  tutoring WON  the problem (n={len(ws)}): judge {st.mean(ws):+.3f}")
        if ls and ws:
            gap = st.mean(ws) - st.mean(ls)
            report["lost_won_gap"] = gap
            print(f"  gap (WON - LOST) = {gap:+.3f}")
            print("  A judge worth training against scores WON above LOST. The 1.5B")
            print("  judge gave +0.500 vs +0.545, i.e. it could not tell them apart.")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dialogues", required=True,
                    help="ab_benchmark.json containing saved transcripts")
    ap.add_argument("--split", default="test")
    ap.add_argument("--problems", type=int, default=60)
    ap.add_argument("--arms", default="base,trained")
    ap.add_argument("--quantize-4bit", action="store_true",
                    help="Required for a 7B judge on a 16 GB T4: bf16 needs "
                         "~15.2 GB and leaves no room, 4-bit needs ~4.6 GB.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="judge_validation.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from sahai.core.dataset import load_mbpp
    from sahai.core.models import load_for_inference, resolve_device
    from sahai.reward.hint_judge import HintJudge

    device = resolve_device(args.device)
    print(f"judge   : {args.judge}")
    print(f"device  : {device}  4-bit: {args.quantize_4bit}")

    model, tok = load_for_inference(args.judge, "bfloat16", device,
                                    quantize_4bit=args.quantize_4bit)
    judge = HintJudge(model, tok)

    bank = load_mbpp(split=args.split, max_problems=args.problems)
    problems = {p.id: p for p in bank.problems}
    dialogues = json.load(open(args.dialogues))

    report = validate(judge, dialogues, problems,
                      [a.strip() for a in args.arms.split(",") if a.strip()])
    report["judge"] = args.judge
    report["quantize_4bit"] = args.quantize_4bit
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
