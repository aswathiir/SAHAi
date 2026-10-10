"""Is SAHAi on a pedagogy/outcome frontier, or is it simply dominated?

Every arm this project has ever scored used one checkpoint,
Qwen2.5-1.5B-Instruct, with the adapter on or off. No other model has been
evaluated as a tutor, so the claim that the system outperforms available models
has no measurements behind it at all.

Solve rate alone is also the wrong axis. MathTutorBench (Macina et al., EMNLP
2025, arXiv:2502.18940) reports that subject expertise does not translate into
good teaching and that pedagogy and expertise form a trade-off navigated by how
specialised for tutoring a model is. Our own oracle arm is the degenerate
corner of exactly that trade-off: disclose the solution and the student reaches
0.783 with leakage 1.0, which is an answer key rather than a tutor.

So this scores each tutor on two axes at once, student outcome and answer
non-disclosure, and reports where each sits. Three outcomes are possible and
only the first is good for the project:

  on the frontier   a stronger prompted model solves higher but leaks much
                    more, so the trained small model holds a defensible point
  dominated         a stronger prompted model solves higher AND leaks less,
                    in which case training bought nothing and we say so
  indistinguishable  everything lands together, and the trade-off is not
                    being navigated by anything we did

Tutors are loaded one at a time and released, so peak memory is one tutor plus
one student rather than all of them. A 7B tutor is not included: 14 GB of
weights beside a student does not fit in 16 GB of unified memory.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, ".")
sys.path.insert(0, "libs/sahai-core")

logger = logging.getLogger("frontier")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", type=int, default=60)
    ap.add_argument("--split", default="test")
    ap.add_argument("--adapter", default="artifacts/retrain/best_model")
    ap.add_argument(
        "--prompted", default="Qwen/Qwen2.5-3B-Instruct",
        help="A stronger existing model, used with no training at all.",
    )
    ap.add_argument("--out", default="artifacts/frontier/frontier.json")
    ap.add_argument("--solve-context", default="hints", choices=["full", "hints", "quoted"])
    ap.add_argument(
        "--device", default="auto",
        help="auto resolves cuda -> mps -> cpu. On a T4 this is cuda and the "
             "arms run at a few seconds per generation; on 16 GB of unified "
             "memory the 3B arm swapped and reached 9 minutes per problem.",
    )
    args = ap.parse_args()

    os.environ["SAHAI_SOLVE_CONTEXT"] = args.solve_context
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import torch

    from sahai.agents.student import StudentPersona, StudentSimulator
    from sahai.agents.tutor import TutorPolicy
    from sahai.core.dataset import load_mbpp
    from sahai.core.models import load_for_inference, resolve_device
    from sahai.eval.ab_benchmark import mcnemar, min_detectable, run
    from sahai.eval.benchmark import Evaluator
    from sahai.reward.pedagogy import PedagogyReward
    from sahai.settings import Settings

    settings = Settings.kaggle()
    device = resolve_device(args.device)
    print(f"  device: {device} (requested {args.device})", flush=True)
    train_bank = load_mbpp(split="train", max_problems=settings.max_problems)
    bank = load_mbpp(split=args.split, max_problems=args.problems)
    n = len(bank.problems)

    print("=" * 70)
    print(f"TUTOR FRONTIER   n={n}   framing={args.solve_context}")
    print("=" * 70)
    print(f"  at a true rate of 0.35, n={n} resolves an improvement of "
          f"{min_detectable(n, 0.35):+.1%} or larger at 80% power\n", flush=True)

    evaluator = Evaluator(settings, solution_corpus=[p.solution for p in train_bank.problems])
    pedagogy = PedagogyReward(use_rules=True)

    student_model, student_tok = load_for_inference(
        settings.model.student, settings.model.dtype, device,
        quantize_4bit=False,
    )
    persona = StudentPersona(ability_level=2, code_mixing_ratio=0.3,
                             language="hinglish", persistence=0.7)

    def fresh_student():
        """A new tracer per arm. Carrying one over would let an earlier arm's
        mastery updates move the ability baseline the next arm is scored on."""
        s = StudentSimulator(student_model, student_tok, persona,
                             max_new_tokens=settings.model.student_max_new_tokens)
        s.tracer.reset()
        torch.manual_seed(settings.seed)
        return s

    results = {}

    def score(arm: str, tutor) -> None:
        t0 = time.time()
        results[arm] = run(bank, tutor, fresh_student(), evaluator, pedagogy, arm)
        print(f"  [{arm}] done in {(time.time()-t0)/60:.1f} min", flush=True)

    # --- arms that need no tutor model -------------------------------------
    print("-- unaided (the floor any tutor must beat) --", flush=True)
    score("unaided", None)
    print("-- oracle (solution disclosed: the degenerate corner) --", flush=True)
    score("oracle", None)

    # --- our trained 1.5B --------------------------------------------------
    print(f"-- sahai-trained ({settings.model.tutor} + adapter) --", flush=True)
    tm, tt = load_for_inference(settings.model.tutor, settings.model.dtype, device)
    from peft import PeftModel

    tm = PeftModel.from_pretrained(tm, args.adapter)
    tm.eval()
    score("sahai-trained", TutorPolicy(tm, tt,
                                       max_new_tokens=settings.model.tutor_max_new_tokens))
    print("-- sahai-base (same weights, adapter off) --", flush=True)
    tm.disable_adapter_layers()
    score("sahai-base", TutorPolicy(tm, tt,
                                    max_new_tokens=settings.model.tutor_max_new_tokens))
    tm.enable_adapter_layers()
    del tm, tt
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()

    # --- a stronger existing model, prompted only --------------------------
    print(f"-- prompted ({args.prompted}, no training) --", flush=True)
    pm, pt = load_for_inference(args.prompted, settings.model.dtype, device)
    pm.eval()
    score("prompted-3b", TutorPolicy(pm, pt,
                                     max_new_tokens=settings.model.tutor_max_new_tokens))
    del pm, pt
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device.startswith("cuda"):
        torch.cuda.empty_cache()

    # --- report ------------------------------------------------------------
    order = ["unaided", "sahai-base", "sahai-trained", "prompted-3b", "oracle"]
    print("\n" + "=" * 70)
    print("THE TWO AXES")
    print("=" * 70)
    print(f"{'arm':<16}{'solve':>8}{'partial':>9}{'leak':>8}{'ped':>8}  {'min':>6}")
    print("-" * 62)
    rows = {}
    for a in order:
        r = results.get(a)
        if r is None:
            continue
        ped = [x for x in r.r_ped if x == x]
        rows[a] = {
            "solve": r.solve_rate,
            "partial": r.partial_rate,
            "leak": sum(r.leak) / max(len(r.leak), 1),
            "ped": (sum(ped) / len(ped)) if ped else None,
            "solved": r.solved,
            "minutes": r.seconds / 60,
        }
        p = rows[a]["ped"]
        print(f"{a:<16}{r.solve_rate:>8.3f}{r.partial_rate:>9.3f}"
              f"{rows[a]['leak']:>8.3f}{(p if p is not None else float('nan')):>8.3f}"
              f"  {r.seconds/60:>6.1f}")

    print("\nPAIRED COMPARISONS (McNemar, exact)")
    pairs = [
        ("unaided", "sahai-trained"),
        ("unaided", "prompted-3b"),
        ("sahai-trained", "prompted-3b"),
        ("sahai-base", "sahai-trained"),
    ]
    stats = {}
    for x, y in pairs:
        if x in results and y in results:
            m = mcnemar(results[x].solved, results[y].solved)
            stats[f"{x}->{y}"] = m
            print(f"  {x:<15} -> {y:<15} {m['b01']}:{m['b10']} discordant  p={m['p_value']:.4f}")

    print("\nREADING")
    t, pr = rows.get("sahai-trained"), rows.get("prompted-3b")
    if t and pr:
        if pr["solve"] > t["solve"] and pr["leak"] > t["leak"]:
            print("  The stronger prompted model solves higher and leaks more.")
            print("  SAHAi holds a point on the frontier: that is a comparative result.")
        elif pr["solve"] >= t["solve"] and pr["leak"] <= t["leak"]:
            print("  The stronger prompted model is at least as good on BOTH axes.")
            print("  SAHAi is dominated. Training bought nothing a prompt did not.")
        else:
            print("  Mixed. Read the two axes directly rather than from a rule.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"n": n, "framing": args.solve_context, "arms": rows, "mcnemar": stats},
              open(args.out, "w"), indent=1, default=float)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
