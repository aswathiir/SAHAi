"""Three-arm paired benchmark: no tutor, base tutor, trained tutor.

Answers the only question that matters about the adapter: does GRPO training
make the tutor better at getting a student to solve a problem than the
untrained model it started from, and is either better than no tutor at all?

Three design choices, each because the project's earlier evaluations could not
support the conclusions drawn from them.

**Paired, not independent.** Every arm sees the same problems in the same order
with the same seed. Problem difficulty is by far the largest source of variance
in this metric -- a 20-problem mean moved +-0.08 purely on which problems were
drawn -- and pairing removes it entirely. The comparison is then McNemar's test
on the problems where two arms disagree, which ignores the problems both get
right and both get wrong, because those carry no information about which arm is
better.

**Ablation, not just a baseline.** "Our model vs no tutor" cannot separate the
value of the training from the value of having any tutor at all. The base arm
is the same checkpoint with the LoRA adapters switched off, so the only
difference between arms 2 and 3 is the 160 optimizer steps.

**Powered, or honest about not being.** The script reports the smallest
difference the chosen n could detect before it runs, and refuses to describe a
null result as "no difference" when the design could not have found one.

Run:
    python -m sahai.eval.ab_benchmark --problems 60 --adapter /path/to/final_model
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# Two-sided alpha and the z for 80% power, used only to report what the run can
# and cannot resolve. Nothing branches on these.
Z_ALPHA = 1.959964
Z_POWER = 0.841621


@dataclass
class ArmResult:
    name: str
    solved: list[int] = field(default_factory=list)      # per problem, 0/1
    partial: list[float] = field(default_factory=list)   # per problem, fraction of tests
    r_ped: list[float] = field(default_factory=list)
    leak: list[float] = field(default_factory=list)
    seconds: float = 0.0
    # The transcript behind each outcome. The first run of this benchmark
    # produced a significant result it could not explain, because only the
    # scores were kept and the dialogues were thrown away; diagnosing it meant
    # borrowing training rollouts from a different problem set.
    dialogues: list[dict] = field(default_factory=list)

    @property
    def solve_rate(self) -> float:
        return sum(self.solved) / max(len(self.solved), 1)

    @property
    def partial_rate(self) -> float:
        return sum(self.partial) / max(len(self.partial), 1)


def mcnemar(a: list[int], b: list[int]) -> dict:
    """Exact McNemar on paired binary outcomes.

    b01 = problems arm A missed and arm B solved; b10 = the reverse. Problems
    both arms got right, or both got wrong, are deliberately ignored: they say
    nothing about which arm is better, and including them is what makes an
    unpaired test on this data so insensitive.
    """
    b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "discordant": 0, "p_value": 1.0,
                "note": "no problem was solved by one arm and missed by the other"}
    # Exact binomial two-sided, p=0.5 under the null.
    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return {"b01": b01, "b10": b10, "discordant": n,
            "p_value": min(1.0, 2 * tail)}


def min_detectable(n: int, p1: float) -> float:
    """Smallest improvement over p1 this n could detect at 80% power."""
    def need(p2: float) -> float:
        pbar = (p1 + p2) / 2
        num = (Z_ALPHA * math.sqrt(2 * pbar * (1 - pbar))
               + Z_POWER * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        return num / max((p2 - p1) ** 2, 1e-12)
    lo, hi = p1 + 1e-4, 0.99
    for _ in range(80):
        mid = (lo + hi) / 2
        if need(mid) <= n:
            hi = mid
        else:
            lo = mid
    return hi - p1


def run(problems, tutor, student, evaluator, pedagogy, arm: str) -> ArmResult:
    """Score one arm over the problem bank.

    `tutor=None` is the unaided arm: the student attempts the problem with an
    empty dialogue, which is the floor any tutor has to beat to justify itself.
    """
    from sahai.core.dialogue import Dialogue, DialogueEngine

    res = ArmResult(name=arm)
    engine = DialogueEngine(max_turns=evaluator.settings.training.max_turns)
    t0 = time.time()

    for i, problem in enumerate(problems.problems, 1):
        if arm == "oracle":
            # A ceiling probe, not a tutor. The transcript hands over the
            # reference solution verbatim, so the student has only to copy it.
            #
            # This exists to separate two explanations of the same measurement.
            # Five runs say tutoring lowers solve rate. That is consistent with
            # "this tutor is bad" and equally with "this student cannot use a
            # dialogue at all", and no amount of reward engineering helps in
            # the second case. If solve here does not approach 1.0, the
            # environment carries no signal for any method to find, because
            # nothing a tutor could possibly say beats being given the answer.
            dialogue = Dialogue(problem_id=problem.id)
            dialogue.add("student", "I do not know how to start this one.")
            dialogue.add(
                "tutor",
                "Here is a correct solution. Study it, then write it out:\n\n"
                f"```python\n{problem.solution.strip()}\n```",
            )
        elif tutor is None:
            # Empty transcript, not a short one: the student sees the problem
            # and nothing else. `problem_id` is required on Dialogue.
            dialogue = Dialogue(problem_id=problem.id)
        else:
            dialogue = engine.run(tutor, student, problem)

        outcome = evaluator.solve_reward.compute(student, dialogue, problem)
        res.solved.append(1 if outcome.solved >= 1.0 else 0)
        res.partial.append(float(outcome.score))
        res.r_ped.append(
            pedagogy.evaluate(dialogue) if tutor is not None else float("nan")
        )
        res.leak.append(
            evaluator.leakage_estimator.estimate(
                dialogue, problem.solution,
                problem_text=f"{problem.title} {problem.description}",
            ) if tutor is not None else 0.0
        )
        res.dialogues.append({
            "problem_id": problem.id,
            "solved": res.solved[-1],
            "partial": res.partial[-1],
            "turns": [
                {"role": t.role, "content": t.content} for t in dialogue.turns
            ],
        })
        logger.info("[%s] %d/%d %s solved=%d partial=%.2f turns=%d",
                    arm, i, len(problems.problems), problem.id,
                    res.solved[-1], res.partial[-1], len(dialogue.turns))

    res.seconds = time.time() - t0
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", type=int, default=60)
    ap.add_argument("--adapter", default=os.getenv("SAHAI_ADAPTER_PATH", ""))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="ab_benchmark.json")
    ap.add_argument("--arms", default="unaided,base,trained")
    ap.add_argument(
        "--device", default="auto",
        help="auto resolves cuda -> mps -> cpu. settings.device says 'cuda' "
             "because that is what Kaggle has; auto lets the same script run "
             "on Apple Silicon without editing settings.",
    )
    ap.add_argument(
        "--solve-context", default=None, choices=["full", "hints"],
        help="How tutoring reaches the solve attempt. Sets SAHAI_SOLVE_CONTEXT.",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.solve_context:
        os.environ["SAHAI_SOLVE_CONTEXT"] = args.solve_context
    logger.info("solve context framing: %s",
                os.getenv("SAHAI_SOLVE_CONTEXT", "full"))

    import torch
    from sahai.settings import Settings
    from sahai.core.dataset import load_mbpp
    from sahai.core.models import load_for_inference, resolve_device
    from sahai.agents.tutor import TutorPolicy
    from sahai.agents.student import StudentSimulator, StudentPersona
    from sahai.reward.pedagogy import PedagogyReward
    from sahai.eval.benchmark import Evaluator

    settings = Settings.kaggle()
    train_bank = load_mbpp(split="train", max_problems=settings.max_problems)
    bank = load_mbpp(split=args.split, max_problems=args.problems)
    n = len(bank.problems)

    print(f"\n{'='*64}\nPAIRED THREE-ARM BENCHMARK  (n={n} problems, same bank every arm)")
    print(f"{'='*64}")
    for p1 in (0.10, 0.15):
        print(f"  at a true rate of {p1:.2f}, n={n} can detect an improvement of "
              f"{min_detectable(n, p1):+.1%} or larger (80% power)")
    print("  smaller real differences will not be distinguishable from noise.\n")

    evaluator = Evaluator(settings, solution_corpus=[p.solution for p in train_bank.problems])
    pedagogy = PedagogyReward(use_rules=True)

    # Inference-only base weights, then the trained adapter on top. Not
    # `load_for_training`: that attaches a *fresh, untrained* LoRA, which would
    # make the "trained" arm silently identical to the base arm and the whole
    # benchmark a comparison of a model with itself.
    device = resolve_device(args.device)
    logger.info("device: %s (requested %s)", device, args.device)
    tutor_model, tutor_tok = load_for_inference(
        settings.model.tutor, settings.model.dtype, device
    )
    if args.adapter:
        from peft import PeftModel
        tutor_model = PeftModel.from_pretrained(tutor_model, args.adapter)
        tutor_model.eval()
        logger.info("loaded adapter from %s", args.adapter)
    elif "trained" in args.arms or "base" in args.arms:
        raise SystemExit("--arms needs a model arm but no --adapter was given")

    student_model, student_tok = load_for_inference(
        settings.model.student, settings.model.dtype, device,
        quantize_4bit=settings.model.student_quantize_4bit,
    )
    persona = StudentPersona(ability_level=2, code_mixing_ratio=0.3,
                             language="hinglish", persistence=0.7)

    results: dict[str, ArmResult] = {}
    wanted = [a.strip() for a in args.arms.split(",") if a.strip()]

    for arm in wanted:
        # A fresh student every arm. Carrying one over would let an earlier
        # arm's BKT updates change the ability estimate the next arm is scored
        # against, which is the one way to make this comparison meaningless.
        student = StudentSimulator(student_model, student_tok, persona,
                                   max_new_tokens=settings.model.student_max_new_tokens)
        student.tracer.reset()
        torch.manual_seed(settings.seed)

        if arm in ("unaided", "oracle"):
            tutor = None
        else:
            tutor = TutorPolicy(tutor_model, tutor_tok,
                                max_new_tokens=settings.model.tutor_max_new_tokens)

        if arm == "base" and args.adapter:
            # Same weights, adapters off: isolates the 160 optimizer steps.
            tutor_model.disable_adapter_layers()
        try:
            results[arm] = run(bank, tutor, student, evaluator, pedagogy, arm)
        finally:
            if arm == "base" and args.adapter:
                tutor_model.enable_adapter_layers()

    print(f"\n{'arm':<10} {'solve':>8} {'partial':>9} {'ped':>7} {'leak':>7} {'min':>7}")
    print("-" * 56)
    for a in wanted:
        r = results[a]
        ped = [x for x in r.r_ped if x == x]
        print(f"{a:<10} {r.solve_rate:>8.3f} {r.partial_rate:>9.3f} "
              f"{(sum(ped)/len(ped) if ped else float('nan')):>7.3f} "
              f"{sum(r.leak)/max(len(r.leak),1):>7.3f} {r.seconds/60:>7.1f}")

    print("\nPAIRED COMPARISONS (McNemar, exact)")
    pairs = [("unaided", "trained"), ("base", "trained"), ("unaided", "base")]
    stats = {}
    for x, y in pairs:
        if x in results and y in results:
            m = mcnemar(results[x].solved, results[y].solved)
            stats[f"{x}_vs_{y}"] = m
            d = results[y].solve_rate - results[x].solve_rate
            verdict = ("significant" if m["p_value"] < 0.05
                       else "NOT resolvable at this n")
            print(f"  {x:>8} -> {y:<8} delta={d:+.3f}  "
                  f"{y} won {m['b01']}, {x} won {m['b10']}, "
                  f"p={m['p_value']:.3f}  [{verdict}]")

    out = {"n": n, "split": args.split, "adapter": args.adapter,
           "device": device,
           "solve_context": os.getenv("SAHAI_SOLVE_CONTEXT", "full"),
           "arms": {k: asdict(v) for k, v in results.items()},
           "mcnemar": stats,
           "min_detectable_at_0.15": min_detectable(n, 0.15)}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
