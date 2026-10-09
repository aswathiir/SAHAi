"""Rejection-sampling fine-tuning on the dialogues that already worked.

Finding #29 measured `lora_B` at 0.000762 mean absolute weight against an
initialisation of exactly zero: across eight epochs and 128 optimiser steps the
policy moved about 0.2%, and KL to the reference never left noise. The GRPO
surrogate is the reason. Its advantage is a within-group z-score, so advantages
sum to zero and the measured policy loss ran 0.0008 to 0.0042. There is almost
nothing there to differentiate.

RAFT (Xiong et al., arXiv:2504.11343) trains only on positively rewarded
samples and is reported competitive with GRPO and PPO, with the pointed finding
that GRPO's advantage comes from discarding prompts whose responses are all
wrong rather than from reward normalisation. For us the decisive property is
different: an ordinary cross-entropy objective has loss of order 1, not 0.002,
so the policy will actually move.

The data already exists. Of 256 saved rollouts, 67 ended with the student
solving the problem, and 45 of those also had zero leakage and full pedagogy
compliance. Those 45 are dialogues where the tutor taught without disclosing
and the student then solved. No new generation is required.

Two things are reported honestly rather than assumed:

* `lora_B` is measured before and after. If it does not move under
  cross-entropy either, the problem is not the objective.
* 45 dialogues cover 15 distinct problems. That is a severe overfitting risk
  and the probe is the only thing that can speak to it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, ".")
sys.path.insert(0, "libs/sahai-core")


def adapter_movement(model) -> dict:
    """Mean absolute weight of each LoRA factor.

    `lora_B` initialises to exactly zero, so its magnitude is the whole of what
    training changed. `lora_A` is a random draw and serves only as a scale.
    """
    import torch

    tot = {"A": 0.0, "B": 0.0}
    cnt = {"A": 0, "B": 0}
    mx = 0.0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_A" in name:
                tot["A"] += param.abs().sum().item(); cnt["A"] += param.numel()
            elif "lora_B" in name:
                tot["B"] += param.abs().sum().item(); cnt["B"] += param.numel()
                mx = max(mx, param.abs().max().item())
    return {
        "lora_A_mean_abs": tot["A"] / max(cnt["A"], 1),
        "lora_B_mean_abs": tot["B"] / max(cnt["B"], 1),
        "lora_B_max_abs": mx,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="artifacts/retrain/rollouts")
    ap.add_argument("--out", default="artifacts/raft")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument(
        "--require-clean", action="store_true", default=True,
        help="Keep only solved dialogues that also had zero leakage and full "
             "pedagogy. Training on a dialogue that solved BECAUSE the tutor "
             "disclosed the answer would teach exactly the behaviour the "
             "reward exists to suppress.",
    )
    args = ap.parse_args()

    os.environ.setdefault("SAHAI_SOLVE_CONTEXT", "hints")

    import torch

    from sahai.agents.student import StudentPersona, StudentSimulator
    from sahai.agents.tutor import TutorPolicy
    from sahai.core.dataset import load_mbpp
    from sahai.core.dialogue import Dialogue, DialogueEngine
    from sahai.core.models import load_for_inference, load_for_training
    from sahai.reward.solve import CodeVerifier, SolveReward
    from sahai.settings import Settings
    from sahai.training.grpo import seed_everything

    settings = Settings.local_mps()
    seed_everything(settings.seed)

    # ---- select the training set ----------------------------------------
    rows = [r for f in sorted(glob.glob(f"{args.rollouts}/epoch-*.json"))
            for r in json.load(open(f))]
    kept = [r for r in rows if r["solved"] >= 1.0]
    if args.require_clean:
        kept = [r for r in kept if r["leakage"] == 0.0 and r["r_ped"] >= 1.0]
    probs = {r["problem_id"] for r in kept}
    print(f"rollouts available        : {len(rows)}")
    print(f"kept (solved{' and clean' if args.require_clean else ''}) : {len(kept)}")
    print(f"distinct problems covered : {len(probs)}")
    if not kept:
        raise SystemExit("nothing to train on")

    train_bank = load_mbpp(split="train", max_problems=settings.max_problems)
    by_id = {p.id: p for p in train_bank.problems}
    kept = [r for r in kept if r["problem_id"] in by_id]
    print(f"usable after id match     : {len(kept)}\n", flush=True)

    # ---- model ------------------------------------------------------------
    tutor_model, tutor_tok = load_for_training(settings.model.tutor, settings.model, "mps")
    tutor = TutorPolicy(tutor_model, tutor_tok)
    before = adapter_movement(tutor_model)
    print("adapter before:", {k: round(v, 6) for k, v in before.items()}, flush=True)

    opt = torch.optim.AdamW(
        [p for p in tutor_model.parameters() if p.requires_grad], lr=args.lr
    )

    # ---- supervised pass over the tutor's turns --------------------------
    t0 = time.time()
    history = []
    tutor_model.train()
    for epoch in range(args.epochs):
        import random

        order = list(range(len(kept)))
        random.shuffle(order)
        tot, nb = 0.0, 0
        for step, idx in enumerate(order, 1):
            r = kept[idx]
            problem = by_id[r["problem_id"]]
            d = Dialogue(problem_id=problem.id)
            for t in r["turns"]:
                d.add(t["role"], t["content"])

            # Cross-entropy over the tutor's tokens only, which is exactly the
            # masked quantity GRPO was taking log-ratios of.
            log_probs, mask = tutor.compute_log_probs(d, problem)
            denom = mask.sum().clamp(min=1.0)
            loss = -(log_probs.sum() / denom)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in tutor_model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            opt.zero_grad()
            tot += loss.item(); nb += 1
        mid = adapter_movement(tutor_model)
        history.append({"epoch": epoch, "mean_loss": tot / max(nb, 1), **mid})
        print(f"epoch {epoch}: mean CE loss {tot/max(nb,1):.4f}   "
              f"lora_B mean|w| {mid['lora_B_mean_abs']:.6f}", flush=True)

    after = adapter_movement(tutor_model)
    print(f"\nadapter after : {({k: round(v, 6) for k, v in after.items()})}")
    grew = after["lora_B_mean_abs"] / max(before["lora_B_mean_abs"], 1e-12)
    print(f"lora_B grew by a factor of {grew:.1f} "
          f"(GRPO reached 0.000762 over 128 steps)")

    os.makedirs(args.out, exist_ok=True)
    tutor_model.save_pretrained(f"{args.out}/raft_model")
    tutor_tok.save_pretrained(f"{args.out}/raft_model")

    # ---- probe, same protocol as the trainer -----------------------------
    tutor_model.eval()
    student_model, student_tok = load_for_inference(
        settings.model.student, settings.model.dtype, "mps", quantize_4bit=False
    )
    student = StudentSimulator(student_model, student_tok, StudentPersona())
    engine = DialogueEngine(max_turns=settings.training.max_turns)
    verifier = CodeVerifier(timeout=settings.reward.exec_timeout,
                            memory_mb=settings.reward.exec_memory_mb)
    solve = SolveReward(verifier, settings.reward.num_solve_samples)

    probe = load_mbpp(split=settings.training.probe_split,
                      max_problems=settings.training.probe_problems)
    solved = []
    with torch.no_grad():
        for i, p in enumerate(probe.problems, 1):
            d = engine.run(tutor, student, p)
            out = solve.compute(student, d, p)
            solved.append(1 if out.solved >= 1.0 else 0)
            print(f"[probe {i:3d}/{len(probe.problems)}] {p.id:12s} solved={solved[-1]}",
                  flush=True)

    k, n = sum(solved), len(solved)
    print("\n" + "=" * 64)
    print(f"RAFT   {len(kept)} dialogues, {args.epochs} epochs, "
          f"{(time.time()-t0)/60:.1f} min")
    print("=" * 64)
    print(f"  probe solve, RAFT model : {k}/{n} = {k/n:.3f}")
    print(f"  GRPO probes for the same split: 0.150, 0.250, 0.350, 0.250")
    print(f"  lora_B mean|w|: {before['lora_B_mean_abs']:.6f} -> "
          f"{after['lora_B_mean_abs']:.6f}")
    print("\n  The movement figure is the point. A probe over 20 problems")
    print("  resolves about 0.29 at 80% power, so it cannot settle performance.")

    json.dump({"kept": len(kept), "problems": len(probs), "history": history,
               "before": before, "after": after,
               "probe_solved": solved, "probe_rate": k / n},
              open(f"{args.out}/raft.json", "w"), indent=1)
    print(f"\nwritten to {args.out}/raft.json")


if __name__ == "__main__":
    main()
