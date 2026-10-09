"""What does the UNTRAINED tutor score on the probe set?

The retrain's probe rose 0.150 -> 0.250 -> 0.350 across three measurements and
early stopping never fired. That trend is uninterpretable on its own, for the
reason this project has already paid for once: a solve rate with no control is
a number, not a result. The untrained adapter scores 0.350 under this same
framing on the held-out 60, so two readings fit the same data.

  training worked   the probe climbed above what the untrained policy scores
  training recovered  the probe climbed up to it, having started below

This scores the base checkpoint with the LoRA adapter disabled, on the same
problems, in the same order, under the same framing, with the same strict
all-or-nothing criterion the trainer's own probe uses. `disable_adapter` is the
same mechanism the KL reference policy uses, so "base" here means exactly what
it means inside the training loop.

The BKT tracer matters. The trainer's probe snapshots and restores the learner
model so held-out problems cannot leak into the next epoch's curriculum, and it
runs from whatever mastery state that epoch reached. This runs from a fresh
tracer, which is the epoch-0 state. That is a difference worth naming: the
comparison is against an untrained policy with an untouched learner model, not
against the same policy at the moment each probe fired.
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

from sahai.agents.student import StudentPersona, StudentSimulator
from sahai.agents.tutor import TutorPolicy
from sahai.core.dataset import load_mbpp
from sahai.core.models import load_for_inference, load_for_training
from sahai.reward.solve import CodeVerifier, SolveReward
from sahai.settings import Settings

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
OUT = sys.argv[2] if len(sys.argv) > 2 else "artifacts/retrain/base_probe.json"

settings = Settings.local_mps()
print(f"framing: SAHAI_SOLVE_CONTEXT={os.getenv('SAHAI_SOLVE_CONTEXT', 'full')}")
print(f"probe split: {settings.training.probe_split}, {N} problems\n", flush=True)

bank = load_mbpp(split=settings.training.probe_split, max_problems=N)
problems = bank.problems

tutor_model, tutor_tok = load_for_training(settings.model.tutor, settings.model, "mps")
tutor = TutorPolicy(tutor_model, tutor_tok)
student_model, student_tok = load_for_inference(
    settings.model.student, settings.model.dtype, "mps",
    quantize_4bit=settings.model.student_quantize_4bit,
)
student = StudentSimulator(student_model, student_tok, StudentPersona())

from sahai.core.dialogue import DialogueEngine
from sahai.reward.pedagogy import PedagogyReward

engine = DialogueEngine(max_turns=settings.training.max_turns)
verifier = CodeVerifier(
    timeout=settings.reward.exec_timeout,
    memory_mb=settings.reward.exec_memory_mb,
)
solve_reward = SolveReward(verifier, settings.reward.num_solve_samples)
_ = PedagogyReward(use_rules=True)

rows = []
t0 = time.time()
tutor.model.eval()
# Adapter off: the policy as it was before any gradient step. Same mechanism
# the KL reference uses inside the training loop.
with tutor.model.disable_adapter():
    with torch.no_grad():
        for i, p in enumerate(problems, 1):
            dialogue = engine.run(tutor, student, p)
            outcome = solve_reward.compute(student, dialogue, p)
            solved = 1 if outcome.solved >= 1.0 else 0
            rows.append({
                "problem_id": p.id,
                "solved": solved,
                "partial": float(outcome),
                "n_turns": len(dialogue.turns),
            })
            print(f"[{i:3d}/{len(problems)}] {p.id:12s} solved={solved} "
                  f"partial={float(outcome):.2f}", flush=True)

json.dump(rows, open(OUT, "w"), indent=1)
k = sum(r["solved"] for r in rows)
n = len(rows)
print("\n" + "=" * 64)
print(f"BASE-PROBE CONTROL   n={n}   {(time.time()-t0)/60:.1f} min")
print("=" * 64)
print(f"  untrained policy, strict solve : {k}/{n} = {k/n:.3f}")
print(f"  trained probes were             : 0.150 (ep1), 0.250 (ep3), 0.350 (ep5)")
print()
if n:
    import math
    se = math.sqrt((k / n) * (1 - k / n) / n) if 0 < k < n else 0.0
    print(f"  binomial s.e. at this rate      : {se:.3f}  (so +/-{1.96*se:.3f} at 95%)")
print("  reading: trained probe above this -> training gained;")
print("           at or below              -> the rise was recovery, not gain.")
print(f"\nwritten to {OUT}")
