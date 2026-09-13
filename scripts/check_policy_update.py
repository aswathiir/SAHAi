"""Exercise `GRPOTrainer._policy_update` on a tiny model before spending a GPU.

    KMP_DUPLICATE_LIB_OK=TRUE python3 scripts/check_policy_update.py

Kaggle quota is the binding constraint on this project, and a shape or sign
error in the surrogate costs a nine-hour run to discover. This runs the real
code path — `_old_log_probs`, `_ref_log_probs`, the clipped objective, the
backward and the optimizer step — against a randomly initialised two-layer
model, so it is a correctness check rather than a training run.

It deliberately lives in `scripts/` and not `tests/`: `pytest tests/` is
torch-free by design (see docs/07-operations-kaggle.md) and importing torch
under this Mac's duplicated libomp aborts the process rather than raising.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from sahai.agents.tutor import TutorPolicy
from sahai.core.data import Problem, ProblemBank, TestCase
from sahai.core.dialogue import Dialogue
from sahai.settings import Settings
from sahai.training.grpo import GRPOTrainer, Rollout

BASE = "Qwen/Qwen2.5-0.5B-Instruct"   # tokenizer only; weights are random
CLIP_EPS = 0.2


def _tiny_tutor():
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    cfg = AutoConfig.from_pretrained(BASE, local_files_only=True)
    cfg.num_hidden_layers = 2
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    model = AutoModelForCausalLM.from_config(cfg)
    # Dropout off: the ratio must measure policy movement, and rollouts are
    # generated under `model.eval()` anyway.
    model = get_peft_model(
        model,
        LoraConfig(
            r=4, lora_alpha=8, lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
        ),
    )
    return TutorPolicy(model, tok, max_new_tokens=16)


def _fixture():
    problem = Problem(
        id="p1", title="Sum a list", description="Return the sum.", difficulty=1,
        skills=["arrays"], function_name="s", function_signature="def s(xs):",
        test_cases=[TestCase(input={"xs": [1, 2]}, expected=3)],
        solution="def s(xs):\n    return sum(xs)",
    )
    dialogue = Dialogue(problem_id="p1")
    dialogue.add("student", "Mujhe samajh nahi aa raha.")
    dialogue.add("tutor", "What happens if you look at one element at a time?")
    dialogue.add("student", "I would add them up.")
    dialogue.add("tutor", "Good. What should the running total start at?")
    return problem, dialogue


def _trainer(tutor, problem):
    settings = Settings()
    settings.training.clip_epsilon = CLIP_EPS
    settings.training.gradient_accumulation_steps = 1
    settings.training.learning_rate = 1e-2       # visible movement in one step
    settings.training.kl_coeff = 0.0             # isolate the surrogate
    return GRPOTrainer(
        settings=settings, tutor=tutor, student=None, pedagogy_reward=None,
        problem_bank=ProblemBank(problems=[problem]),
    )


def _mean_log_prob(tutor, dialogue, problem) -> float:
    with torch.no_grad():
        lp, mask = tutor.compute_log_probs(dialogue, problem)
    return ((lp * mask).sum() / mask.sum().clamp(min=1)).item()


def check_surrogate_formula() -> None:
    """The clipped objective, on tensors, independent of any model."""
    adv = torch.tensor(1.0)
    for ratio_value, expected in [
        (1.0, -1.0),                       # unmoved policy: plain -A
        (5.0, -(1.0 + CLIP_EPS)),          # moved up on a good rollout: clipped
        (0.1, -0.1),                       # moved down on a good rollout: NOT clipped
    ]:
        ratio = torch.tensor([[ratio_value]])
        mask = torch.ones_like(ratio)
        loss = -(torch.min(ratio * adv, torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv) * mask).sum() / mask.sum()
        assert abs(loss.item() - expected) < 1e-6, (ratio_value, loss.item(), expected)

    adv = torch.tensor(-1.0)
    ratio = torch.tensor([[5.0]])          # moved up on a bad rollout: NOT clipped,
    mask = torch.ones_like(ratio)          # so the penalty stays large. This asymmetry
    loss = -(torch.min(ratio * adv, torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv) * mask).sum() / mask.sum()
    assert abs(loss.item() - 5.0) < 1e-6, loss.item()
    print("ok   clipped objective: clips a good rollout that ran away, not a bad one")


def check_ratio_is_one_before_any_step(trainer, dialogue, problem) -> None:
    rollouts = [Rollout(problem=problem, dialogue=dialogue, advantage=1.0)]
    old = trainer._old_log_probs(rollouts)[0]
    trainer.tutor.model.train()
    cur, mask = trainer.tutor.compute_log_probs(dialogue, problem)
    ratio = torch.exp(cur.detach() - old)
    off = ((ratio - 1.0).abs() * mask).max().item()
    assert off < 1e-4, f"ratio departs from 1 by {off} before any update"
    print(f"ok   ratio == 1 before the first optimizer step (max dev {off:.2e})")


def check_sign(trainer, dialogue, problem, advantage: float) -> None:
    before = _mean_log_prob(trainer.tutor, dialogue, problem)
    loss, _ = trainer._policy_update(
        [Rollout(problem=problem, dialogue=dialogue, advantage=advantage)]
    )
    after = _mean_log_prob(trainer.tutor, dialogue, problem)
    assert torch.isfinite(torch.tensor(loss)), f"non-finite loss {loss}"
    moved = after - before
    direction = "up" if advantage > 0 else "down"
    ok = moved > 0 if advantage > 0 else moved < 0
    assert ok, f"advantage {advantage:+.1f} moved log-prob the wrong way ({moved:+.5f})"
    print(
        f"ok   advantage {advantage:+.1f} -> tutor log-prob {direction} "
        f"({before:.4f} -> {after:.4f}, loss {loss:+.4f})"
    )


def check_later_rollouts_are_off_policy(trainer, dialogue, problem) -> None:
    """The condition the ratio exists to correct.

    32 rollouts are collected under one policy and 16 sequential optimizer
    steps are taken over them. Only the first is on-policy; this shows the
    departure is real and not a rounding artefact, so the clip has something
    to do.
    """
    rollouts = [
        Rollout(problem=problem, dialogue=dialogue, advantage=a)
        for a in (1.0, -1.0, 1.0, -1.0)
    ]
    old = trainer._old_log_probs(rollouts)[0]
    for _ in range(4):                       # four steps on stale samples
        trainer._policy_update([rollouts[0]])
    trainer.tutor.model.train()
    with torch.no_grad():
        cur, mask = trainer.tutor.compute_log_probs(dialogue, problem)
    ratio = torch.exp(cur - old)
    drift = ((ratio - 1.0).abs() * mask).max().item()
    clipped = int((((ratio > 1 + CLIP_EPS) | (ratio < 1 - CLIP_EPS)).float() * mask).sum())
    assert drift > 1e-3, "policy did not move; this check proves nothing"
    print(
        f"ok   after 4 steps on stale samples max|rho-1| = {drift:.3f}, "
        f"{clipped}/{int(mask.sum())} tokens clipped "
        f"(lr here is 1e-2, not the run's 1e-4 — the magnitude is exaggerated "
        f"on purpose, the departure is the point)"
    )


def check_dropout_is_off(trainer) -> None:
    """With dropout on, the two sides of the ratio are different functions."""
    active = [
        m for m in trainer.tutor.model.modules()
        if isinstance(m, torch.nn.Dropout) and m.p > 0
    ]
    assert not active, f"{len(active)} dropout layers with p>0 would add noise to rho"
    print("ok   no active dropout: rho measures policy movement, not sampling")


def main() -> int:
    torch.manual_seed(0)
    problem, dialogue = _fixture()

    check_surrogate_formula()

    tutor = _tiny_tutor()
    trainer = _trainer(tutor, problem)
    check_ratio_is_one_before_any_step(trainer, dialogue, problem)
    check_sign(trainer, dialogue, problem, advantage=+1.0)

    tutor = _tiny_tutor()
    trainer = _trainer(tutor, problem)
    check_sign(trainer, dialogue, problem, advantage=-1.0)

    tutor = _tiny_tutor()
    trainer = _trainer(tutor, problem)
    check_dropout_is_off(trainer)
    check_later_rollouts_are_off_policy(trainer, dialogue, problem)

    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
