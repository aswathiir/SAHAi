from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from sahai.core.dialogue import DialogueEngine
from sahai.reward.combined import SAHAIReward
from sahai.reward.leakage import LeakageEstimator
from sahai.reward.solve import CodeVerifier, SolveReward, tutor_code_solves

if TYPE_CHECKING:
    from sahai.agents.student import StudentSimulator
    from sahai.agents.tutor import TutorPolicy
    from sahai.core.data import Problem, ProblemBank
    from sahai.core.dialogue import Dialogue
    from sahai.reward.pedagogy import PedagogyReward
    from sahai.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class Rollout:
    problem: Problem
    dialogue: Dialogue
    r_sol: float = 0.0
    r_ped: float = 0.0
    leakage: float = 0.0
    reward: float = 0.0
    advantage: float = 0.0


@dataclass
class TrainMetrics:
    epoch: int
    step: int
    policy_loss: float
    kl_loss: float
    mean_reward: float
    mean_solve_rate: float
    mean_ped_rate: float
    mean_leakage: float


class GRPOTrainer:
    def __init__(
        self,
        settings: Settings,
        tutor: TutorPolicy,
        student: StudentSimulator,
        pedagogy_reward: PedagogyReward,
        problem_bank: ProblemBank,
    ):
        self.settings = settings
        self.tutor = tutor
        self.student = student
        self.pedagogy_reward = pedagogy_reward
        self.problem_bank = problem_bank

        self.dialogue_engine = DialogueEngine(max_turns=settings.training.max_turns)
        self.verifier = CodeVerifier(
            timeout=settings.reward.exec_timeout,
            memory_mb=settings.reward.exec_memory_mb,
        )
        self.solve_reward = SolveReward(self.verifier, settings.reward.num_solve_samples)
        # Fitted on the problem bank so "rare" is measured against the whole
        # corpus. The evaluator must be fitted on this SAME corpus or the two
        # score leakage by different rules.
        self.leakage_estimator = LeakageEstimator(
            solution_corpus=[p.solution for p in problem_bank.problems]
        )
        self.sahai_reward = SAHAIReward(settings.reward)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.tutor.model.parameters()),
            lr=settings.training.learning_rate,
        )

    def _ref_log_probs(self, dialogue: Dialogue, problem: Problem) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute reference log-probs by disabling LoRA adapters."""
        self.tutor.model.disable_adapter_layers()
        try:
            with torch.no_grad():
                log_probs, mask = self.tutor.compute_log_probs(dialogue, problem)
        finally:
            self.tutor.model.enable_adapter_layers()
        return log_probs.detach(), mask.detach()

    def _rollout_batch(self, problems: list[Problem]) -> list[Rollout]:
        rollouts = []
        G = self.settings.training.group_size
        self.tutor.model.eval()

        with torch.no_grad():
            for problem in problems:
                for _ in range(G):
                    dialogue = self.dialogue_engine.run(self.tutor, self.student, problem)
                    rollouts.append(Rollout(problem=problem, dialogue=dialogue))

        return rollouts

    def _compute_rewards(self, rollouts: list[Rollout]) -> None:
        ability = self.student.tracer.get_ability()

        for rollout in rollouts:
            rollout.r_sol = self.solve_reward.compute(
                self.student, rollout.dialogue, rollout.problem
            )
            rollout.r_ped = self.pedagogy_reward.evaluate(rollout.dialogue)
            rollout.leakage = self.leakage_estimator.estimate(
                rollout.dialogue,
                rollout.problem.solution,
                problem_text=f"{rollout.problem.title} {rollout.problem.description}",
                tutor_code_solves=self._tutor_code_solves(rollout),
            )
            components = self.sahai_reward.compute(
                rollout.r_sol, rollout.r_ped, rollout.leakage, ability
            )
            rollout.reward = components.r_sahai

    def _tutor_code_solves(self, rollout: Rollout) -> bool:
        """Shared with the evaluator — see sahai.reward.solve.tutor_code_solves.

        This used to be implemented here only, and `Evaluator` simply never
        called it, so held-out leakage was scored by a different rule than
        training leakage for every run to date.
        """
        return tutor_code_solves(
            rollout.dialogue, rollout.problem, self.verifier, self.leakage_estimator
        )

    def _compute_advantages(self, rollouts: list[Rollout]) -> None:
        G = self.settings.training.group_size
        for i in range(0, len(rollouts), G):
            group = rollouts[i : i + G]
            rewards = [r.reward for r in group]
            mean_r = sum(rewards) / len(rewards)
            var = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
            std_r = var ** 0.5
            for rollout in group:
                rollout.advantage = (rollout.reward - mean_r) / (std_r + 1e-8)

    def _policy_update(self, rollouts: list[Rollout]) -> tuple[float, float]:
        self.tutor.model.train()
        total_policy_loss = 0.0
        total_kl_loss = 0.0
        n = len(rollouts)

        self.optimizer.zero_grad()
        accum_steps = self.settings.training.gradient_accumulation_steps
        effective_n = 0

        for idx, rollout in enumerate(rollouts):
            ref_log_probs, ref_mask = self._ref_log_probs(rollout.dialogue, rollout.problem)

            self.tutor.model.train()
            log_probs, mask = self.tutor.compute_log_probs(rollout.dialogue, rollout.problem)

            advantage = torch.tensor(
                rollout.advantage, device=log_probs.device, dtype=log_probs.dtype
            )

            token_count = mask.sum().clamp(min=1)
            mean_lp = (log_probs * mask).sum() / token_count
            policy_loss = -advantage * mean_lp

            kl = ((log_probs - ref_log_probs) * mask).sum() / token_count
            kl_loss = self.settings.training.kl_coeff * kl

            loss = (policy_loss + kl_loss) / accum_steps
            loss.backward()

            total_policy_loss += policy_loss.item()
            total_kl_loss += kl_loss.item()
            effective_n += 1

            if (idx + 1) % accum_steps == 0 or idx == n - 1:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.tutor.model.parameters()),
                    self.settings.training.max_grad_norm,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

        return total_policy_loss / max(effective_n, 1), total_kl_loss / max(effective_n, 1)

    def train(self) -> list[TrainMetrics]:
        all_metrics = []
        output_dir = Path(self.settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        step = 0

        for epoch in range(self.settings.training.epochs):
            batch_size = self.settings.training.batch_size
            in_band = len(self.problem_bank.zpd_candidates(self.student.tracer))
            problems = self.problem_bank.zpd_sample(self.student.tracer, batch_size)

            # A thin band used to shrink the batch instead of being reported:
            # epoch 5 of three separate runs trained on 2 problems rather than
            # 4 and said so only as a number in a routine log line. The batch
            # is topped up now, so the band size has to be stated explicitly or
            # the fact that the epoch was padded disappears entirely.
            if in_band < batch_size:
                logger.warning(
                    f"Only {in_band} problems in the ZPD band (need {batch_size}); "
                    f"topped up with the nearest {batch_size - in_band} outside it"
                )

            logger.info(f"Epoch {epoch}: rollout on {len(problems)} problems x G={self.settings.training.group_size}")
            rollouts = self._rollout_batch(problems)

            logger.info("Computing rewards...")
            self._compute_rewards(rollouts)
            self._compute_advantages(rollouts)
            self._log_group_spread(rollouts, self.settings.training.group_size)
            self._dump_rollouts(output_dir, epoch, rollouts)

            logger.info("Policy update...")
            policy_loss, kl_loss = self._policy_update(rollouts)

            metrics = TrainMetrics(
                epoch=epoch,
                step=step,
                policy_loss=policy_loss,
                kl_loss=kl_loss,
                mean_reward=sum(r.reward for r in rollouts) / len(rollouts),
                mean_solve_rate=sum(r.r_sol for r in rollouts) / len(rollouts),
                mean_ped_rate=sum(r.r_ped for r in rollouts) / len(rollouts),
                mean_leakage=sum(r.leakage for r in rollouts) / len(rollouts),
            )
            all_metrics.append(metrics)
            self._write_metrics(output_dir, all_metrics)

            logger.info(
                f"Epoch {epoch} | loss={policy_loss:.4f} kl={kl_loss:.4f} "
                f"reward={metrics.mean_reward:.4f} solve={metrics.mean_solve_rate:.4f} "
                f"ped={metrics.mean_ped_rate:.4f} leak={metrics.mean_leakage:.4f}"
            )

            for rollout in rollouts:
                solved = rollout.r_sol > 0.5
                self.student.tracer.update_batch(rollout.problem.skills, solved)

            step += 1
            if step % self.settings.training.checkpoint_every == 0:
                self._checkpoint(output_dir, epoch, step)

        self._checkpoint(output_dir, epoch, step)
        return all_metrics

    def _write_metrics(self, output_dir: Path, metrics: list[TrainMetrics]) -> None:
        """Rewrite metrics.json after every epoch.

        The notebook also writes this at the end, but only if training returns.
        A multi-hour run that is killed at hour 8 would otherwise leave no
        metrics at all — rewriting each epoch means whatever finished is kept.
        """
        payload = [
            {
                "epoch": m.epoch,
                "step": m.step,
                "policy_loss": m.policy_loss,
                "kl_loss": m.kl_loss,
                "mean_reward": m.mean_reward,
                "mean_solve_rate": m.mean_solve_rate,
                "mean_ped_rate": m.mean_ped_rate,
                "mean_leakage": m.mean_leakage,
            }
            for m in metrics
        ]
        (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2))

    def _dump_rollouts(self, output_dir: Path, epoch: int, rollouts: list[Rollout]) -> None:
        """Write every dialogue with its reward breakdown.

        Without this the only way to inspect a run is to paste a transcript by
        hand, which makes it impossible to tell a real behaviour change from a
        sampling artifact.
        """
        path = output_dir / "rollouts"
        path.mkdir(parents=True, exist_ok=True)

        records = []
        for i, r in enumerate(rollouts):
            records.append(
                {
                    "epoch": epoch,
                    "rollout": i,
                    "problem_id": r.problem.id,
                    "problem_title": r.problem.title,
                    "num_turns": len(r.dialogue),
                    "r_sol": r.r_sol,
                    "r_ped": r.r_ped,
                    "leakage": r.leakage,
                    "reward": r.reward,
                    "advantage": r.advantage,
                    "turns": [
                        {"role": t.role, "content": t.content, "complete": t.complete}
                        for t in r.dialogue.turns
                    ],
                }
            )

        out = path / f"epoch-{epoch}.json"
        out.write_text(json.dumps(records, indent=2))
        logger.info(f"Rollout transcripts saved: {out}")

    @staticmethod
    def _log_group_spread(rollouts: list[Rollout], group_size: int) -> None:
        """Warn when a group has zero reward variance.

        Identical rewards inside a group drive every advantage to 0, so the
        epoch contributes no gradient regardless of what the dialogues looked
        like. This is silent in the loss value alone.
        """
        degenerate = 0
        total = 0
        for i in range(0, len(rollouts), group_size):
            group = rollouts[i : i + group_size]
            if not group:
                continue
            total += 1
            rewards = [r.reward for r in group]
            if max(rewards) - min(rewards) < 1e-8:
                degenerate += 1

        if degenerate:
            logger.warning(
                f"{degenerate}/{total} groups had zero reward spread "
                f"(advantages are 0 — those groups contribute no gradient)"
            )

    def _checkpoint(self, output_dir: Path, epoch: int, step: int) -> None:
        path = output_dir / f"checkpoint-e{epoch}-s{step}"
        path.mkdir(parents=True, exist_ok=True)
        self.tutor.model.save_pretrained(path)
        self.tutor.tokenizer.save_pretrained(path)
        logger.info(f"Checkpoint saved: {path}")
