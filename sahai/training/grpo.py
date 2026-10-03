from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from sahai.core.dialogue import DialogueEngine
from sahai.training.early_stop import EarlyStopper
from sahai.reward.combined import SAHAIReward
from sahai.reward.correctness import evaluate as evaluate_correctness
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
    # All-or-nothing pass rate. `r_sol` carries partial credit now, so this is
    # the only field comparable to the solve rates of earlier runs.
    solved: float = 0.0
    r_ped: float = 0.0
    leakage: float = 0.0
    # Agreement between the tutor's technical claims and the reference
    # solution. Recorded even when mu_correct is 0 so a run reports this
    # term's coverage instead of leaving it to be guessed at.
    correctness: float = 0.0
    correctness_claims: int = 0
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
    # Fraction of attempts passing every test. `mean_solve_rate` is partial
    # credit now; this is the series that lines up with runs before that.
    mean_solved: float = 0.0
    mean_correctness: float = 0.0
    # Share of rollouts in which the tutor made any checkable claim. If this
    # stays near 0.10 the correctness term is inert whatever its weight.
    correctness_coverage: float = 0.0
    # Strict solve rate on the held-out probe split, present only on the epochs
    # where the probe ran. This, not `policy_loss`, is the stopping signal.
    probe_solve: float | None = None


class GRPOTrainer:
    def __init__(
        self,
        settings: Settings,
        tutor: TutorPolicy,
        student: StudentSimulator,
        pedagogy_reward: PedagogyReward,
        problem_bank: ProblemBank,
        hint_judge=None,
    ):
        self.settings = settings
        self.tutor = tutor
        self.student = student
        self.pedagogy_reward = pedagogy_reward
        self.problem_bank = problem_bank
        # Optional. When present it replaces the technique-vocabulary detector
        # for the correctness term: both return a score in [-1, 1] with the
        # same meaning, but the vocabulary one fires on 10% of dialogues and a
        # 7B judge on 38-72%, with a WON-minus-LOST gap of +0.409 against the
        # vocabulary detector's untestable sparsity. See docs/04-findings.md.
        self.hint_judge = hint_judge

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
            outcome = self.solve_reward.compute(
                self.student, rollout.dialogue, rollout.problem
            )
            rollout.r_sol = outcome.score
            rollout.solved = outcome.solved
            rollout.r_ped = self.pedagogy_reward.evaluate(rollout.dialogue)
            rollout.leakage = self.leakage_estimator.estimate(
                rollout.dialogue,
                rollout.problem.solution,
                problem_text=f"{rollout.problem.title} {rollout.problem.description}",
                tutor_code_solves=self._tutor_code_solves(rollout),
            )
            if self.hint_judge is not None and self.hint_judge.available:
                judged = self.hint_judge.evaluate(rollout.dialogue, rollout.problem)
                rollout.correctness = judged.score
                rollout.correctness_claims = judged.judged
            else:
                outcome_c = evaluate_correctness(rollout.dialogue, rollout.problem)
                rollout.correctness = outcome_c.score
                rollout.correctness_claims = outcome_c.claims
            components = self.sahai_reward.compute(
                rollout.r_sol, rollout.r_ped, rollout.leakage, ability,
                correctness=rollout.correctness,
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

    def _old_log_probs(self, rollouts: list[Rollout]) -> list[torch.Tensor]:
        """Log-probs under the policy that generated these rollouts.

        Must be computed for *every* rollout before the first optimizer step,
        because after that step the policy is no longer the one that sampled
        them. That is the whole point of the ratio below.
        """
        self.tutor.model.eval()
        cached = []
        with torch.no_grad():
            for rollout in rollouts:
                log_probs, _ = self.tutor.compute_log_probs(
                    rollout.dialogue, rollout.problem
                )
                cached.append(log_probs.detach())
        return cached

    def _policy_update(self, rollouts: list[Rollout]) -> tuple[float, float]:
        """One GRPO update over a batch of rollouts.

        This was plain REINFORCE — `-advantage * mean_log_prob`, with no
        importance ratio and no clipping, while `clip_epsilon` sat in
        `settings.py` referenced nowhere in the codebase. On its own that is a
        defensible estimator only when a single gradient step is taken per
        batch of samples. It was not: 32 rollouts are collected under one
        policy and then 16 sequential optimizer steps are taken over them
        (`gradient_accumulation_steps=2`), so steps 2..16 pushed on samples
        from a policy that no longer existed, uncorrected. Every run shows the
        symptom — KL to the reference climbing monotonically (0.002 -> 0.026)
        while held-out performance does not move.

        The surrogate is now the standard clipped one, per token:

            min(rho_t * A, clip(rho_t, 1-eps, 1+eps) * A)

        where `rho_t = pi_theta(t) / pi_theta_old(t)`. Where the policy has
        already moved far on a token, the clip flattens the objective and that
        token stops contributing gradient, which is exactly the protection the
        sequential steps needed.

        Note the reported `policy_loss` changes meaning: it was O(advantage x
        mean log-prob), a number around 0.01-0.2, and is now O(advantage) with
        rho near 1. The *gradient* scale is unchanged at rho ~= 1, so the
        learning rate does not need retuning — but the loss column is not
        comparable to earlier runs.
        """
        self.tutor.model.train()
        total_policy_loss = 0.0
        total_kl_loss = 0.0
        n = len(rollouts)

        old_log_probs = self._old_log_probs(rollouts)

        self.optimizer.zero_grad()
        accum_steps = self.settings.training.gradient_accumulation_steps
        clip_eps = self.settings.training.clip_epsilon
        effective_n = 0

        for idx, rollout in enumerate(rollouts):
            ref_log_probs, ref_mask = self._ref_log_probs(rollout.dialogue, rollout.problem)

            self.tutor.model.train()
            log_probs, mask = self.tutor.compute_log_probs(rollout.dialogue, rollout.problem)

            advantage = torch.tensor(
                rollout.advantage, device=log_probs.device, dtype=log_probs.dtype
            )

            token_count = mask.sum().clamp(min=1)

            # Both tensors are already masked, so a padded position gives
            # 0 - 0 = 0 and a ratio of exactly 1; `mask` below drops it anyway.
            ratio = torch.exp(log_probs - old_log_probs[idx])
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
            policy_loss = -(torch.min(unclipped, clipped) * mask).sum() / token_count

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
        epoch = -1
        cfg = self.settings.training
        stopper = EarlyStopper(
            patience=cfg.early_stop_patience, min_delta=cfg.early_stop_min_delta
        )

        for epoch in range(self.settings.training.epochs):
            batch_size = self.settings.training.batch_size
            in_band = len(self.problem_bank.zpd_candidates(self.student.tracer))
            problems = self.problem_bank.zpd_sample(self.student.tracer, batch_size)

            # Selection is a ranking on difficulty now, so the batch is always
            # full and this number is pure diagnostics: it says how much of the
            # bank actually sits at the learner's level. Under the old
            # mastery-only band it read 0 for seven of ten epochs and the
            # curriculum was decided by the top-up ordering instead.
            ability = self.student.tracer.get_ability()
            target = self.student.tracer.target_difficulty(
                self.problem_bank.max_difficulty()
            )
            logger.info(
                f"Epoch {epoch}: ability={ability:.3f} target_difficulty={target:.2f} "
                f"({in_band} of {len(self.problem_bank.problems)} problems in band)"
            )
            if in_band < batch_size:
                logger.warning(
                    f"Only {in_band} problems within one difficulty level of the "
                    f"target (need {batch_size}); drawing the nearest available"
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
                mean_correctness=sum(r.correctness for r in rollouts) / len(rollouts),
                correctness_coverage=sum(
                    1 for r in rollouts if r.correctness_claims > 0
                ) / len(rollouts),
                mean_solved=sum(r.solved for r in rollouts) / len(rollouts),
            )
            all_metrics.append(metrics)
            self._write_metrics(output_dir, all_metrics)

            logger.info(
                f"Epoch {epoch} | loss={policy_loss:.4f} kl={kl_loss:.4f} "
                f"reward={metrics.mean_reward:.4f} solve={metrics.mean_solve_rate:.4f} "
                f"solved={metrics.mean_solved:.4f} "
                f"ped={metrics.mean_ped_rate:.4f} leak={metrics.mean_leakage:.4f}"
            )

            for rollout in rollouts:
                # BKT observes "did they get it right", which is the
                # all-or-nothing reading. Thresholding partial credit at 0.5
                # would count an attempt passing 3 of 5 tests as mastery
                # evidence and drift the ability estimate upward.
                self.student.tracer.update_batch(
                    rollout.problem.skills, rollout.solved > 0.5
                )

            step += 1
            if step % self.settings.training.checkpoint_every == 0:
                self._checkpoint(output_dir, epoch, step)

            # Probe, select, and possibly stop. Deliberately after the BKT
            # update above so the probe sees the same learner state the next
            # epoch's curriculum will.
            if cfg.eval_every and (epoch + 1) % cfg.eval_every == 0:
                score = self._probe()
                if score is not None:
                    metrics.probe_solve = score
                    self._write_metrics(output_dir, all_metrics)
                    d = stopper.observe(epoch, score)
                    if d.improved:
                        logger.info("Probe solve %.3f is a new best", score)
                        self._save_best(output_dir, epoch, score)
                    else:
                        logger.info(
                            "Probe solve %.3f does not improve on %.3f "
                            "(%d/%d probes without improvement)",
                            score, d.best_score, d.stale, cfg.early_stop_patience,
                        )
                    if d.should_stop:
                        logger.info(
                            "Early stop at epoch %d: %d probes without improvement",
                            epoch, d.stale,
                        )
                        break

        self._checkpoint(output_dir, epoch, step)

        # The final-epoch weights are not necessarily the best ones, and
        # nothing used to say so.
        logger.info("Checkpoint selection: %s", stopper.summary(epoch))
        return all_metrics

    def _write_metrics(self, output_dir: Path, metrics: list[TrainMetrics]) -> None:
        """Rewrite metrics.json after every epoch.

        The notebook also writes this at the end, but only if training returns.
        A multi-hour run that is killed at hour 8 would otherwise leave no
        metrics at all — rewriting each epoch means whatever finished is kept.
        """
        # asdict, not a hand-kept list of keys. The previous form named every
        # field explicitly and silently dropped any added later: mean_solved,
        # mean_correctness, correctness_coverage and probe_solve were all
        # computed correctly during the 2026-10-03 run and none reached the
        # file, which made a working experiment look like it had not applied.
        payload = [asdict(m) for m in metrics]
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
                    "solved": r.solved,
                    "r_ped": r.r_ped,
                    "leakage": r.leakage,
                    "correctness": r.correctness,
                    "correctness_claims": r.correctness_claims,
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

    def _probe_bank(self) -> ProblemBank | None:
        """Held-out problems for early stopping, loaded once.

        A different split from the one the final evaluation reports on. Picking
        a checkpoint by its score on the test set turns that score into a
        selection artefact rather than a held-out measurement.
        """
        if getattr(self, "_probe_cache", None) is not None:
            return self._probe_cache
        cfg = self.settings.training
        try:
            from sahai.core.dataset import load_mbpp

            self._probe_cache = load_mbpp(
                split=cfg.probe_split, max_problems=cfg.probe_problems
            )
            logger.info(
                "Probe set: %d problems from the '%s' split",
                len(self._probe_cache.problems), cfg.probe_split,
            )
        except Exception as exc:  # noqa: BLE001 - a probe is not a precondition
            logger.warning(
                "Could not load the '%s' split (%s); early stopping disabled",
                cfg.probe_split, type(exc).__name__,
            )
            self._probe_cache = None
        return self._probe_cache

    def _probe(self) -> float | None:
        """Strict solve rate on held-out problems, for the stopping decision.

        Returns the all-or-nothing rate, not partial credit, because that is
        the series every earlier run is comparable on.

        Two pieces of state have to survive this untouched. The tutor goes back
        to train() afterwards, and the student's BKT skills are snapshotted and
        restored: the probe runs dialogues, and letting those update the
        learner model would feed held-out problems back into the curriculum the
        next epoch samples from.
        """
        bank = self._probe_bank()
        if not bank or not bank.problems:
            return None

        saved_skills = dict(self.student.tracer.skills)
        was_training = self.tutor.model.training
        self.tutor.model.eval()
        solved = 0
        try:
            with torch.no_grad():
                for problem in bank.problems:
                    dialogue = self.dialogue_engine.run(self.tutor, self.student, problem)
                    outcome = self.solve_reward.compute(self.student, dialogue, problem)
                    solved += 1 if outcome.solved >= 1.0 else 0
        finally:
            self.student.tracer.skills = saved_skills
            if was_training:
                self.tutor.model.train()

        return solved / len(bank.problems)

    def _save_best(self, output_dir: Path, epoch: int, score: float) -> None:
        path = output_dir / "best_model"
        path.mkdir(parents=True, exist_ok=True)
        self.tutor.model.save_pretrained(path)
        self.tutor.tokenizer.save_pretrained(path)
        (path / "selection.json").write_text(
            json.dumps(
                {
                    "epoch": epoch,
                    "probe_solve": score,
                    "probe_split": self.settings.training.probe_split,
                    "probe_problems": self.settings.training.probe_problems,
                    "criterion": "strict held-out solve rate, maximised",
                },
                indent=2,
            )
        )
        logger.info("New best (probe solve=%.3f at epoch %d) saved to %s",
                    score, epoch, path)
