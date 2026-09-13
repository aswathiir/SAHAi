from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from sahai.core.dialogue import DialogueEngine
from sahai.reward.combined import SAHAIReward
from sahai.reward.leakage import LeakageEstimator
from sahai.reward.solve import CodeVerifier, SolveReward, tutor_code_solves

if TYPE_CHECKING:
    from sahai.agents.student import StudentSimulator
    from sahai.agents.tutor import TutorPolicy
    from sahai.core.data import ProblemBank
    from sahai.reward.pedagogy import PedagogyReward
    from sahai.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class MetricsReport:
    model_name: str
    num_problems: int = 0
    solve_rate: float = 0.0
    """Fraction of held-out problems the student fully solved after tutoring.

    Deliberately the **all-or-nothing** reading, not the partial credit the
    reward now uses: this is the series every run since v6 is reported on, and
    the only one a comparison against 16.3% can be made against. Partial credit
    would raise it for reasons that have nothing to do with the policy."""
    partial_credit: float = 0.0
    """Mean fraction of test cases passed — the quantity GRPO optimises."""
    ped_acceptance: float = 0.0
    leakage_rate: float = 0.0
    mean_reward: float = 0.0
    per_difficulty: dict[int, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Evaluator:
    def __init__(self, settings: Settings, solution_corpus: list[str] | None = None):
        """`solution_corpus` must be the *training* problem bank's solutions —
        the same corpus the trainer fits on. Leakage counts a token only if it
        is rare across that corpus, so fitting on the 20-problem eval set
        instead would make "rare" mean something entirely different and put
        held-out leakage back on a different scale from training leakage.

        Left optional so existing callers keep working, but the notebook passes
        it; unfitted, leakage falls back to counting every token (old behaviour)
        rather than silently applying a filter with no corpus behind it.
        """
        self.settings = settings
        self.dialogue_engine = DialogueEngine(max_turns=settings.training.max_turns)
        self.verifier = CodeVerifier(
            timeout=settings.reward.exec_timeout,
            memory_mb=settings.reward.exec_memory_mb,
        )
        self.solve_reward = SolveReward(self.verifier, settings.reward.num_solve_samples)
        self.leakage_estimator = LeakageEstimator(solution_corpus=solution_corpus)
        self.sahai_reward = SAHAIReward(settings.reward)

    def evaluate_model(
        self,
        model_name: str,
        tutor: TutorPolicy,
        student: StudentSimulator,
        pedagogy_reward: PedagogyReward,
        problem_bank: ProblemBank,
    ) -> MetricsReport:
        report = MetricsReport(model_name=model_name, num_problems=len(problem_bank.problems))
        difficulty_buckets: dict[int, list[dict]] = {}

        for problem in problem_bank.problems:
            dialogue = self.dialogue_engine.run(tutor, student, problem)

            outcome = self.solve_reward.compute(student, dialogue, problem)
            # The reward is scored on partial credit because that is what the
            # policy was trained against; the report leads with the strict
            # pass rate because that is what earlier runs are comparable on.
            r_sol = outcome.score
            r_ped = pedagogy_reward.evaluate(dialogue)
            # Both keyword arguments were previously omitted, so eval silently
            # used the defaults (problem_text="", tutor_code_solves=False) and
            # scored leakage by a *different rule than training*:
            #   - no problem-statement subtraction, inflating leak by 34%
            #     relative (measured: 0.1495 -> 0.2006 over 304 v14 dialogues)
            #   - no execution check, so a tutor writing a complete working
            #     solution fell back to token overlap instead of scoring 1.0
            # The two errors point opposite ways and partly cancelled, which is
            # why the headline number never looked wrong. Held-out leakage from
            # runs before this fix is not comparable to training leakage.
            leak = self.leakage_estimator.estimate(
                dialogue,
                problem.solution,
                problem_text=f"{problem.title} {problem.description}",
                tutor_code_solves=tutor_code_solves(
                    dialogue, problem, self.verifier, self.leakage_estimator
                ),
            )
            ability = student.tracer.get_ability()
            components = self.sahai_reward.compute(r_sol, r_ped, leak, ability)

            report.solve_rate += outcome.solved
            report.partial_credit += r_sol
            report.ped_acceptance += r_ped
            report.leakage_rate += leak
            report.mean_reward += components.r_sahai

            bucket = difficulty_buckets.setdefault(problem.difficulty, [])
            bucket.append(
                {
                    "solved": outcome.solved,
                    "r_sol": r_sol,
                    "r_ped": r_ped,
                    "leak": leak,
                }
            )

            logger.info(
                f"  {problem.id}: solved={outcome.solved:.2f} r_sol={r_sol:.2f} "
                f"r_ped={r_ped:.2f} leak={leak:.2f} reward={components.r_sahai:.4f}"
            )

        n = max(report.num_problems, 1)
        report.solve_rate /= n
        report.partial_credit /= n
        report.ped_acceptance /= n
        report.leakage_rate /= n
        report.mean_reward /= n

        for diff, entries in difficulty_buckets.items():
            k = len(entries)
            report.per_difficulty[diff] = {
                "solve_rate": sum(e["solved"] for e in entries) / k,
                "partial_credit": sum(e["r_sol"] for e in entries) / k,
                "ped_acceptance": sum(e["r_ped"] for e in entries) / k,
                "leakage_rate": sum(e["leak"] for e in entries) / k,
                "count": k,
            }

        return report

    @staticmethod
    def compare(reports: list[MetricsReport]) -> str:
        header = (
            f"{'Model':<30} {'Solve':>8} {'Partial':>9} {'Ped':>8} "
            f"{'Leak':>8} {'Reward':>10}"
        )
        lines = [header, "-" * len(header)]
        for r in reports:
            lines.append(
                f"{r.model_name:<30} {r.solve_rate:>8.3f} {r.partial_credit:>9.3f} "
                f"{r.ped_acceptance:>8.3f} {r.leakage_rate:>8.3f} {r.mean_reward:>10.4f}"
            )
        lines.append("")
        lines.append(
            "Solve is all-or-nothing and comparable to every previous run "
            "(best so far 0.163)."
        )
        lines.append("Partial is the mean fraction of tests passed — what GRPO optimises.")
        return "\n".join(lines)

    @staticmethod
    def save_report(report: MetricsReport, path: str) -> None:
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
