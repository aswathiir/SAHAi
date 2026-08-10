from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from sahai.core.dialogue import DialogueEngine
from sahai.reward.combined import SAHAIReward
from sahai.reward.leakage import LeakageEstimator
from sahai.reward.solve import CodeVerifier, SolveReward

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
    ped_acceptance: float = 0.0
    leakage_rate: float = 0.0
    mean_reward: float = 0.0
    per_difficulty: dict[int, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Evaluator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.dialogue_engine = DialogueEngine(max_turns=settings.training.max_turns)
        self.verifier = CodeVerifier(
            timeout=settings.reward.exec_timeout,
            memory_mb=settings.reward.exec_memory_mb,
        )
        self.solve_reward = SolveReward(self.verifier, settings.reward.num_solve_samples)
        self.leakage_estimator = LeakageEstimator()
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

            r_sol = self.solve_reward.compute(student, dialogue, problem)
            r_ped = pedagogy_reward.evaluate(dialogue)
            leak = self.leakage_estimator.estimate(dialogue, problem.solution)
            ability = student.tracer.get_ability()
            components = self.sahai_reward.compute(r_sol, r_ped, leak, ability)

            report.solve_rate += r_sol
            report.ped_acceptance += r_ped
            report.leakage_rate += leak
            report.mean_reward += components.r_sahai

            bucket = difficulty_buckets.setdefault(problem.difficulty, [])
            bucket.append({"r_sol": r_sol, "r_ped": r_ped, "leak": leak})

            logger.info(
                f"  {problem.id}: r_sol={r_sol:.2f} r_ped={r_ped:.2f} "
                f"leak={leak:.2f} reward={components.r_sahai:.4f}"
            )

        n = max(report.num_problems, 1)
        report.solve_rate /= n
        report.ped_acceptance /= n
        report.leakage_rate /= n
        report.mean_reward /= n

        for diff, entries in difficulty_buckets.items():
            k = len(entries)
            report.per_difficulty[diff] = {
                "solve_rate": sum(e["r_sol"] for e in entries) / k,
                "ped_acceptance": sum(e["r_ped"] for e in entries) / k,
                "leakage_rate": sum(e["leak"] for e in entries) / k,
                "count": k,
            }

        return report

    @staticmethod
    def compare(reports: list[MetricsReport]) -> str:
        header = f"{'Model':<30} {'Solve':>8} {'Ped':>8} {'Leak':>8} {'Reward':>10}"
        lines = [header, "-" * len(header)]
        for r in reports:
            lines.append(
                f"{r.model_name:<30} {r.solve_rate:>8.3f} {r.ped_acceptance:>8.3f} "
                f"{r.leakage_rate:>8.3f} {r.mean_reward:>10.4f}"
            )
        return "\n".join(lines)

    @staticmethod
    def save_report(report: MetricsReport, path: str) -> None:
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
