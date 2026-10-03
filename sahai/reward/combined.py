from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sahai.settings import RewardSettings


@dataclass
class RewardComponents:
    r_sol: float
    r_ped: float
    leakage: float
    ability_baseline: float
    r_sahai: float
    # Agreement between the tutor's technical claims and the reference
    # solution, in [-1, 1], with 0.0 when it made no checkable claim. Recorded
    # on every rollout even when mu_correct is 0, so its coverage can be read
    # off a run instead of guessed at.
    correctness: float = 0.0


class SAHAIReward:
    def __init__(self, settings: RewardSettings):
        self.lambda_ped = settings.lambda_ped
        self.gamma_leak = settings.gamma_leak
        self.hard_penalty = settings.hard_penalty
        self.mu_correct = getattr(settings, "mu_correct", 0.0)

    def compute(
        self,
        r_sol: float,
        r_ped: float,
        leakage: float,
        ability_baseline: float,
        correctness: float = 0.0,
    ) -> RewardComponents:
        if self.hard_penalty and r_ped == 0.0:
            r_sahai = -self.lambda_ped
        else:
            # Correctness is signed, so it adds rather than subtracts: a tutor
            # that says nothing checkable scores 0 and is neither rewarded nor
            # punished, one that is right gains, one that is wrong loses. A
            # penalty-only form would make silence optimal, and the policy is
            # already drifting there -- mean turn length fell 574 -> 393
            # characters over v24.
            r_sahai = (
                (r_sol - ability_baseline)
                + (r_ped - 1) * self.lambda_ped
                - self.gamma_leak * leakage
                + self.mu_correct * correctness
            )

        return RewardComponents(
            r_sol=r_sol,
            r_ped=r_ped,
            leakage=leakage,
            ability_baseline=ability_baseline,
            r_sahai=r_sahai,
            correctness=correctness,
        )

    def compute_batch(
        self,
        r_sols: list[float],
        r_peds: list[float],
        leakages: list[float],
        ability_baseline: float,
    ) -> list[RewardComponents]:
        return [
            self.compute(r_sol, r_ped, leak, ability_baseline)
            for r_sol, r_ped, leak in zip(r_sols, r_peds, leakages)
        ]
