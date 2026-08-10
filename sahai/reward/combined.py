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


class SAHAIReward:
    def __init__(self, settings: RewardSettings):
        self.lambda_ped = settings.lambda_ped
        self.gamma_leak = settings.gamma_leak
        self.hard_penalty = settings.hard_penalty

    def compute(
        self,
        r_sol: float,
        r_ped: float,
        leakage: float,
        ability_baseline: float,
    ) -> RewardComponents:
        if self.hard_penalty and r_ped == 0.0:
            r_sahai = -self.lambda_ped
        else:
            r_sahai = (
                (r_sol - ability_baseline)
                + (r_ped - 1) * self.lambda_ped
                - self.gamma_leak * leakage
            )

        return RewardComponents(
            r_sol=r_sol,
            r_ped=r_ped,
            leakage=leakage,
            ability_baseline=ability_baseline,
            r_sahai=r_sahai,
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
