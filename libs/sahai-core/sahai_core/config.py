"""Configuration primitives shared by every service.

Only settings that describe *domain behaviour* live here. Anything describing
deployment (ports, DSNs, model paths) belongs to the individual service, so a
service never carries config it does not use.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RewardSettings(BaseModel):
    lambda_ped: float = 1.0
    gamma_leak: float = 0.5
    num_solve_samples: int = 8
    num_judges: int = 2
    hard_penalty: bool = True
    use_rule_judge: bool = False


class TracerSettings(BaseModel):
    p_init: float = 0.3
    p_learn: float = 0.1
    p_guess: float = 0.2
    p_slip: float = 0.1
    zpd_low: float = 0.3
    zpd_high: float = 0.7


class ExecutionLimits(BaseModel):
    """Bounds applied to untrusted, model-generated code."""

    timeout_seconds: int = Field(default=5, ge=1, le=30)
    memory_mb: int = Field(default=256, ge=64, le=2048)
    max_output_bytes: int = Field(default=64_000, ge=1_000)
