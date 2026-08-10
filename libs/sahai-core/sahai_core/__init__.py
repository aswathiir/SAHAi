"""SAHAi shared domain library.

Pure domain logic and scoring. No I/O, no network, no GPU, no code execution —
every service can depend on this without inheriting those capabilities.
"""

from sahai_core.config import ExecutionLimits, RewardSettings, TracerSettings
from sahai_core.domain.dialogue import TERMINATION_PHRASES, Dialogue, Turn
from sahai_core.domain.mastery import BKTTracer
from sahai_core.domain.problem import Problem, ProblemBank, TestCase
from sahai_core.reward.combined import RewardComponents, SAHAIReward
from sahai_core.reward.leakage import LeakageEstimator
from sahai_core.reward.pedagogy import RuleBasedJudge
from sahai_core.reward.solve import (
    AttemptResult,
    ExecutionOutcome,
    SolveScorer,
    TestResult,
)

__version__ = "0.1.0"

__all__ = [
    "TERMINATION_PHRASES",
    "AttemptResult",
    "BKTTracer",
    "Dialogue",
    "ExecutionLimits",
    "ExecutionOutcome",
    "LeakageEstimator",
    "Problem",
    "ProblemBank",
    "RewardComponents",
    "RewardSettings",
    "RuleBasedJudge",
    "SAHAIReward",
    "SolveScorer",
    "TestCase",
    "TestResult",
    "TracerSettings",
    "Turn",
    "__version__",
]
