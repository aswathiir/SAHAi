"""Seeding, because its absence voided a 16-GPU-hour experiment.

`settings.seed` existed from the first commit and was applied nowhere in the
training path. `zpd_sample` draws randomly inside its difficulty window, so two
runs of an identical configuration took different problems in epoch 0, before a
single gradient step. The 2026-10-03 correctness experiment was built as a
paired comparison and was not one: its two arms differed by training
nondeterminism as well as by the single setting under test.

These tests cover the part that can be checked without a GPU -- that seeding
Python's `random` makes problem selection reproducible, and that the trainer
calls the seeding function at all.
"""

from __future__ import annotations

import pathlib
import random

from sahai.agents.tracer import BKTTracer
from sahai.core.data import Problem, ProblemBank


def bank(n: int = 60) -> ProblemBank:
    return ProblemBank(problems=[
        Problem(id=f"p{i}", title=f"t{i}", description="d", function_name="f",
                function_signature="def f():", solution="def f():\n    return 1\n",
                test_cases=[], skills=["arrays"], difficulty=(i % 5) + 1)
        for i in range(n)
    ])


def draws(seed: int, epochs: int = 5, batch: int = 4) -> list[list[str]]:
    random.seed(seed)
    b, tracer = bank(), BKTTracer()
    return [[p.id for p in b.zpd_sample(tracer, batch)] for _ in range(epochs)]


class TestProblemSelection:
    def test_same_seed_gives_the_same_curriculum(self):
        assert draws(42) == draws(42)

    def test_different_seeds_diverge(self):
        """If this ever fails, selection stopped depending on the seed and the
        reproducibility above is vacuous."""
        assert draws(42) != draws(99)

    def test_divergence_starts_at_the_first_epoch(self):
        """The precise failure: the arms differed before any gradient step, so
        nothing downstream was attributable."""
        assert draws(42)[0] != draws(99)[0]

    def test_selection_is_not_constant_across_epochs(self):
        """A seeded run must still vary its curriculum; freezing it would be a
        different bug wearing the same fix."""
        d = draws(42, epochs=6)
        assert len({tuple(x) for x in d}) > 1


class TestTrainerSeeds:
    """grpo.py by source: it imports torch, which this venv does not have."""

    @staticmethod
    def _src() -> str:
        return pathlib.Path("sahai/training/grpo.py").read_text()

    def test_the_trainer_seeds_on_construction(self):
        assert "seed_everything(settings.seed)" in self._src()

    def test_every_source_of_randomness_is_covered(self):
        """random drives zpd_sample, torch drives both agents' sampled turns,
        and transformers reaches for numpy in places this code does not own."""
        block = self._src()
        block = block[block.index("def seed_everything"):block.index("@dataclass")]
        for call in ("random.seed(seed)", "torch.manual_seed(seed)",
                     "torch.cuda.manual_seed_all(seed)", "np.random.seed(seed)"):
            assert call in block, f"{call} missing from seed_everything"

    def test_numpy_is_optional(self):
        """Seeding must not become a new import dependency of training."""
        block = self._src()
        block = block[block.index("def seed_everything"):block.index("@dataclass")]
        assert "except ImportError" in block

    def test_the_docstring_does_not_promise_bitwise_reproducibility(self):
        """cuBLAS reductions are not deterministic. Claiming identical runs
        would invite the next confound to go unnoticed."""
        block = self._src()
        block = block[block.index("def seed_everything"):block.index("@dataclass")]
        assert "comparable, not identical" in block
