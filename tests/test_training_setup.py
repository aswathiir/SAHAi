"""The training entry point must seed before it loads, and validate its bank.

Two gaps found on 2026-10-07 while preparing the retrain.

`GRPOTrainer.__init__` calls `seed_everything`, but the CLI loads both models
first and `load_for_training` applies the LoRA adapter, whose `lora_A` is drawn
from a Kaiming uniform. `lora_B` is zeros, so step 0 is identical either way and
no epoch-0 metric shows the difference; everything after step 1 diverges. That
is the seed-never-applied defect one layer further in.

`load_mbpp` defaults to `validate=False` deliberately, so evaluation stays
comparable with runs measured before validation existed. Training is the case
that needs it: a group whose reference fails its own tests has no reward
variance, so every z-scored advantage in it is zero and it buys no gradient.
"""

from __future__ import annotations

import pathlib


class TestSeedOrder:
    def test_the_cli_seeds_before_loading_a_model(self):
        src = pathlib.Path("sahai/cli.py").read_text()
        body = src[src.index("def train("):]
        assert "seed_everything(settings.seed)" in body
        assert body.index("seed_everything(settings.seed)") < body.index(
            "load_for_training("
        ), "LoRA initialisation draws from the RNG, so seeding after it is too late"

    def test_the_trainer_still_seeds_itself(self):
        """Belt and braces: the notebook path does not go through the CLI."""
        src = pathlib.Path("sahai/training/grpo.py").read_text()
        assert "seed_everything(settings.seed)" in src


class TestTrainingValidatesItsBank:
    def test_the_cli_validates_for_training(self):
        src = pathlib.Path("sahai/cli.py").read_text()
        assert "_load_problems(settings, validate=True)" in src

    def test_validation_is_still_opt_in_at_the_loader(self):
        """Evaluation must keep the old default or old numbers stop comparing."""
        assert "validate: bool = False" in pathlib.Path("sahai/core/dataset.py").read_text()
        assert "def _load_problems(settings, validate: bool = False)" in (
            pathlib.Path("sahai/cli.py").read_text()
        )


class TestCorrectedPreset:
    """The corrected preset changes three things and only three.

    Run v14 of this project changed three variables at once, came out net
    negative, and nothing could be attributed to any of them. The comparison
    against `local_mps()` is only interpretable if everything else is equal, so
    this pins which fields may differ.
    """

    def _presets(self):
        from sahai.settings import Settings

        return Settings.local_mps(), Settings.local_mps_corrected()

    def test_the_three_hyperparameters_match_the_source_paper(self):
        _, c = self._presets()
        assert c.training.batch_size == 8
        assert c.training.learning_rate == 5e-7
        assert c.training.kl_coeff == 0.001

    def test_nothing_else_in_the_training_block_moves(self):
        a, c = self._presets()
        changed = {"batch_size", "learning_rate", "kl_coeff"}
        for field in a.training.model_dump():
            if field in changed:
                continue
            assert getattr(a.training, field) == getattr(c.training, field), (
                f"{field} differs; the comparison stops isolating the three"
            )

    def test_the_model_and_reward_blocks_are_identical(self):
        a, c = self._presets()
        assert a.model.model_dump() == c.model.model_dump()
        assert a.reward.model_dump() == c.reward.model_dump()
        assert a.seed == c.seed

    def test_accumulation_is_not_touched_so_steps_double_as_a_consequence(self):
        a, c = self._presets()
        assert a.training.gradient_accumulation_steps == c.training.gradient_accumulation_steps
        steps = lambda s: s.training.batch_size * s.training.group_size // s.training.gradient_accumulation_steps
        assert steps(a) == 16 and steps(c) == 32

    def test_it_writes_somewhere_else(self):
        a, c = self._presets()
        assert a.output_dir != c.output_dir

    def test_the_stated_risk_is_recorded_not_just_known(self):
        """The likely failure mode is inertia, not instability, and the preset
        says so with the number that would change if it happens."""
        import inspect

        from sahai.settings import Settings

        doc = inspect.getdoc(Settings.local_mps_corrected) or ""
        assert "2e-5" in doc and "KL" in doc
