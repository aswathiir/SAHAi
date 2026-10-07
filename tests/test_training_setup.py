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
