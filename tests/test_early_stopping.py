"""Early stopping and best-checkpoint selection.

The trainer used to run a fixed epoch count and save whichever weights came
last. In the v24 run that meant epoch 9 was shipped while epoch 5 had scored
better on training rollouts, with nothing recording the difference.

The decision rule lives in `sahai.training.early_stop`, which imports no
torch, so it is exercised directly here rather than through the trainer. The
wiring in grpo.py is checked by source, following the convention in
test_reward.py: that module pulls in torch, which the test venv does not have.
"""

from __future__ import annotations

import pathlib

import pytest

from sahai.settings import Settings
from sahai.training.early_stop import EarlyStopper


class TestStoppingRule:
    def test_stops_after_patience_probes_without_improvement(self):
        s = EarlyStopper(patience=3)
        assert s.observe(1, 0.20).improved
        assert not s.observe(3, 0.10).should_stop   # stale 1
        assert not s.observe(5, 0.10).should_stop   # stale 2
        assert s.observe(7, 0.10).should_stop       # stale 3 -> stop

    def test_improvement_resets_the_counter(self):
        s = EarlyStopper(patience=3)
        s.observe(1, 0.10)
        s.observe(3, 0.10)
        s.observe(5, 0.10)                           # stale 2
        assert s.observe(7, 0.20).improved          # resets
        assert s.observe(9, 0.10).stale == 1
        assert not s.observe(11, 0.10).should_stop

    def test_best_is_kept_not_the_latest(self):
        """The v24 failure mode: a mid-run epoch beats the final one."""
        s = EarlyStopper(patience=9)
        for ep, sc in [(1, 0.10), (3, 0.40), (5, 0.20), (7, 0.20), (9, 0.20)]:
            s.observe(ep, sc)
        assert s.best_epoch == 3
        assert s.best_score == 0.40

    def test_all_zero_probes_still_record_a_best(self):
        """A run that never solves anything must still select a checkpoint.

        best_score starts below zero for exactly this case; at 0.0 nothing
        would ever count as an improvement and no model would be saved.
        """
        s = EarlyStopper(patience=9)
        d = s.observe(1, 0.0)
        assert d.improved
        assert s.best_epoch == 1 and s.best_score == 0.0

    def test_min_delta_suppresses_trivial_improvements(self):
        s = EarlyStopper(patience=2, min_delta=0.05)
        s.observe(1, 0.20)
        assert not s.observe(3, 0.22).improved, "+0.02 is under the 0.05 floor"
        assert s.observe(5, 0.30).improved

    def test_equal_score_is_not_an_improvement(self):
        s = EarlyStopper(patience=5)
        s.observe(1, 0.25)
        assert not s.observe(3, 0.25).improved
        assert s.best_epoch == 1, "ties keep the earlier, cheaper checkpoint"

    def test_patience_below_one_is_rejected(self):
        with pytest.raises(ValueError):
            EarlyStopper(patience=0)


class TestSummary:
    def test_names_the_gap_when_best_is_not_last(self):
        s = EarlyStopper(patience=9)
        s.observe(3, 0.40)
        out = s.summary(final_epoch=9)
        assert "epoch 3" in out and "best_model" in out

    def test_says_so_when_they_agree(self):
        s = EarlyStopper(patience=9)
        s.observe(9, 0.40)
        assert "agree" in s.summary(final_epoch=9)

    def test_says_so_when_no_probe_ran(self):
        assert "no probe ran" in EarlyStopper(patience=3).summary(final_epoch=9)


class TestProbeSettings:
    def test_probe_split_is_not_the_reported_split(self):
        """Selecting a checkpoint on the split the final number comes from
        makes that number a selection artefact, not a held-out measurement."""
        t = Settings.kaggle().training
        assert t.probe_split == "validation"
        assert t.probe_split != "test"

    def test_probing_is_enabled(self):
        t = Settings.kaggle().training
        assert t.eval_every > 0 and t.probe_problems > 0

    def test_probe_budget_fits_the_session_cap(self):
        """0.893 min/problem, measured on a T4 in the v24 run (60 problems in
        53.6 min). Training was 4.75h and the final evaluation 0.9h."""
        t = Settings.kaggle().training
        probes = t.epochs // t.eval_every
        hours = probes * t.probe_problems * 0.893 / 60
        assert hours + 4.75 + 0.9 < 12


class TestTrainerWiring:
    """grpo.py by source: it imports torch, which this venv does not have."""

    @staticmethod
    def _src() -> str:
        return pathlib.Path("sahai/training/grpo.py").read_text()

    def test_trainer_uses_the_shared_stopper(self):
        src = self._src()
        assert "from sahai.training.early_stop import EarlyStopper" in src
        assert "stopper.observe(" in src

    def test_trainer_probes_and_can_break(self):
        src = self._src()
        assert "self._probe()" in src
        assert "d.should_stop" in src and "break" in src

    def test_probe_restores_the_learner_model(self):
        """A probe that left its BKT updates behind would feed held-out
        problems into the curriculum the next epoch samples from."""
        src = self._src()
        assert "saved_skills" in src
        assert "self.student.tracer.skills = saved_skills" in src

    def test_probe_restores_training_mode(self):
        src = self._src()
        assert "self.tutor.model.eval()" in src
        assert "self.tutor.model.train()" in src

    def test_best_model_is_saved_separately_from_final(self):
        src = self._src()
        assert '"best_model"' in src
        assert "selection.json" in src

    def test_stopping_does_not_read_the_loss(self):
        """The whole point. policy_loss is ~0 every epoch by construction."""
        src = self._src()
        probe_block = src[src.index("if cfg.eval_every"):src.index("self._checkpoint(output_dir, epoch, step)\n\n")]
        assert "policy_loss" not in probe_block
        assert "mean_reward" not in probe_block
