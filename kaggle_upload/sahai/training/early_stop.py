"""When to stop training, and which checkpoint to keep.

Separate from grpo.py on purpose. That module imports torch, which the test
venv deliberately does not have, so a decision rule living there could only be
tested by reading its source. The rule is the part most worth testing: it
decides which weights get served.

What it does not use, and why:

* **Not the loss.** The GRPO surrogate is `-min(rho*A, clip(rho)*A)` with the
  advantage z-scored inside its group. Advantages therefore sum to zero, and
  `old_log_probs` come from the very policy that generated the rollouts, so
  `rho = 1` and the reported loss is ~0 at every epoch by construction. Over
  the v24 run it ranged -0.00096 to +0.00406 and changed sign four times in
  nine transitions. There is no descending curve to converge.
* **Not the reward.** Reward climbing while the outcome it exists to produce
  falls is this project's documented failure, not a hypothetical: v15 -> v16
  moved held-out solve from 16.3% to 6.2% while mean reward reached its first
  positive value. Stopping on reward would stop at precisely the wrong epoch.

So the signal is held-out solve rate, measured on a split the final evaluation
does not report on.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StopDecision:
    """What to do after one probe."""

    improved: bool
    should_stop: bool
    stale: int
    best_score: float
    best_epoch: int


class EarlyStopper:
    """Tracks the best held-out score and how long since it improved.

    `best_score` starts below zero rather than at zero so that a run whose
    every probe scores 0.000 still registers a best and saves a checkpoint.
    Starting at 0.0 would mean nothing ever "improved", and such a run would
    finish with no selected model at all -- which is the case where having one
    matters most, because the final epoch is no more trustworthy than any
    other.
    """

    def __init__(self, patience: int, min_delta: float = 0.0):
        if patience < 1:
            raise ValueError("patience must be at least 1")
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -1.0
        self.best_epoch = -1
        self.stale = 0

    def observe(self, epoch: int, score: float) -> StopDecision:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.stale = 0
            return StopDecision(True, False, 0, self.best_score, self.best_epoch)

        self.stale += 1
        return StopDecision(
            improved=False,
            should_stop=self.stale >= self.patience,
            stale=self.stale,
            best_score=self.best_score,
            best_epoch=self.best_epoch,
        )

    def summary(self, final_epoch: int) -> str:
        """One line for the log, naming the gap this exists to close.

        The trainer used to save the last epoch's weights as `final_model` with
        nothing recording whether a better epoch had gone by. In the v24 run
        epoch 5 scored 0.333 on training rollouts, epoch 8 scored 0.073, and
        epoch 9 was shipped because it was last.
        """
        if self.best_epoch < 0:
            return "no probe ran; final_model is the last epoch and unranked"
        if self.best_epoch == final_epoch:
            return (
                f"best probe was the final epoch ({final_epoch}, "
                f"{self.best_score:.3f}); final_model and best_model agree"
            )
        return (
            f"best probe was epoch {self.best_epoch} ({self.best_score:.3f}), "
            f"but final_model is epoch {final_epoch}; prefer best_model"
        )
