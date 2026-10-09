#!/bin/zsh
# GRPO retrain with the RL hyperparameters matched to the source paper.
#
# Same corrected solve framing as the previous run. The difference is three
# numbers that had never been compared against Dinucu-Jianu et al. (EMNLP
# 2025, arXiv:2505.15607), the paper this method was taken from:
#
#   problems per batch   4 -> 8      (reference 16; 8 is a compute choice)
#   learning rate        1e-4 -> 5e-7
#   KL coefficient       0.05 -> 0.001
#
# Nothing else moves, so the comparison against artifacts/retrain isolates
# them. Steps per epoch double from 16 to 32 as a consequence of the batch,
# not as a fourth change.
#
# Stated risk, recorded in the preset's docstring: total parameter movement is
# about a hundred times smaller than the run this replaces, so the likely
# failure mode is inertia rather than instability. Watch KL. If it stays near
# zero the learning rate is the thing to raise, and the estimate that matches
# the reference's cumulative movement at this step budget is near 2e-5.
#
# Why this run exists. The outcome term is computed from the student's final
# attempt, and finding #25 showed that attempt's framing moves solve rate by
# 0.184, which is larger than any effect this project was built to detect.
# Every run to date therefore optimised a reward whose outcome term was
# distorted by it, penalising dialogues in proportion to how much the student
# wrote. This is the first run whose r_sol means what it says.
#
# Why `hints` and not `quoted`. `hints` is the only framing shown to recover
# the deficit. `quoted`, which keeps every character and removes only the
# assistant role, does not: on the discordant problems it tracked the harmful
# framing 14 times to 8, and in the trained arm on all 8 (finding #27). So
# removing the student's text is doing the work and re-roling it is not enough.
# We cannot say why, and finding #27 records that both explanations we offered
# are falsified.
#
# Everything else is inherited from Settings.kaggle() through local_mps(), so
# this is comparable with the Kaggle runs rather than a new configuration.
set -u
REPO="/Users/aswathiranjith/Documents/sem 7/rl/SAHAi"
OUT="$REPO/artifacts/retrain_corrected"
cd "$REPO" || exit 1
mkdir -p "$OUT"

# Explicit interpreter: `python3` is /usr/bin/python3 on this machine in some
# shells, which is 3.9 and has no torch.
# The OpenMP workaround has to be exported BEFORE the guard, not after: two
# copies of libomp are linked into this environment and a bare `import torch`
# aborts without it, so the guard was failing on a healthy interpreter.
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH="$REPO:$REPO/libs/sahai-core"
export TOKENIZERS_PARALLELISM=false

PY_BIN="${PY_BIN:-$HOME/miniforge3/bin/python3}"
if ! "$PY_BIN" -c "import torch, sys; sys.exit(0)" >/dev/null 2>&1; then
  echo "FATAL: $PY_BIN cannot import torch. Tried with KMP_DUPLICATE_LIB_OK set."
  "$PY_BIN" -c "import torch" 2>&1 | tail -3
  exit 1
fi

# The whole point of the run. Set in the environment so the configuration is
# recorded rather than inferred.
export SAHAI_SOLVE_CONTEXT=hints

# Left off, matching every run recorded so far. Turning it on is a separate
# measured change and would have to be set for the services too.
export SAHAI_LEARNER_CONTEXT=0

echo "===== SAHAI GRPO retrain, corrected hyperparameters ====="
echo "started      $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "interpreter  $PY_BIN ($("$PY_BIN" -V 2>&1))"
echo "framing      SAHAI_SOLVE_CONTEXT=$SAHAI_SOLVE_CONTEXT"
echo "learner ctx  SAHAI_LEARNER_CONTEXT=$SAHAI_LEARNER_CONTEXT"
echo "output       $OUT"
echo

nice -n 5 "$PY_BIN" -m sahai.cli train --corrected --output "$OUT"
echo "train exit=$? $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "RETRAIN DONE"
