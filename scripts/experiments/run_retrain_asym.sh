#!/bin/zsh
# A tutor that knows more than its student.
#
# Tutor and student were the same checkpoint for the whole project. A tutor
# with its student's exact weights has no knowledge the student lacks, so the
# only thing it can contribute is organisation, and organisation measured
# 0.350 against 0.367 for no tutoring at all. The oracle arm shows the channel
# is open (0.783 when the solution is disclosed), so the student can absorb
# information; there was none to send.
#
#   student  1.5B -> 0.5B    creates the asymmetry, and lowers the unaided
#                            baseline so there is headroom to measure
#   kl_coeff 0.05 -> 0.001   matches arXiv:2505.15607 and removes most of the
#                            force opposing an already negligible gradient
#   batch    4 -> 8          coverage was 26 of 198 problems
#
# The learning rate is deliberately unchanged. Finding #29 measured lora_B at
# 0.000762 against an initialisation of exactly zero, so at 1e-4 the policy did
# not move and 5e-7 would freeze it.
#
# Watch lora_B after epoch 1. If it is still near zero the configuration is not
# the problem and the GRPO surrogate magnitude is, and the next thing to try is
# rejection-sampling fine-tuning.
set -u
REPO="/Users/aswathiranjith/Documents/sem 7/rl/SAHAi"
OUT="$REPO/artifacts/retrain_asym"
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

echo "===== SAHAI GRPO retrain, asymmetric tutor/student ====="
echo "started      $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "interpreter  $PY_BIN ($("$PY_BIN" -V 2>&1))"
echo "framing      SAHAI_SOLVE_CONTEXT=$SAHAI_SOLVE_CONTEXT"
echo "learner ctx  SAHAI_LEARNER_CONTEXT=$SAHAI_LEARNER_CONTEXT"
echo "output       $OUT"
echo

nice -n 5 "$PY_BIN" -m sahai.cli train --asymmetric --output "$OUT"
echo "train exit=$? $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "RETRAIN DONE"
