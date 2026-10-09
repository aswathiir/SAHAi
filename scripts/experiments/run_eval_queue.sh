#!/bin/zsh
# Waits for training to exit, then runs the two measurements that decide
# whether the retrain changed anything. Sequential, never concurrent.
#
# 1. Base-probe control. The probe rose 0.150 -> 0.250 -> 0.350 across the
#    run, and that trend is uninterpretable on its own: the untrained adapter
#    scores 0.350 under this same framing on the held-out 60, so the probe may
#    be climbing up to baseline rather than above it. This scores the base
#    checkpoint with the adapter disabled on the SAME 20 validation problems
#    under the SAME framing. Without it the trend repeats the project's oldest
#    mistake, a number with no control.
#
# 2. Four-arm paired benchmark on the 60 held-out problems: unaided, base,
#    trained (best_model), oracle. This is the measurement that decides.
set -u
REPO="/Users/aswathiranjith/Documents/sem 7/rl/SAHAi"
OUT="$REPO/artifacts/retrain"
cd "$REPO" || exit 1
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH="$REPO:$REPO/libs/sahai-core"
export TOKENIZERS_PARALLELISM=false
export SAHAI_SOLVE_CONTEXT=hints      # must match what training optimised
export SAHAI_LEARNER_CONTEXT=0
PY_BIN="${PY_BIN:-$HOME/miniforge3/bin/python3}"

echo "waiting for training to exit..."
while pgrep -f "sahai.cli train" >/dev/null 2>&1; do sleep 60; done
echo "training exited at $(date '+%H:%M:%S')"

echo "=== 1/2 base-probe control (20 validation problems, hints) ==="
nice -n 5 "$PY_BIN" scripts/experiments/base_probe.py 20 "$OUT/base_probe.json" \
  > "$OUT/base_probe.log" 2>&1
echo "base probe exit=$? $(date '+%H:%M:%S')"

echo "=== 2/2 four-arm paired benchmark (60 held-out, hints) ==="
nice -n 5 "$PY_BIN" -m sahai.eval.ab_benchmark \
  --adapter "$OUT/best_model" --problems 60 --split test \
  --arms unaided,base,trained,oracle --device mps --solve-context hints \
  --out "$OUT/benchmark.json" > "$OUT/benchmark.log" 2>&1
echo "benchmark exit=$? $(date '+%H:%M:%S')"
echo "EVAL QUEUE DONE"
