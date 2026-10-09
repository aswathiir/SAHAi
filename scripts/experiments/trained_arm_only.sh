#!/bin/zsh
# Only the arm we do not already have.
#
# A four-arm benchmark costs about two hours and two of its arms are already
# measured on these exact 60 test problems under this exact framing:
#
#   unaided      0.367 (22/60). Framing-invariant: an empty dialogue produces
#                no solve context under either setting, so the stored vector is
#                the same measurement.
#   base, hints  0.350 (21/60), from the framing replay.
#   oracle       0.783 under the default framing. A ceiling, not today's
#                question, and re-measuring it buys nothing for the retrain.
#
# So this runs the trained arm alone, 60 generations rather than 240, and the
# paired comparisons are computed against the stored per-problem vectors.
set -u
REPO="/Users/aswathiranjith/Documents/sem 7/rl/SAHAi"
cd "$REPO" || exit 1
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$REPO:$REPO/libs/sahai-core"
export TOKENIZERS_PARALLELISM=false SAHAI_SOLVE_CONTEXT=hints SAHAI_LEARNER_CONTEXT=0
PY_BIN="${PY_BIN:-$HOME/miniforge3/bin/python3}"

echo "waiting for the base probe..."
while pgrep -f "base_probe.py" >/dev/null 2>&1; do sleep 30; done
echo "base probe done at $(date '+%H:%M:%S')"

echo "=== trained arm, 60 held-out problems, hints framing, best_model (epoch 5) ==="
nice -n 5 "$PY_BIN" -m sahai.eval.ab_benchmark \
  --adapter "$REPO/artifacts/retrain/best_model" --problems 60 --split test \
  --arms trained --device mps --solve-context hints \
  --out "$REPO/artifacts/retrain/trained_arm.json" \
  > "$REPO/artifacts/retrain/trained_arm.log" 2>&1
echo "exit=$? $(date '+%H:%M:%S')"
echo "TRAINED ARM DONE"
