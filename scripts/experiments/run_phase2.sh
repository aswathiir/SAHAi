#!/bin/zsh
# Phase 2 GPU experiments, strictly sequential.
#
# Never run these concurrently with each other or with another project's
# training. This machine has 16 GB and a 10 GB swap; two 1.5B models at once
# put both processes into uninterruptible sleep for twenty minutes on
# 2026-10-06 and neither advanced a single problem. `nice` so an interactive
# session stays usable if one is open.
#
# Outputs land in artifacts/phase2/ rather than a session scratchpad, because
# a scratchpad under /private/tmp does not survive.
set -u
REPO="/Users/aswathiranjith/Documents/sem 7/rl/SAHAi"
OUT="$REPO/artifacts/phase2"
cd "$REPO" || exit 1
mkdir -p "$OUT"
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH="$REPO:$REPO/libs/sahai-core"

echo "started $(date '+%Y-%m-%d %H:%M:%S %Z')"

echo "=== 1/2 oracle decomposition (60 problems) ==="
nice -n 10 python3 scripts/experiments/oracle_decompose.py 60 "$OUT/oracle.json" \
  > "$OUT/oracle.log" 2>&1
echo "oracle exit=$? $(date '+%H:%M:%S')"

echo "=== 2/2 solve-context framing replay (2 arms x 60 x 2 framings) ==="
nice -n 10 python3 scripts/experiments/framing_replay.py \
  kaggle_upload/dialogues/ab_benchmark.json "$OUT/framing.json" 60 \
  > "$OUT/framing.log" 2>&1
echo "framing exit=$? $(date '+%H:%M:%S')"

echo "ALL DONE $(date '+%Y-%m-%d %H:%M:%S %Z')"
