#!/bin/zsh
# Launch the corrected run once the evaluation queue has finished. Never
# concurrent: two 1.5B models on 16 GB deadlocked both processes twice.
set -u
REPO="/Users/aswathiranjith/Documents/sem 7/rl/SAHAi"
cd "$REPO" || exit 1
echo "waiting for the evaluation queue..."
while pgrep -f "run_eval_queue.sh" >/dev/null 2>&1 \
   || pgrep -f "base_probe.py" >/dev/null 2>&1 \
   || pgrep -f "ab_benchmark" >/dev/null 2>&1 \
   || pgrep -f "sahai.cli train" >/dev/null 2>&1; do
  sleep 60
done
echo "queue clear at $(date '+%Y-%m-%d %H:%M:%S')"
exec ./scripts/experiments/run_retrain_corrected.sh
