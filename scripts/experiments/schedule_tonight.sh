#!/bin/zsh
# Tonight's three runs, in the order that puts a usable result first.
#
# 1. Tutor frontier, ~1.5 h. Inference only. The only item that can produce a
#    comparative result, and the only one that scores a model other than this
#    project's own checkpoint. Runs first so that if the night is cut short
#    there is still something to show.
# 2. RAFT, ~30 min. Cross-entropy on the 45 saved dialogues where the tutor
#    taught without disclosing and the student then solved. Tests whether the
#    policy can move at all, which finding #29 showed GRPO never achieved.
# 3. Asymmetric GRPO, the rest of the night. 1.5B tutor against a 0.5B student,
#    so the tutor finally knows something the student does not.
#
# Strictly sequential. Two 1.5B models at once deadlocked this 16 GB machine
# twice, and a 3B tutor beside a student is tighter still.
set -u
REPO="/Users/aswathiranjith/Documents/sem 7/rl/SAHAi"
cd "$REPO" || exit 1
START_HOUR=23

export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH="$REPO:$REPO/libs/sahai-core"
export TOKENIZERS_PARALLELISM=false
export SAHAI_LEARNER_CONTEXT=0
PY_BIN="${PY_BIN:-$HOME/miniforge3/bin/python3}"
if ! "$PY_BIN" -c "import torch" >/dev/null 2>&1; then
  echo "FATAL: $PY_BIN cannot import torch"; exit 1
fi

echo "holding until ${START_HOUR}:00 local. now $(date '+%Y-%m-%d %H:%M %Z')"
while true; do
  h=$(date +%H)
  if [ "$h" -ge "$START_HOUR" ] || [ "$h" -lt 6 ]; then break; fi
  sleep 300
done
echo "released at $(date '+%Y-%m-%d %H:%M %Z')"

while pgrep -f "train_rccr.py" >/dev/null 2>&1 \
   || pgrep -f "scripts.run_local" >/dev/null 2>&1; do
  echo "waiting: rccr-genomics training is active $(date '+%H:%M')"
  sleep 300
done

mkdir -p artifacts/frontier artifacts/raft artifacts/retrain_asym

echo "=== 1/3 tutor frontier $(date '+%H:%M') ==="
nice -n 5 "$PY_BIN" scripts/experiments/tutor_frontier.py \
  --problems 60 --adapter artifacts/retrain/best_model \
  --prompted Qwen/Qwen2.5-3B-Instruct --solve-context hints \
  --out artifacts/frontier/frontier.json \
  > artifacts/frontier/frontier.log 2>&1
echo "frontier exit=$? $(date '+%H:%M')"

echo "=== 2/3 RAFT $(date '+%H:%M') ==="
nice -n 5 "$PY_BIN" scripts/experiments/raft_finetune.py \
  --rollouts artifacts/retrain/rollouts --out artifacts/raft --epochs 3 \
  > artifacts/raft/raft.log 2>&1
echo "raft exit=$? $(date '+%H:%M')"

echo "=== 3/3 asymmetric GRPO $(date '+%H:%M') ==="
./scripts/experiments/run_retrain_asym.sh > artifacts/retrain_asym/run.log 2>&1
echo "asym exit=$? $(date '+%H:%M')"

echo "TONIGHT DONE $(date '+%Y-%m-%d %H:%M %Z')"
