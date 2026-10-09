#!/bin/zsh
# Hold until 23:00 IST, then run tonight's work.
#
# A clock comparison rather than one long sleep: if the laptop suspends, a
# sleep(n) finishes late by however long it was out, while this fires as soon
# as it wakes past the hour.
set -u
REPO="/Users/aswathiranjith/Documents/sem 7/rl/SAHAi"
cd "$REPO" || exit 1
START_HOUR=23

echo "holding until ${START_HOUR}:00 local. now $(date '+%Y-%m-%d %H:%M %Z')"
while true; do
  h=$(date +%H)
  # Fire from 23:00 up to 05:59, so a wake-up after midnight still runs.
  if [ "$h" -ge "$START_HOUR" ] || [ "$h" -lt 6 ]; then break; fi
  sleep 300
done
echo "released at $(date '+%Y-%m-%d %H:%M %Z')"

# Never start on top of another project's training: that deadlocked both
# processes twice on this 16 GB machine.
while pgrep -f "train_rccr.py" >/dev/null 2>&1 \
   || pgrep -f "scripts.run_local" >/dev/null 2>&1; do
  echo "waiting: rccr-genomics training is active $(date '+%H:%M')"
  sleep 300
done

exec ./scripts/experiments/run_retrain_asym.sh
