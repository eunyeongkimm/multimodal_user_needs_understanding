#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
source venv/bin/activate

LOG="outputs/stage2_bd_v3_autorun.log"
POLL_INTERVAL=90

echo "$(date '+%Y-%m-%d %H:%M:%S') stage2_bd_v3 autorun started (pid $$)" >> "$LOG"

while true; do
  OUT=$(python3 scripts/stage2_bd_v3_check.py 2>&1)
  echo "$(date '+%Y-%m-%d %H:%M:%S') $OUT" >> "$LOG"

  if echo "$OUT" | grep -qE "ALL_CHUNKS_COMPLETE|ALL_CHUNKS_DONE_WITH_FAILURES"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') AUTORUN_STOPPED" >> "$LOG"
    break
  fi

  if echo "$OUT" | grep -qE "Traceback|ERROR: OPENAI_API_KEY"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') stage2_bd_v3 autorun stopped: crash/config error, needs human check" >> "$LOG"
    break
  fi

  sleep "$POLL_INTERVAL"
done
