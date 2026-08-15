#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
source venv/bin/activate

LOG="outputs/v_unified_autorun.log"
POLL_INTERVAL=60

echo "$(date '+%Y-%m-%d %H:%M:%S') v_unified autorun started (pid $$)" >> "$LOG"

while true; do
  OUT=$(python3 scripts/v_unified_check.py 2>&1)
  echo "$(date '+%Y-%m-%d %H:%M:%S') $OUT" >> "$LOG"

  if echo "$OUT" | grep -qE "ALL_CHUNKS_COMPLETE|ALL_CHUNKS_DONE_WITH_FAILURES"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') AUTORUN_STOPPED" >> "$LOG"
    break
  fi
  if echo "$OUT" | grep -qE "Traceback|ERROR: OPENAI_API_KEY"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') crash/config error, needs human check" >> "$LOG"
    break
  fi
  sleep "$POLL_INTERVAL"
done
