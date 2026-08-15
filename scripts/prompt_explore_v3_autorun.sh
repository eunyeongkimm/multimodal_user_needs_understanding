#!/bin/bash
cd "/Users/eunyeongkim/Desktop/skku-ads/1.논문"
source venv/bin/activate
LOG="outputs/prompt_explore_v3_autorun.log"
while true; do
  python scripts/prompt_explore_v3_check.py >> "$LOG" 2>&1
  if grep -q "ALL_CHUNKS_COMPLETE\|ALL_CHUNKS_DONE_WITH_FAILURES" "$LOG"; then
    echo "AUTORUN_STOPPED_$(date +%s)" >> "$LOG"
    break
  fi
  sleep 30
done
