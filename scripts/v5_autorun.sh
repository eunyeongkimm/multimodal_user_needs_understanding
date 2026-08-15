#!/bin/bash
# v5_check.py를 반복 폴링해 청크 완료를 감지하고, 모든 청크가 끝나면
# 자동으로 v5_merge_final.py(병합 + v2/v4/v5 비교표 출력)까지 실행한다.

cd "$(dirname "$0")/.." || exit 1
source venv/bin/activate

LOG="outputs/v5_autorun.log"
POLL_INTERVAL=120  # seconds

echo "$(date '+%Y-%m-%d %H:%M:%S') v5 autorun started (pid $$)" >> "$LOG"

while true; do
  OUT=$(python3 scripts/v5_check.py 2>&1)
  echo "$(date '+%Y-%m-%d %H:%M:%S') $OUT" >> "$LOG"

  if echo "$OUT" | grep -qE "ALL_CHUNKS_COMPLETE|ALL_CHUNKS_DONE_WITH_FAILURES"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') 재분류 완료 - merge 자동 진행" >> "$LOG"

    echo "$(date '+%Y-%m-%d %H:%M:%S') === v5_merge_final.py 시작 ===" >> "$LOG"
    python3 scripts/v5_merge_final.py >> "$LOG" 2>&1
    if [ $? -ne 0 ]; then
      echo "$(date '+%Y-%m-%d %H:%M:%S') merge 실패. 사람 확인 필요." >> "$LOG"
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') ALL_STEPS_COMPLETE v5 autorun finished, loop stopping" >> "$LOG"
    break
  fi

  if echo "$OUT" | grep -qE "Traceback|ERROR: OPENAI_API_KEY"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') v5 autorun stopped: script crashed / config error, needs human check" >> "$LOG"
    break
  fi

  sleep "$POLL_INTERVAL"
done
