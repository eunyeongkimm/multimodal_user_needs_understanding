#!/bin/bash
# stage2m_check.py(B/D x N=2,3,5, 성별 재정규화 버전)를 동시-제출 파이프라인으로
# 반복 폴링하고, 모든 청크가 끝나면 자동으로 before/after 병합 + 분석까지 실행한다.

cd "$(dirname "$0")/.." || exit 1
source venv/bin/activate

LOG="outputs/stage2m_autorun.log"
POLL_INTERVAL=90

echo "$(date '+%Y-%m-%d %H:%M:%S') stage2m autorun started (pid $$)" >> "$LOG"

while true; do
  OUT=$(python3 scripts/stage2m_check.py 2>&1)
  echo "$(date '+%Y-%m-%d %H:%M:%S') $OUT" >> "$LOG"

  if echo "$OUT" | grep -qE "ALL_CHUNKS_COMPLETE|ALL_CHUNKS_DONE_WITH_FAILURES"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') B/D 재실행 완료 - before/after 분석 자동 진행" >> "$LOG"

    echo "$(date '+%Y-%m-%d %H:%M:%S') === stage2n_before_after_analysis.py 시작 ===" >> "$LOG"
    python3 scripts/stage2n_before_after_analysis.py >> "$LOG" 2>&1
    if [ $? -ne 0 ]; then
      echo "$(date '+%Y-%m-%d %H:%M:%S') 분석 실패. 사람 확인 필요." >> "$LOG"
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') ALL_STEPS_COMPLETE stage2m autorun finished, loop stopping" >> "$LOG"
    break
  fi

  if echo "$OUT" | grep -qE "Traceback|ERROR: OPENAI_API_KEY"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') stage2m autorun stopped: script crashed / config error, needs human check" >> "$LOG"
    break
  fi

  sleep "$POLL_INTERVAL"
done
