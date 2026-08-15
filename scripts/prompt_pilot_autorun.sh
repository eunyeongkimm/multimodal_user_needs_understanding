#!/bin/bash
# prompt_pilot_check.py를 반복 폴링하며 동시-제출 파이프라인(최대 8개 안팎 청크
# 동시 in-flight, 완료되는 대로 즉시 backfill)을 굴린다. 모든 청크가
# 터미널 상태에 도달하면 자동으로 prompt_pilot_build_excel.py(비교표 출력)까지 실행한다.

cd "$(dirname "$0")/.." || exit 1
source venv/bin/activate

LOG="outputs/prompt_pilot_autorun.log"
POLL_INTERVAL=90  # seconds - 동시 여러 청크가 끝날 수 있어 기존보다 짧게

echo "$(date '+%Y-%m-%d %H:%M:%S') stage2d autorun started (pid $$)" >> "$LOG"

while true; do
  OUT=$(python3 scripts/prompt_pilot_check.py 2>&1)
  echo "$(date '+%Y-%m-%d %H:%M:%S') $OUT" >> "$LOG"

  if echo "$OUT" | grep -qE "ALL_CHUNKS_COMPLETE|ALL_CHUNKS_DONE_WITH_FAILURES"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') 프롬프트 파일럿 GPT 완료 - 엑셀 집계 자동 진행" >> "$LOG"

    echo "$(date '+%Y-%m-%d %H:%M:%S') === prompt_pilot_build_excel.py 시작 ===" >> "$LOG"
    python3 scripts/prompt_pilot_build_excel.py >> "$LOG" 2>&1
    if [ $? -ne 0 ]; then
      echo "$(date '+%Y-%m-%d %H:%M:%S') 분석 실패. 사람 확인 필요." >> "$LOG"
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') ALL_STEPS_COMPLETE stage2d autorun finished, loop stopping" >> "$LOG"
    break
  fi

  if echo "$OUT" | grep -qE "Traceback|ERROR: OPENAI_API_KEY"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') stage2d autorun stopped: script crashed / config error, needs human check" >> "$LOG"
    break
  fi

  sleep "$POLL_INTERVAL"
done
