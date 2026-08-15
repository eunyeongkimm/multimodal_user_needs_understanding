#!/bin/bash
# v3_check.py를 주기적으로 실행해 청크 완료를 감지하고 즉시 다음 청크를
# 제출하는 자동화 루프 (batch1_autorun.sh와 동일한 패턴).
#
# 개별 청크의 실패/재시도/스킵은 v3_check.py 안에서 처리되므로
# (전체 파이프라인을 막지 않음), 이 루프는 아래 두 경우에만 스스로 멈춘다:
#   1) ALL_CHUNKS_COMPLETE / ALL_CHUNKS_DONE_WITH_FAILURES - 모든 청크가
#      터미널 상태(완료 또는 재시도 소진)에 도달해 더 할 일이 없음.
#   2) 스크립트 자체가 크래시(Traceback)했거나 API 키가 없는 등,
#      재시도로 해결되지 않는 설정 문제 - 사람 확인 필요.

cd "$(dirname "$0")/.." || exit 1
source venv/bin/activate

LOG="outputs/v3_autorun.log"
POLL_INTERVAL=120  # seconds

echo "$(date '+%Y-%m-%d %H:%M:%S') v3 autorun started (pid $$)" >> "$LOG"

while true; do
  OUT=$(python3 scripts/v3_check.py 2>&1)
  echo "$(date '+%Y-%m-%d %H:%M:%S') $OUT" >> "$LOG"

  if echo "$OUT" | grep -qE "ALL_CHUNKS_COMPLETE|ALL_CHUNKS_DONE_WITH_FAILURES"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') v3 autorun finished, loop stopping (see log above for success/failure detail)" >> "$LOG"
    break
  fi

  if echo "$OUT" | grep -qE "Traceback|ERROR: OPENAI_API_KEY"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') v3 autorun stopped: script crashed / config error, needs human check" >> "$LOG"
    break
  fi

  sleep "$POLL_INTERVAL"
done
