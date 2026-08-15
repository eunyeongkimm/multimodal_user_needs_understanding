#!/bin/bash
# 1단계(v4_check.py 반복 폴링)를 자동으로 끝까지 돌리고, 완료되면 이어서
# 2단계(merge), 3단계(self-consistency), 4단계(suspect scan)까지 자동 실행한다.
#
# 1단계 개별 청크의 실패/재시도/스킵은 v4_check.py 안에서 처리되므로
# (전체 파이프라인을 막지 않음), 이 루프는 아래 두 경우에만 스스로 멈춘다:
#   1) ALL_CHUNKS_COMPLETE / ALL_CHUNKS_DONE_WITH_FAILURES 감지 시
#      -> merge/self-consistency/suspect-scan을 순서대로 실행하고 종료.
#   2) 스크립트 자체가 크래시(Traceback)했거나 API 키가 없는 등,
#      재시도로 해결되지 않는 설정 문제 - 사람 확인 필요, 이후 단계는 실행 안 함.

cd "$(dirname "$0")/.." || exit 1
source venv/bin/activate

LOG="outputs/v4_autorun.log"
POLL_INTERVAL=120  # seconds

echo "$(date '+%Y-%m-%d %H:%M:%S') v4 autorun started (pid $$)" >> "$LOG"

while true; do
  OUT=$(python3 scripts/v4_check.py 2>&1)
  echo "$(date '+%Y-%m-%d %H:%M:%S') $OUT" >> "$LOG"

  if echo "$OUT" | grep -qE "ALL_CHUNKS_COMPLETE|ALL_CHUNKS_DONE_WITH_FAILURES"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') 1단계(재라벨링) 완료 - 2,3단계 자동 진행 시작" >> "$LOG"

    echo "$(date '+%Y-%m-%d %H:%M:%S') === v4_merge_final.py 시작 ===" >> "$LOG"
    python3 scripts/v4_merge_final.py >> "$LOG" 2>&1
    if [ $? -ne 0 ]; then
      echo "$(date '+%Y-%m-%d %H:%M:%S') merge 실패, 이후 단계 중단. 사람 확인 필요." >> "$LOG"
      break
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') === v4_self_consistency.py 시작 ===" >> "$LOG"
    python3 scripts/v4_self_consistency.py >> "$LOG" 2>&1
    if [ $? -ne 0 ]; then
      echo "$(date '+%Y-%m-%d %H:%M:%S') self-consistency 실패. 사람 확인 필요 (suspect scan은 계속 진행)." >> "$LOG"
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') === v4_suspect_scan.py 시작 ===" >> "$LOG"
    python3 scripts/v4_suspect_scan.py >> "$LOG" 2>&1
    if [ $? -ne 0 ]; then
      echo "$(date '+%Y-%m-%d %H:%M:%S') suspect scan 실패. 사람 확인 필요." >> "$LOG"
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') ALL_STEPS_COMPLETE v4 autorun finished, loop stopping" >> "$LOG"
    break
  fi

  if echo "$OUT" | grep -qE "Traceback|ERROR: OPENAI_API_KEY"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') v4 autorun stopped: script crashed / config error, needs human check" >> "$LOG"
    break
  fi

  sleep "$POLL_INTERVAL"
done
