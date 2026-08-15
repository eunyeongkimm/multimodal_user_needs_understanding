"""batch1(19,848콜)을 조직의 Batch 큐 토큰 한도(gpt-4.1-mini 2,000,000 토큰)를
넘지 않도록 여러 청크로 나눈다. 토큰 예산 기준 greedy bin-packing으로 청크를 만들고,
outputs/batch1_chunks_plan.json에 {chunk_index: [call_id, ...]} 형태로 저장한다.

한 번 저장되면 재실행해도 덮어쓰지 않는다 (청크 경계가 바뀌면 이후 상태 추적이 꼬이므로).
"""

import json
import sys
from pathlib import Path

import pandas as pd
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relabel_v2_and_kappa import make_prompt

BASE_DIR = Path(__file__).resolve().parent.parent
CALL_TEXTS_PATH = BASE_DIR / "outputs" / "batch1_call_texts.parquet"
PLAN_PATH = BASE_DIR / "outputs" / "batch1_chunks_plan.json"

TOKEN_BUDGET_PER_CHUNK = 1_500_000  # 조직 한도 2,000,000 대비 안전 마진


def main():
    if PLAN_PATH.exists():
        with open(PLAN_PATH, "r", encoding="utf-8") as f:
            plan = json.load(f)
        print(f"이미 청크 계획이 존재합니다: {PLAN_PATH} ({len(plan)}개 청크). 재생성하지 않습니다.")
        return

    call_texts = pd.read_parquet(CALL_TEXTS_PATH)
    enc = tiktoken.get_encoding("o200k_base")
    call_texts = call_texts.copy()
    call_texts["tokens"] = call_texts["call_text"].apply(lambda t: len(enc.encode(make_prompt(t))))

    chunks = []
    current_ids = []
    current_tokens = 0
    for row in call_texts.itertuples():
        if current_ids and current_tokens + row.tokens > TOKEN_BUDGET_PER_CHUNK:
            chunks.append(current_ids)
            current_ids = []
            current_tokens = 0
        current_ids.append(row.call_id)
        current_tokens += row.tokens
    if current_ids:
        chunks.append(current_ids)

    with open(PLAN_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    total = sum(len(c) for c in chunks)
    print(f"청크 {len(chunks)}개 생성, 총 {total}콜 (콜당 청크 크기: {[len(c) for c in chunks]})")
    print(f"저장: {PLAN_PATH}")


if __name__ == "__main__":
    main()
