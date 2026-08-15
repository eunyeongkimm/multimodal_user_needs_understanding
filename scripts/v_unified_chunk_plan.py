"""v_unified 프롬프트로 층화 샘플 2,000건 라벨링 요청 생성.
입력 텍스트는 v2/v4/v5가 쓰던 것과 동일한 batch1_call_texts.parquet.

출력: outputs/v_unified_request_index.parquet, outputs/v_unified_requests_plan.json
"""

import json
from pathlib import Path

import pandas as pd
import tiktoken

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v_unified_prompt import make_prompt

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_PATH = BASE_DIR / "outputs" / "v_unified_sample_call_ids.csv"
CALL_TEXTS_PATH = BASE_DIR / "outputs" / "batch1_call_texts.parquet"
REQUEST_INDEX_PATH = BASE_DIR / "outputs" / "v_unified_request_index.parquet"
PLAN_PATH = BASE_DIR / "outputs" / "v_unified_requests_plan.json"

CHUNK_TOKEN_BUDGET = 200_000
PRICE_INPUT_PER_M = 0.40
PRICE_OUTPUT_PER_M = 1.60
EST_OUTPUT_TOKENS_PER_CALL = 10


def main():
    sample = pd.read_csv(SAMPLE_PATH)
    call_texts = pd.read_parquet(CALL_TEXTS_PATH)
    subset = sample.merge(call_texts, on="call_id", how="left")
    n_missing = subset["call_text"].isna().sum()
    print(f"샘플 {len(sample)}건, call_text 누락: {n_missing}건")
    assert n_missing == 0, "call_text 누락된 call_id가 있습니다."

    enc = tiktoken.get_encoding("o200k_base")
    subset = subset.copy()
    subset["prompt"] = subset["call_text"].apply(make_prompt)
    subset["tokens"] = subset["prompt"].apply(lambda p: len(enc.encode(p)))

    idx = subset[["call_id", "prompt", "tokens", "gold_actual"]].rename(columns={"call_id": "custom_id"})
    idx["call_id"] = subset["call_id"]
    idx.to_parquet(REQUEST_INDEX_PATH, index=False)

    n = len(idx)
    total_input = idx["tokens"].sum()
    total_output = n * EST_OUTPUT_TOKENS_PER_CALL
    in_cost = total_input / 1_000_000 * PRICE_INPUT_PER_M * 0.5
    out_cost = total_output / 1_000_000 * PRICE_OUTPUT_PER_M * 0.5
    print(f"\n=== 예상 비용 (gpt-4.1-mini, Batch API 50% 할인, {n:,}건) ===")
    print(f"input {total_input:,} tokens (${in_cost:.4f}) + output 추정 {total_output:,} tokens (${out_cost:.4f}) "
          f"= ${in_cost+out_cost:.4f}")

    chunks = []
    current_ids, current_tokens = [], 0
    for row in idx.itertuples():
        if current_ids and current_tokens + row.tokens > CHUNK_TOKEN_BUDGET:
            chunks.append(current_ids)
            current_ids, current_tokens = [], 0
        current_ids.append(row.custom_id)
        current_tokens += row.tokens
    if current_ids:
        chunks.append(current_ids)

    with open(PLAN_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    print(f"\n청크 {len(chunks)}개 생성")
    print(f"저장: {PLAN_PATH}, {REQUEST_INDEX_PATH}")


if __name__ == "__main__":
    main()
