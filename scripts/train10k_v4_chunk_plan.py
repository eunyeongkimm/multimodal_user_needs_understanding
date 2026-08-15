"""train10k v2 라벨 중 환불요청/배송확인인 콜만 v4(항의 게이트) 프롬프트로 재라벨링
하기 위한 요청 생성. v4_chunk_plan.py와 동일 로직, 대상만 train10k로 교체.

출력: outputs/train10k_v4_request_index.parquet, outputs/train10k_v4_requests_plan.json
"""

import json
from pathlib import Path

import pandas as pd
import tiktoken

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v4_prompt import make_prompt

BASE_DIR = Path(__file__).resolve().parent.parent
V2_PATH = BASE_DIR / "outputs" / "train10k_gpt_labels.parquet"
CALL_TEXTS_PATH = BASE_DIR / "outputs" / "train10k_call_texts.parquet"
TARGET_IDS_PATH = BASE_DIR / "outputs" / "train10k_v4_target_call_ids.csv"
REQUEST_INDEX_PATH = BASE_DIR / "outputs" / "train10k_v4_request_index.parquet"
PLAN_PATH = BASE_DIR / "outputs" / "train10k_v4_requests_plan.json"

CHUNK_TOKEN_BUDGET = 200_000
PRICE_INPUT_PER_M = 0.40
PRICE_OUTPUT_PER_M = 1.60
EST_OUTPUT_TOKENS_PER_CALL = 20


def main():
    v2 = pd.read_parquet(V2_PATH).rename(columns={"gold_actual": "label"})
    target = v2[v2["label"].isin(["환불요청", "배송확인"])][["call_id", "label"]]
    target.to_csv(TARGET_IDS_PATH, index=False)
    print(f"대상 콜: {len(target)}건 (환불요청 {(target['label']=='환불요청').sum()}건, "
          f"배송확인 {(target['label']=='배송확인').sum()}건)")

    call_texts = pd.read_parquet(CALL_TEXTS_PATH)
    subset = target.merge(call_texts, on="call_id", how="left")
    assert subset["call_text"].isna().sum() == 0, "call_text 누락된 call_id가 있습니다."

    enc = tiktoken.get_encoding("o200k_base")
    subset = subset.copy()
    subset["prompt"] = subset["call_text"].apply(make_prompt)
    subset["tokens"] = subset["prompt"].apply(lambda p: len(enc.encode(p)))

    idx = subset[["call_id", "prompt", "tokens"]].rename(columns={"call_id": "custom_id"})
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
