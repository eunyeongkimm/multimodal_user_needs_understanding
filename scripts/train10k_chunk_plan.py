"""train10k 콜에 v2 프롬프트(기존 gold_actual과 100% 동일, relabel_v2_and_kappa.py의
PROMPT_TEMPLATE_V2 - 통화 전체 텍스트, 7개 카테고리)를 적용하기 위한 요청 생성.
--test 플래그를 주면 10건만 뽑아 스모크 테스트용 계획을 만든다(STEP1 패치 검증용).

출력(기본): outputs/train10k_requests_plan.json, outputs/train10k_request_index.parquet
출력(--test): outputs/train10k_test10_requests_plan.json, outputs/train10k_test10_request_index.parquet
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import tiktoken

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from relabel_v2_and_kappa import make_prompt

BASE_DIR = Path(__file__).resolve().parent.parent
CALL_TEXTS_PATH = BASE_DIR / "outputs" / "train10k_call_texts.parquet"

CHUNK_TOKEN_BUDGET = 200_000
PRICE_INPUT_PER_M = 0.40
PRICE_OUTPUT_PER_M = 1.60
EST_OUTPUT_TOKENS_PER_CALL = 8


def build(call_texts: pd.DataFrame, enc) -> pd.DataFrame:
    rows = []
    for row in call_texts.itertuples():
        prompt = make_prompt(row.call_text)
        tokens = len(enc.encode(prompt))
        rows.append({"custom_id": row.call_id, "call_id": row.call_id, "prompt": prompt, "tokens": tokens})
    return pd.DataFrame(rows)


def print_cost(idx: pd.DataFrame):
    n = len(idx)
    total_input = idx["tokens"].sum()
    total_output = n * EST_OUTPUT_TOKENS_PER_CALL
    in_cost = total_input / 1_000_000 * PRICE_INPUT_PER_M * 0.5
    out_cost = total_output / 1_000_000 * PRICE_OUTPUT_PER_M * 0.5
    print(f"=== 예상 비용 (gpt-4.1-mini, Batch API 50% 할인, {n:,}건) ===")
    print(f"input {total_input:,} tokens (${in_cost:.4f}) + output 추정 {total_output:,} tokens (${out_cost:.4f}) "
          f"= ${in_cost+out_cost:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="10건만 뽑아 스모크 테스트 계획 생성")
    args = parser.parse_args()

    call_texts = pd.read_parquet(CALL_TEXTS_PATH)
    if args.test:
        call_texts = call_texts.head(10)
        plan_path = BASE_DIR / "outputs" / "train10k_test10_requests_plan.json"
        idx_path = BASE_DIR / "outputs" / "train10k_test10_request_index.parquet"
    else:
        plan_path = BASE_DIR / "outputs" / "train10k_requests_plan.json"
        idx_path = BASE_DIR / "outputs" / "train10k_request_index.parquet"

    enc = tiktoken.get_encoding("o200k_base")
    idx = build(call_texts, enc)
    idx.to_parquet(idx_path, index=False)
    print(f"요청 {len(idx)}건 생성")
    print_cost(idx)

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

    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    print(f"청크 {len(chunks)}개 생성")
    print(f"저장: {plan_path}, {idx_path}")


if __name__ == "__main__":
    main()
