"""gold_actual_batch1_v4.parquet에서 has_complaint=False인 콜(11,127건)만
v5 프롬프트(순수 6개 카테고리 분류, 게이트 없음)로 재분류하기 위해 청크로 나누고
예상 비용을 미리 계산한다. has_complaint=True(452건)는 이미 '불만제기'로 확정된
상태라 이 파이프라인 대상에서 제외한다.

한 번 저장되면 재실행해도 덮어쓰지 않는다.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v5_prompt import make_prompt

BASE_DIR = Path(__file__).resolve().parent.parent
V4_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_v4.parquet"
CALL_TEXTS_PATH = BASE_DIR / "outputs" / "batch1_call_texts.parquet"
TARGET_IDS_PATH = BASE_DIR / "outputs" / "v5_target_call_ids.csv"
PLAN_PATH = BASE_DIR / "outputs" / "v5_chunks_plan.json"

TOKEN_BUDGET_PER_CHUNK = 1_500_000

PRICE_INPUT_PER_M = 0.40
PRICE_OUTPUT_PER_M = 1.60
EST_OUTPUT_TOKENS_PER_CALL = 10  # 카테고리명만 출력하므로 v4보다 짧음


def build_target(enc):
    v4 = pd.read_parquet(V4_PATH)
    target = v4[v4["has_complaint"] == False][["call_id"]]  # noqa: E712
    target.to_csv(TARGET_IDS_PATH, index=False)
    print(f"대상 콜(has_complaint=False): {len(target)}건")

    call_texts = pd.read_parquet(CALL_TEXTS_PATH)
    subset = target.merge(call_texts, on="call_id", how="left")
    assert subset["call_text"].isna().sum() == 0, "call_text 누락된 call_id가 있습니다."

    subset = subset.copy()
    subset["tokens"] = subset["call_text"].apply(lambda t: len(enc.encode(make_prompt(t))))
    return subset


def print_cost_estimate(subset: pd.DataFrame):
    n = len(subset)
    total_input_tokens = subset["tokens"].sum()
    total_output_tokens = n * EST_OUTPUT_TOKENS_PER_CALL

    in_cost = total_input_tokens / 1_000_000 * PRICE_INPUT_PER_M * 0.5  # Batch API 50% 할인
    out_cost = total_output_tokens / 1_000_000 * PRICE_OUTPUT_PER_M * 0.5
    total = in_cost + out_cost

    print(f"\n=== 예상 비용 (gpt-4.1-mini, Batch API 50% 할인, {n:,}건) ===")
    print(f"input {total_input_tokens:,} tokens (${in_cost:.4f}) + "
          f"output 추정 {total_output_tokens:,} tokens (${out_cost:.4f}) = ${total:.4f}\n")


def main():
    enc = tiktoken.get_encoding("o200k_base")

    if PLAN_PATH.exists():
        with open(PLAN_PATH, "r", encoding="utf-8") as f:
            plan = json.load(f)
        print(f"이미 청크 계획이 존재합니다: {PLAN_PATH} ({len(plan)}개 청크). 재생성하지 않습니다.")
        subset = build_target(enc)
        print_cost_estimate(subset)
        return

    subset = build_target(enc)
    print_cost_estimate(subset)

    chunks = []
    current_ids = []
    current_tokens = 0
    for row in subset.itertuples():
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
    print(f"청크 {len(chunks)}개 생성, 총 {total}콜 (청크당 크기: {[len(c) for c in chunks]})")
    print(f"저장: {PLAN_PATH}")


if __name__ == "__main__":
    main()
