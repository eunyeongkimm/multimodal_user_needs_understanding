"""gold_actual_batch1.parquet에서 label이 환불요청/배송확인인 콜(11,579건)을
v4 프롬프트(항의 게이트 2단계 판단 + JSON 출력)로 재라벨링하기 위해 청크로 나누고,
예상 비용을 미리 계산해서 출력한다.

한 번 저장되면 재실행해도 덮어쓰지 않는다 (청크 경계가 바뀌면 이후 상태 추적이 꼬이므로).
"""

import json
import sys
from pathlib import Path

import pandas as pd
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v4_prompt import make_prompt

BASE_DIR = Path(__file__).resolve().parent.parent
GOLD_V2_PATH = BASE_DIR / "outputs" / "gold_actual_batch1.parquet"
CALL_TEXTS_PATH = BASE_DIR / "outputs" / "batch1_call_texts.parquet"
TARGET_IDS_PATH = BASE_DIR / "outputs" / "v4_target_call_ids.csv"
PLAN_PATH = BASE_DIR / "outputs" / "v4_chunks_plan.json"

TOKEN_BUDGET_PER_CHUNK = 1_500_000  # 조직 한도 2,000,000 대비 안전 마진

# gpt-4.1-mini 표준(동기) 단가. OpenAI Batch API는 이 단가의 50%.
PRICE_INPUT_PER_M = 0.40
PRICE_OUTPUT_PER_M = 1.60
EST_OUTPUT_TOKENS_PER_CALL = 20  # JSON 응답 {"has_complaint": ..., "label": "..."} 기준 여유있게 추정


def build_target(enc):
    gold_v2 = pd.read_parquet(GOLD_V2_PATH)
    target = gold_v2[gold_v2["label"].isin(["환불요청", "배송확인"])][["call_id", "label"]]
    target.to_csv(TARGET_IDS_PATH, index=False)
    print(f"대상 콜: {len(target)}건 (환불요청 {(target['label']=='환불요청').sum()}건, "
          f"배송확인 {(target['label']=='배송확인').sum()}건)")

    call_texts = pd.read_parquet(CALL_TEXTS_PATH)
    subset = target.merge(call_texts, on="call_id", how="left")
    assert subset["call_text"].isna().sum() == 0, "call_text 누락된 call_id가 있습니다."

    subset = subset.copy()
    subset["tokens"] = subset["call_text"].apply(lambda t: len(enc.encode(make_prompt(t))))
    return subset


def print_cost_estimate(subset: pd.DataFrame, n_self_consistency: int = 200):
    n = len(subset)
    total_input_tokens = subset["tokens"].sum()
    total_output_tokens = n * EST_OUTPUT_TOKENS_PER_CALL

    # 1단계 본 배치(Batch API, 50% 할인) + 2단계 자기일관성 재실행(200건, 동기 API, 정가)
    batch_in_cost = total_input_tokens / 1_000_000 * PRICE_INPUT_PER_M * 0.5
    batch_out_cost = total_output_tokens / 1_000_000 * PRICE_OUTPUT_PER_M * 0.5
    batch_total = batch_in_cost + batch_out_cost

    avg_tokens_per_call = total_input_tokens / n
    sc_input_tokens = avg_tokens_per_call * n_self_consistency
    sc_output_tokens = n_self_consistency * EST_OUTPUT_TOKENS_PER_CALL
    sc_in_cost = sc_input_tokens / 1_000_000 * PRICE_INPUT_PER_M
    sc_out_cost = sc_output_tokens / 1_000_000 * PRICE_OUTPUT_PER_M
    sc_total = sc_in_cost + sc_out_cost

    print(f"\n=== 예상 비용 (gpt-4.1-mini 기준) ===")
    print(f"1단계 본 배치 (Batch API, 정가 50% 할인, {n:,}건):")
    print(f"  input {total_input_tokens:,} tokens (${batch_in_cost:.4f}) + "
          f"output 추정 {total_output_tokens:,} tokens (${batch_out_cost:.4f}) = ${batch_total:.4f}")
    print(f"2단계 자기일관성 재실행 (동기 API, 정가, {n_self_consistency}건):")
    print(f"  input 추정 {sc_input_tokens:,.0f} tokens (${sc_in_cost:.4f}) + "
          f"output 추정 {sc_output_tokens:,} tokens (${sc_out_cost:.4f}) = ${sc_total:.4f}")
    print(f"합계 예상 비용: ${batch_total + sc_total:.4f}")
    print("(실제 청구는 출력 토큰 수 등에 따라 달라질 수 있음 - 위는 보수적 추정치)\n")


def main():
    enc = tiktoken.get_encoding("o200k_base")

    if PLAN_PATH.exists():
        with open(PLAN_PATH, "r", encoding="utf-8") as f:
            plan = json.load(f)
        print(f"이미 청크 계획이 존재합니다: {PLAN_PATH} ({len(plan)}개 청크). 재생성하지 않습니다.")
        subset = build_target(enc)  # 비용 재출력을 위해 다시 계산(대상 집합은 동일)
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
