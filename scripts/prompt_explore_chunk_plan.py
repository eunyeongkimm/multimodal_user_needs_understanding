"""프롬프트 4버전(base/v2/v3/v4) 탐색 - 샘플 B(200, 자연분포 mismatch) 재사용.
v_base는 outputs/prompt_pilot_gpt_predictions.parquet(sample=='B', version=='base')에
이미 있으면 그대로 재사용하고, v2/v3/v4만 새로 호출한다.

출력: outputs/prompt_explore_request_index.parquet, outputs/prompt_explore_requests_plan.json
"""

import json
from pathlib import Path

import pandas as pd
import tiktoken

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2d_prompt import make_prompt

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_B_PATH = BASE_DIR / "outputs" / "prompt_pilot_sample_b.csv"
WINDOWS_NL_PATH = BASE_DIR / "outputs" / "stage2_windows_nl_v2.parquet"
EXISTING_PILOT_PRED_PATH = BASE_DIR / "outputs" / "prompt_pilot_gpt_predictions.parquet"
REQUEST_INDEX_PATH = BASE_DIR / "outputs" / "prompt_explore_request_index.parquet"
PLAN_PATH = BASE_DIR / "outputs" / "prompt_explore_requests_plan.json"
BASE_REUSE_PATH = BASE_DIR / "outputs" / "prompt_explore_base_reused.parquet"

WINDOW_N = 2
WINDOW_VERSION = "customer_only"
NEW_VERSIONS = ["v2", "v3", "v4"]
CHUNK_TOKEN_BUDGET = 200_000

PRICE_INPUT_PER_M = 0.40
PRICE_OUTPUT_PER_M = 1.60
EST_OUTPUT_TOKENS_PER_CALL = 60  # v3/v4는 분석 텍스트 포함이라 base보다 김


def main():
    sample_b = pd.read_csv(SAMPLE_B_PATH)
    windows = pd.read_parquet(WINDOWS_NL_PATH)
    win_sub = windows[(windows["window_n"] == WINDOW_N) & (windows["window_version"] == WINDOW_VERSION)]

    # === v_base 기존 결과 확인 ===
    if EXISTING_PILOT_PRED_PATH.exists():
        existing = pd.read_parquet(EXISTING_PILOT_PRED_PATH)
        base_existing = existing[(existing["sample"] == "B") & (existing["version"] == "base")]
        base_existing = base_existing[base_existing["call_id"].isin(sample_b["call_id"])]
        print(f"기존 v_base 결과 발견: {len(base_existing)}건 (샘플B {len(sample_b)}건 중) -> 재사용")
    else:
        base_existing = pd.DataFrame(columns=["call_id", "predicted_label"])
        print("경고: 기존 prompt_pilot_gpt_predictions.parquet 없음 -> v_base도 새로 호출 필요")

    base_reused = base_existing[["call_id", "predicted_label"]].rename(columns={"predicted_label": "base"})
    base_reused.to_parquet(BASE_REUSE_PATH, index=False)

    missing_base_ids = set(sample_b["call_id"]) - set(base_existing["call_id"])
    if missing_base_ids:
        print(f"v_base 없는 콜 {len(missing_base_ids)}건 발견 -> 이번엔 v2/v3/v4만 돌리므로 "
              f"이 콜들의 base는 결측으로 남을 수 있음(별도 처리 필요시 알려주세요)")

    enc = tiktoken.get_encoding("o200k_base")
    rows = []
    for call_id in sample_b["call_id"]:
        g = win_sub[win_sub["call_id"] == call_id].sort_values("dialog_idx")
        utt_rows = g[["speaker_group", "text_clean", "nl_description"]].to_dict("records")
        if not utt_rows:
            print(f"경고: {call_id} 윈도우 없음 -> 스킵")
            continue
        for version in NEW_VERSIONS:
            prompt = make_prompt(utt_rows, with_acoustic=True, version=version)
            tokens = len(enc.encode(prompt))
            custom_id = f"{call_id}::{version}"
            rows.append({"custom_id": custom_id, "call_id": call_id, "version": version,
                         "prompt": prompt, "tokens": tokens})

    idx = pd.DataFrame(rows)
    idx.to_parquet(REQUEST_INDEX_PATH, index=False)

    print(f"\n신규 요청 수: {len(idx)}건")
    print(idx.groupby("version").size())

    total_input = idx["tokens"].sum()
    total_output = len(idx) * EST_OUTPUT_TOKENS_PER_CALL
    in_cost = total_input / 1_000_000 * PRICE_INPUT_PER_M * 0.5
    out_cost = total_output / 1_000_000 * PRICE_OUTPUT_PER_M * 0.5
    print(f"\n=== 예상 비용 (Batch API 50% 할인) ===")
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
    print(f"저장: {PLAN_PATH}, {REQUEST_INDEX_PATH}, {BASE_REUSE_PATH}")


if __name__ == "__main__":
    main()
