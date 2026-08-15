"""v3 변형 4개(v3_orig/v3a/v3b/v3c) 탐색 - 샘플 B(200, 자연분포 mismatch) 재사용.
v_base와 v3_orig(=기존 v3, 파싱버그 수정 후 재파싱된 결과)는 이미 있으므로 재사용하고,
v3a/v3b/v3c만 새로 호출한다.

출력: outputs/prompt_explore_v3_request_index.parquet, outputs/prompt_explore_v3_requests_plan.json
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
EXISTING_EXPLORE_PRED_PATH = BASE_DIR / "outputs" / "prompt_explore_gpt_predictions.parquet"
REQUEST_INDEX_PATH = BASE_DIR / "outputs" / "prompt_explore_v3_request_index.parquet"
PLAN_PATH = BASE_DIR / "outputs" / "prompt_explore_v3_requests_plan.json"
REUSE_PATH = BASE_DIR / "outputs" / "prompt_explore_v3_reused.parquet"

WINDOW_N = 2
WINDOW_VERSION = "customer_only"
NEW_VERSIONS = ["v3a", "v3b", "v3c"]
CHUNK_TOKEN_BUDGET = 200_000

PRICE_INPUT_PER_M = 0.40
PRICE_OUTPUT_PER_M = 1.60
EST_OUTPUT_TOKENS_PER_CALL = 60


def main():
    sample_b = pd.read_csv(SAMPLE_B_PATH)
    windows = pd.read_parquet(WINDOWS_NL_PATH)
    win_sub = windows[(windows["window_n"] == WINDOW_N) & (windows["window_version"] == WINDOW_VERSION)]

    # === v_base, v3_orig 재사용 ===
    pilot_existing = pd.read_parquet(EXISTING_PILOT_PRED_PATH)
    base_existing = pilot_existing[(pilot_existing["sample"] == "B") & (pilot_existing["version"] == "base")]
    base_existing = base_existing[base_existing["call_id"].isin(sample_b["call_id"])]
    print(f"기존 v_base 결과: {len(base_existing)}건 (샘플B {len(sample_b)}건 중) -> 재사용")

    explore_existing = pd.read_parquet(EXISTING_EXPLORE_PRED_PATH)
    v3_existing = explore_existing[explore_existing["version"] == "v3"]
    v3_existing = v3_existing[v3_existing["call_id"].isin(sample_b["call_id"])]
    n_v3_fail = v3_existing["parse_error"].notna().sum()
    print(f"기존 v3(=v3_orig) 결과: {len(v3_existing)}건 -> 재사용 (파싱실패 {n_v3_fail}건 포함)")

    reused = base_existing[["call_id", "predicted_label"]].rename(columns={"predicted_label": "base"})
    reused = reused.merge(
        v3_existing[["call_id", "predicted_label", "parse_error"]].rename(
            columns={"predicted_label": "v3_orig", "parse_error": "v3_orig_parse_error"}),
        on="call_id", how="outer",
    )
    reused.to_parquet(REUSE_PATH, index=False)

    missing_base = set(sample_b["call_id"]) - set(base_existing["call_id"])
    missing_v3 = set(sample_b["call_id"]) - set(v3_existing["call_id"])
    if missing_base:
        print(f"경고: v_base 없는 콜 {len(missing_base)}건")
    if missing_v3:
        print(f"경고: v3_orig 없는 콜 {len(missing_v3)}건")

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
    print(f"저장: {PLAN_PATH}, {REQUEST_INDEX_PATH}, {REUSE_PATH}")


if __name__ == "__main__":
    main()
