"""프롬프트 파일럿(샘플A: base+kr_full, 샘플B: base+kr_slim) 요청 생성.
N=2, customer_only, Track A(stage2_windows_nl_v2) 음성서술 사용.
700건 전체가 청크 1개 토큰 예산(200,000) 안에 들어오면 1청크, 아니면 자동 분할.

출력: outputs/prompt_pilot_request_index.parquet, outputs/prompt_pilot_requests_plan.json
"""

import json
from pathlib import Path

import pandas as pd
import tiktoken

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2d_prompt import make_prompt

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_A_PATH = BASE_DIR / "outputs" / "prompt_pilot_sample_a.csv"
SAMPLE_B_PATH = BASE_DIR / "outputs" / "prompt_pilot_sample_b.csv"
WINDOWS_NL_PATH = BASE_DIR / "outputs" / "stage2_windows_nl_v2.parquet"
REQUEST_INDEX_PATH = BASE_DIR / "outputs" / "prompt_pilot_request_index.parquet"
PLAN_PATH = BASE_DIR / "outputs" / "prompt_pilot_requests_plan.json"

WINDOW_N = 2
WINDOW_VERSION = "customer_only"
CHUNK_TOKEN_BUDGET = 200_000

PRICE_INPUT_PER_M = 0.40
PRICE_OUTPUT_PER_M = 1.60
EST_OUTPUT_TOKENS_PER_CALL = 8


def build_utt_rows(windows, call_id):
    g = windows[windows["call_id"] == call_id].sort_values("dialog_idx")
    return g[["speaker_group", "text_clean", "nl_description"]].to_dict("records")


def main():
    sample_a = pd.read_csv(SAMPLE_A_PATH)
    sample_b = pd.read_csv(SAMPLE_B_PATH)
    windows = pd.read_parquet(WINDOWS_NL_PATH)
    win_sub = windows[(windows["window_n"] == WINDOW_N) & (windows["window_version"] == WINDOW_VERSION)]

    enc = tiktoken.get_encoding("o200k_base")
    rows = []

    plan_specs = [
        (sample_a, "A", ["base", "kr_full"]),
        (sample_b, "B", ["base", "kr_slim"]),
    ]
    for sample_df, sample_name, versions in plan_specs:
        for call_id in sample_df["call_id"]:
            utt_rows = build_utt_rows(win_sub, call_id)
            if not utt_rows:
                print(f"경고: {call_id}의 N=2 윈도우가 비어있음 -> 스킵")
                continue
            for version in versions:
                prompt = make_prompt(utt_rows, with_acoustic=True, version=version)
                tokens = len(enc.encode(prompt))
                custom_id = f"{call_id}::{sample_name}::{version}"
                rows.append({
                    "custom_id": custom_id, "call_id": call_id, "sample": sample_name,
                    "version": version, "prompt": prompt, "tokens": tokens,
                })

    idx = pd.DataFrame(rows)
    idx.to_parquet(REQUEST_INDEX_PATH, index=False)

    print(f"전체 요청 수: {len(idx)}건")
    print(idx.groupby(["sample", "version"]).size().unstack())

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
    print(f"저장: {PLAN_PATH}, {REQUEST_INDEX_PATH}")


if __name__ == "__main__":
    main()
