"""B/D 조건(재정규화 A트랙) x N in {2,3,5}을 v3_orig(reasoning) 프롬프트로 재실행하기
위한 요청 생성. stage2m_chunk_plan_v2.py와 동일 스코프(전체 19,847콜 x B/D x N{2,3,5}),
프롬프트만 version="v3"(reasoning, <분석>+<카테고리> 태그출력)로 교체.
비교 기준(v_base+재정규화)은 이미 outputs/stage2m_gpt_predictions_BD.parquet에 있음(재사용).

출력: outputs/stage2_bd_v3_request_index.parquet, outputs/stage2_bd_v3_requests_plan.json
"""

import json
from pathlib import Path

import pandas as pd
import tiktoken

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2d_prompt import make_prompt

BASE_DIR = Path(__file__).resolve().parent.parent
WINDOWS_NL_V2_PATH = BASE_DIR / "outputs" / "stage2_windows_nl_v2.parquet"
BASELINE_BD_PATH = BASE_DIR / "outputs" / "stage2m_gpt_predictions_BD.parquet"
REQUEST_INDEX_PATH = BASE_DIR / "outputs" / "stage2_bd_v3_request_index.parquet"
PLAN_PATH = BASE_DIR / "outputs" / "stage2_bd_v3_requests_plan.json"

N_VALUES = [2, 3, 5]
CHUNK_TOKEN_BUDGET = 200_000

PRICE_INPUT_PER_M = 0.40
PRICE_OUTPUT_PER_M = 1.60
EST_OUTPUT_TOKENS_PER_CALL = 60  # v3는 <분석> 텍스트 포함이라 v_base(단답)보다 김


def build_requests(windows: pd.DataFrame, enc) -> pd.DataFrame:
    rows = []
    for n in N_VALUES:
        sub_n = windows[windows["window_n"] == n]
        for version, condition in [("customer_only", "B"), ("all", "D")]:
            sub_v = sub_n[sub_n["window_version"] == version]
            for call_id, g in sub_v.groupby("call_id"):
                g = g.sort_values("dialog_idx")
                utt_rows = g[["speaker_group", "text_clean", "nl_description"]].to_dict("records")
                if not utt_rows:
                    continue
                prompt = make_prompt(utt_rows, with_acoustic=True, version="v3")
                tokens = len(enc.encode(prompt))
                custom_id = f"{call_id}::{condition}::N{n}"
                rows.append({
                    "custom_id": custom_id, "call_id": call_id, "condition": condition,
                    "n": n, "window_version": version, "acoustic": True,
                    "prompt": prompt, "tokens": tokens,
                })
    return pd.DataFrame(rows)


def print_cost(idx: pd.DataFrame):
    n = len(idx)
    total_input = idx["tokens"].sum()
    total_output = n * EST_OUTPUT_TOKENS_PER_CALL
    in_cost = total_input / 1_000_000 * PRICE_INPUT_PER_M * 0.5
    out_cost = total_output / 1_000_000 * PRICE_OUTPUT_PER_M * 0.5
    print(f"\n=== 예상 비용 (gpt-4.1-mini, Batch API 50% 할인, {n:,}건) ===")
    print(f"input {total_input:,} tokens (${in_cost:.4f}) + output 추정 {total_output:,} tokens (${out_cost:.4f}) "
          f"= ${in_cost+out_cost:.4f}")


def main():
    baseline = pd.read_parquet(BASELINE_BD_PATH)
    print(f"비교 기준(v_base+재정규화) 스코프: {len(baseline)}행, "
          f"{baseline['call_id'].nunique()}개 콜, condition x n:")
    print(baseline.groupby(["condition", "n"]).size())

    if PLAN_PATH.exists() and REQUEST_INDEX_PATH.exists():
        idx = pd.read_parquet(REQUEST_INDEX_PATH)
        with open(PLAN_PATH, "r", encoding="utf-8") as f:
            plan = json.load(f)
        print(f"\n이미 계획이 존재합니다: {PLAN_PATH} ({len(plan)}개 청크, {len(idx)}건 요청). 재생성하지 않습니다.")
        print_cost(idx)
        return

    windows = pd.read_parquet(WINDOWS_NL_V2_PATH)
    enc = tiktoken.get_encoding("o200k_base")

    idx = build_requests(windows, enc)
    idx.to_parquet(REQUEST_INDEX_PATH, index=False)

    print(f"\n신규 요청(v3_orig, B/D, N=2,3,5): {len(idx)}건")
    print(idx.groupby(["condition", "n"]).size().unstack())

    scope_match = set(idx["call_id"]) == set(baseline["call_id"]) and len(idx) == len(baseline)
    print(f"\n스코프 baseline(v_base+재정규화)과 100% 일치: {scope_match} "
          f"(신규 {len(idx)}건 vs 기준 {len(baseline)}건, call_id셋 일치: {set(idx['call_id']) == set(baseline['call_id'])})")

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

    with open(PLAN_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    print(f"\n청크 {len(chunks)}개 생성")
    print(f"저장: {PLAN_PATH}, {REQUEST_INDEX_PATH}")


if __name__ == "__main__":
    main()
