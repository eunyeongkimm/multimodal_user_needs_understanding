"""B/D 조건(acoustic 포함) x N in {2,3,5}만 재정규화된 자연어 서술로 재실행하기
위한 요청 생성. A/C 조건은 acoustic 서술을 프롬프트에 쓰지 않으므로(text only)
재정규화해도 결과가 바뀔 수 없어 재실행 대상에서 제외한다 (기존
stage2d_gpt_predictions.parquet의 A/C 값을 그대로 재사용).

N=1은 mismatch subset 정의(조건A, N=1) 자체에 쓰이는 기준이라 재실행 대상에서
제외 (사용자 지시).

프롬프트 템플릿/포맷은 stage2d_prompt.py를 그대로 재사용 (변경 없음) -
입력 데이터만 stage2_windows_nl_v2.parquet(재정규화된 nl_description)로 교체.

출력:
  - outputs/stage2m_request_index.parquet
  - outputs/stage2m_requests_plan.json
  - 콘솔에 예상 비용
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
REQUEST_INDEX_PATH = BASE_DIR / "outputs" / "stage2m_request_index.parquet"
PLAN_PATH = BASE_DIR / "outputs" / "stage2m_requests_plan.json"

N_VALUES = [2, 3, 5]  # N=1 제외
CHUNK_TOKEN_BUDGET = 200_000

PRICE_INPUT_PER_M = 0.40
PRICE_OUTPUT_PER_M = 1.60
EST_OUTPUT_TOKENS_PER_CALL = 8


def build_requests(windows: pd.DataFrame, enc) -> pd.DataFrame:
    rows = []
    for n in N_VALUES:
        sub_n = windows[windows["window_n"] == n]
        # B = customer_only + acoustic, D = all + acoustic (A/C는 재실행 안 함)
        for version, condition in [("customer_only", "B"), ("all", "D")]:
            sub_v = sub_n[sub_n["window_version"] == version]
            for call_id, g in sub_v.groupby("call_id"):
                g = g.sort_values("dialog_idx")
                utt_rows = g[["speaker_group", "text_clean", "nl_description"]].to_dict("records")
                if not utt_rows:
                    continue
                prompt = make_prompt(utt_rows, with_acoustic=True)
                tokens = len(enc.encode(prompt))
                custom_id = f"{call_id}::{condition}::N{n}"
                rows.append({
                    "custom_id": custom_id, "call_id": call_id, "condition": condition,
                    "n": n, "version": version, "acoustic": True,
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
    if PLAN_PATH.exists() and REQUEST_INDEX_PATH.exists():
        idx = pd.read_parquet(REQUEST_INDEX_PATH)
        with open(PLAN_PATH, "r", encoding="utf-8") as f:
            plan = json.load(f)
        print(f"이미 계획이 존재합니다: {PLAN_PATH} ({len(plan)}개 청크, {len(idx)}건 요청). 재생성하지 않습니다.")
        print_cost(idx)
        return

    windows = pd.read_parquet(WINDOWS_NL_V2_PATH)
    enc = tiktoken.get_encoding("o200k_base")

    idx = build_requests(windows, enc)
    idx.to_parquet(REQUEST_INDEX_PATH, index=False)

    print(f"전체 요청 수: {len(idx)}건")
    print(idx.groupby(["condition", "n"]).size().unstack())
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

    print(f"\n청크 {len(chunks)}개 생성 (청크당 최대 {CHUNK_TOKEN_BUDGET:,}토큰)")
    print(f"저장: {PLAN_PATH}, {REQUEST_INDEX_PATH}")


if __name__ == "__main__":
    main()
