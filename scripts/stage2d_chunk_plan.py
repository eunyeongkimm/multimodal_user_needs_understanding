"""Stage 2-4a: A/B/C/D x N(1,2,3,5) = 16개 조건의 모든 콜에 대해 프롬프트를
만들고, 동시-제출 파이프라인에 맞게 작은 청크(기본 20만 토큰)로 나눈다.

조건 이름:
  A = custonly_text          (고객only + Text only)
  B = custonly_textacoustic  (고객only + Text&Acoustic)
  C = all_text                (고객&상담사 + Text only)
  D = all_textacoustic        (고객&상담사 + Text&Acoustic)

고객only(A,B) 조건은 해당 콜에 고객 발화가 1개도 없으면 그 N에서는 제외한다
(윈도우가 비어 요청 자체를 만들 수 없음).

custom_id 형식: "{call_id}::{condition}::N{n}" (다운로드 후 파싱해서 call_id/조건/N 복원)

출력:
  - outputs/stage2d_requests_plan.json : [[custom_id, ...], ...] 청크별 custom_id 리스트
  - outputs/stage2d_request_index.parquet : custom_id -> call_id, condition, n 매핑 + 프롬프트 텍스트
  - 콘솔에 예상 비용/청크 수 출력
"""

import json
from pathlib import Path

import pandas as pd
import tiktoken

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2d_prompt import make_prompt

BASE_DIR = Path(__file__).resolve().parent.parent
WINDOWS_NL_PATH = BASE_DIR / "outputs" / "stage2_windows_nl.parquet"
REQUEST_INDEX_PATH = BASE_DIR / "outputs" / "stage2d_request_index.parquet"
PLAN_PATH = BASE_DIR / "outputs" / "stage2d_requests_plan.json"

N_VALUES = [1, 2, 3, 5]
CHUNK_TOKEN_BUDGET = 200_000  # 동시 제출을 위해 작게 나눔 (기존 1.5M 청크 대비 훨씬 잘게)

PRICE_INPUT_PER_M = 0.40
PRICE_OUTPUT_PER_M = 1.60
EST_OUTPUT_TOKENS_PER_CALL = 8  # 카테고리명만 출력


def build_requests(windows: pd.DataFrame, enc) -> pd.DataFrame:
    rows = []
    for n in N_VALUES:
        sub_n = windows[windows["window_n"] == n]
        for version, cond_prefix in [("customer_only", "cust"), ("all", "all")]:
            sub_v = sub_n[sub_n["window_version"] == version]
            for call_id, g in sub_v.groupby("call_id"):
                g = g.sort_values("dialog_idx")
                utt_rows = g[["speaker_group", "text_clean", "nl_description"]].to_dict("records")
                if not utt_rows:
                    continue  # 고객only인데 고객 발화가 아예 없는 콜
                for acoustic, cond_suffix in [(False, "text"), (True, "textacoustic")]:
                    condition = "A" if (version == "customer_only" and not acoustic) else \
                                "B" if (version == "customer_only" and acoustic) else \
                                "C" if (version == "all" and not acoustic) else "D"
                    prompt = make_prompt(utt_rows, with_acoustic=acoustic)
                    tokens = len(enc.encode(prompt))
                    custom_id = f"{call_id}::{condition}::N{n}"
                    rows.append({
                        "custom_id": custom_id, "call_id": call_id, "condition": condition,
                        "n": n, "version": version, "acoustic": acoustic,
                        "prompt": prompt, "tokens": tokens,
                    })
    return pd.DataFrame(rows)


def main():
    if PLAN_PATH.exists() and REQUEST_INDEX_PATH.exists():
        idx = pd.read_parquet(REQUEST_INDEX_PATH)
        with open(PLAN_PATH, "r", encoding="utf-8") as f:
            plan = json.load(f)
        print(f"이미 계획이 존재합니다: {PLAN_PATH} ({len(plan)}개 청크, {len(idx)}건 요청). 재생성하지 않습니다.")
        print_cost(idx)
        return

    windows = pd.read_parquet(WINDOWS_NL_PATH)
    enc = tiktoken.get_encoding("o200k_base")

    idx = build_requests(windows, enc)
    idx.to_parquet(REQUEST_INDEX_PATH, index=False)

    print(f"전체 요청 수: {len(idx)}건")
    print(idx.groupby(["condition", "n"]).size().unstack())
    print_cost(idx)

    chunks = []
    current_ids = []
    current_tokens = 0
    for row in idx.itertuples():
        if current_ids and current_tokens + row.tokens > CHUNK_TOKEN_BUDGET:
            chunks.append(current_ids)
            current_ids = []
            current_tokens = 0
        current_ids.append(row.custom_id)
        current_tokens += row.tokens
    if current_ids:
        chunks.append(current_ids)

    with open(PLAN_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    print(f"\n청크 {len(chunks)}개 생성 (청크당 최대 {CHUNK_TOKEN_BUDGET:,}토큰)")
    print(f"저장: {PLAN_PATH}, {REQUEST_INDEX_PATH}")


def print_cost(idx: pd.DataFrame):
    n = len(idx)
    total_input = idx["tokens"].sum()
    total_output = n * EST_OUTPUT_TOKENS_PER_CALL
    in_cost = total_input / 1_000_000 * PRICE_INPUT_PER_M * 0.5
    out_cost = total_output / 1_000_000 * PRICE_OUTPUT_PER_M * 0.5
    print(f"\n=== 예상 비용 (gpt-4.1-mini, Batch API 50% 할인, {n:,}건) ===")
    print(f"input {total_input:,} tokens (${in_cost:.4f}) + output 추정 {total_output:,} tokens (${out_cost:.4f}) "
          f"= ${in_cost+out_cost:.4f}")


if __name__ == "__main__":
    main()
