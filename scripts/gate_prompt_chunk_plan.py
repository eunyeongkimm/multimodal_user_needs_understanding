"""v3(reasoning) 프롬프트 vs v3+게이트(강/중/약 신호 주입) 비교용 요청 생성.
샘플: v_unified_sample_call_ids.csv의 층화 2,000건(불만 오버샘플 250건 포함,
기존 v_unified 검증과 동일 샘플 재사용) 중 N=2 customer_only 윈도우가 있는 콜만.

게이트 문장은 voice_guide(VOICE_GUIDE_V3) 블록 뒤, [발췌된 통화 시작 부분] 바로
앞에 삽입: "이 통화의 음성은 불만 패턴과 {강/중/약} 일치합니다."

출력: outputs/gate_prompt_request_index.parquet, outputs/gate_prompt_requests_plan.json
"""

import json
from pathlib import Path

import pandas as pd
import tiktoken

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2d_prompt import make_prompt

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_PATH = BASE_DIR / "outputs" / "v_unified_sample_call_ids.csv"
GATE_PATH = BASE_DIR / "outputs" / "gate_signal_inference.parquet"
WINDOWS_NL_V2_PATH = BASE_DIR / "outputs" / "stage2_windows_nl_v2.parquet"
REQUEST_INDEX_PATH = BASE_DIR / "outputs" / "gate_prompt_request_index.parquet"
PLAN_PATH = BASE_DIR / "outputs" / "gate_prompt_requests_plan.json"

WINDOW_N = 2
WINDOW_VERSION = "customer_only"
CHUNK_TOKEN_BUDGET = 200_000
PRICE_INPUT_PER_M = 0.40
PRICE_OUTPUT_PER_M = 1.60
EST_OUTPUT_TOKENS_PER_CALL = 60

GATE_SENTENCE_TMPL = "이 통화의 음성은 불만 패턴과 {level} 일치합니다.\n\n"
INSERT_MARK = "[발췌된 통화 시작 부분]"


def build_gated_prompt(base_prompt: str, level: str) -> str:
    sentence = GATE_SENTENCE_TMPL.format(level=level)
    assert base_prompt.count(INSERT_MARK) == 1, "삽입 지점이 유일해야 함"
    return base_prompt.replace(INSERT_MARK, sentence + INSERT_MARK, 1)


def main():
    sample = pd.read_csv(SAMPLE_PATH)
    gate = pd.read_parquet(GATE_PATH)[["call_id", "gate_level"]]
    sample = sample.merge(gate, on="call_id", how="left")
    n_no_gate = sample["gate_level"].isna().sum()
    print(f"샘플 {len(sample)}건, 게이트 신호 없는 콜: {n_no_gate}건")
    sample = sample.dropna(subset=["gate_level"])

    windows = pd.read_parquet(WINDOWS_NL_V2_PATH)
    win_sub = windows[(windows["window_n"] == WINDOW_N) & (windows["window_version"] == WINDOW_VERSION)]

    enc = tiktoken.get_encoding("o200k_base")
    rows_out = []
    n_no_window = 0
    for row in sample.itertuples():
        g = win_sub[win_sub["call_id"] == row.call_id].sort_values("dialog_idx")
        utt_rows = g[["speaker_group", "text_clean", "nl_description"]].to_dict("records")
        if not utt_rows:
            n_no_window += 1
            continue
        base_prompt = make_prompt(utt_rows, with_acoustic=True, version="v3")
        gated_prompt = build_gated_prompt(base_prompt, row.gate_level)

        for variant, prompt in [("base", base_prompt), ("gate", gated_prompt)]:
            tokens = len(enc.encode(prompt))
            custom_id = f"{row.call_id}::{variant}"
            rows_out.append({"custom_id": custom_id, "call_id": row.call_id, "variant": variant,
                             "gate_level": row.gate_level, "gold_actual": row.gold_actual,
                             "prompt": prompt, "tokens": tokens})

    print(f"윈도우 없는 콜(제외): {n_no_window}건")
    idx = pd.DataFrame(rows_out)
    idx.to_parquet(REQUEST_INDEX_PATH, index=False)

    print(f"\n신규 요청 수: {len(idx)}건 (콜당 base+gate 2건)")
    print(idx.groupby("variant").size())
    print(idx.groupby("gate_level").size())

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
