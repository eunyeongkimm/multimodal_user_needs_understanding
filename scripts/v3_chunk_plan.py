"""gold_actual_batch1.parquet에서 label이 환불요청/배송확인인 콜(11,579건)을
v3 프롬프트(불만제기 우선판단 규칙 추가)로 재라벨링하기 위해 여러 청크로 나눈다.
batch1_chunk_plan.py와 동일한 토큰 예산 기준 greedy bin-packing 방식.

한 번 저장되면 재실행해도 덮어쓰지 않는다 (청크 경계가 바뀌면 이후 상태 추적이 꼬이므로).
"""

import json
import sys
from pathlib import Path

import pandas as pd
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_prompt import make_prompt

BASE_DIR = Path(__file__).resolve().parent.parent
GOLD_V2_PATH = BASE_DIR / "outputs" / "gold_actual_batch1.parquet"
CALL_TEXTS_PATH = BASE_DIR / "outputs" / "batch1_call_texts.parquet"
TARGET_IDS_PATH = BASE_DIR / "outputs" / "v3_target_call_ids.csv"
PLAN_PATH = BASE_DIR / "outputs" / "v3_chunks_plan.json"

TOKEN_BUDGET_PER_CHUNK = 1_500_000  # 조직 한도 2,000,000 대비 안전 마진


def main():
    if PLAN_PATH.exists():
        with open(PLAN_PATH, "r", encoding="utf-8") as f:
            plan = json.load(f)
        print(f"이미 청크 계획이 존재합니다: {PLAN_PATH} ({len(plan)}개 청크). 재생성하지 않습니다.")
        return

    gold_v2 = pd.read_parquet(GOLD_V2_PATH)
    target = gold_v2[gold_v2["label"].isin(["환불요청", "배송확인"])][["call_id", "label"]]
    target.to_csv(TARGET_IDS_PATH, index=False)
    print(f"대상 콜: {len(target)}건 (환불요청 {(target['label']=='환불요청').sum()}건, "
          f"배송확인 {(target['label']=='배송확인').sum()}건)")

    call_texts = pd.read_parquet(CALL_TEXTS_PATH)
    subset = target.merge(call_texts, on="call_id", how="left")
    assert subset["call_text"].isna().sum() == 0, "call_text 누락된 call_id가 있습니다."

    enc = tiktoken.get_encoding("o200k_base")
    subset = subset.copy()
    subset["tokens"] = subset["call_text"].apply(lambda t: len(enc.encode(make_prompt(t))))

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
