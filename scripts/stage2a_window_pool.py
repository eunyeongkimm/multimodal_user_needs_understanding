"""Stage 2-1: Surface 윈도우 추출.

gold_actual_batch1_final.parquet의 19,847개 콜에 대해 d04_dialog_index.parquet에서
r_gold_valid_flag==True인 dialog만 dialog_idx 순서로 정렬한 뒤,
- overall_rank: 화자 무관 전체 발화 순번 (1-based)
- customer_rank: 고객 발화만의 순번 (1-based, 고객 발화가 아니면 -1)
을 매겨서 N=1,2,3,5 "고객only"/"고객&상담사" 윈도우를 나중에 언제든 재구성할 수
있는 최소 pool(각 콜에서 필요한 unique dialog만 모은 테이블)을 만든다.

N=5가 N=1,2,3을 포함하므로, overall_rank<=5 OR customer_rank in [1,5]인
dialog만 모으면 모든 N x 버전 조합을 커버할 수 있다.

출력: outputs/stage2_utterance_pool.parquet
"""

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
D04_PATH = BASE_DIR / "outputs" / "d04_dialog_index.parquet"
GOLD_FINAL_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_final.parquet"
OUT_PATH = BASE_DIR / "outputs" / "stage2_utterance_pool.parquet"
PROGRESS_PATH = BASE_DIR / "outputs" / "stage2_progress.json"

MAX_N = 5


def mark_progress(key: str, value):
    progress = {}
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            progress = json.load(f)
    progress[key] = value
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def main():
    d04 = pd.read_parquet(D04_PATH)
    final = pd.read_parquet(GOLD_FINAL_PATH)
    batch1_ids = set(final["call_id"])

    sub = d04[d04["call_id"].isin(batch1_ids) & (d04["r_gold_valid_flag"] == True)].copy()  # noqa: E712
    sub = sub.sort_values(["call_id", "dialog_idx"]).reset_index(drop=True)
    sub["is_customer"] = sub["speaker_type"].str.startswith("고객")

    sub["overall_rank"] = sub.groupby("call_id").cumcount() + 1
    cust_rank = sub[sub["is_customer"]].groupby("call_id").cumcount() + 1
    sub["customer_rank"] = cust_rank.reindex(sub.index).fillna(-1).astype(int)

    pool = sub[(sub["overall_rank"] <= MAX_N) | (sub["customer_rank"].between(1, MAX_N))].copy()

    keep_cols = [
        "call_id", "dialog_idx", "speaker_type", "is_customer",
        "overall_rank", "customer_rank", "duration", "audioPath",
        "r_gold", "text", "text_clean",
    ]
    pool = pool[keep_cols].reset_index(drop=True)
    pool.to_parquet(OUT_PATH, index=False)

    n_calls = pool["call_id"].nunique()
    n_no_customer = len(batch1_ids) - sub[sub["is_customer"]]["call_id"].nunique()
    cust_counts = sub[sub["is_customer"]].groupby("call_id").size()
    n_lt5_customer = (cust_counts < MAX_N).sum()

    print(f"저장: {OUT_PATH} ({len(pool)}행, {n_calls}개 콜)")
    print(f"콜당 평균 pool 크기: {len(pool)/n_calls:.2f}")
    print(f"고객 유효 발화가 전혀 없는 콜: {n_no_customer}건 (고객only 윈도우 전부 비게 됨)")
    print(f"고객 유효 발화가 {MAX_N}개 미만인 콜: {n_lt5_customer}건 (N=5 고객only 윈도우가 {MAX_N}개보다 작게 채워짐)")

    mark_progress("window_pool", "done")


if __name__ == "__main__":
    main()
