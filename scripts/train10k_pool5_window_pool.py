"""train10k(10,000콜) N<=5 발화 pool 추출. stage2a_window_pool.py와 동일 로직
(MAX_N=5), 대상만 train_10k_final_labeled.parquet(10,000콜)의 call_id로 교체.

출력: outputs/train10k_pool5_utterance_pool.parquet
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
D04_PATH = BASE_DIR / "outputs" / "d04_dialog_index.parquet"
CALL_IDS_PATH = BASE_DIR / "outputs" / "train_10k_final_labeled.parquet"
OUT_PATH = BASE_DIR / "outputs" / "train10k_pool5_utterance_pool.parquet"

MAX_N = 5


def main():
    d04 = pd.read_parquet(D04_PATH)
    call_ids = set(pd.read_parquet(CALL_IDS_PATH)["call_id"])
    print(f"대상 콜: {len(call_ids)}건")

    sub = d04[d04["call_id"].isin(call_ids) & (d04["r_gold_valid_flag"] == True)].copy()  # noqa: E712
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
    print(f"저장: {OUT_PATH} ({len(pool)}행, {n_calls}개 콜)")
    print(f"콜당 평균 pool 크기: {len(pool)/n_calls:.2f}")

    print("\nTRAIN10K_POOL5_WINDOW_POOL_COMPLETE")


if __name__ == "__main__":
    main()
