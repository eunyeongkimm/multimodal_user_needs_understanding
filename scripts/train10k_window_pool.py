"""train10k(게이트 프로토타입 B용 train 1만 콜)의 N=2 발화 pool 추출.
stage2a_window_pool.py와 동일 로직이나, 대상이 gold_actual_batch1_final(19,847)이
아니라 outputs/train10k_call_ids.csv(10,000, batch2 풀에서 샘플링)이고 MAX_N=2만
필요(게이트 프로토타입은 N=2만 사용). 오디오 파일 자체는 읽지 않음(경로/메타데이터만
정리) - WAV 추출은 별도 단계(train10k_acoustic_extract.py, 외장하드 필요)에서 수행.

출력: outputs/train10k_utterance_pool.parquet
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
D04_PATH = BASE_DIR / "outputs" / "d04_dialog_index.parquet"
CALL_IDS_PATH = BASE_DIR / "outputs" / "train10k_call_ids.csv"
OUT_PATH = BASE_DIR / "outputs" / "train10k_utterance_pool.parquet"

MAX_N = 2


def main():
    d04 = pd.read_parquet(D04_PATH)
    call_ids = set(pd.read_csv(CALL_IDS_PATH)["call_id"])
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
    n_no_customer = len(call_ids) - sub[sub["is_customer"]]["call_id"].nunique()
    cust_counts = sub[sub["is_customer"]].groupby("call_id").size()
    n_lt2_customer = (cust_counts < MAX_N).sum()

    print(f"저장: {OUT_PATH} ({len(pool)}행, {n_calls}개 콜)")
    print(f"콜당 평균 pool 크기: {len(pool)/n_calls:.2f}")
    print(f"고객 유효 발화가 전혀 없는 콜: {n_no_customer}건 (고객only 윈도우 전부 비게 됨)")
    print(f"고객 유효 발화가 {MAX_N}개 미만인 콜: {n_lt2_customer}건")
    print(f"오디오 로딩 필요 행 수(다음 단계 WAV 추출 대상): {len(pool)}건")

    print("\nTRAIN10K_WINDOW_POOL_COMPLETE")


if __name__ == "__main__":
    main()
