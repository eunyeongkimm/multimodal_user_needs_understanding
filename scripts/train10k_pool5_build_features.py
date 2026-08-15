"""train10k pool5(N<=5) 콜 단위 피처 테이블 구축. probing_build_features.py의
scope_pool5 로직과 동일(전체 pool을 그대로 콜당 mean/std로 집계, 랭크 필터 없음,
정규화 안 함).

출력: outputs/train_10k_pool5_labeled.parquet (call_id, pool5_* 음성피처 12개, gold_actual)
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
ACOUSTIC_PATH = BASE_DIR / "outputs" / "train10k_pool5_acoustic_features.parquet"
POOL_PATH = BASE_DIR / "outputs" / "train10k_pool5_utterance_pool.parquet"
GOLD_PATH = BASE_DIR / "outputs" / "train_10k_final_labeled.parquet"
OUT_PATH = BASE_DIR / "outputs" / "train_10k_pool5_labeled.parquet"

PRIMARY_METRICS = ["f0_mean", "f0_std", "e_mean_db", "e_std_db", "r_gold", "s_top_db30"]
AUX_METRICS = ["e_mean_amp", "e_std_amp"]


def log_column_check(df, name, cols):
    print(f"[{name}] 행수={len(df)}, 컬럼={df.columns.tolist()}")
    for c in cols:
        status = "OK" if c in df.columns else "!! 없음"
        print(f"  - {c}: {status}")


def aggregate_scope(df: pd.DataFrame, scope_name: str) -> pd.DataFrame:
    all_metrics = PRIMARY_METRICS + AUX_METRICS
    agg = df.groupby("call_id")[all_metrics].agg(["mean", "std"])
    agg.columns = [f"{scope_name}_{m}_{stat}" for m, stat in agg.columns]
    agg = agg.reset_index()
    std_cols = [c for c in agg.columns if c.endswith("_std")]
    agg[std_cols] = agg[std_cols].fillna(0.0)
    n_utt = df.groupby("call_id").size().rename(f"{scope_name}_n_utt")
    agg = agg.merge(n_utt, on="call_id", how="left")
    return agg


def main():
    acoustic = pd.read_parquet(ACOUSTIC_PATH)
    pool = pd.read_parquet(POOL_PATH)
    gold = pd.read_parquet(GOLD_PATH)

    log_column_check(acoustic, "train10k_pool5_acoustic_features",
                      ["call_id", "dialog_idx"] + PRIMARY_METRICS + AUX_METRICS)
    log_column_check(pool, "train10k_pool5_utterance_pool", ["call_id", "dialog_idx"])
    log_column_check(gold, "train_10k_final_labeled", ["call_id", "gold_actual"])

    r_gold_map = pool[["call_id", "dialog_idx", "r_gold"]]
    acoustic = acoustic.merge(r_gold_map, on=["call_id", "dialog_idx"], how="left")
    n_rgold_missing = acoustic["r_gold"].isna().sum()
    print(f"\nr_gold join 후 결측: {n_rgold_missing}/{len(acoustic)}건 (0이어야 정상)")

    pool5_scope = aggregate_scope(acoustic, "pool5")
    print(f"\nscope_pool5 콜 수: {pool5_scope['call_id'].nunique()}")

    table = gold.merge(pool5_scope, on="call_id", how="left")
    table["is_complaint"] = (table["gold_actual"] == "불만제기").astype(int)

    n_no_pool5 = table["pool5_n_utt"].isna().sum()
    print(f"scope_pool5 데이터 없는 콜: {n_no_pool5}/{len(table)}건")

    table.to_parquet(OUT_PATH, index=False)
    print(f"\n저장: {OUT_PATH} ({len(table)}행, {len(table.columns)}컬럼)")
    print(f"불만제기(is_complaint=1): {table['is_complaint'].sum()}건")

    print("\nTRAIN10K_POOL5_FEATURES_COMPLETE")


if __name__ == "__main__":
    main()
