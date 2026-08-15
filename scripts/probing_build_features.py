"""Probing 실험용 콜 단위 피처 테이블 구축 (WAV 재추출 없음 - 기존 stage2b
산출물 재사용).

스코프 2개:
  - scope_n2: 첫 2발화(overall_rank<=2, 화자 무관) - 게이트 실작동 시점, 핵심 비교
  - scope_pool5: outputs/stage2_acoustic_features.parquet 전체(135,340행,
    콜당 평균 6.8개, N<=5 pool 그대로) - "콜 전체"가 아니라 N<=5 pool임을
    명확히 함. N이 커지면 신호가 강해지는지 비교용.

피처(6개, 콜 단위 mean/std로 집계 -> 스코프당 12피처):
  f0_mean, f0_std, e_mean_db, e_std_db(에너지는 dB 스케일 우선 사용 - 로그
  스케일이 인지적 정합성 높음. e_mean_amp/e_std_amp는 보조 컬럼으로만 별도 보존),
  r_gold(d04/pool에서 join, acoustic_features엔 없음), s_top_db30.

정규화/스케일링은 여기서 하지 않음 (실험 스크립트에서 fold 내부 fit - 누수 방지).

출력: outputs/probing_call_features.parquet
"""

from pathlib import Path

import pandas as pd
import tiktoken

BASE_DIR = Path(__file__).resolve().parent.parent
ACOUSTIC_PATH = BASE_DIR / "outputs" / "stage2_acoustic_features.parquet"
POOL_PATH = BASE_DIR / "outputs" / "stage2_utterance_pool.parquet"
GOLD_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_final.parquet"
PRED_PATH = BASE_DIR / "outputs" / "stage2d_gpt_predictions.parquet"
OUT_PATH = BASE_DIR / "outputs" / "probing_call_features.parquet"

PRIMARY_METRICS = ["f0_mean", "f0_std", "e_mean_db", "e_std_db", "r_gold", "s_top_db30"]
AUX_METRICS = ["e_mean_amp", "e_std_amp"]  # 보조 컬럼(참고용, 실험엔 미사용)

print("=== 에너지 스케일 선택: dB 우선(e_mean_db/e_std_db) 사용, "
      "amp(e_mean_amp/e_std_amp)는 보조 컬럼으로만 보존 ===")


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
    agg[std_cols] = agg[std_cols].fillna(0.0)  # 표본 1개라 std 계산 불가 -> 0
    n_utt = df.groupby("call_id").size().rename(f"{scope_name}_n_utt")
    agg = agg.merge(n_utt, on="call_id", how="left")
    return agg


def main():
    acoustic = pd.read_parquet(ACOUSTIC_PATH)
    pool = pd.read_parquet(POOL_PATH)
    gold = pd.read_parquet(GOLD_PATH)[["call_id", "label"]].rename(columns={"label": "gold_actual"})
    preds = pd.read_parquet(PRED_PATH)

    log_column_check(acoustic, "stage2_acoustic_features(WAV 재추출 없이 재사용)",
                      ["call_id", "dialog_idx"] + PRIMARY_METRICS + AUX_METRICS)
    log_column_check(pool, "stage2_utterance_pool", ["call_id", "dialog_idx", "overall_rank"])
    log_column_check(gold, "gold_actual", ["call_id", "gold_actual"])
    log_column_check(preds, "stage2d_gpt_predictions", ["call_id", "condition", "n", "predicted_label"])

    # r_gold는 acoustic_features에 없으므로 pool에서 join (pool은 이미 d04의 r_gold를 갖고 있음)
    r_gold_map = pool[["call_id", "dialog_idx", "r_gold", "overall_rank"]]
    acoustic = acoustic.merge(r_gold_map, on=["call_id", "dialog_idx"], how="left")
    n_rgold_missing = acoustic["r_gold"].isna().sum()
    print(f"\nr_gold join 후 결측: {n_rgold_missing}/{len(acoustic)}건 "
          f"(0이어야 정상 - acoustic_features는 pool의 부분집합이므로)")

    # === scope_pool5: stage2_acoustic_features 전체 그대로 (콜당 평균 6.8개, N<=5 pool) ===
    pool5_scope = aggregate_scope(acoustic, "pool5")

    # === scope_n2: overall_rank<=2 ===
    n2_acoustic = acoustic[acoustic["overall_rank"] <= 2]
    n2_scope = aggregate_scope(n2_acoustic, "n2")

    print(f"\nscope_n2 콜 수: {n2_scope['call_id'].nunique()}, "
          f"scope_pool5 콜 수: {pool5_scope['call_id'].nunique()}")

    # === 예측 컬럼 (A_N2/B_N2/A_N5) ===
    pred_specs = [("A", 2, "A_N2_pred"), ("B", 2, "B_N2_pred"), ("A", 5, "A_N5_pred")]
    pred_wide = None
    missing_preds = []
    for cond, n, colname in pred_specs:
        sub = preds[(preds["condition"] == cond) & (preds["n"] == n)][["call_id", "predicted_label"]]
        if sub.empty:
            print(f"경고: 조건={cond}, n={n} 예측이 없음 -> {colname} 생성 불가")
            missing_preds.append(colname)
            continue
        sub = sub.rename(columns={"predicted_label": colname})
        pred_wide = sub if pred_wide is None else pred_wide.merge(sub, on="call_id", how="outer")

    print(f"\n예측 컬럼 존재 여부: A_N2_pred={'A_N2_pred' not in missing_preds}, "
          f"B_N2_pred={'B_N2_pred' not in missing_preds}, A_N5_pred={'A_N5_pred' not in missing_preds}")

    # === N=2 발화 총 토큰수 (실험 F의 'short' 신호용) ===
    enc = tiktoken.get_encoding("o200k_base")
    n2_pool_text = pool[pool["overall_rank"] <= 2]
    tok_counts = n2_pool_text.groupby("call_id")["text_clean"].apply(
        lambda texts: sum(len(enc.encode(str(t))) for t in texts)
    ).rename("n2_total_tokens")

    # === 병합 ===
    table = gold.merge(n2_scope, on="call_id", how="left").merge(pool5_scope, on="call_id", how="left")
    if pred_wide is not None:
        table = table.merge(pred_wide, on="call_id", how="left")
    table = table.merge(tok_counts, on="call_id", how="left")
    table["is_complaint"] = (table["gold_actual"] == "불만제기").astype(int)

    n_before = len(table)
    table = table.dropna(subset=["gold_actual"])
    print(f"\ngold_actual 결측 제외: {n_before} -> {len(table)}")

    n_no_n2 = table["n2_n_utt"].isna().sum()
    n_no_pool5 = table["pool5_n_utt"].isna().sum()
    print(f"scope_n2 데이터 없는 콜: {n_no_n2}건, scope_pool5 데이터 없는 콜: {n_no_pool5}건 "
          f"(고객/전체 발화 자체가 부족한 극소수 콜 - 실험에서 자동 제외됨)")

    table.to_parquet(OUT_PATH, index=False)
    print(f"\n저장: {OUT_PATH} ({len(table)}행, {len(table.columns)}컬럼)")
    print(f"\ngold_actual 분포:")
    print(table["gold_actual"].value_counts().to_string())
    print("\nPROBING_FEATURES_COMPLETE")


if __name__ == "__main__":
    main()
