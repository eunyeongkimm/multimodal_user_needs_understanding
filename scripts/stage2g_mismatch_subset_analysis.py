"""Stage 2-7: "surface(N=1) != gold_actual" mismatch 콜만 골라, 그 서브셋에서
A vs B, C vs D 정확도를 N=1,2,3,5별로 재계산한다.

surface(N=1) 정의: 조건 A(고객only + Text only, N=1)의 예측 라벨을 사용한다.
"통화 시작 시 고객이 표면적으로 표현한 니즈"를 가장 직접적으로 나타내는
조건이라 판단해 이걸 기준으로 mismatch 콜을 정의했다 (다른 정의를 원하면
SURFACE_CONDITION 상수만 바꾸면 됨).

주의: 이 정의상 서브셋 내에서 A@N=1 정확도는 항상 0%가 된다(정의 자체가
"A@N=1이 틀린 콜"이므로). 이는 자명한 값이라 트렌드 해석 시 참고만 할 것.
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PRED_PATH = BASE_DIR / "outputs" / "stage2d_gpt_predictions.parquet"
GOLD_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_final.parquet"
OUT_PATH = BASE_DIR / "outputs" / "stage2_mismatch_subset_accuracy.csv"

SURFACE_CONDITION = "A"  # 고객only + Text only, N=1
N_VALUES = [1, 2, 3, 5]


def main():
    preds = pd.read_parquet(PRED_PATH)
    gold = pd.read_parquet(GOLD_PATH)[["call_id", "label"]].rename(columns={"label": "actual_label"})
    merged = preds.merge(gold, on="call_id", how="left")

    surface = merged[(merged["condition"] == SURFACE_CONDITION) & (merged["n"] == 1)]
    surface_valid = surface.dropna(subset=["predicted_label"])
    mismatch_ids = set(surface_valid[surface_valid["predicted_label"] != surface_valid["actual_label"]]["call_id"])
    match_ids = set(surface_valid[surface_valid["predicted_label"] == surface_valid["actual_label"]]["call_id"])

    print(f"surface(N=1, 조건 {SURFACE_CONDITION}) 기준 전체 유효 콜: {len(surface_valid)}건")
    print(f"mismatch 콜(surface != actual): {len(mismatch_ids)}건 "
          f"({len(mismatch_ids)/len(surface_valid)*100:.1f}%)")
    print(f"match 콜(surface == actual): {len(match_ids)}건 ({len(match_ids)/len(surface_valid)*100:.1f}%)")

    sub = merged[merged["call_id"].isin(mismatch_ids) & merged["n"].isin(N_VALUES)].copy()
    sub["match"] = sub["predicted_label"] == sub["actual_label"]

    acc = (
        sub.groupby(["n", "condition"])["match"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n_calls"})
        .reset_index()
    )
    acc.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n=== mismatch 서브셋({len(mismatch_ids)}건)에서 N x 조건별 정확도 ===")
    print(acc.to_string(index=False))

    pivot = acc.pivot(index="n", columns="condition", values="accuracy")
    print("\n=== 정확도 피벗 (행: N, 열: 조건 A/B/C/D) - mismatch 서브셋 ===")
    print(pivot.to_string())

    pivot["acoustic_effect_customer_only(B-A)"] = pivot["B"] - pivot["A"]
    pivot["acoustic_effect_all(D-C)"] = pivot["D"] - pivot["C"]
    print("\n=== Acoustic 효과 (mismatch 서브셋, N=1은 A가 정의상 0%라 참고용) ===")
    print(pivot[["acoustic_effect_customer_only(B-A)", "acoustic_effect_all(D-C)"]].to_string())

    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
