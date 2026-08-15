"""Stage 2-5: A/B/C/D x N(1,2,3,5) 16개 조건의 surface 예측을 gold_actual과
비교해 mismatch 비율/정확도를 정리하고, acoustic 유무에 따른 성능 차이를
하이라이트한다.

A = 고객only + Text only       B = 고객only + Text&Acoustic
C = 고객&상담사 + Text only     D = 고객&상담사 + Text&Acoustic

acoustic 효과 = accuracy(B) - accuracy(A)  (고객only)
             = accuracy(D) - accuracy(C)  (고객&상담사)
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PRED_PATH = BASE_DIR / "outputs" / "stage2d_gpt_predictions.parquet"
GOLD_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_final.parquet"
OUT_SUMMARY_PATH = BASE_DIR / "outputs" / "stage2_condition_summary.csv"

CONDITION_LABELS = {
    "A": "고객only+Text",
    "B": "고객only+Text&Acoustic",
    "C": "고객&상담사+Text",
    "D": "고객&상담사+Text&Acoustic",
}


def main():
    preds = pd.read_parquet(PRED_PATH)
    gold = pd.read_parquet(GOLD_PATH)[["call_id", "label"]].rename(columns={"label": "actual_label"})

    merged = preds.merge(gold, on="call_id", how="left")
    merged["valid"] = merged["predicted_label"].notna()
    merged["match"] = merged["predicted_label"] == merged["actual_label"]
    merged["mismatch"] = merged["valid"] & ~merged["match"]

    rows = []
    for (condition, n), g in merged.groupby(["condition", "n"]):
        valid = g[g["valid"]]
        n_total = len(g)
        n_valid = len(valid)
        accuracy = valid["match"].mean() if n_valid else float("nan")
        mismatch_rate = 1 - accuracy if n_valid else float("nan")
        rows.append({
            "condition": condition, "condition_label": CONDITION_LABELS[condition], "n": n,
            "n_total": n_total, "n_valid": n_valid,
            "accuracy": accuracy, "mismatch_rate": mismatch_rate,
        })

    summary = pd.DataFrame(rows).sort_values(["n", "condition"])
    summary.to_csv(OUT_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print("=== N x 조건별 정확도/mismatch 비율 ===")
    print(summary.to_string(index=False))

    acc_pivot = summary.pivot(index="n", columns="condition", values="accuracy")
    print("\n=== 정확도 피벗 (행: N, 열: 조건 A/B/C/D) ===")
    print(acc_pivot.to_string())

    acc_pivot["acoustic_effect_customer_only(B-A)"] = acc_pivot["B"] - acc_pivot["A"]
    acc_pivot["acoustic_effect_all(D-C)"] = acc_pivot["D"] - acc_pivot["C"]
    print("\n=== Acoustic 효과 (양수면 acoustic 추가가 정확도 개선) ===")
    print(acc_pivot[["acoustic_effect_customer_only(B-A)", "acoustic_effect_all(D-C)"]].to_string())

    best_customer = acc_pivot["acoustic_effect_customer_only(B-A)"].idxmax()
    best_all = acc_pivot["acoustic_effect_all(D-C)"].idxmax()
    print(f"\n고객only 조건에서 acoustic 효과가 가장 큰 N: {best_customer} "
          f"(+{acc_pivot.loc[best_customer, 'acoustic_effect_customer_only(B-A)']:.4f})")
    print(f"고객&상담사 조건에서 acoustic 효과가 가장 큰 N: {best_all} "
          f"(+{acc_pivot.loc[best_all, 'acoustic_effect_all(D-C)']:.4f})")

    print(f"\n저장: {OUT_SUMMARY_PATH}")
    print("STAGE2E_COMPLETE")


if __name__ == "__main__":
    main()
