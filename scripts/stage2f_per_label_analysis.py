"""Stage 2-6: N=2,3,5 x (A vs B, C vs D)에 대해 gold_actual label별로
정확도를 따로 계산하고, acoustic 유무에 따른 정확도 차이가 카테고리별로
어떻게 다른지(특히 불만제기) 비교한다.
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PRED_PATH = BASE_DIR / "outputs" / "stage2d_gpt_predictions.parquet"
GOLD_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_final.parquet"
OUT_PATH = BASE_DIR / "outputs" / "stage2_per_label_acoustic_effect.csv"

N_VALUES = [2, 3, 5]
CATEGORY_ORDER = ["환불요청", "주문취소", "불만제기", "배송확인", "교환반품", "구매진행", "서비스이용"]


def main():
    preds = pd.read_parquet(PRED_PATH)
    gold = pd.read_parquet(GOLD_PATH)[["call_id", "label"]].rename(columns={"label": "actual_label"})
    merged = preds.merge(gold, on="call_id", how="left")
    merged["match"] = merged["predicted_label"] == merged["actual_label"]
    merged = merged[merged["n"].isin(N_VALUES)]

    acc = (
        merged.groupby(["n", "condition", "actual_label"])["match"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n_calls"})
        .reset_index()
    )

    rows = []
    for n in N_VALUES:
        for pair_name, cond_a, cond_b in [("customer_only", "A", "B"), ("all", "C", "D")]:
            sub_a = acc[(acc["n"] == n) & (acc["condition"] == cond_a)].set_index("actual_label")
            sub_b = acc[(acc["n"] == n) & (acc["condition"] == cond_b)].set_index("actual_label")
            for label in CATEGORY_ORDER:
                if label not in sub_a.index or label not in sub_b.index:
                    continue
                acc_a = sub_a.loc[label, "accuracy"]
                acc_b = sub_b.loc[label, "accuracy"]
                n_calls = sub_a.loc[label, "n_calls"]
                rows.append({
                    "n": n, "pair": pair_name, "cond_no_acoustic": cond_a, "cond_acoustic": cond_b,
                    "label": label, "n_calls": int(n_calls),
                    "accuracy_no_acoustic": acc_a, "accuracy_acoustic": acc_b,
                    "acoustic_effect": acc_b - acc_a,
                })

    result = pd.DataFrame(rows)
    result.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("=== N x pair(customer_only/all) x 카테고리별 정확도 및 acoustic 효과 ===")
    print(result.to_string(index=False))

    print("\n=== 카테고리별 |acoustic_effect| 평균 (N=2,3,5, customer_only+all 통합) - 큰 순서 ===")
    magnitude = result.groupby("label")["acoustic_effect"].agg(
        mean_effect="mean", mean_abs_effect=lambda s: s.abs().mean()
    ).reindex(CATEGORY_ORDER).sort_values("mean_abs_effect", ascending=False)
    print(magnitude.to_string())

    print("\n=== 불만제기만 따로 ===")
    complaint = result[result["label"] == "불만제기"]
    print(complaint.to_string(index=False))

    overall_mean_abs = result[result["label"] != "불만제기"]["acoustic_effect"].abs().mean()
    complaint_mean_abs = complaint["acoustic_effect"].abs().mean()
    print(f"\n불만제기 |acoustic_effect| 평균: {complaint_mean_abs:.4f}")
    print(f"불만제기 제외 나머지 카테고리 |acoustic_effect| 평균: {overall_mean_abs:.4f}")
    print(f"불만제기가 {'더 큼' if complaint_mean_abs > overall_mean_abs else '더 작거나 비슷함'}")

    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
