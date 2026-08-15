"""v_unified 층화샘플 2,000건 예측을 gold_actual과 비교 검증.
- Cohen's Kappa (7카테고리)
- 불만제기 recall/precision (샘플 내 gold 불만 대비)
- 클래스별 일치율 + 7x7 혼동행렬
- 불만제기(gold) -> 환불요청/배송확인(pred) 흡수 재발 여부 확인

출력: outputs/validation_vs_gold.txt
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
LABELS_PATH = BASE_DIR / "outputs" / "v_unified_sample_labels.parquet"
SAMPLE_PATH = BASE_DIR / "outputs" / "v_unified_sample_call_ids.csv"
OUT_PATH = BASE_DIR / "outputs" / "validation_vs_gold.txt"

CATEGORY_ORDER = ["환불요청", "주문취소", "불만제기", "배송확인", "교환반품", "구매진행", "서비스이용"]
PARSE_FAIL_SENTINEL = "PARSE_FAIL"
KAPPA_THRESHOLD = 0.8
RECALL_THRESHOLD = 0.7


def main():
    df = pd.read_parquet(LABELS_PATH)
    sample_meta = pd.read_csv(SAMPLE_PATH)

    lines = []
    lines.append("v_unified 검증 결과 (gold_actual_batch1_final 대비)")
    lines.append(f"샘플 {len(df)}건, seed=42")
    lines.append("")
    lines.append("=== 실제 층별 추출 수 ===")
    lines.append(sample_meta["gold_actual"].value_counts().to_string())
    lines.append("")

    n_fail = df["predicted_label"].isna().sum()
    lines.append(f"파싱 실패/누락: {n_fail}/{len(df)}건 ({n_fail/len(df)*100:.2f}%) - 오답으로 계산 포함")
    df["predicted_label"] = df["predicted_label"].fillna(PARSE_FAIL_SENTINEL)

    kappa = cohen_kappa_score(df["gold_actual"], df["predicted_label"], labels=CATEGORY_ORDER)
    acc = (df["gold_actual"] == df["predicted_label"]).mean()
    lines.append(f"\n=== 1) Cohen's Kappa (7카테고리) ===")
    lines.append(f"Kappa = {kappa:.4f}  (accuracy = {acc:.4f})")
    lines.append(f"판정 기준(Kappa>=0.8): {'충족' if kappa >= KAPPA_THRESHOLD else '미달'}")

    y_true_complaint = (df["gold_actual"] == "불만제기").astype(int)
    y_pred_complaint = (df["predicted_label"] == "불만제기").astype(int)
    recall = recall_score(y_true_complaint, y_pred_complaint, zero_division=0)
    precision = precision_score(y_true_complaint, y_pred_complaint, zero_division=0)
    f1 = f1_score(y_true_complaint, y_pred_complaint, zero_division=0)
    n_gold_complaint = y_true_complaint.sum()
    n_pred_complaint = y_pred_complaint.sum()
    lines.append(f"\n=== 2) 불만제기 재현 (샘플 내 gold 불만 {n_gold_complaint}건 대비) ===")
    lines.append(f"recall={recall:.4f}, precision={precision:.4f}, f1={f1:.4f} "
                 f"(예측 불만 건수={n_pred_complaint})")
    lines.append(f"판정 기준(recall>=0.7): {'충족' if recall >= RECALL_THRESHOLD else '미달'}")

    lines.append(f"\n=== 3) 클래스별 일치율(recall) ===")
    per_class_recall = {}
    for cls in CATEGORY_ORDER:
        sub = df[df["gold_actual"] == cls]
        if len(sub) == 0:
            continue
        r = (sub["predicted_label"] == cls).mean()
        per_class_recall[cls] = (r, len(sub))
        lines.append(f"  {cls:8s}: {r:.4f} (n={len(sub)})")

    lines.append(f"\n=== 4) 7x7 혼동행렬 (행=gold, 열=predicted) ===")
    labels_for_cm = CATEGORY_ORDER + (
        [PARSE_FAIL_SENTINEL] if (df["predicted_label"] == PARSE_FAIL_SENTINEL).any() else []
    )
    cm = confusion_matrix(df["gold_actual"], df["predicted_label"], labels=labels_for_cm)
    cm_df = pd.DataFrame(cm, index=[f"gold_{c}" for c in labels_for_cm],
                          columns=[f"pred_{c}" for c in labels_for_cm])
    lines.append(cm_df.to_string())

    lines.append(f"\n=== 5) 핵심 점검: 불만제기(gold) -> 환불요청/배송확인(pred) 흡수 재발 여부 ===")
    complaint_gold = df[df["gold_actual"] == "불만제기"]
    n_to_refund = (complaint_gold["predicted_label"] == "환불요청").sum()
    n_to_delivery = (complaint_gold["predicted_label"] == "배송확인").sum()
    n_to_complaint = (complaint_gold["predicted_label"] == "불만제기").sum()
    lines.append(f"gold 불만제기 {len(complaint_gold)}건 중:")
    lines.append(f"  -> 불만제기(정답 유지): {n_to_complaint}건 ({n_to_complaint/len(complaint_gold)*100:.1f}%)")
    lines.append(f"  -> 환불요청으로 흡수: {n_to_refund}건 ({n_to_refund/len(complaint_gold)*100:.1f}%)")
    lines.append(f"  -> 배송확인으로 흡수: {n_to_delivery}건 ({n_to_delivery/len(complaint_gold)*100:.1f}%)")
    absorb_rate = (n_to_refund + n_to_delivery) / len(complaint_gold)
    lines.append(f"흡수율(환불+배송 합) = {absorb_rate*100:.1f}% "
                 f"({'재발 우려' if absorb_rate > 0.1 else '재발 낮음'})")

    verdict = "통합 성공 - 다음부터 단일 라벨링 가능" if (kappa >= KAPPA_THRESHOLD and recall >= RECALL_THRESHOLD) \
        else "미달 - 1단계 게이트 보강 필요"
    lines.append(f"\n=== 최종 판정 ===")
    lines.append(verdict)

    output = "\n".join(lines)
    print(output)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(output + "\n")
    print(f"\n저장: {OUT_PATH}")

    # === 요약 3~5줄 ===
    print("\n" + "=" * 60)
    print("요약")
    print("=" * 60)
    print(f"1. Kappa={kappa:.4f}({'충족' if kappa>=KAPPA_THRESHOLD else '미달'}, 기준0.8), "
          f"불만 recall={recall:.4f}({'충족' if recall>=RECALL_THRESHOLD else '미달'}, 기준0.7), "
          f"precision={precision:.4f}")
    print(f"2. 불만제기 흡수 재발: gold 불만 {len(complaint_gold)}건 중 환불요청 {n_to_refund}건 + "
          f"배송확인 {n_to_delivery}건 = {absorb_rate*100:.1f}% "
          f"({'v4가 막았던 흡수 문제 재발함' if absorb_rate > 0.1 else '흡수 재발 낮음 - v4 게이트 없이도 억제됨'})")
    print(f"3. 클래스별 최저 일치율: " +
          min(per_class_recall.items(), key=lambda kv: kv[1][0])[0] +
          f" ({min(per_class_recall.items(), key=lambda kv: kv[1][0])[1][0]:.3f})")
    print(f"4. 파싱 실패율: {n_fail}/{len(df)}건 ({n_fail/len(df)*100:.2f}%)")
    print(f"5. 판정: {verdict}")

    print("\nV_UNIFIED_VALIDATE_COMPLETE")


if __name__ == "__main__":
    main()
