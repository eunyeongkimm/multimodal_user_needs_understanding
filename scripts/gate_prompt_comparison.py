"""v3(게이트 없음) vs v3+게이트(강/중/약 신호 주입) 비교. 층화 2,000 샘플(불만
오버샘플) 기준. 파싱 실패는 오답으로 처리(제외 금지).

출력: outputs/gate_prompt_comparison.txt
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

BASE_DIR = Path(__file__).resolve().parent.parent
PRED_PATH = BASE_DIR / "outputs" / "gate_prompt_gpt_predictions.parquet"
OUT_PATH = BASE_DIR / "outputs" / "gate_prompt_comparison.txt"

CATEGORY_ORDER = ["환불요청", "주문취소", "불만제기", "배송확인", "교환반품", "구매진행", "서비스이용"]
PARSE_FAIL_SENTINEL = "PARSE_FAIL"


def main():
    df = pd.read_parquet(PRED_PATH)
    n_fail = df["predicted_label"].isna().sum()
    df["predicted_label"] = df["predicted_label"].fillna(PARSE_FAIL_SENTINEL)

    lines = []
    lines.append("v3(게이트 없음) vs v3+게이트(강/중/약) 비교 - 층화 2,000 샘플(불만 오버샘플)")
    lines.append(f"파싱 실패(오답 처리): {n_fail}/{len(df)}건 ({n_fail/len(df)*100:.2f}%)")
    lines.append("")

    metrics = {}
    for variant in ["base", "gate"]:
        sub = df[df["variant"] == variant]
        y_true = sub["gold_actual"]
        y_pred = sub["predicted_label"]
        acc = (y_true == y_pred).mean()
        macro_f1 = f1_score(y_true, y_pred, labels=CATEGORY_ORDER, average="macro", zero_division=0)
        yt = (y_true == "불만제기").astype(int)
        yp = (y_pred == "불만제기").astype(int)
        recall = recall_score(yt, yp, zero_division=0)
        precision = precision_score(yt, yp, zero_division=0)
        f1 = f1_score(yt, yp, zero_division=0)
        n_pred_complaint = yp.sum()
        metrics[variant] = {"n": len(sub), "acc": acc, "macro_f1": macro_f1,
                             "recall": recall, "precision": precision, "f1": f1,
                             "n_pred_complaint": n_pred_complaint}

    lines.append("=== 1) 불만제기 recall/precision/F1 ===")
    lines.append(f"{'variant':<8} {'n':>6} {'recall':>8} {'precision':>10} {'f1':>8} {'예측불만건수':>10}")
    for variant in ["base", "gate"]:
        m = metrics[variant]
        lines.append(f"{variant:<8} {m['n']:>6} {m['recall']:>8.4f} {m['precision']:>10.4f} "
                     f"{m['f1']:>8.4f} {m['n_pred_complaint']:>10}")
    d_recall = metrics["gate"]["recall"] - metrics["base"]["recall"]
    d_precision = metrics["gate"]["precision"] - metrics["base"]["precision"]
    lines.append(f"\nΔrecall(gate-base) = {d_recall:+.4f}, Δprecision(gate-base) = {d_precision:+.4f}")
    verdict_1 = "recall 상승 & precision 유지/상승 - 게이트 효과 긍정적" if (d_recall > 0 and d_precision >= -0.02) \
        else "recall 상승했으나 precision 하락(오탐 증가 의심)" if (d_recall > 0 and d_precision < -0.02) \
        else "recall 개선 없음/하락"
    lines.append(f"판정: {verdict_1}")

    lines.append(f"\n=== 2) 전체 7-way accuracy / macro-F1 ===")
    lines.append(f"base: accuracy={metrics['base']['acc']:.4f}, macro_f1={metrics['base']['macro_f1']:.4f}")
    lines.append(f"gate: accuracy={metrics['gate']['acc']:.4f}, macro_f1={metrics['gate']['macro_f1']:.4f}")
    lines.append(f"Δmacro_f1 = {metrics['gate']['macro_f1'] - metrics['base']['macro_f1']:+.4f}")

    lines.append(f"\n=== 3) 게이트 신호별(강/중/약) 실제 불만 비율 (이 샘플 기준, gold_actual) ===")
    base_sub = df[df["variant"] == "base"]
    for level in ["강", "중", "약"]:
        sub = base_sub[base_sub["gate_level"] == level]
        rate = (sub["gold_actual"] == "불만제기").mean()
        n_c = (sub["gold_actual"] == "불만제기").sum()
        lines.append(f"  {level}: {rate*100:.2f}% ({n_c}/{len(sub)}건)")

    lines.append(f"\n=== 4) 게이트 레벨별 v3+게이트 예측 불만 recall (강에서 recall 더 높아야 타당) ===")
    gate_sub = df[df["variant"] == "gate"]
    for level in ["강", "중", "약"]:
        sub = gate_sub[gate_sub["gate_level"] == level]
        yt = (sub["gold_actual"] == "불만제기").astype(int)
        yp = (sub["predicted_label"] == "불만제기").astype(int)
        r = recall_score(yt, yp, zero_division=0) if yt.sum() > 0 else float("nan")
        lines.append(f"  {level}: recall={r:.4f} (gold 불만 {yt.sum()}건 중)")

    output = "\n".join(lines)
    print(output)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(output + "\n")
    print(f"\n저장: {OUT_PATH}")

    print("\n" + "=" * 60)
    print("요약")
    print("=" * 60)
    print(f"1. 불만 recall: base={metrics['base']['recall']:.4f} -> gate={metrics['gate']['recall']:.4f} "
          f"({d_recall:+.4f})")
    print(f"2. 불만 precision: base={metrics['base']['precision']:.4f} -> gate={metrics['gate']['precision']:.4f} "
          f"({d_precision:+.4f}) - {'유지/개선' if d_precision >= -0.02 else '하락(오탐 증가 의심)'}")
    print(f"3. 전체 macro-F1: base={metrics['base']['macro_f1']:.4f} -> gate={metrics['gate']['macro_f1']:.4f}")
    print(f"4. 강/중/약 신호 타당성(이 샘플 기준): 강 vs 약 실제불만비율 차이로 위 표 참고")
    print(f"5. 판정: {verdict_1}")

    print("\nGATE_PROMPT_COMPARISON_COMPLETE")


if __name__ == "__main__":
    main()
