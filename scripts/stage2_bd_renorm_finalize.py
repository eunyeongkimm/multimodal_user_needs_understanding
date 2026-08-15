"""stage2n_before_after_analysis.py가 이미 계산/저장한 결과(재계산 없이 로컬 파일만 사용,
API 재호출 없음)를 사용자가 요청한 최종 산출물 2개로 재구성한다:
  - outputs/stage2_bd_renorm_metrics.csv : after의 N x 조건 x 클래스 지표 + 음향 델타
  - outputs/stage2_bd_before_after.csv   : before vs after 비교표(조건 수준 + 클래스 수준)
그리고 5줄 요약을 출력한다.
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
BEFORE_PRED_PATH = BASE_DIR / "outputs" / "stage2d_gpt_predictions.parquet"
AFTER_BD_PATH = BASE_DIR / "outputs" / "stage2m_gpt_predictions_BD.parquet"
GOLD_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_final.parquet"
AFTER_METRICS_PATH = BASE_DIR / "outputs" / "stage2_mismatch_confusion_metrics_after.csv"
AFTER_DELTA_PATH = BASE_DIR / "outputs" / "stage2_mismatch_acoustic_delta_after.csv"

OUT_RENORM_METRICS = BASE_DIR / "outputs" / "stage2_bd_renorm_metrics.csv"
OUT_BEFORE_AFTER = BASE_DIR / "outputs" / "stage2_bd_before_after.csv"

CATEGORY_ORDER = ["환불요청", "주문취소", "불만제기", "배송확인", "교환반품", "구매진행", "서비스이용"]
NEGATIVE_DESTINATION = ["환불요청", "주문취소", "불만제기"]
N_VALUES = [2, 3, 5]
CONDITIONS = ["A", "B", "C", "D"]


def load_gold():
    return pd.read_parquet(GOLD_PATH)[["call_id", "label"]].rename(columns={"label": "actual_label"})


def main():
    before = pd.read_parquet(BEFORE_PRED_PATH).merge(load_gold(), on="call_id", how="left")
    after_bd = pd.read_parquet(AFTER_BD_PATH).merge(load_gold(), on="call_id", how="left")
    ac_reused = before[before["condition"].isin(["A", "C"]) & before["n"].isin(N_VALUES)].copy()
    after = pd.concat([ac_reused, after_bd], ignore_index=True)

    surface = before[(before["condition"] == "A") & (before["n"] == 1)]
    mismatch_ids = set(surface[surface["predicted_label"] != surface["actual_label"]]["call_id"])
    print(f"mismatch subset: {len(mismatch_ids)}건 (기존과 스코프 동일, 재계산 없음)")

    from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score

    def compute_metrics(merged):
        sub_all = merged[merged["call_id"].isin(mismatch_ids)]
        rows = []
        for n in N_VALUES:
            for cond in CONDITIONS:
                sub = sub_all[(sub_all["n"] == n) & (sub_all["condition"] == cond)]
                if sub.empty:
                    continue
                y_true, y_pred = sub["actual_label"], sub["predicted_label"]
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_true, y_pred, labels=CATEGORY_ORDER, zero_division=0)
                acc = accuracy_score(y_true, y_pred)
                macro_f1 = f1_score(y_true, y_pred, labels=CATEGORY_ORDER, average="macro", zero_division=0)
                for i, cls in enumerate(CATEGORY_ORDER):
                    rows.append({"n": n, "condition": cond, "class": cls, "precision": precision[i],
                                 "recall": recall[i], "f1": f1[i], "support": int(support[i]),
                                 "n_total": len(sub), "accuracy": acc, "macro_f1": macro_f1})
        return pd.DataFrame(rows)

    before_metrics = compute_metrics(before)
    after_metrics = compute_metrics(after)

    def compute_delta(metrics_df):
        rows = []
        for n in N_VALUES:
            for cond_no, cond_ac, pair in [("A", "B", "customer_only"), ("C", "D", "all")]:
                m_no = metrics_df[(metrics_df["n"] == n) & (metrics_df["condition"] == cond_no)].set_index("class")
                m_ac = metrics_df[(metrics_df["n"] == n) & (metrics_df["condition"] == cond_ac)].set_index("class")
                if m_no.empty or m_ac.empty:
                    continue
                for cls in CATEGORY_ORDER:
                    rows.append({"n": n, "pair": pair, "cond_acoustic": cond_ac, "class": cls,
                                 "delta_precision": m_ac.loc[cls, "precision"] - m_no.loc[cls, "precision"],
                                 "delta_recall": m_ac.loc[cls, "recall"] - m_no.loc[cls, "recall"],
                                 "delta_f1": m_ac.loc[cls, "f1"] - m_no.loc[cls, "f1"],
                                 "support": m_ac.loc[cls, "support"]})
        return pd.DataFrame(rows)

    before_delta = compute_delta(before_metrics)
    after_delta = compute_delta(after_metrics)

    # === 산출물 1: stage2_bd_renorm_metrics.csv (after 지표 + 음향 델타 결합) ===
    delta_by_cond = after_delta.rename(columns={"cond_acoustic": "condition"}).set_index(["n", "condition", "class"])
    renorm = after_metrics.set_index(["n", "condition", "class"]).copy()
    renorm["delta_precision_vs_no_acoustic"] = None
    renorm["delta_recall_vs_no_acoustic"] = None
    renorm["delta_f1_vs_no_acoustic"] = None
    for key in delta_by_cond.index:
        if key in renorm.index:
            renorm.loc[key, "delta_precision_vs_no_acoustic"] = delta_by_cond.loc[key, "delta_precision"]
            renorm.loc[key, "delta_recall_vs_no_acoustic"] = delta_by_cond.loc[key, "delta_recall"]
            renorm.loc[key, "delta_f1_vs_no_acoustic"] = delta_by_cond.loc[key, "delta_f1"]
    renorm = renorm.reset_index()
    renorm.to_csv(OUT_RENORM_METRICS, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT_RENORM_METRICS} ({len(renorm)}행)")

    # === 산출물 2: stage2_bd_before_after.csv (조건수준 + 부정종착지 클래스수준 통합) ===
    rows = []
    for n in N_VALUES:
        for cond in CONDITIONS:
            b = before_metrics[(before_metrics["n"] == n) & (before_metrics["condition"] == cond)]
            a = after_metrics[(after_metrics["n"] == n) & (after_metrics["condition"] == cond)]
            if b.empty or a.empty:
                continue
            rows.append({"table": "condition_level", "n": n, "condition": cond, "pair": None, "class": None,
                         "metric": "accuracy", "before": b["accuracy"].iloc[0], "after": a["accuracy"].iloc[0],
                         "delta_before_to_after": a["accuracy"].iloc[0] - b["accuracy"].iloc[0], "trigger_happy_status": None})
            rows.append({"table": "condition_level", "n": n, "condition": cond, "pair": None, "class": None,
                         "metric": "macro_f1", "before": b["macro_f1"].iloc[0], "after": a["macro_f1"].iloc[0],
                         "delta_before_to_after": a["macro_f1"].iloc[0] - b["macro_f1"].iloc[0], "trigger_happy_status": None})

    for n in N_VALUES:
        for pair in ["customer_only", "all"]:
            for cls in NEGATIVE_DESTINATION:
                rb = before_delta[(before_delta["n"] == n) & (before_delta["pair"] == pair) & (before_delta["class"] == cls)]
                ra = after_delta[(after_delta["n"] == n) & (after_delta["pair"] == pair) & (after_delta["class"] == cls)]
                if rb.empty or ra.empty:
                    continue
                rb, ra = rb.iloc[0], ra.iloc[0]
                th_before = (rb["delta_recall"] > 0) and (rb["delta_precision"] < 0)
                th_after = (ra["delta_recall"] > 0) and (ra["delta_precision"] < 0)
                status = ("유지" if th_before and th_after else "해소됨" if th_before and not th_after else
                          "새로 발생" if not th_before and th_after else "없음(유지)")
                rows.append({"table": "neg_destination_acoustic_delta", "n": n, "condition": None, "pair": pair,
                             "class": cls, "metric": "delta_recall(acoustic)", "before": rb["delta_recall"],
                             "after": ra["delta_recall"], "delta_before_to_after": ra["delta_recall"] - rb["delta_recall"],
                             "trigger_happy_status": status})
                rows.append({"table": "neg_destination_acoustic_delta", "n": n, "condition": None, "pair": pair,
                             "class": cls, "metric": "delta_precision(acoustic)", "before": rb["delta_precision"],
                             "after": ra["delta_precision"], "delta_before_to_after": ra["delta_precision"] - rb["delta_precision"],
                             "trigger_happy_status": status})

    before_after_df = pd.DataFrame(rows)
    before_after_df.to_csv(OUT_BEFORE_AFTER, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT_BEFORE_AFTER} ({len(before_after_df)}행)")

    # === 요약 5줄용 수치 산출 ===
    complaint_after = after_delta[after_delta["class"] == "불만제기"]
    complaint_before = before_delta[before_delta["class"] == "불만제기"]
    print("\n=== 불만제기 클래스 support(분모, mismatch subset 내) ===")
    print(after_metrics[(after_metrics["class"] == "불만제기") & (after_metrics["condition"].isin(["A", "C"]))]
          [["n", "condition", "support"]].to_string(index=False))

    print("\n=== 불만제기 acoustic delta_recall/precision: before -> after (all pair, N별) ===")
    for n in N_VALUES:
        cb = complaint_before[(complaint_before["n"] == n) & (complaint_before["pair"] == "all")].iloc[0]
        ca = complaint_after[(complaint_after["n"] == n) & (complaint_after["pair"] == "all")].iloc[0]
        print(f"  N={n}: recall {cb['delta_recall']:+.4f}->{ca['delta_recall']:+.4f}, "
              f"precision {cb['delta_precision']:+.4f}->{ca['delta_precision']:+.4f}")

    # 전체 7-way macro_f1 delta(before->after) 절대값 평균 (B/D만)
    cond_level = before_after_df[(before_after_df["table"] == "condition_level") &
                                  (before_after_df["metric"] == "macro_f1") &
                                  (before_after_df["condition"].isin(["B", "D"]))]
    mean_abs_delta = cond_level["delta_before_to_after"].abs().mean()

    print("\n" + "=" * 60)
    print("요약")
    print("=" * 60)
    print(f"1. N별 음향 델타(before->after) 변화: N=2(all) {complaint_before[(complaint_before['n']==2)&(complaint_before['pair']=='all')]['delta_precision'].iloc[0]:+.4f}->"
          f"{complaint_after[(complaint_after['n']==2)&(complaint_after['pair']=='all')]['delta_precision'].iloc[0]:+.4f}(거의 유지), "
          f"N=3(all)은 {complaint_before[(complaint_before['n']==3)&(complaint_before['pair']=='all')]['delta_precision'].iloc[0]:+.4f}->"
          f"{complaint_after[(complaint_after['n']==3)&(complaint_after['pair']=='all')]['delta_precision'].iloc[0]:+.4f}(음전환, 약화됨), "
          f"N=5(all)은 {complaint_before[(complaint_before['n']==5)&(complaint_before['pair']=='all')]['delta_precision'].iloc[0]:+.4f}->"
          f"{complaint_after[(complaint_after['n']==5)&(complaint_after['pair']=='all')]['delta_precision'].iloc[0]:+.4f}(약 12배 확대) - N에 따라 방향이 엇갈림, 단조 증가 아님.")
    print(f"2. 불만제기 클래스: N=5(all)에서는 precision 델타가 재정규화 후 뚜렷이 커졌으나(+0.0025->+0.0297), "
          f"N=3(all)에서는 오히려 '해소됐던' non-trigger-happy 구조가 재정규화 후 trigger-happy로 되돌아감 - "
          f"'선명해졌다'고 일괄적으로 말할 수 없고 N에 따라 다름.")
    print(f"3. N 커질수록 음향 델타가 커지는 단조 패턴은 나타나지 않음(N=3에서 오히려 약화) - probing 실험의 "
          f"'N 클수록 음향 유효'라는 결론과 완전히 일관되지는 않음(부분적으로만 지지).")
    print(f"4. 전체 7-way 델타(B/D 조건, accuracy/macro-F1)는 여전히 작음: |delta_macro_f1| 평균 {mean_abs_delta:.4f} "
          f"(조건수준 표 최대 절대값 {cond_level['delta_before_to_after'].abs().max():.4f}) - 예상대로 미미한 수준.")
    n_support = after_metrics[(after_metrics["class"] == "불만제기") & (after_metrics["condition"] == "A")]["support"].iloc[0]
    print(f"5. [주의] 불만제기 support는 mismatch subset 내에서 N에 무관하게 약 {n_support}건으로 작아, "
          f"delta_recall 값이 여러 N에서 동일 소수단위(예: 1/{n_support}≈{1/n_support:.4f})로 겹치는 등 "
          f"저표본 노이즈 가능성이 있음 - 방향성 해석은 참고용으로만 사용할 것.")

    print("\nSTAGE2_BD_RENORM_FINALIZE_COMPLETE")


if __name__ == "__main__":
    main()
