"""Stage 2-12: 성별 이원화 재정규화 전(before)/후(after) mismatch subset
confusion/precision/recall/F1 + acoustic 델타 비교.

- before: 기존 outputs/stage2d_gpt_predictions.parquet (v1 정규화 기반)
- after: N=2,3,5의 조건 B/D만 outputs/stage2m_gpt_predictions_BD.parquet(재정규화
  기반 재실행 결과)로 교체하고, 조건 A/C(N=2,3,5)는 acoustic을 안 쓰므로
  before 값을 그대로 재사용해서 구성.
- mismatch subset 정의(조건A, N=1)는 재실행하지 않았으므로 before/after 동일
  (before 데이터에서 그대로 산출).
- N=1은 정의상 자명값이라 이 비교에서 제외 (사용자 지시).

출력:
  - outputs/stage2_mismatch_confusion_metrics_after.csv
  - outputs/stage2_mismatch_acoustic_delta_after.csv
  - outputs/stage2_before_after_comparison.csv
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
BEFORE_PRED_PATH = BASE_DIR / "outputs" / "stage2d_gpt_predictions.parquet"
AFTER_BD_PATH = BASE_DIR / "outputs" / "stage2m_gpt_predictions_BD.parquet"
GOLD_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_final.parquet"

OUT_METRICS_AFTER = BASE_DIR / "outputs" / "stage2_mismatch_confusion_metrics_after.csv"
OUT_DELTA_AFTER = BASE_DIR / "outputs" / "stage2_mismatch_acoustic_delta_after.csv"
OUT_COMPARISON = BASE_DIR / "outputs" / "stage2_before_after_comparison.csv"

CATEGORY_ORDER = ["환불요청", "주문취소", "불만제기", "배송확인", "교환반품", "구매진행", "서비스이용"]
NEGATIVE_DESTINATION = ["환불요청", "주문취소", "불만제기"]
N_VALUES = [2, 3, 5]  # N=1 제외
CONDITIONS = ["A", "B", "C", "D"]
LOW_SUPPORT_THRESHOLD = 30


def load_gold():
    return pd.read_parquet(GOLD_PATH)[["call_id", "label"]].rename(columns={"label": "actual_label"})


def load_before():
    preds = pd.read_parquet(BEFORE_PRED_PATH)
    gold = load_gold()
    return preds.merge(gold, on="call_id", how="left")


def load_after(before: pd.DataFrame):
    after_bd = pd.read_parquet(AFTER_BD_PATH)
    gold = load_gold()
    after_bd = after_bd.merge(gold, on="call_id", how="left")

    ac_reused = before[before["condition"].isin(["A", "C"]) & before["n"].isin(N_VALUES)].copy()
    after = pd.concat([ac_reused, after_bd], ignore_index=True)
    return after


def build_mismatch_ids(before: pd.DataFrame) -> set:
    surface = before[(before["condition"] == "A") & (before["n"] == 1)]
    return set(surface[surface["predicted_label"] != surface["actual_label"]]["call_id"])


def compute_condition_n_metrics(merged: pd.DataFrame, mismatch_ids: set) -> pd.DataFrame:
    sub_all = merged[merged["call_id"].isin(mismatch_ids)]
    rows = []
    for n in N_VALUES:
        for cond in CONDITIONS:
            sub = sub_all[(sub_all["n"] == n) & (sub_all["condition"] == cond)]
            if sub.empty:
                continue
            y_true, y_pred = sub["actual_label"], sub["predicted_label"]
            n_out_of_taxonomy = (~y_pred.isin(CATEGORY_ORDER)).sum()
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, labels=CATEGORY_ORDER, zero_division=0
            )
            acc = accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(y_true, y_pred, labels=CATEGORY_ORDER, average="macro", zero_division=0)
            weighted_f1 = f1_score(y_true, y_pred, labels=CATEGORY_ORDER, average="weighted", zero_division=0)
            for i, cls in enumerate(CATEGORY_ORDER):
                rows.append({
                    "n": n, "condition": cond, "class": cls,
                    "precision": precision[i], "recall": recall[i], "f1": f1[i],
                    "support": int(support[i]), "low_support_flag": support[i] < LOW_SUPPORT_THRESHOLD,
                    "n_total": len(sub), "accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
                    "n_out_of_taxonomy_pred": int(n_out_of_taxonomy),
                })
    return pd.DataFrame(rows)


def compute_acoustic_delta(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n in N_VALUES:
        for cond_no, cond_ac, pair_name in [("A", "B", "customer_only"), ("C", "D", "all")]:
            m_no = metrics_df[(metrics_df["n"] == n) & (metrics_df["condition"] == cond_no)].set_index("class")
            m_ac = metrics_df[(metrics_df["n"] == n) & (metrics_df["condition"] == cond_ac)].set_index("class")
            if m_no.empty or m_ac.empty:
                continue
            for cls in CATEGORY_ORDER:
                if cls not in m_no.index or cls not in m_ac.index:
                    continue
                d_prec = m_ac.loc[cls, "precision"] - m_no.loc[cls, "precision"]
                d_rec = m_ac.loc[cls, "recall"] - m_no.loc[cls, "recall"]
                d_f1 = m_ac.loc[cls, "f1"] - m_no.loc[cls, "f1"]
                trigger_happy = (d_rec > 0) and (d_prec < 0)
                rows.append({
                    "n": n, "pair": pair_name, "cond_no_acoustic": cond_no, "cond_acoustic": cond_ac,
                    "class": cls, "is_negative_destination": cls in NEGATIVE_DESTINATION,
                    "precision_no_ac": m_no.loc[cls, "precision"], "precision_ac": m_ac.loc[cls, "precision"],
                    "delta_precision": d_prec,
                    "recall_no_ac": m_no.loc[cls, "recall"], "recall_ac": m_ac.loc[cls, "recall"],
                    "delta_recall": d_rec,
                    "f1_no_ac": m_no.loc[cls, "f1"], "f1_ac": m_ac.loc[cls, "f1"], "delta_f1": d_f1,
                    "support": m_no.loc[cls, "support"], "low_support_flag": m_no.loc[cls, "low_support_flag"],
                    "trigger_happy_flag": trigger_happy,
                })
    return pd.DataFrame(rows)


def main():
    before = load_before()
    after = load_after(before)
    mismatch_ids = build_mismatch_ids(before)
    print(f"mismatch subset: {len(mismatch_ids)}건 (before/after 동일, N=1 재실행 안 함)\n")

    print("=" * 70)
    print("BEFORE (기존 정규화) 지표 계산 중...")
    before_metrics = compute_condition_n_metrics(before, mismatch_ids)
    before_delta = compute_acoustic_delta(before_metrics)

    print("AFTER (성별 이원화 재정규화) 지표 계산 중...")
    after_metrics = compute_condition_n_metrics(after, mismatch_ids)
    after_delta = compute_acoustic_delta(after_metrics)

    after_metrics.to_csv(OUT_METRICS_AFTER, index=False, encoding="utf-8-sig")
    after_delta.to_csv(OUT_DELTA_AFTER, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT_METRICS_AFTER}")
    print(f"저장: {OUT_DELTA_AFTER}")

    # === 1) 조건x N별 accuracy/macroF1 before vs after ===
    print("\n" + "=" * 70)
    print("조건 x N별 accuracy / macro-F1: before vs after")
    print("=" * 70)
    acc_rows = []
    for n in N_VALUES:
        for cond in CONDITIONS:
            b = before_metrics[(before_metrics["n"] == n) & (before_metrics["condition"] == cond)]
            a = after_metrics[(after_metrics["n"] == n) & (after_metrics["condition"] == cond)]
            if b.empty or a.empty:
                continue
            acc_rows.append({
                "n": n, "condition": cond,
                "accuracy_before": b["accuracy"].iloc[0], "accuracy_after": a["accuracy"].iloc[0],
                "delta_accuracy": a["accuracy"].iloc[0] - b["accuracy"].iloc[0],
                "macro_f1_before": b["macro_f1"].iloc[0], "macro_f1_after": a["macro_f1"].iloc[0],
                "delta_macro_f1": a["macro_f1"].iloc[0] - b["macro_f1"].iloc[0],
                "changed": (a["accuracy"].iloc[0] != b["accuracy"].iloc[0]),
            })
    acc_df = pd.DataFrame(acc_rows)
    print(acc_df.to_string(index=False))
    print("\n(참고: A/C 조건은 재실행 안 했으므로 delta=0이 정상. B/D만 변화 있어야 함)")

    # === 2) 부정 종착지 클래스 Δrecall/Δprecision: before-delta vs after-delta ===
    print("\n" + "=" * 70)
    print("부정 종착지 클래스(환불요청/주문취소/불만제기): acoustic 델타의 before vs after")
    print("=" * 70)
    neg_b = before_delta[before_delta["is_negative_destination"]].copy()
    neg_a = after_delta[after_delta["is_negative_destination"]].copy()
    comp_rows = []
    for _, rb in neg_b.iterrows():
        ra = neg_a[(neg_a["n"] == rb["n"]) & (neg_a["pair"] == rb["pair"]) & (neg_a["class"] == rb["class"])]
        if ra.empty:
            continue
        ra = ra.iloc[0]
        comp_rows.append({
            "n": rb["n"], "pair": rb["pair"], "class": rb["class"],
            "delta_recall_before": rb["delta_recall"], "delta_recall_after": ra["delta_recall"],
            "delta_precision_before": rb["delta_precision"], "delta_precision_after": ra["delta_precision"],
            "trigger_happy_before": rb["trigger_happy_flag"], "trigger_happy_after": ra["trigger_happy_flag"],
            "trigger_happy_status": (
                "유지" if rb["trigger_happy_flag"] and ra["trigger_happy_flag"] else
                "해소됨" if rb["trigger_happy_flag"] and not ra["trigger_happy_flag"] else
                "새로 발생" if not rb["trigger_happy_flag"] and ra["trigger_happy_flag"] else
                "없음(유지)"
            ),
        })
    comp_df = pd.DataFrame(comp_rows)
    print(comp_df.to_string(index=False))
    comp_df.to_csv(OUT_COMPARISON, index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT_COMPARISON}")

    # === 3) 불만제기: customer_only(trigger-happy) vs all(precision도 상승) 구조 유지 여부 ===
    print("\n" + "=" * 70)
    print("특별 점검: 불만제기의 'customer_only=trigger-happy vs all=precision도 상승' 구조")
    print("=" * 70)
    for phase, delta_df in [("BEFORE", before_delta), ("AFTER", after_delta)]:
        sub = delta_df[delta_df["class"] == "불만제기"]
        print(f"\n[{phase}]")
        print(sub[["n", "pair", "delta_precision", "delta_recall", "trigger_happy_flag"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("STAGE2N_COMPLETE")


if __name__ == "__main__":
    main()
