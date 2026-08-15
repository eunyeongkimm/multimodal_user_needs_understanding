"""Stage 2-11: mismatch subset(surface != gold_actual) 한정, 조건(A/B/C/D)×N별
혼동행렬 + 클래스별 precision/recall/F1, 그리고 acoustic 추가(A->B, C->D)가
"부정 종착지" 클래스(환불요청/주문취소/불만제기)의 recall/precision을 어떻게
바꾸는지(trigger-happy 여부) 분석한다.

용어 매핑 (사용자 확인 완료):
  - gold_actual := gold_actual_batch1_final.parquet의 'label' 컬럼
  - gold_surface := stage2d_gpt_predictions.parquet에서 condition=='A' & n==1인
    행의 predicted_label (별도 사람 정답 컬럼이 없어, "첫 발화 하나만 본 모델
    예측"을 surface proxy로 사용 — 기존 stage2g/h/i/j 분석과 동일 정의)
  - mismatch subset := gold_surface != gold_actual 인 call_id 집합

데이터 이상: predicted_label 중 taxonomy(7개 카테고리) 밖 값이 있으면 삭제/보정하지
않고 그대로 두되 "taxonomy 밖 예측"으로 별도 카운트해서 보고한다.

출력: 콘솔에 마크다운 표 + outputs/stage2_mismatch_confusion_metrics.csv,
      outputs/stage2_mismatch_acoustic_delta.csv
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
PRED_PATH = BASE_DIR / "outputs" / "stage2d_gpt_predictions.parquet"
GOLD_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_final.parquet"
OUT_METRICS_PATH = BASE_DIR / "outputs" / "stage2_mismatch_confusion_metrics.csv"
OUT_DELTA_PATH = BASE_DIR / "outputs" / "stage2_mismatch_acoustic_delta.csv"

CATEGORY_ORDER = ["환불요청", "주문취소", "불만제기", "배송확인", "교환반품", "구매진행", "서비스이용"]
NEGATIVE_DESTINATION = ["환불요청", "주문취소", "불만제기"]
N_VALUES = [1, 2, 3, 5]
CONDITIONS = ["A", "B", "C", "D"]
LOW_SUPPORT_THRESHOLD = 30


def load_data():
    preds = pd.read_parquet(PRED_PATH)
    gold = pd.read_parquet(GOLD_PATH)[["call_id", "label"]].rename(columns={"label": "actual_label"})
    merged = preds.merge(gold, on="call_id", how="left")
    return merged


def build_mismatch_ids(merged: pd.DataFrame) -> set:
    surface = merged[(merged["condition"] == "A") & (merged["n"] == 1)]
    return set(surface[surface["predicted_label"] != surface["actual_label"]]["call_id"])


def report_stage2(merged: pd.DataFrame, mismatch_ids: set):
    surface = merged[(merged["condition"] == "A") & (merged["n"] == 1)]
    n_surface_total = len(surface)
    n_mismatch = len(mismatch_ids)

    print("=" * 70)
    print("2단계: mismatch subset 구성")
    print("=" * 70)
    print(f"surface(조건A, N=1) 유효 콜: {n_surface_total}건")
    print(f"mismatch 콜(surface != gold_actual): {n_mismatch}건 "
          f"({n_mismatch/n_surface_total*100:.1f}% of surface 유효 콜, "
          f"{n_mismatch/19847*100:.1f}% of 전체 19,847건)")

    sub_gold = merged[merged["call_id"].isin(mismatch_ids) & (merged["condition"] == "A") & (merged["n"] == 1)]
    dist = sub_gold["actual_label"].value_counts().reindex(CATEGORY_ORDER, fill_value=0)
    dist_pct = (dist / dist.sum() * 100).round(2)
    print("\nmismatch subset 내 gold_actual 클래스 분포:")
    tbl = pd.DataFrame({"건수": dist, "비율(%)": dist_pct})
    print(tbl.to_string())
    neg_sum = dist.loc[NEGATIVE_DESTINATION].sum()
    print(f"\n부정 종착지(환불요청+주문취소+불만제기) 합계: {neg_sum}건 "
          f"({neg_sum/dist.sum()*100:.2f}%) vs 전체 gold_actual에서의 비율 "
          f"{(8563+775+484)/19847*100:.2f}%")
    print()


def compute_condition_n_metrics(merged: pd.DataFrame, mismatch_ids: set) -> pd.DataFrame:
    sub_all = merged[merged["call_id"].isin(mismatch_ids)]
    rows = []

    for n in N_VALUES:
        for cond in CONDITIONS:
            sub = sub_all[(sub_all["n"] == n) & (sub_all["condition"] == cond)]
            if sub.empty:
                continue

            y_true = sub["actual_label"]
            y_pred = sub["predicted_label"]

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
                    "support": int(support[i]),
                    "low_support_flag": support[i] < LOW_SUPPORT_THRESHOLD,
                    "n_total": len(sub), "accuracy": acc,
                    "macro_f1": macro_f1, "weighted_f1": weighted_f1,
                    "n_out_of_taxonomy_pred": int(n_out_of_taxonomy),
                })
    return pd.DataFrame(rows)


def print_confusion_and_metrics(merged: pd.DataFrame, mismatch_ids: set, n: int, cond: str):
    sub = merged[merged["call_id"].isin(mismatch_ids) & (merged["n"] == n) & (merged["condition"] == cond)]
    y_true = sub["actual_label"]
    y_pred = sub["predicted_label"]

    ct = pd.crosstab(y_true, y_pred, dropna=False)
    ct = ct.reindex(index=CATEGORY_ORDER, fill_value=0)
    extra_cols = [c for c in ct.columns if c not in CATEGORY_ORDER]
    col_order = CATEGORY_ORDER + extra_cols
    ct = ct.reindex(columns=col_order, fill_value=0)

    print(f"\n--- N={n}, 조건={cond} (표본 {len(sub)}건) 혼동행렬 (행=gold_actual, 열=predicted) ---")
    print(ct.to_string())
    if extra_cols:
        print(f"  (taxonomy 밖 예측 컬럼: {extra_cols})")


def print_metrics_table(metrics_df: pd.DataFrame, n: int, cond: str):
    sub = metrics_df[(metrics_df["n"] == n) & (metrics_df["condition"] == cond)]
    if sub.empty:
        return
    display = sub[["class", "precision", "recall", "f1", "support", "low_support_flag"]].copy()
    display["precision"] = display["precision"].round(4)
    display["recall"] = display["recall"].round(4)
    display["f1"] = display["f1"].round(4)
    print(f"N={n}, 조건={cond}: accuracy={sub['accuracy'].iloc[0]:.4f}, "
          f"macro-F1={sub['macro_f1'].iloc[0]:.4f}, weighted-F1={sub['weighted_f1'].iloc[0]:.4f}, "
          f"taxonomy 밖 예측={sub['n_out_of_taxonomy_pred'].iloc[0]}건")
    print(display.to_string(index=False))


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
                    "support": m_no.loc[cls, "support"],
                    "low_support_flag": m_no.loc[cls, "low_support_flag"],
                    "trigger_happy_flag": trigger_happy,
                })
    return pd.DataFrame(rows)


def print_delta_table(delta_df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("4단계: 음향 효과 델타 (A->B, C->D) — 부정 종착지 클래스 강조")
    print("=" * 70)

    for n in N_VALUES:
        for pair in ["customer_only", "all"]:
            sub = delta_df[(delta_df["n"] == n) & (delta_df["pair"] == pair)]
            if sub.empty:
                continue
            cond_no = sub["cond_no_acoustic"].iloc[0]
            cond_ac = sub["cond_acoustic"].iloc[0]
            print(f"\n--- N={n}, {pair}({cond_no}->{cond_ac}) ---")
            disp = sub[["class", "is_negative_destination", "precision_no_ac", "precision_ac",
                        "delta_precision", "recall_no_ac", "recall_ac", "delta_recall",
                        "delta_f1", "support", "trigger_happy_flag"]].copy()
            for c in ["precision_no_ac", "precision_ac", "delta_precision",
                      "recall_no_ac", "recall_ac", "delta_recall", "delta_f1"]:
                disp[c] = disp[c].round(4)
            print(disp.to_string(index=False))

    print("\n--- trigger-happy (recall↑ & precision↓) 로 플래그된 케이스만 모아보기 ---")
    trig = delta_df[delta_df["trigger_happy_flag"]]
    if len(trig):
        print(trig[["n", "pair", "class", "is_negative_destination",
                     "delta_precision", "delta_recall", "delta_f1"]].to_string(index=False))
    else:
        print("(없음)")

    print("\n--- 부정 종착지 클래스(환불요청/주문취소/불만제기)만 모아보기 ---")
    neg = delta_df[delta_df["is_negative_destination"]]
    print(neg[["n", "pair", "class", "delta_precision", "delta_recall", "delta_f1",
               "support", "trigger_happy_flag"]].to_string(index=False))


def main():
    merged = load_data()
    mismatch_ids = build_mismatch_ids(merged)
    report_stage2(merged, mismatch_ids)

    print("=" * 70)
    print("3단계: 조건x N별 혼동행렬 + 클래스별 precision/recall/F1 (mismatch subset)")
    print("=" * 70)
    metrics_df = compute_condition_n_metrics(merged, mismatch_ids)
    for n in N_VALUES:
        for cond in CONDITIONS:
            print_confusion_and_metrics(merged, mismatch_ids, n, cond)
            print_metrics_table(metrics_df, n, cond)

    metrics_df.to_csv(OUT_METRICS_PATH, index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT_METRICS_PATH}")

    delta_df = compute_acoustic_delta(metrics_df)
    print_delta_table(delta_df)
    delta_df.to_csv(OUT_DELTA_PATH, index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT_DELTA_PATH}")

    print("\n" + "=" * 70)
    print("요약")
    print("=" * 70)
    neg = delta_df[delta_df["is_negative_destination"]]
    trig_neg = neg[neg["trigger_happy_flag"]]
    good_neg = neg[(neg["delta_recall"] > 0) & (neg["delta_precision"] >= 0)]
    if len(trig_neg):
        print("Trigger-happy(부정 종착지, recall↑ precision↓):")
        for _, r in trig_neg.iterrows():
            print(f"  N={r['n']}, {r['pair']}, {r['class']}: "
                  f"recall {r['delta_recall']:+.4f}, precision {r['delta_precision']:+.4f}")
    else:
        print("부정 종착지 클래스 중 trigger-happy로 플래그된 케이스 없음.")
    if len(good_neg):
        print("Precision 희생 없이 recall만 오른(또는 유지된) 부정 종착지 케이스:")
        for _, r in good_neg.iterrows():
            print(f"  N={r['n']}, {r['pair']}, {r['class']}: "
                  f"recall {r['delta_recall']:+.4f}, precision {r['delta_precision']:+.4f}")


if __name__ == "__main__":
    main()
