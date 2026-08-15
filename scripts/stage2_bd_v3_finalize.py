"""B/D x N{2,3,5} v3_orig(reasoning, 재정규화 A트랙) 결과를 v_base(같은 재정규화, 기존
stage2m_gpt_predictions_BD.parquet) 대비로 비교한다. 핵심 지표는 불만제기 클래스.
파싱 실패(형식오류/태그누락)는 제외하지 않고 "오답"(PARSE_FAIL sentinel)으로 처리해
정확도/정밀도/재현율 계산에 그대로 포함시킨다.

출력:
  - outputs/stage2_bd_v3_metrics.csv       : v3_orig(after) N x 조건 x 클래스 지표 + 음향 델타
  - outputs/stage2_bd_v3_vs_base.csv       : 불만제기 중심 v_base vs v3_orig 비교
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
BEFORE_PRED_PATH = BASE_DIR / "outputs" / "stage2d_gpt_predictions.parquet"  # A/C 재사용용(원본)
BASE_BD_PATH = BASE_DIR / "outputs" / "stage2m_gpt_predictions_BD.parquet"   # v_base + 재정규화
V3_BD_PATH = BASE_DIR / "outputs" / "stage2_bd_v3_gpt_predictions.parquet"  # v3_orig + 재정규화
GOLD_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_final.parquet"

OUT_METRICS = BASE_DIR / "outputs" / "stage2_bd_v3_metrics.csv"
OUT_VS_BASE = BASE_DIR / "outputs" / "stage2_bd_v3_vs_base.csv"

CATEGORY_ORDER = ["환불요청", "주문취소", "불만제기", "배송확인", "교환반품", "구매진행", "서비스이용"]
N_VALUES = [2, 3, 5]
CONDITIONS = ["A", "B", "C", "D"]
PARSE_FAIL_SENTINEL = "PARSE_FAIL"


def load_gold():
    return pd.read_parquet(GOLD_PATH)[["call_id", "label"]].rename(columns={"label": "actual_label"})


def build_full(bd_path, is_v3: bool):
    """A/C(재사용, 원본 predicted_label 그대로) + B/D(신규) 결합. v3는 파싱실패를 sentinel로 채움."""
    gold = load_gold()
    before_all = pd.read_parquet(BEFORE_PRED_PATH)
    ac_reused = before_all[before_all["condition"].isin(["A", "C"]) & before_all["n"].isin(N_VALUES)].copy()

    bd = pd.read_parquet(bd_path).copy()
    if is_v3:
        n_fail = bd["predicted_label"].isna().sum()
        print(f"v3_orig 파싱 실패(오답 처리): {n_fail}/{len(bd)}건 ({n_fail/len(bd)*100:.2f}%)")
        print(bd["parse_error"].value_counts(dropna=False).head(10))
        bd["predicted_label"] = bd["predicted_label"].fillna(PARSE_FAIL_SENTINEL)

    full = pd.concat([ac_reused, bd], ignore_index=True)
    full = full.merge(gold, on="call_id", how="left")
    return full, (bd["parse_error"].notna().sum() if is_v3 else 0), (len(bd) if is_v3 else 0)


def build_mismatch_ids():
    before_all = pd.read_parquet(BEFORE_PRED_PATH)
    surface = before_all[(before_all["condition"] == "A") & (before_all["n"] == 1)]
    gold = load_gold()
    surface = surface.merge(gold, on="call_id", how="left")
    return set(surface[surface["predicted_label"] != surface["actual_label"]]["call_id"])


def compute_metrics(merged: pd.DataFrame, mismatch_ids: set) -> pd.DataFrame:
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


def compute_delta(metrics_df: pd.DataFrame) -> pd.DataFrame:
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
                             "delta_f1": m_ac.loc[cls, "f1"] - m_no.loc[cls, "f1"]})
    return pd.DataFrame(rows)


def main():
    mismatch_ids = build_mismatch_ids()
    print(f"mismatch subset: {len(mismatch_ids)}건\n")

    base_full, _, _ = build_full(BASE_BD_PATH, is_v3=False)
    v3_full, n_parse_fail, n_v3_total = build_full(V3_BD_PATH, is_v3=True)

    base_metrics = compute_metrics(base_full, mismatch_ids)
    v3_metrics = compute_metrics(v3_full, mismatch_ids)
    base_delta = compute_delta(base_metrics)
    v3_delta = compute_delta(v3_metrics)

    # === 산출물 1: v3_orig(after) 지표 + 음향 델타 ===
    delta_idx = v3_delta.rename(columns={"cond_acoustic": "condition"}).set_index(["n", "condition", "class"])
    out1 = v3_metrics.set_index(["n", "condition", "class"]).copy()
    for col in ["delta_precision_vs_no_acoustic", "delta_recall_vs_no_acoustic", "delta_f1_vs_no_acoustic"]:
        out1[col] = None
    for key in delta_idx.index:
        if key in out1.index:
            out1.loc[key, "delta_precision_vs_no_acoustic"] = delta_idx.loc[key, "delta_precision"]
            out1.loc[key, "delta_recall_vs_no_acoustic"] = delta_idx.loc[key, "delta_recall"]
            out1.loc[key, "delta_f1_vs_no_acoustic"] = delta_idx.loc[key, "delta_f1"]
    out1 = out1.reset_index()
    out1.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT_METRICS} ({len(out1)}행)")

    # === 산출물 2: 불만제기 중심 v_base vs v3_orig 비교 ===
    rows = []
    # (1) 불만제기 recall/precision/F1, N x 조건
    for n in N_VALUES:
        for cond in CONDITIONS:
            b = base_metrics[(base_metrics["n"] == n) & (base_metrics["condition"] == cond) & (base_metrics["class"] == "불만제기")]
            v = v3_metrics[(v3_metrics["n"] == n) & (v3_metrics["condition"] == cond) & (v3_metrics["class"] == "불만제기")]
            if b.empty or v.empty:
                continue
            b, v = b.iloc[0], v.iloc[0]
            for metric in ["recall", "precision", "f1"]:
                rows.append({"section": "불만제기_클래스", "n": n, "condition": cond, "pair": None, "metric": metric,
                             "v_base": b[metric], "v3_orig": v[metric], "delta": v[metric] - b[metric]})
    # (2) 7-way accuracy/macro_f1, N x 조건 (참고)
    for n in N_VALUES:
        for cond in CONDITIONS:
            b = base_metrics[(base_metrics["n"] == n) & (base_metrics["condition"] == cond)]
            v = v3_metrics[(v3_metrics["n"] == n) & (v3_metrics["condition"] == cond)]
            if b.empty or v.empty:
                continue
            b, v = b.iloc[0], v.iloc[0]
            for metric in ["accuracy", "macro_f1"]:
                rows.append({"section": "7way_전체_참고", "n": n, "condition": cond, "pair": None, "metric": metric,
                             "v_base": b[metric], "v3_orig": v[metric], "delta": v[metric] - b[metric]})
    # (3) 음향 델타(B-A, D-C) 비교: v_base일 때 vs v3_orig일 때 (불만제기 위주 + 참고로 전체 클래스)
    for n in N_VALUES:
        for pair in ["customer_only", "all"]:
            for cls in CATEGORY_ORDER:
                b = base_delta[(base_delta["n"] == n) & (base_delta["pair"] == pair) & (base_delta["class"] == cls)]
                v = v3_delta[(v3_delta["n"] == n) & (v3_delta["pair"] == pair) & (v3_delta["class"] == cls)]
                if b.empty or v.empty:
                    continue
                b, v = b.iloc[0], v.iloc[0]
                section = "음향델타_불만제기" if cls == "불만제기" else "음향델타_기타클래스"
                for metric in ["delta_recall", "delta_precision", "delta_f1"]:
                    rows.append({"section": section, "n": n, "condition": None, "pair": pair, "class": cls,
                                 "metric": f"acoustic_{metric}", "v_base": b[metric], "v3_orig": v[metric],
                                 "delta": v[metric] - b[metric]})

    vs_base_df = pd.DataFrame(rows)
    vs_base_df.to_csv(OUT_VS_BASE, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT_VS_BASE} ({len(vs_base_df)}행)")

    # === 요약 5줄 ===
    complaint_recall = vs_base_df[(vs_base_df["section"] == "불만제기_클래스") & (vs_base_df["metric"] == "recall")]
    complaint_prec = vs_base_df[(vs_base_df["section"] == "불만제기_클래스") & (vs_base_df["metric"] == "precision")]
    complaint_f1 = vs_base_df[(vs_base_df["section"] == "불만제기_클래스") & (vs_base_df["metric"] == "f1")]
    seven_way = vs_base_df[vs_base_df["section"] == "7way_전체_참고"]
    ac_delta_complaint = vs_base_df[vs_base_df["section"] == "음향델타_불만제기"]

    print("\n" + "=" * 60)
    print("요약")
    print("=" * 60)
    for n in N_VALUES:
        r = complaint_recall[complaint_recall["n"] == n]
        p = complaint_prec[complaint_prec["n"] == n]
        rb_str = ", ".join(f"{row['condition']}:{row['v_base']:.3f}->{row['v3_orig']:.3f}({row['delta']:+.3f})" for _, row in r.iterrows())
        pb_str = ", ".join(f"{row['condition']}:{row['v_base']:.3f}->{row['v3_orig']:.3f}({row['delta']:+.3f})" for _, row in p.iterrows())
        print(f"1-N={n}. 불만제기 recall[{rb_str}] / precision[{pb_str}]")

    max_abs_7way = seven_way["delta"].abs().max()
    mean_abs_7way = seven_way["delta"].abs().mean()
    print(f"2. 7-way 전체(accuracy/macro-F1) 델타는 |delta| 평균 {mean_abs_7way:.4f}, 최대 {max_abs_7way:.4f} - "
          f"{'예상대로 미미함' if max_abs_7way < 0.02 else '예상보다 큰 변화 있음, 확인 필요'}")

    ac_rec = ac_delta_complaint[ac_delta_complaint["metric"] == "acoustic_delta_recall"]
    ac_prec = ac_delta_complaint[ac_delta_complaint["metric"] == "acoustic_delta_precision"]
    print(f"3. 음향 델타(B-A,D-C) reasoning으로 확대 여부 - 불만제기 recall: " +
          ", ".join(f"N{r['n']}({r['pair']}):{r['v_base']:+.4f}->{r['v3_orig']:+.4f}" for _, r in ac_rec.iterrows()))
    print(f"   불만제기 precision: " +
          ", ".join(f"N{r['n']}({r['pair']}):{r['v_base']:+.4f}->{r['v3_orig']:+.4f}" for _, r in ac_prec.iterrows()))

    print(f"4. N별 불만 F1 개선폭(v3_orig-v_base): " +
          ", ".join(f"N{row['n']}({row['condition']}):{row['delta']:+.4f}" for _, row in complaint_f1.iterrows()))

    print(f"5. 파싱 실패율(형식오류/태그누락, 오답 처리됨): {n_parse_fail}/{n_v3_total}건 "
          f"({n_parse_fail/n_v3_total*100:.2f}%)")

    print("\nSTAGE2_BD_V3_FINALIZE_COMPLETE")


if __name__ == "__main__":
    main()
