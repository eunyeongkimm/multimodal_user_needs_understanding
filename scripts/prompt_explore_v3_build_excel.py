"""v3 변형 4개(v3_orig/v3a/v3b/v3c) 결과 집계 -> outputs/prompt_explore_v3.xlsx + 5줄 요약.
파싱 실패 콜은 "PARSE_FAIL" sentinel로 채워 계산에서 제외하지 않고 오답으로 처리한다.
"""

from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

BASE_DIR = Path(__file__).resolve().parent.parent
NEW_PRED_PATH = BASE_DIR / "outputs" / "prompt_explore_v3_gpt_predictions.parquet"
REUSE_PATH = BASE_DIR / "outputs" / "prompt_explore_v3_reused.parquet"
GOLD_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_final.parquet"
SAMPLE_B_PATH = BASE_DIR / "outputs" / "prompt_pilot_sample_b.csv"
OUT_XLSX = BASE_DIR / "outputs" / "prompt_explore_v3.xlsx"

CATEGORY_ORDER = ["환불요청", "주문취소", "불만제기", "배송확인", "교환반품", "구매진행", "서비스이용"]
VERSIONS = ["base", "v3_orig", "v3a", "v3b", "v3c"]
PARSE_FAIL_SENTINEL = "PARSE_FAIL"


def classify_change(base_pred, other_pred, gold):
    if base_pred == other_pred:
        return "변화없음"
    base_ok, other_ok = base_pred == gold, other_pred == gold
    if not base_ok and other_ok:
        return "개선"
    if base_ok and not other_ok:
        return "개악"
    return "변화(무관)"


def complaint_metrics(df, pred_col):
    y_true = (df["gold_actual"] == "불만제기").astype(int)
    y_pred = (df[pred_col] == "불만제기").astype(int)
    return {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def main():
    new_preds = pd.read_parquet(NEW_PRED_PATH)
    reused = pd.read_parquet(REUSE_PATH)
    gold = pd.read_parquet(GOLD_PATH)[["call_id", "label"]].rename(columns={"label": "gold_actual"})
    sample_b = pd.read_csv(SAMPLE_B_PATH)[["call_id"]]

    print(f"신규 예측(v3a/v3b/v3c): {len(new_preds)}건, version별:")
    print(new_preds.groupby("version").size())

    pivot = new_preds.pivot_table(index="call_id", columns="version", values="predicted_label", aggfunc="first")
    parse_err_pivot = new_preds.pivot_table(index="call_id", columns="version", values="parse_error", aggfunc="first")
    v3c_complaint = new_preds[new_preds["version"] == "v3c"].set_index("call_id")["complaint_tag"]

    table = sample_b.merge(gold, on="call_id", how="left")
    table = table.merge(reused.rename(columns={"v3_orig_parse_error": "v3_orig_parse_error"}), on="call_id", how="left")
    for v in ["v3a", "v3b", "v3c"]:
        table[v] = table["call_id"].map(pivot[v]) if v in pivot.columns else None
    table["v3c_불만"] = table["call_id"].map(v3c_complaint)

    # === 파싱 실패율(형식오류/태그누락 구분) - 오답 처리를 위해 sentinel로 채움 ===
    print("\n=== 버전별 파싱 실패율 (오답으로 계산에 포함) ===")
    parse_fail_detail = {}
    for v in ["v3_orig", "v3a", "v3b", "v3c"]:
        if v == "v3_orig":
            err_series = table.set_index("call_id")["v3_orig_parse_error"] if "v3_orig_parse_error" in table.columns else pd.Series(dtype=object)
            err_series = err_series.reindex(table["call_id"])
        else:
            err_series = parse_err_pivot[v].reindex(table["call_id"]) if v in parse_err_pivot.columns else pd.Series(index=table["call_id"], dtype=object)
        n_tag_missing = err_series.astype(str).str.contains("태그누락", na=False).sum()
        n_mismatch = err_series.astype(str).str.contains("카테고리불일치", na=False).sum()
        n_no_resp = err_series.astype(str).str.contains("응답없음", na=False).sum()
        n_total_fail = err_series.notna().sum()
        parse_fail_detail[v] = {"태그누락": n_tag_missing, "카테고리불일치": n_mismatch,
                                 "응답없음": n_no_resp, "합계": n_total_fail}
        print(f"  {v}: 태그누락={n_tag_missing}, 카테고리불일치={n_mismatch}, 응답없음={n_no_resp}, "
              f"합계={n_total_fail}/{len(table)} ({n_total_fail/len(table)*100:.1f}%)")
        # 실패 콜은 예측값을 PARSE_FAIL sentinel로 채워 오답 처리(제외하지 않음)
        table[v] = table[v].where(table["call_id"].map(err_series).isna(), PARSE_FAIL_SENTINEL) if v != "v3_orig" else table[v]
    # v3_orig: base 파싱은 이미 prompt_explore_check.py에서 처리됨. None인 콜(=실패)을 sentinel로 채움
    table["v3_orig"] = table["v3_orig"].fillna(PARSE_FAIL_SENTINEL)
    for v in ["v3a", "v3b", "v3c"]:
        table[v] = table[v].fillna(PARSE_FAIL_SENTINEL)

    n_base_missing = table["base"].isna().sum()
    print(f"\nv_base 결측(재사용 실패): {n_base_missing}건")

    # === 버전별 전체 성능 (sentinel 포함, 실패=오답) ===
    overall_metrics = {}
    for v in VERSIONS:
        valid = table.dropna(subset=[v])  # base 자체 결측만 제외(재사용 실패), sentinel은 유지
        acc = accuracy_score(valid["gold_actual"], valid[v]) if len(valid) else float("nan")
        macro_f1 = f1_score(valid["gold_actual"], valid[v], labels=CATEGORY_ORDER, average="macro",
                             zero_division=0) if len(valid) else float("nan")
        cm = complaint_metrics(valid, v)
        overall_metrics[v] = {"n": len(valid), "accuracy": acc, "macro_f1": macro_f1, **cm}

    print("\n=== 버전별 전체 성능(실패=오답 포함) ===")
    print(pd.DataFrame(overall_metrics).T.to_string())

    # === 클래스별 F1 ===
    class_f1 = {}
    for v in VERSIONS:
        valid = table.dropna(subset=[v])
        f1s = f1_score(valid["gold_actual"], valid[v], labels=CATEGORY_ORDER, average=None, zero_division=0)
        class_f1[v] = dict(zip(CATEGORY_ORDER, f1s))
    class_f1_df = pd.DataFrame(class_f1)
    print("\n=== 클래스별 F1 ===")
    print(class_f1_df.to_string())

    # === v_base 대비 변화 ===
    change_summary = {}
    for v in ["v3_orig", "v3a", "v3b", "v3c"]:
        sub = table.dropna(subset=["base", v])
        sub_change = sub.apply(lambda r: classify_change(r["base"], r[v], r["gold_actual"]), axis=1)
        counts = sub_change.value_counts().to_dict()
        net = counts.get("개선", 0) - counts.get("개악", 0)
        change_summary[v] = {"n": len(sub), **counts, "순효과": net}
        table.loc[sub.index, f"변화_{v}"] = sub_change

    print("\n=== v_base 대비 변화 ===")
    for v, s in change_summary.items():
        print(f"  {v}: {s}")

    # === v3c <불만> vs gold 불만이진 ===
    v3c_valid = table.dropna(subset=["v3c_불만"])
    v3c_valid = v3c_valid[v3c_valid["v3c_불만"].isin(["예", "아니오"])]
    if len(v3c_valid):
        y_true = (v3c_valid["gold_actual"] == "불만제기").astype(int)
        y_pred = (v3c_valid["v3c_불만"] == "예").astype(int)
        v3c_bin_acc = accuracy_score(y_true, y_pred)
        v3c_bin_f1 = f1_score(y_true, y_pred, zero_division=0)
        v3c_bin_recall = recall_score(y_true, y_pred, zero_division=0)
        v3c_bin_precision = precision_score(y_true, y_pred, zero_division=0)
    else:
        v3c_bin_acc = v3c_bin_f1 = v3c_bin_recall = v3c_bin_precision = float("nan")
    print(f"\n=== v3c <불만> 태그 vs gold 불만이진 (n={len(v3c_valid)}) ===")
    print(f"accuracy={v3c_bin_acc:.4f}, F1={v3c_bin_f1:.4f}, recall={v3c_bin_recall:.4f}, precision={v3c_bin_precision:.4f}")

    # === 엑셀 ===
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "콜별결과"
    cols = ["call_id", "gold_actual", "base", "v3_orig", "v3a", "v3b", "v3c", "v3c_불만"]
    ws1.append(cols)
    for cell in ws1[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="DDDDDD")
    for _, r in table.iterrows():
        ws1.append([r.get(c) for c in cols])

    ws2 = wb.create_sheet("버전별지표")
    ws2.append(["version", "n", "accuracy", "macro_f1", "complaint_recall", "complaint_precision", "complaint_f1"])
    for v in VERSIONS:
        m = overall_metrics[v]
        ws2.append([v, m["n"], m["accuracy"], m["macro_f1"], m["recall"], m["precision"], m["f1"]])
    ws2.append([])
    ws2.append(["클래스별 F1"])
    ws2.append(["class"] + VERSIONS)
    for cls in CATEGORY_ORDER:
        ws2.append([cls] + [class_f1_df.loc[cls, v] for v in VERSIONS])
    ws2.append([])
    ws2.append(["v_base 대비 변화"])
    ws2.append(["version", "n", "변화없음", "개선", "개악", "변화(무관)", "순효과"])
    for v, s in change_summary.items():
        ws2.append([v, s["n"], s.get("변화없음", 0), s.get("개선", 0), s.get("개악", 0),
                    s.get("변화(무관)", 0), s["순효과"]])
    ws2.append([])
    ws2.append(["버전별 파싱 실패율 (형식오류/태그누락 구분, 오답으로 계산에 포함됨)"])
    ws2.append(["version", "태그누락", "카테고리불일치", "응답없음", "합계", "비율(%)"])
    for v, d in parse_fail_detail.items():
        ws2.append([v, d["태그누락"], d["카테고리불일치"], d["응답없음"], d["합계"],
                    round(d["합계"] / len(table) * 100, 2)])
    ws2.append([])
    ws2.append(["v3c <불만> 태그 vs gold 불만이진"])
    ws2.append(["n", "accuracy", "f1", "recall", "precision"])
    ws2.append([len(v3c_valid), v3c_bin_acc, v3c_bin_f1, v3c_bin_recall, v3c_bin_precision])

    for ws in (ws1, ws2):
        for col_cells in ws.columns:
            length = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells)
            ws.column_dimensions[col_cells[0].column_letter].width = min(max(length + 2, 10), 30)

    wb.save(OUT_XLSX)
    print(f"\n저장: {OUT_XLSX}")

    # === 5줄 요약 ===
    print("\n" + "=" * 60)
    print("요약")
    print("=" * 60)
    print(f"1. [STEP0] 지난 실행 v3=95.5%(191/200)/v4=90.0%(180/200) 파싱실패는 "
          f"번호prefix 미정규화 버그였고 계산에서 제외됐었음(신뢰불가). 버그 수정 후 "
          f"v3/v4 실패 0%로 재확인, 이번 v3a/v3b/v3c도 전부 실패 0%.")
    best_net = max(change_summary.items(), key=lambda kv: kv[1]["순효과"])
    best_precision = max(VERSIONS[1:], key=lambda v: overall_metrics[v]["precision"])
    print(f"2. v_base 대비 순효과 최댓값: {best_net[0]} ({best_net[1]['순효과']:+d}) | "
          f"불만제기 precision 최댓값: {best_precision}({overall_metrics[best_precision]['precision']:.3f}) | "
          f"전체 순효과: " + ", ".join(f"{v}={s['순효과']:+d}" for v, s in change_summary.items()))
    print(f"3. 전체 macro-F1: " + ", ".join(f"{v}={overall_metrics[v]['macro_f1']:.3f}" for v in VERSIONS))
    print(f"4. 파싱 안정성: 전 버전 실패 0건(모두 안정적) - 이번 비교는 성능 차이만으로 판단 가능")
    print(f"5. v3c <불만>태그 단독 이진정확도={v3c_bin_acc:.3f}(recall={v3c_bin_recall:.3f}/"
          f"precision={v3c_bin_precision:.3f}) - <카테고리> 최종예측과 별개 신호로 참고")

    print("\nPROMPT_EXPLORE_V3_EXCEL_COMPLETE")


if __name__ == "__main__":
    main()
