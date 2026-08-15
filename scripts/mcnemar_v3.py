"""prompt_explore_v3 예측(API 재사용, 신규 호출 없음)으로 McNemar 검정.
비교쌍: base vs v3_orig, base vs v3b, base vs v3c, v3_orig vs v3b
기준: (1) 7-way 정답여부, (2) 불만제기 이진 정답여부
출력: outputs/mcnemar_results.txt
"""

from pathlib import Path

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

BASE_DIR = Path(__file__).resolve().parent.parent
NEW_PRED_PATH = BASE_DIR / "outputs" / "prompt_explore_v3_gpt_predictions.parquet"
REUSE_PATH = BASE_DIR / "outputs" / "prompt_explore_v3_reused.parquet"
GOLD_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_final.parquet"
SAMPLE_B_PATH = BASE_DIR / "outputs" / "prompt_pilot_sample_b.csv"
OUT_PATH = BASE_DIR / "outputs" / "mcnemar_results.txt"

PAIRS = [("base", "v3_orig"), ("base", "v3b"), ("base", "v3c"), ("v3_orig", "v3b")]
PARSE_FAIL_SENTINEL = "PARSE_FAIL"
ALPHA = 0.05


def build_table():
    new_preds = pd.read_parquet(NEW_PRED_PATH)
    reused = pd.read_parquet(REUSE_PATH)
    gold = pd.read_parquet(GOLD_PATH)[["call_id", "label"]].rename(columns={"label": "gold_actual"})
    sample_b = pd.read_csv(SAMPLE_B_PATH)[["call_id"]]

    pivot = new_preds.pivot_table(index="call_id", columns="version", values="predicted_label", aggfunc="first")
    parse_err_pivot = new_preds.pivot_table(index="call_id", columns="version", values="parse_error", aggfunc="first")

    table = sample_b.merge(gold, on="call_id", how="left")
    table = table.merge(reused, on="call_id", how="left")
    for v in ["v3a", "v3b", "v3c"]:
        table[v] = table["call_id"].map(pivot[v]) if v in pivot.columns else None

    for v in ["v3a", "v3b", "v3c"]:
        err_series = parse_err_pivot[v].reindex(table["call_id"]).values if v in parse_err_pivot.columns else None
        if err_series is not None:
            table.loc[pd.notna(err_series), v] = PARSE_FAIL_SENTINEL
        table[v] = table[v].fillna(PARSE_FAIL_SENTINEL)
    table["v3_orig"] = table["v3_orig"].fillna(PARSE_FAIL_SENTINEL)

    n_base_missing = table["base"].isna().sum()
    if n_base_missing:
        print(f"경고: v_base 결측 {n_base_missing}건 -> 해당 콜 전체 비교에서 제외")
    table = table.dropna(subset=["base"])
    return table


def mcnemar_test(correct_a, correct_b):
    n11 = int((correct_a & correct_b).sum())
    n10 = int((correct_a & ~correct_b).sum())
    n01 = int((~correct_a & correct_b).sum())
    n00 = int((~correct_a & ~correct_b).sum())
    disagree = n10 + n01
    exact = disagree < 25
    cont_table = [[n11, n10], [n01, n00]]
    if disagree == 0:
        return n10, n01, float("nan"), 1.0, exact
    res = mcnemar(cont_table, exact=exact, correction=True)
    return n10, n01, float(res.statistic), float(res.pvalue), exact


def main():
    table = build_table()
    n = len(table)
    print(f"비교 대상 콜: {n}건")

    lines = []
    lines.append(f"McNemar 검정 결과 (n={n}, alpha={ALPHA})")
    lines.append("API 재호출 없음 - prompt_explore_v3 기존 예측 재사용")
    lines.append("")

    results_7way = {}
    results_complaint = {}

    lines.append("=== 1) 7-way 정답여부 기준 ===")
    lines.append(f"{'비교쌍':<20} {'n10':>5} {'n01':>5} {'검정방식':>8} {'statistic':>10} {'p-value':>10} {'유의(a=.05)':>12}")
    for a, b in PAIRS:
        correct_a = (table[a] == table["gold_actual"])
        correct_b = (table[b] == table["gold_actual"])
        n10, n01, stat, p, exact = mcnemar_test(correct_a, correct_b)
        sig = "유의함" if p < ALPHA else "유의안함"
        method = "exact" if exact else "chi2+cc"
        results_7way[(a, b)] = {"n10": n10, "n01": n01, "stat": stat, "p": p, "sig": sig, "method": method}
        lines.append(f"{a+' vs '+b:<20} {n10:>5} {n01:>5} {method:>8} {stat:>10.4f} {p:>10.4f} {sig:>12}")

    lines.append("")
    lines.append("=== 2) 불만제기 이진 정답여부 기준 (pred=='불만제기' == gold=='불만제기') ===")
    lines.append(f"{'비교쌍':<20} {'n10':>5} {'n01':>5} {'검정방식':>8} {'statistic':>10} {'p-value':>10} {'유의(a=.05)':>12}")
    gold_is_complaint = (table["gold_actual"] == "불만제기")
    for a, b in PAIRS:
        correct_a = (table[a] == "불만제기") == gold_is_complaint
        correct_b = (table[b] == "불만제기") == gold_is_complaint
        n10, n01, stat, p, exact = mcnemar_test(correct_a, correct_b)
        sig = "유의함" if p < ALPHA else "유의안함"
        method = "exact" if exact else "chi2+cc"
        results_complaint[(a, b)] = {"n10": n10, "n01": n01, "stat": stat, "p": p, "sig": sig, "method": method}
        lines.append(f"{a+' vs '+b:<20} {n10:>5} {n01:>5} {method:>8} {stat:>10.4f} {p:>10.4f} {sig:>12}")

    # === 요약 ===
    r_base_v3b_7 = results_7way[("base", "v3b")]
    r_base_v3b_c = results_complaint[("base", "v3b")]
    r_orig_v3b_7 = results_7way[("v3_orig", "v3b")]
    r_orig_v3b_c = results_complaint[("v3_orig", "v3b")]

    lines.append("")
    lines.append("=== 요약 ===")
    lines.append(f"1. base vs v3b: 7-way {r_base_v3b_7['sig']}(p={r_base_v3b_7['p']:.4f}, "
                  f"n10={r_base_v3b_7['n10']}/n01={r_base_v3b_7['n01']}), "
                  f"불만이진 {r_base_v3b_c['sig']}(p={r_base_v3b_c['p']:.4f}, "
                  f"n10={r_base_v3b_c['n10']}/n01={r_base_v3b_c['n01']})")
    lines.append(f"2. v3_orig vs v3b(few-shot 추가 단독 효과): 7-way {r_orig_v3b_7['sig']}(p={r_orig_v3b_7['p']:.4f}, "
                  f"n10={r_orig_v3b_7['n10']}/n01={r_orig_v3b_7['n01']}), "
                  f"불만이진 {r_orig_v3b_c['sig']}(p={r_orig_v3b_c['p']:.4f}, "
                  f"n10={r_orig_v3b_c['n10']}/n01={r_orig_v3b_c['n01']})")
    any_not_sig = any(r["sig"] == "유의안함" for r in list(results_7way.values()) + list(results_complaint.values()))
    small_disagree = [f"{a} vs {b}(7way n10+n01={results_7way[(a,b)]['n10']+results_7way[(a,b)]['n01']}, "
                       f"불만이진 n10+n01={results_complaint[(a,b)]['n10']+results_complaint[(a,b)]['n01']})"
                       for a, b in PAIRS]
    lines.append(f"3. 불일치 셀(n10+n01) 크기: " + " / ".join(small_disagree))
    lines.append("4. [해석 주의] McNemar는 두 모델이 '서로 다른 콜에서' 틀리는 패턴만 봄. "
                 "순효과(개선-개악)가 작으면(v3b는 base 대비 +5) 불일치 셀도 작아 유의하게 안 나올 수 있음. "
                 "유의하지 않다는 것은 '차이가 없다'가 아니라 '이 표본(200건)으로는 방향을 확정할 수 없다'는 뜻이며, "
                 "그 경우 표본을 늘려 재검정할지 판단이 필요함.")

    output = "\n".join(lines)
    print("\n" + output)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(output + "\n")
    print(f"\n저장: {OUT_PATH}")
    print("\nMCNEMAR_COMPLETE")


if __name__ == "__main__":
    main()
