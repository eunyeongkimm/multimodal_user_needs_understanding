"""Stage 2-8: mismatch 서브셋(9,658건, surface(A@N=1) != gold_actual)에서
acoustic 효과의 통계적 유의성을 McNemar test로 검정한다.

비교쌍 4개:
  N=1: A(고객only+Text) vs B(고객only+Text&Acoustic)
  N=2: C(고객&상담사+Text) vs D(고객&상담사+Text&Acoustic)
  N=3: C vs D
  N=5: C vs D

각 비교쌍마다:
  - 2x2 분할표(둘다맞음/1만맞음/2만맞음/둘다틀림)
  - McNemar 통계량 + p-value (불일치 셀 중 작은 쪽이 25 미만이면 exact=True)
  - 정확도 차이(cond2-cond1)의 95% CI (paired bootstrap, resample=10000)

출력: outputs/stage2_mcnemar_results.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

BASE_DIR = Path(__file__).resolve().parent.parent
PRED_PATH = BASE_DIR / "outputs" / "stage2d_gpt_predictions.parquet"
GOLD_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_final.parquet"
OUT_PATH = BASE_DIR / "outputs" / "stage2_mcnemar_results.csv"

SURFACE_CONDITION = "A"
N_BOOTSTRAP = 10_000
SEED = 42

COMPARISONS = [
    (1, "A", "B"),
    (2, "C", "D"),
    (3, "C", "D"),
    (5, "C", "D"),
]


def get_mismatch_ids(merged: pd.DataFrame) -> set:
    surface = merged[(merged["condition"] == SURFACE_CONDITION) & (merged["n"] == 1)]
    surface_valid = surface.dropna(subset=["predicted_label"])
    return set(surface_valid[surface_valid["predicted_label"] != surface_valid["actual_label"]]["call_id"])


def paired_bootstrap_ci(correct1: np.ndarray, correct2: np.ndarray, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(correct1)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[b] = correct2[idx].mean() - correct1[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return lo, hi


def run_comparison(sub_mismatch: pd.DataFrame, n: int, cond1: str, cond2: str):
    p1 = sub_mismatch[(sub_mismatch["n"] == n) & (sub_mismatch["condition"] == cond1)][
        ["call_id", "predicted_label", "actual_label"]
    ].rename(columns={"predicted_label": "pred1"})
    p2 = sub_mismatch[(sub_mismatch["n"] == n) & (sub_mismatch["condition"] == cond2)][
        ["call_id", "predicted_label"]
    ].rename(columns={"predicted_label": "pred2"})

    paired = p1.merge(p2, on="call_id", how="inner").dropna(subset=["pred1", "pred2"])
    correct1 = (paired["pred1"] == paired["actual_label"]).to_numpy()
    correct2 = (paired["pred2"] == paired["actual_label"]).to_numpy()

    both_correct = int((correct1 & correct2).sum())
    only1_correct = int((correct1 & ~correct2).sum())
    only2_correct = int((~correct1 & correct2).sum())
    both_wrong = int((~correct1 & ~correct2).sum())
    n_total = len(paired)

    table = np.array([[both_correct, only1_correct], [only2_correct, both_wrong]])
    discordant_min = min(only1_correct, only2_correct)
    use_exact = discordant_min < 25
    result = mcnemar(table, exact=use_exact, correction=not use_exact)

    acc1 = correct1.mean()
    acc2 = correct2.mean()
    diff = acc2 - acc1
    ci_lo, ci_hi = paired_bootstrap_ci(correct1.astype(float), correct2.astype(float))

    return {
        "n": n, "cond1": cond1, "cond2": cond2,
        "n_total": n_total,
        "both_correct": both_correct, "only_cond1_correct": only1_correct,
        "only_cond2_correct": only2_correct, "both_wrong": both_wrong,
        "accuracy_cond1": acc1, "accuracy_cond2": acc2,
        "accuracy_diff(cond2-cond1)": diff,
        "ci95_lo": ci_lo, "ci95_hi": ci_hi,
        "mcnemar_stat": result.statistic, "p_value": result.pvalue,
        "exact_test": use_exact,
        "significant(p<0.05)": result.pvalue < 0.05,
    }


def main():
    preds = pd.read_parquet(PRED_PATH)
    gold = pd.read_parquet(GOLD_PATH)[["call_id", "label"]].rename(columns={"label": "actual_label"})
    merged = preds.merge(gold, on="call_id", how="left")

    mismatch_ids = get_mismatch_ids(merged)
    print(f"mismatch 서브셋: {len(mismatch_ids)}건")
    sub_mismatch = merged[merged["call_id"].isin(mismatch_ids)]

    rows = []
    for n, cond1, cond2 in COMPARISONS:
        r = run_comparison(sub_mismatch, n, cond1, cond2)
        rows.append(r)

        print(f"\n=== N={n}: {cond1} vs {cond2} ===")
        print(f"2x2 분할표 (표본 {r['n_total']}건):")
        print(f"                {cond2}=맞음   {cond2}=틀림")
        print(f"  {cond1}=맞음      {r['both_correct']:>6}      {r['only_cond1_correct']:>6}")
        print(f"  {cond1}=틀림      {r['only_cond2_correct']:>6}      {r['both_wrong']:>6}")
        print(f"정확도: {cond1}={r['accuracy_cond1']:.4f}, {cond2}={r['accuracy_cond2']:.4f}, "
              f"차이({cond2}-{cond1})={r['accuracy_diff(cond2-cond1)']:.4f}")
        print(f"McNemar: statistic={r['mcnemar_stat']:.4f}, p-value={r['p_value']:.6f} "
              f"(exact={r['exact_test']})")
        print(f"95% CI(bootstrap): [{r['ci95_lo']:.4f}, {r['ci95_hi']:.4f}]")
        print(f"유의함(p<0.05): {r['significant(p<0.05)']}")

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("\n\n=== 전체 요약 ===")
    display_cols = ["n", "cond1", "cond2", "n_total", "accuracy_diff(cond2-cond1)",
                     "p_value", "ci95_lo", "ci95_hi", "significant(p<0.05)"]
    print(summary[display_cols].to_string(index=False))
    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
