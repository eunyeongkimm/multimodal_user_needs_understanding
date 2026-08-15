"""Stage 2-9: mismatch 서브셋(9,658건)에서 N=1,2,3,5 전부에 대해 A vs B, C vs D
둘 다(8개 비교쌍) McNemar test를 실행한다. stage2h_mcnemar_test.py는 N=1은
A/B만, N=2/3/5는 C/D만 봤는데, 이번엔 모든 N에서 두 버전(고객only/고객&상담사)을
동일하게 비교해 일관된 그림을 만든다.

출력: outputs/stage2_mcnemar_full.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

BASE_DIR = Path(__file__).resolve().parent.parent
PRED_PATH = BASE_DIR / "outputs" / "stage2d_gpt_predictions.parquet"
GOLD_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_final.parquet"
OUT_PATH = BASE_DIR / "outputs" / "stage2_mcnemar_full.csv"

SURFACE_CONDITION = "A"
N_BOOTSTRAP = 10_000
SEED = 42

N_VALUES = [1, 2, 3, 5]
PAIRS = [("A", "B"), ("C", "D")]


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

    version = "고객only" if {cond1, cond2} == {"A", "B"} else "고객&상담사"

    return {
        "n": n, "version": version, "cond1": cond1, "cond2": cond2,
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
    print(f"mismatch 서브셋: {len(mismatch_ids)}건 (surface=조건{SURFACE_CONDITION}@N=1 != actual)\n")
    sub_mismatch = merged[merged["call_id"].isin(mismatch_ids)]

    rows = []
    for n in N_VALUES:
        for cond1, cond2 in PAIRS:
            r = run_comparison(sub_mismatch, n, cond1, cond2)
            rows.append(r)
            print(f"=== N={n}: {cond1} vs {cond2} ({r['version']}) ===")
            print(f"  표본 {r['n_total']}건 | 정확도 {cond1}={r['accuracy_cond1']:.4f}, "
                  f"{cond2}={r['accuracy_cond2']:.4f}, 차이={r['accuracy_diff(cond2-cond1)']:+.4f}")
            print(f"  McNemar p-value={r['p_value']:.6f} (exact={r['exact_test']}), "
                  f"95% CI=[{r['ci95_lo']:+.4f}, {r['ci95_hi']:+.4f}], "
                  f"유의함={r['significant(p<0.05)']}\n")

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("=== N x 조건쌍 매트릭스: 정확도 차이(cond2-cond1) ===")
    diff_pivot = summary.pivot(index="n", columns="version", values="accuracy_diff(cond2-cond1)")
    print(diff_pivot.to_string())

    print("\n=== N x 조건쌍 매트릭스: p-value ===")
    p_pivot = summary.pivot(index="n", columns="version", values="p_value")
    print(p_pivot.to_string())

    print("\n=== N x 조건쌍 매트릭스: 유의 여부(p<0.05) ===")
    sig_pivot = summary.pivot(index="n", columns="version", values="significant(p<0.05)")
    print(sig_pivot.to_string())

    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
