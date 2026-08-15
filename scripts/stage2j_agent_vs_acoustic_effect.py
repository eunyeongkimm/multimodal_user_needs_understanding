"""Stage 2-10: mismatch 서브셋(9,658건)에서 "상담사 정보 추가"(C-A)와
"acoustic 정보 추가"(B-A, D-C) 효과를 나란히 McNemar test로 비교한다.

- 상담사 추가 효과: A(고객only+Text) -> C(고객&상담사+Text), 텍스트 조건 고정, 화자 범위만 확장
- acoustic 추가 효과(고객only): A -> B
- acoustic 추가 효과(고객&상담사): C -> D

세 효과를 같은 mismatch 서브셋, 같은 통계적 절차(McNemar + paired bootstrap CI)로
계산해서 "어느 쪽 정보 추가가 더 크게 기여하는지" 직접 비교 가능하게 만든다.

출력: outputs/stage2_agent_vs_acoustic_effect.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

BASE_DIR = Path(__file__).resolve().parent.parent
PRED_PATH = BASE_DIR / "outputs" / "stage2d_gpt_predictions.parquet"
GOLD_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_final.parquet"
OUT_PATH = BASE_DIR / "outputs" / "stage2_agent_vs_acoustic_effect.csv"

SURFACE_CONDITION = "A"
N_BOOTSTRAP = 10_000
SEED = 42

N_VALUES = [1, 2, 3, 5]
EFFECTS = [
    ("agent_add(C-A)", "A", "C"),
    ("acoustic_add_customer_only(B-A)", "A", "B"),
    ("acoustic_add_all(D-C)", "C", "D"),
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


def run_comparison(sub_mismatch: pd.DataFrame, n: int, effect_name: str, cond1: str, cond2: str):
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

    acc1, acc2 = correct1.mean(), correct2.mean()
    diff = acc2 - acc1
    ci_lo, ci_hi = paired_bootstrap_ci(correct1.astype(float), correct2.astype(float))

    return {
        "n": n, "effect": effect_name, "cond1": cond1, "cond2": cond2, "n_total": n_total,
        "both_correct": both_correct, "only_cond1_correct": only1_correct,
        "only_cond2_correct": only2_correct, "both_wrong": both_wrong,
        "accuracy_cond1": acc1, "accuracy_cond2": acc2,
        "accuracy_diff": diff, "ci95_lo": ci_lo, "ci95_hi": ci_hi,
        "mcnemar_stat": result.statistic, "p_value": result.pvalue,
        "exact_test": use_exact, "significant(p<0.05)": result.pvalue < 0.05,
    }


def main():
    preds = pd.read_parquet(PRED_PATH)
    gold = pd.read_parquet(GOLD_PATH)[["call_id", "label"]].rename(columns={"label": "actual_label"})
    merged = preds.merge(gold, on="call_id", how="left")

    mismatch_ids = get_mismatch_ids(merged)
    print(f"mismatch 서브셋: {len(mismatch_ids)}건\n")
    sub_mismatch = merged[merged["call_id"].isin(mismatch_ids)]

    rows = []
    for n in N_VALUES:
        for effect_name, cond1, cond2 in EFFECTS:
            r = run_comparison(sub_mismatch, n, effect_name, cond1, cond2)
            rows.append(r)
            print(f"N={n} | {effect_name} ({cond1}->{cond2}) | 표본 {r['n_total']} | "
                  f"{r['accuracy_cond1']:.4f}->{r['accuracy_cond2']:.4f} "
                  f"(diff {r['accuracy_diff']:+.4f}) | p={r['p_value']:.6f} | "
                  f"CI=[{r['ci95_lo']:+.4f},{r['ci95_hi']:+.4f}] | 유의={r['significant(p<0.05)']}")
        print()

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("=== 효과 크기 매트릭스 (행: N, 열: 효과 종류) - accuracy_diff ===")
    diff_pivot = summary.pivot(index="n", columns="effect", values="accuracy_diff")
    diff_pivot = diff_pivot[[e[0] for e in EFFECTS]]
    print(diff_pivot.to_string())

    print("\n=== p-value 매트릭스 ===")
    p_pivot = summary.pivot(index="n", columns="effect", values="p_value")
    p_pivot = p_pivot[[e[0] for e in EFFECTS]]
    print(p_pivot.to_string())

    print("\n=== N별 '상담사 추가' vs 'acoustic 추가' 중 어느 쪽이 더 큰가 ===")
    for n in N_VALUES:
        agent = summary[(summary["n"] == n) & (summary["effect"] == "agent_add(C-A)")]["accuracy_diff"].iloc[0]
        acoustic_co = summary[(summary["n"] == n) &
                               (summary["effect"] == "acoustic_add_customer_only(B-A)")]["accuracy_diff"].iloc[0]
        winner = "상담사 추가" if agent > acoustic_co else "acoustic 추가"
        print(f"  N={n}: 상담사추가(C-A)={agent:+.4f} vs acoustic추가(B-A)={acoustic_co:+.4f} "
              f"-> {winner}가 더 큼 (차이 {abs(agent-acoustic_co):.4f})")

    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
