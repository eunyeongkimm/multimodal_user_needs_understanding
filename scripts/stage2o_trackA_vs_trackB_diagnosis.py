"""Track A(성별 population 개별 정규화)와 Track B(n_obs 2~3 shrinkage) 사이에
5등급 자연어 서술이 실제로 얼마나 달라지는지 API 호출 없이 로컬에서 진단한다.

N=2,3에 한정 (사용자 지시 — 이 구간이 shrinkage(n_obs 2~3)가 걸리는 핵심 구간).

출력: 콘솔에 feature별 변경 콜 비율 + 변경 방향(등급 이동) 요약,
      outputs/stage2_trackA_vs_trackB_diagnosis.csv 저장.
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
TRACK_A_PATH = BASE_DIR / "outputs" / "stage2_windows_nl_v2.parquet"
TRACK_B_PATH = BASE_DIR / "outputs" / "stage2_windows_nl_v3_shrinkage.parquet"
OUT_PATH = BASE_DIR / "outputs" / "stage2_trackA_vs_trackB_diagnosis.csv"

N_VALUES = [2, 3]
METRICS = ["f0_mean", "f0_std", "e_mean_db", "e_std_db", "r_gold", "s_top_db30"]
KEY_COLS = ["call_id", "window_n", "window_version", "dialog_idx"]


def main():
    a = pd.read_parquet(TRACK_A_PATH)
    b = pd.read_parquet(TRACK_B_PATH)

    a = a[a["window_n"].isin(N_VALUES)]
    b = b[b["window_n"].isin(N_VALUES)]

    level_cols_a = KEY_COLS + [f"{m}_level" for m in METRICS] + ["speaker_group"]
    level_cols_b = KEY_COLS + [f"{m}_level" for m in METRICS]

    merged = a[level_cols_a].merge(b[level_cols_b], on=KEY_COLS, suffixes=("_A", "_B"), how="inner")
    print(f"비교 대상 행수: {len(merged)} (N=2,3, Track A/B 공통 key 기준)")

    summary_rows = []
    for n in N_VALUES:
        for version in ["customer_only", "all"]:
            sub = merged[(merged["window_n"] == n) & (merged["window_version"] == version)]
            if sub.empty:
                continue
            n_calls = sub["call_id"].nunique()
            for m in METRICS:
                col_a, col_b = f"{m}_level_A", f"{m}_level_B"
                row_changed = sub[col_a].astype(str) != sub[col_b].astype(str)
                changed_calls = sub.loc[row_changed, "call_id"].nunique()
                pct_calls = changed_calls / n_calls * 100 if n_calls else 0

                summary_rows.append({
                    "n": n, "version": version, "feature": m,
                    "n_calls": n_calls, "changed_calls": changed_calls,
                    "changed_call_pct": round(pct_calls, 2),
                    "n_rows": len(sub), "changed_rows": int(row_changed.sum()),
                })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("\n=== N x 버전 x feature별 '등급이 달라진 콜' 비율 ===")
    pivot = summary.pivot_table(index=["n", "version"], columns="feature", values="changed_call_pct")
    pivot = pivot[METRICS]
    print(pivot.to_string())

    max_pct = summary["changed_call_pct"].max()
    print(f"\n최대 변경 비율: {max_pct:.2f}%")
    if max_pct < 5:
        print(">> 전체적으로 5% 미만 -> Track B는 Track A와 사실상 거의 동일 (shrinkage 영향 미미).")
    else:
        print(">> 5% 이상인 구간이 있음 -> 아래 방향(등급 이동) 요약 참고.")

    print("\n=== 등급 이동 방향 (변경된 행들만, feature x N x 버전) ===")
    for n in N_VALUES:
        for version in ["customer_only", "all"]:
            sub = merged[(merged["window_n"] == n) & (merged["window_version"] == version)]
            if sub.empty:
                continue
            for m in METRICS:
                col_a, col_b = f"{m}_level_A", f"{m}_level_B"
                diff = sub[sub[col_a].astype(str) != sub[col_b].astype(str)]
                if diff.empty:
                    continue
                pct = len(diff) / len(sub) * 100
                if pct < 1:  # 너무 미미한 건 생략
                    continue
                trans = diff.groupby([col_a, col_b], observed=True).size().sort_values(ascending=False)
                print(f"\nN={n}, {version}, {m} (행 기준 변경 {len(diff)}/{len(sub)}, {pct:.2f}%):")
                print(trans.head(10).to_string())

    print(f"\n저장: {OUT_PATH}")
    print("STAGE2O_COMPLETE")


if __name__ == "__main__":
    main()
