"""이원화 정규화(Stage 2 재정규화)의 population 기준 통계를 계산한다.

population = stage2_acoustic_features.parquet(135,340건, batch1 stage2 pool,
r_gold_valid_flag==True 이미 적용됨)에 d04_dialog_index의 speaker_gender를 join.
성별(여/남)별 6개 지표 평균/표준편차 + gender가 없는 경우를 위한 전체(성별무관)
fallback 통계도 함께 계산해서 저장한다.

주의(매핑 결정, 명시): "전체 dialog"를 D04 전체(1,672,062행)가 아니라 이미
acoustic feature가 계산되어 있는 batch1 stage2 pool(135,340건, r_gold_valid_flag
필터 이미 적용됨)로 해석했다 — 전체 D04에 대해 새로 acoustic 추출을 하려면
수 시간이 추가로 필요하고, 기존 global_fallback도 동일하게 이 pool을 "전체
분포"로 취급해온 것과 일관성을 맞추기 위함.

출력: outputs/stage2_gender_population_stats.csv
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
ACOUSTIC_PATH = BASE_DIR / "outputs" / "stage2_acoustic_features.parquet"
D04_PATH = BASE_DIR / "outputs" / "d04_dialog_index.parquet"
OUT_PATH = BASE_DIR / "outputs" / "stage2_gender_population_stats.csv"

METRICS = ["f0_mean", "f0_std", "e_mean_db", "e_std_db", "r_gold", "s_top_db30"]


def main():
    acoustic = pd.read_parquet(ACOUSTIC_PATH)
    d04 = pd.read_parquet(D04_PATH)[["call_id", "dialog_idx", "speaker_gender", "r_gold"]]

    merged = acoustic.merge(d04, on=["call_id", "dialog_idx"], how="left")

    n_gender_null = merged["speaker_gender"].isna().sum()
    print(f"population 계산 대상: {len(merged)}건, speaker_gender 결측: {n_gender_null}건")

    rows = []
    for gender, g in merged.groupby("speaker_gender", dropna=False):
        for m in METRICS:
            rows.append({
                "gender": gender if pd.notna(gender) else "UNKNOWN",
                "metric": m,
                "n": g[m].notna().sum(),
                "mean": g[m].mean(),
                "std": g[m].std(ddof=0),
            })

    # 성별 무관 전체 fallback (gender가 결측인 화자를 위한 최종 안전장치)
    for m in METRICS:
        rows.append({
            "gender": "ALL",
            "metric": m,
            "n": merged[m].notna().sum(),
            "mean": merged[m].mean(),
            "std": merged[m].std(ddof=0),
        })

    stats = pd.DataFrame(rows)
    stats.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n저장: {OUT_PATH}")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
