"""Track B: 화자 발화 2~3개 구간에 shrinkage 정규화를 적용한 버전.
API 호출 없음 — 자연어 서술 데이터만 생성해서 Track A(성별 population 개별 정규화,
stage2_windows_nl_v2.parquet)와 비교하기 위한 것.

정규화 규칙 (Track A와의 차이는 n_obs in {2,3} 구간뿐):
  - n_obs >= 4: within-speaker (Track A와 동일 — 화자 본인 발화들의 평균/표준편차)
  - n_obs == 1: 성별 population 정규화 (Track A와 동일 — 개별값을 그대로 사용,
    "소표본 평균"이 표본 1개의 평균이므로 본인 값 자체와 같음)
  - n_obs in {2,3}: shrinkage — 그 화자의 윈도우 내 소표본 평균(own_small_mean)을
    구하고, 그 평균을 "성별 population의 표준편차"로 나눠 정규화한다:
      z = (own_small_mean - gender_mean) / gender_std
    주의: 소표본 자체의 표준편차로 나누지 않는다 (이게 기존 ±1.0 아티팩트의
    원인이었음). 같은 (call_id, window_n, window_version, speaker_group) 소그룹
    안의 모든 발화가 지표별로 동일한 z값(및 동일한 5등급/서술)을 공유하게 된다.

자연어 변환(quintile 5등급, METRIC_DESC 문구 매핑)은 stage2c_normalize_nl.py /
stage2c_v2_normalize_nl_gender.py와 100% 동일한 코드를 그대로 사용한다.

출력: outputs/stage2_windows_nl_v3_shrinkage.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
POOL_PATH = BASE_DIR / "outputs" / "stage2_utterance_pool.parquet"
ACOUSTIC_PATH = BASE_DIR / "outputs" / "stage2_acoustic_features.parquet"
D04_PATH = BASE_DIR / "outputs" / "d04_dialog_index.parquet"
GENDER_STATS_PATH = BASE_DIR / "outputs" / "stage2_gender_population_stats.csv"
OUT_PATH = BASE_DIR / "outputs" / "stage2_windows_nl_v3_shrinkage.parquet"

N_VALUES = [1, 2, 3, 5]
VERSIONS = ["customer_only", "all"]
METRICS = ["f0_mean", "f0_std", "e_mean_db", "e_std_db", "r_gold", "s_top_db30"]
WITHIN_SPEAKER_MIN_N = 4
SHRINKAGE_RANGE = (2, 3)  # n_obs in [2,3] -> shrinkage

LEVEL_LABELS = ["매우낮음", "낮음", "보통", "높음", "매우높음"]

# --- 서술 방식 100% 동일 (변경 금지) ---
METRIC_DESC = {
    "f0_mean": ("음높이는", {"매우낮음": "매우 낮고", "낮음": "낮은 편이고", "보통": "보통이고",
                          "높음": "높은 편이고", "매우높음": "매우 높고"}),
    "f0_std": ("음높이 변동은", {"매우낮음": "매우 작음", "낮음": "작은 편임", "보통": "보통임",
                              "높음": "큰 편임", "매우높음": "매우 큼"}),
    "e_mean_db": ("성량(에너지)은", {"매우낮음": "매우 작고", "낮음": "작은 편이고", "보통": "보통이고",
                                  "높음": "큰 편이고", "매우높음": "매우 크고"}),
    "e_std_db": ("성량 변동은", {"매우낮음": "매우 작음", "낮음": "작은 편임", "보통": "보통임",
                               "높음": "큰 편임", "매우높음": "매우 큼"}),
    "r_gold": ("발화속도는", {"매우낮음": "매우 느림", "낮음": "느린 편임", "보통": "보통임",
                            "높음": "빠른 편임", "매우높음": "매우 빠름"}),
    "s_top_db30": ("무음 비율은", {"매우낮음": "매우 낮음", "낮음": "낮은 편임", "보통": "보통임",
                                 "높음": "높은 편임", "매우높음": "매우 높음"}),
}


def quintile_labels(series: pd.Series) -> pd.Series:
    try:
        return pd.qcut(series, 5, labels=LEVEL_LABELS, duplicates="drop")
    except ValueError:
        return pd.Series(["보통"] * len(series), index=series.index)


def build_nl_description(row) -> str:
    parts = []
    for m in METRICS:
        label = row[f"{m}_level"]
        prefix, phrase_map = METRIC_DESC[m]
        phrase = phrase_map.get(label, "보통")
        parts.append(f"{prefix} {phrase}")
    return ", ".join(parts)


def load_gender_lookup():
    stats = pd.read_csv(GENDER_STATS_PATH)
    lookup = {}
    for _, row in stats.iterrows():
        lookup[(row["gender"], row["metric"])] = {"mean": row["mean"], "std": row["std"]}
    return lookup


def attach_gender_population_columns(df: pd.DataFrame, gender_lookup: dict) -> pd.DataFrame:
    df = df.copy()
    gender_key = df["speaker_gender"].where(df["speaker_gender"].notna(), "ALL")
    for m in METRICS:
        means = gender_key.map(lambda g: gender_lookup.get((g, m), gender_lookup[("ALL", m)])["mean"])
        stds = gender_key.map(lambda g: gender_lookup.get((g, m), gender_lookup[("ALL", m)])["std"])
        df[f"{m}_pop_mean"] = means
        df[f"{m}_pop_std"] = stds
    df["gender_used_fallback"] = df["speaker_gender"].isna()
    return df


def normalize_group_shrinkage(df: pd.DataFrame) -> pd.DataFrame:
    out_frames = []
    for (call_id, n, version, sg), g in df.groupby(
        ["call_id", "window_n", "window_version", "speaker_group"], sort=False
    ):
        g = g.copy()
        n_obs = len(g)
        for m in METRICS:
            n_valid = g[m].notna().sum()

            if n_obs >= WITHIN_SPEAKER_MIN_N and n_valid >= 2 and g[m].std(ddof=0) > 0:
                mean_, std_ = g[m].mean(), g[m].std(ddof=0)
                g[f"{m}_z"] = (g[m] - mean_) / std_
                g[f"{m}_norm_source"] = "within_speaker"

            elif SHRINKAGE_RANGE[0] <= n_obs <= SHRINKAGE_RANGE[1] and n_valid >= 1:
                own_small_mean = g[m].mean()  # NaN 자동 skip
                pop_mean = g[f"{m}_pop_mean"].iloc[0]
                pop_std = g[f"{m}_pop_std"].iloc[0]
                z_shared = (own_small_mean - pop_mean) / pop_std if pop_std and pop_std > 0 else np.nan
                g[f"{m}_z"] = z_shared  # 소그룹 전체가 동일 z 공유
                g[f"{m}_norm_source"] = "shrinkage_population_sd"

            else:
                # n_obs == 1 (또는 위 두 경우 모두 데이터 부족) -> 개별값을 성별
                # population으로 정규화 (Track A의 n_obs==1 케이스와 동일)
                pop_mean = g[f"{m}_pop_mean"]
                pop_std = g[f"{m}_pop_std"]
                g[f"{m}_z"] = np.where(pop_std > 0, (g[m] - pop_mean) / pop_std, np.nan)
                g[f"{m}_norm_source"] = np.where(
                    g["gender_used_fallback"], "population_fallback_no_gender", "population_gender"
                )
        out_frames.append(g)
    return pd.concat(out_frames, ignore_index=True)


def main():
    pool = pd.read_parquet(POOL_PATH)
    acoustic = pd.read_parquet(ACOUSTIC_PATH)
    d04_gender = pd.read_parquet(D04_PATH)[["call_id", "dialog_idx", "speaker_gender"]]
    gender_lookup = load_gender_lookup()

    merged = pool.merge(acoustic, on=["call_id", "dialog_idx"], how="left")
    merged = merged.merge(d04_gender, on=["call_id", "dialog_idx"], how="left")
    merged["speaker_group"] = merged["is_customer"].map({True: "고객", False: "상담사"})
    merged = attach_gender_population_columns(merged, gender_lookup)

    window_frames = []
    for n in N_VALUES:
        for version in VERSIONS:
            if version == "customer_only":
                w = merged[merged["customer_rank"].between(1, n)].copy()
            else:
                w = merged[merged["overall_rank"] <= n].copy()
            w["window_n"] = n
            w["window_version"] = version
            window_frames.append(w)

    windows = pd.concat(window_frames, ignore_index=True)
    print(f"윈도우 전개 후 총 행수: {len(windows)}")

    windows = normalize_group_shrinkage(windows)

    print("\n=== norm_source 전체 분포 (f0_mean 기준) ===")
    print(windows["f0_mean_norm_source"].value_counts())
    print((windows["f0_mean_norm_source"].value_counts(normalize=True) * 100).round(2))

    for m in METRICS:
        windows[f"{m}_level"] = quintile_labels(windows[f"{m}_z"])

    windows["nl_description"] = windows.apply(build_nl_description, axis=1)
    windows = windows.sort_values(["call_id", "window_n", "window_version", "dialog_idx"])

    out_cols = [
        "call_id", "window_n", "window_version", "dialog_idx", "speaker_type",
        "speaker_group", "speaker_gender", "overall_rank", "customer_rank", "text_clean",
    ] + METRICS + [f"{m}_z" for m in METRICS] + [f"{m}_level" for m in METRICS] \
      + [f"{m}_norm_source" for m in METRICS] + ["nl_description"]
    windows = windows[out_cols]

    windows.to_parquet(OUT_PATH, index=False)
    print(f"\n저장: {OUT_PATH} ({len(windows)}행)")

    print("\n=== [필수 검증] 지표별 z가 정확히 ±1.0인 행 비율 ===")
    all_ok = True
    for m in METRICS:
        z = windows[f"{m}_z"]
        n_exact = ((z.round(6) == 1.0) | (z.round(6) == -1.0)).sum()
        pct = n_exact / len(z) * 100
        flag = "" if pct < 5 else "  <<< 경고: 5% 이상, shrinkage 오류 의심"
        if pct >= 5:
            all_ok = False
        print(f"{m}: {n_exact}/{len(z)} ({pct:.3f}%){flag}")

    if not all_ok:
        print("\n*** ±1.0 비율이 예상(0% 근처)보다 크게 나왔습니다. shrinkage 로직 점검 필요 - 아래 단계 중단 권고. ***")
        print("STAGE2C_V3_NEEDS_REVIEW")
    else:
        print("\n±1.0 비율 정상 범위(거의 0%) 확인.")
        print("STAGE2C_V3_COMPLETE")


if __name__ == "__main__":
    main()
