"""Stage 2-3 (v2): 이원화 정규화 + 자연어 변환 (자연어 서술 방식은 기존과 100% 동일).

기존 stage2c_normalize_nl.py 대비 유일한 변경점: 정규화 방식.
- (call_id, window_n, window_version, speaker_group) 그룹에서 해당 화자의
  윈도우 내 발화 수(n_obs)가:
  - 4개 이상: within-speaker 정규화 (그 화자 본인 발화들의 평균/표준편차 사용) -
    norm_source='within_speaker'
  - 3개 이하(또는 위 조건에서 표준편차=0/결측 등으로 계산 불가): 같은 성별
    전체 population 기준 정규화 - norm_source='population_gender'
    (gender가 없으면 성별 무관 전체 fallback - norm_source='population_fallback_no_gender',
    이번 데이터셋엔 gender 결측이 0건이라 이론상 발생하지 않음)

population 통계는 scripts/stage2l_gender_population_stats.py가 미리 계산해둔
outputs/stage2_gender_population_stats.csv를 그대로 사용한다 (재계산 안 함).

quintile 분위수 계산(LEVEL_LABELS 5단계)과 자연어 문장 생성(METRIC_DESC,
build_nl_description)은 stage2c_normalize_nl.py와 완전히 동일한 코드를 그대로
가져왔다 - 서술 방식은 절대 변경하지 않는다는 원칙에 따름.

출력:
  - outputs/stage2_windows_nl_v2.parquet (재정규화된 윈도우, 기존 v1과 동일 스키마
    + norm_source 값만 새로운 카테고리로 바뀜)
  - outputs/stage2_renorm_application_log.csv (화자x지표별로 어느 기준이
    적용됐는지 카운트)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
POOL_PATH = BASE_DIR / "outputs" / "stage2_utterance_pool.parquet"
ACOUSTIC_PATH = BASE_DIR / "outputs" / "stage2_acoustic_features.parquet"
D04_PATH = BASE_DIR / "outputs" / "d04_dialog_index.parquet"
GENDER_STATS_PATH = BASE_DIR / "outputs" / "stage2_gender_population_stats.csv"
OUT_PATH = BASE_DIR / "outputs" / "stage2_windows_nl_v2.parquet"
LOG_PATH = BASE_DIR / "outputs" / "stage2_renorm_application_log.csv"
PROGRESS_PATH = BASE_DIR / "outputs" / "stage2_progress.json"

N_VALUES = [1, 2, 3, 5]
VERSIONS = ["customer_only", "all"]
METRICS = ["f0_mean", "f0_std", "e_mean_db", "e_std_db", "r_gold", "s_top_db30"]
WITHIN_SPEAKER_MIN_N = 4  # 이 이상이면 within-speaker, 미만이면 population

LEVEL_LABELS = ["매우낮음", "낮음", "보통", "높음", "매우높음"]

# --- 아래 METRIC_DESC / quintile_labels / build_nl_description은
# --- stage2c_normalize_nl.py와 100% 동일 (서술 방식 불변 원칙) ---
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


def mark_progress(key: str, value):
    progress = {}
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            progress = json.load(f)
    progress[key] = value
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def load_gender_lookup():
    """{(gender, metric): {'mean':.., 'std':..}} 형태로 population 통계 로드."""
    stats = pd.read_csv(GENDER_STATS_PATH)
    lookup = {}
    for _, row in stats.iterrows():
        lookup[(row["gender"], row["metric"])] = {"mean": row["mean"], "std": row["std"]}
    return lookup


def attach_gender_population_columns(df: pd.DataFrame, gender_lookup: dict) -> pd.DataFrame:
    """각 행에 그 행의 성별(없으면 ALL) 기준 population mean/std를 벡터화해서 붙인다."""
    df = df.copy()
    gender_key = df["speaker_gender"].where(df["speaker_gender"].notna(), "ALL")
    for m in METRICS:
        means = gender_key.map(lambda g: gender_lookup.get((g, m), gender_lookup[("ALL", m)])["mean"])
        stds = gender_key.map(lambda g: gender_lookup.get((g, m), gender_lookup[("ALL", m)])["std"])
        df[f"{m}_pop_mean"] = means
        df[f"{m}_pop_std"] = stds
    df["gender_used_fallback"] = df["speaker_gender"].isna()
    return df


def normalize_group_v2(df: pd.DataFrame) -> pd.DataFrame:
    """(call_id, window_n, window_version, speaker_group) 단위, n_obs 기준
    within-speaker(>=4) vs population_gender(<=3) 이원화 정규화."""
    out_frames = []
    for (call_id, n, version, sg), g in df.groupby(
        ["call_id", "window_n", "window_version", "speaker_group"], sort=False
    ):
        g = g.copy()
        n_obs = len(g)
        for m in METRICS:
            can_within_speaker = (
                n_obs >= WITHIN_SPEAKER_MIN_N
                and g[m].notna().sum() >= 2
                and g[m].std(ddof=0) > 0
            )
            if can_within_speaker:
                mean_, std_ = g[m].mean(), g[m].std(ddof=0)
                g[f"{m}_z"] = (g[m] - mean_) / std_
                g[f"{m}_norm_source"] = "within_speaker"
            else:
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

    n_gender_missing = merged["speaker_gender"].isna().sum()
    print(f"pool({len(merged)}건) 중 speaker_gender 결측: {n_gender_missing}건")

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
    print(f"윈도우 전개 후 총 행수: {len(windows)} "
          f"(콜 {windows['call_id'].nunique()}개 x {len(N_VALUES)}xN x {len(VERSIONS)}버전)")

    windows = normalize_group_v2(windows)

    # 적용 로그: (window_n, window_version, speaker_group) x norm_source 카운트
    log_rows = []
    for m in METRICS:
        vc = windows.groupby(["window_n", "window_version", "speaker_group"])[f"{m}_norm_source"] \
            .value_counts().rename("count").reset_index()
        vc["metric"] = m
        log_rows.append(vc)
    app_log = pd.concat(log_rows, ignore_index=True)
    app_log.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")
    print(f"\n적용 로그 저장: {LOG_PATH}")

    print("\n=== norm_source 전체 분포 (f0_mean 기준) ===")
    print(windows["f0_mean_norm_source"].value_counts())
    print(f"\n비율: {(windows['f0_mean_norm_source'].value_counts(normalize=True)*100).round(2).to_dict()}")

    for m in METRICS:
        windows[f"{m}_level"] = quintile_labels(windows[f"{m}_z"])

    windows["nl_description"] = windows.apply(build_nl_description, axis=1)

    windows = windows.sort_values(["call_id", "window_n", "window_version", "dialog_idx"])

    out_cols = [
        "call_id", "window_n", "window_version", "dialog_idx", "speaker_type",
        "speaker_group", "speaker_gender", "overall_rank", "customer_rank", "text_clean",
    ] + [f"{m}" for m in METRICS] + [f"{m}_z" for m in METRICS] \
      + [f"{m}_level" for m in METRICS] + [f"{m}_norm_source" for m in METRICS] \
      + ["nl_description"]
    windows = windows[out_cols]

    windows.to_parquet(OUT_PATH, index=False)
    print(f"\n저장: {OUT_PATH} ({len(windows)}행)")

    # ±1.0 고정 비율 재측정
    print("\n=== 재계산 후: 지표별 z가 정확히 ±1.0인 행 비율 ===")
    for m in METRICS:
        z = windows[f"{m}_z"]
        n_exact = ((z.round(6) == 1.0) | (z.round(6) == -1.0)).sum()
        print(f"{m}: {n_exact}/{len(z)} ({n_exact/len(z)*100:.2f}%)")
    n_exact_any = windows[[f"{m}_z" for m in METRICS]].apply(
        lambda col: col.round(6).isin([1.0, -1.0])
    ).any(axis=1).sum()
    print(f"6개 지표 중 하나라도 ±1.0: {n_exact_any}/{len(windows)} ({n_exact_any/len(windows)*100:.2f}%)")
    print("(기준선 대비: 27.80% -> 위 수치)")

    mark_progress("normalize_nl_v2_gender", "done")
    print("\nSTAGE2C_V2_COMPLETE")


if __name__ == "__main__":
    main()
