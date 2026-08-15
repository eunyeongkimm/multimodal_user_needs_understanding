"""Stage 2-3: 화자별 정규화 + 자연어 변환 (SpeechCueLLM Description 방식).

입력: stage2_utterance_pool.parquet(발화 메타) + stage2_acoustic_features.parquet(raw
F0/Energy/silence) + d04의 r_gold(pool에 이미 포함).

정규화 방식:
- 콜드스타트 원칙: 콜 외부 정보 사용 금지가 기본 원칙.
- 각 (call_id, window=N x version, speaker_group=고객/상담사) 그룹 내에서
  z = (x - 그룹평균) / 그룹표준편차 로 정규화.
- 그룹 내 표본이 1개뿐이면(표준편차 계산 불가) 데이터셋 전체의 해당
  speaker_group 분포(mean/std)로 대체 정규화 (norm_source='global_fallback'로 표시).
- 정규화 대상: f0_mean, f0_std, e_mean_db, e_std_db, r_gold, s_top_db30 (6개 지표).

자연어 변환:
- 정규화된 z-score를 데이터셋 전체 분포 기준 분위수(quintile)로 5단계 분류:
  매우낮음/낮음/보통/높음/매우높음 (하위 20%씩).
- 감정 해석 없이 값 그대로 서술 (예: "음높이는 높은 편이고 변동이 크며,
  발화속도는 빠름").
- 화자 태그를 붙여 발화 순서대로 나열할 수 있도록 발화 단위 설명 문자열 저장.

윈도우(N x version) 조합별로 정규화 그룹이 달라지므로, 콜 하나가 여러 윈도우에
동시에 속할 수 있어 출력은 (call_id, window_n, window_version, dialog_idx) 단위
long format으로 저장한다.

출력: outputs/stage2_windows_nl.parquet
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
POOL_PATH = BASE_DIR / "outputs" / "stage2_utterance_pool.parquet"
ACOUSTIC_PATH = BASE_DIR / "outputs" / "stage2_acoustic_features.parquet"
OUT_PATH = BASE_DIR / "outputs" / "stage2_windows_nl.parquet"
PROGRESS_PATH = BASE_DIR / "outputs" / "stage2_progress.json"


def mark_progress(key: str, value):
    progress = {}
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            progress = json.load(f)
    progress[key] = value
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

N_VALUES = [1, 2, 3, 5]
VERSIONS = ["customer_only", "all"]
METRICS = ["f0_mean", "f0_std", "e_mean_db", "e_std_db", "r_gold", "s_top_db30"]

LEVEL_LABELS = ["매우낮음", "낮음", "보통", "높음", "매우높음"]

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


def compute_global_fallback(acoustic: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    merged = pool[["call_id", "dialog_idx", "is_customer", "r_gold"]].merge(
        acoustic, on=["call_id", "dialog_idx"], how="left"
    )
    merged["speaker_group"] = merged["is_customer"].map({True: "고객", False: "상담사"})
    global_stats = merged.groupby("speaker_group")[METRICS].agg(["mean", "std"])
    return global_stats


def normalize_group(df: pd.DataFrame, global_stats: pd.DataFrame) -> pd.DataFrame:
    """(call_id, window_n, window_version, speaker_group) 단위로 정규화."""
    out_frames = []
    for (call_id, n, version, sg), g in df.groupby(
        ["call_id", "window_n", "window_version", "speaker_group"], sort=False
    ):
        g = g.copy()
        n_obs = len(g)
        for m in METRICS:
            if n_obs >= 2 and g[m].notna().sum() >= 2 and g[m].std(ddof=0) > 0:
                mean_, std_ = g[m].mean(), g[m].std(ddof=0)
                g[f"{m}_z"] = (g[m] - mean_) / std_
                g[f"{m}_norm_source"] = "within_window"
            else:
                gmean = global_stats.loc[sg, (m, "mean")]
                gstd = global_stats.loc[sg, (m, "std")]
                g[f"{m}_z"] = (g[m] - gmean) / gstd if gstd and gstd > 0 else np.nan
                g[f"{m}_norm_source"] = "global_fallback"
        out_frames.append(g)
    return pd.concat(out_frames, ignore_index=True)


def quintile_labels(series: pd.Series) -> pd.Series:
    try:
        return pd.qcut(series, 5, labels=LEVEL_LABELS, duplicates="drop")
    except ValueError:
        # 값이 너무 적거나 동일해 5구간을 못 나누는 경우 전부 '보통'으로
        return pd.Series(["보통"] * len(series), index=series.index)


def build_nl_description(row) -> str:
    parts = []
    for m in METRICS:
        label = row[f"{m}_level"]
        prefix, phrase_map = METRIC_DESC[m]
        phrase = phrase_map.get(label, "보통")
        parts.append(f"{prefix} {phrase}")
    return ", ".join(parts)


def main():
    pool = pd.read_parquet(POOL_PATH)
    acoustic = pd.read_parquet(ACOUSTIC_PATH)

    global_stats = compute_global_fallback(acoustic, pool)

    merged = pool.merge(acoustic, on=["call_id", "dialog_idx"], how="left")
    merged["speaker_group"] = merged["is_customer"].map({True: "고객", False: "상담사"})

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

    windows = normalize_group(windows, global_stats)

    n_fallback = (windows[[f"{m}_norm_source" for m in METRICS]] == "global_fallback").any(axis=1).sum()
    print(f"global_fallback 적용된 발화: {n_fallback}/{len(windows)}건 "
          f"({n_fallback/len(windows)*100:.2f}%)")

    for m in METRICS:
        windows[f"{m}_level"] = quintile_labels(windows[f"{m}_z"])

    windows["nl_description"] = windows.apply(build_nl_description, axis=1)

    windows = windows.sort_values(["call_id", "window_n", "window_version",
                                    "overall_rank" if False else "dialog_idx"])

    out_cols = [
        "call_id", "window_n", "window_version", "dialog_idx", "speaker_type",
        "speaker_group", "overall_rank", "customer_rank", "text_clean",
    ] + [f"{m}" for m in METRICS] + [f"{m}_z" for m in METRICS] \
      + [f"{m}_level" for m in METRICS] + [f"{m}_norm_source" for m in METRICS] \
      + ["nl_description"]
    windows = windows[out_cols]

    windows.to_parquet(OUT_PATH, index=False)
    print(f"\n저장: {OUT_PATH} ({len(windows)}행)")
    mark_progress("normalize_nl", "done")
    print("STAGE2C_COMPLETE")


if __name__ == "__main__":
    main()
