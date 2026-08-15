"""A·B 공통 오답(gold=불만제기 -> 예측=환불요청) 콜의 초반 고객 arousal 점검.

층화 실험이 아니라 가설 확인용. API 0, 전부 로컬 파일.

arousal = 코퍼스 baseline(성별 분리) 대비 z-score.
  z_pitch  = (콜 초반 고객 F0mean  - baseline_mean) / baseline_std
  z_energy = (콜 초반 고객 RMSmean - baseline_mean) / baseline_std
  arousal  = (z_pitch + z_energy) / 2

주의 - 에너지 지표 선택:
  e_mean_db 는 librosa.amplitude_to_db(rms, ref=np.max) 로, ref가 발화 자신의
  피크라 발화 간 절대 성량 비교에 쓸 수 없다(사실상 dynamic range). 따라서
  z_energy 는 raw linear RMS인 e_mean_amp 로 계산하고, e_mean_db 기반 값은
  z_energy_db 로 참고 저장만 한다.

주의 - baseline 입도:
  stage2_acoustic_features 는 발화 단위 집계값이므로 baseline 은 "코퍼스 전
  프레임의 분포"가 아니라 "발화별 평균값들의 분포"다. 프레임 풀링 대비 std가
  작아 z 절대크기가 달라진다.

출력:
  outputs/arousal_target_percall.parquet
  outputs/arousal_target_summary.md
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

BASE_DIR = Path(__file__).resolve().parent.parent
PERCALL_AB = BASE_DIR / "outputs" / "modality_ab_percall.parquet"
ACOUSTIC_PATH = BASE_DIR / "outputs" / "stage2_acoustic_features.parquet"
D04_PATH = BASE_DIR / "outputs" / "d04_dialog_index.parquet"
WINDOWS_PATH = BASE_DIR / "outputs" / "stage2_windows_nl_v2.parquet"
POOL_PATH = BASE_DIR / "outputs" / "stage2_utterance_pool.parquet"
MD_PATH = BASE_DIR / "outputs" / "arousal_target_summary.md"
PERCALL_PATH = BASE_DIR / "outputs" / "arousal_target_percall.parquet"

COMPLAINT = "불만제기"
REFUND = "환불요청"
WINDOW_N = 5
WINDOW_VERSION = "customer_only"

# (baseline 계산용 지표, 콜 단위 z 컬럼명)
METRICS = [("f0_mean", "z_pitch"), ("e_mean_amp", "z_energy"), ("e_mean_db", "z_energy_db")]
GROUP_ORDER = ["타깃18", "맞춘불만", "전체불만", "비불만전체"]
DIST_COLS = ["z_pitch", "z_energy", "arousal", "z_energy_db"]

log = []


def p(msg=""):
    print(msg)
    log.append(msg)


def main():
    # ---------- 0. 타깃/대조군 ----------
    ab = pd.read_parquet(PERCALL_AB)
    is_c = ab["gold_actual"] == COMPLAINT
    target = is_c & (ab["pred_A"] == REFUND) & (ab["pred_B"] == REFUND)
    hit = is_c & ((ab["pred_A"] == COMPLAINT) | (ab["pred_B"] == COMPLAINT))

    p("## 0. 그룹 정의")
    p("")
    p(f"- 타깃18 = gold={COMPLAINT} & A예측={REFUND} & B예측={REFUND}: **{int(target.sum())}건**")
    p(f"  (A 단독 {int((is_c & (ab['pred_A'] == REFUND)).sum())}건, "
      f"B 단독 {int((is_c & (ab['pred_B'] == REFUND)).sum())}건 -> 교집합 {int(target.sum())}건. "
      f"당초 예상 19건이 아니라 18건)")
    p(f"- (ㄱ) 맞춘불만 = gold={COMPLAINT} & (A 또는 B가 {COMPLAINT}): **{int(hit.sum())}건**")
    p(f"- (ㄴ) 전체불만 = gold={COMPLAINT}: **{int(is_c.sum())}건**")
    p(f"- (ㄷ) 비불만전체 = gold!={COMPLAINT}: **{int((~is_c).sum())}건**")
    p("")
    p("타깃18과 맞춘불만은 겹치지 않는다(예측이 서로 배타적). "
      "전체불만은 타깃18·맞춘불만을 포함하는 상위 집합, 비불만전체는 분리 집합이다.")
    p("")

    ab = ab.assign(is_target=target, is_hit=hit)

    # ---------- 1. 코퍼스 baseline ----------
    ac = pd.read_parquet(ACOUSTIC_PATH)
    d04 = pd.read_parquet(
        D04_PATH, columns=["call_id", "dialog_idx", "speaker_type",
                           "speaker_gender", "r_gold_valid_flag"])
    utt = ac.merge(d04, on=["call_id", "dialog_idx"], how="left")
    assert utt["speaker_gender"].isna().sum() == 0, "성별 결측 발생"

    p("## 1. 코퍼스 baseline")
    p("")
    p(f"원천: `stage2_acoustic_features.parquet` 전체 발화 {len(utt):,}건 "
      f"(batch1 stage2 pool). 성별·화자·유효성은 `d04_dialog_index.parquet`에서 병합.")
    p("")
    n_all = len(utt)
    utt = utt[utt["speaker_type"].astype(str).str.startswith("고객")]
    p(f"- 화자 필터: `speaker_type`이 '고객'으로 시작 (고객1/2/3). "
      f"{n_all:,} -> **{len(utt):,}건** (id 미사용, type 필드만)")
    n_before = len(utt)
    n_invalid = int((utt["r_gold_valid_flag"] != True).sum())  # noqa: E712
    utt = utt[utt["r_gold_valid_flag"] == True]  # noqa: E712
    p(f"- truncated 제외: `r_gold_valid_flag != True` **{n_invalid}건** 제외 "
      f"({n_before:,} -> {len(utt):,}건). stage2 pool이 이미 valid만 담고 있어 0건.")
    p("")
    p("추출기·파라미터 (`stage2b_acoustic_extract.py`, 기존 추출물 재사용 / WAV 재추출 없음):")
    p("")
    p("- **F0**: `parselmouth` (Praat) `Sound.to_pitch(pitch_floor=75.0, pitch_ceiling=500.0)`, "
      "`frequency > 0`인 **유성 프레임만** 사용해 평균 -> 발화별 `f0_mean`. 무음/무성 프레임 제외.")
    p("- **RMS**: `librosa.feature.rms(y=y)` (기본 frame_length=2048, hop_length=512), "
      "`librosa.load(sr=None)`으로 원본 샘플레이트 유지 -> 발화별 `e_mean_amp`(linear RMS 평균).")
    p("- **에너지 지표 선택**: `e_mean_db`는 `amplitude_to_db(rms, ref=np.max)`로 "
      "ref가 발화 자신의 피크라 발화 간 절대 성량 비교에 부적합(사실상 dynamic range). "
      "따라서 `z_energy`는 **`e_mean_amp`(raw linear RMS)** 기준이며, "
      "`e_mean_db` 기반 값은 `z_energy_db`로 참고 저장만 한다.")
    p("- **baseline 입도**: 프레임 단위 풀링이 아니라 **발화별 집계값의 분포**다"
      "(프레임 raw는 저장돼 있지 않고, WAV 재추출은 외장하드 미연결로 불가). "
      "프레임 풀링 대비 std가 작아 z의 절대 크기가 달라진다.")
    p("- **표본 포함 범위**: 이번 250건도 baseline에 포함돼 있다"
      f"(전역 통계 {len(utt):,}건 대비 비중이 작아 영향 미미).")
    p("")

    base = {}
    rows = []
    for gender, g in utt.groupby("speaker_gender"):
        for col, _z in METRICS:
            v = g[col].dropna()
            base[(gender, col)] = (float(v.mean()), float(v.std(ddof=0)))
            rows.append({"성별": gender, "지표": col, "n": len(v),
                         "mean": v.mean(), "std": v.std(ddof=0)})
    bdf = pd.DataFrame(rows)
    p("### baseline 통계 (성별 분리)")
    p("")
    p("| 성별 | 지표 | n | mean | std |")
    p("|---|---|---|---|---|")
    for r in bdf.itertuples():
        p(f"| {r.성별} | {r.지표} | {r.n:,} | {r.mean:.4f} | {r.std:.4f} |")
    p("")

    # ---------- 2. 콜별 초반 arousal ----------
    win = pd.read_parquet(WINDOWS_PATH,
                          columns=["call_id", "window_n", "window_version", "dialog_idx"])
    win = win[(win["window_n"] == WINDOW_N) & (win["window_version"] == WINDOW_VERSION)
              & win["call_id"].isin(ab["call_id"])]
    p("## 2. 콜별 초반 arousal")
    p("")
    p(f"대상 발화 = 프롬프트가 실제로 본 것과 동일한 창"
      f"(`stage2_windows_nl_v2`, window_n={WINDOW_N}, window_version={WINDOW_VERSION}) "
      f"= 초반 5발화 중 고객 발화. 총 {len(win):,}발화 / {win['call_id'].nunique()}콜.")
    p("")

    dur = pd.read_parquet(POOL_PATH, columns=["call_id", "dialog_idx", "duration"])
    seg = win.merge(utt, on=["call_id", "dialog_idx"], how="left").merge(
        dur, on=["call_id", "dialog_idx"], how="left")
    n_seg_missing = int(seg["f0_mean"].isna().sum())

    agg = seg.groupby("call_id").agg(
        n_utt_초반=("dialog_idx", "size"),
        dur_초반_sec=("duration", "sum"),
        gender=("speaker_gender", lambda s: s.mode().iat[0] if len(s.mode()) else None),
        **{c: (c, "mean") for c, _z in METRICS},
    ).reset_index()

    for col, zcol in METRICS:
        mu = agg["gender"].map(lambda g: base[(g, col)][0])
        sd = agg["gender"].map(lambda g: base[(g, col)][1])
        agg[zcol] = (agg[col] - mu) / sd
    agg["arousal"] = agg[["z_pitch", "z_energy"]].mean(axis=1)

    p(f"- 콜 단위 집계: 해당 발화들의 `f0_mean` / `e_mean_amp` / `e_mean_db` **단순 평균** "
      f"(발화 길이 가중 아님).")
    p(f"- 성별은 콜 내 고객 발화의 최빈값 사용.")
    p(f"- z-score는 **해당 콜 성별의 baseline** 기준. `arousal = (z_pitch + z_energy) / 2`.")
    p(f"- 초반 창에서 acoustic 결측 발화: **{n_seg_missing}건**.")
    p(f"- 프레임 수는 저장돼 있지 않아 `n_utt_초반`(발화 수)과 "
      f"`dur_초반_sec`(구간 길이 합)으로 대체 기록한다.")
    p("")

    df = ab.merge(agg, on="call_id", how="left")
    n_no_ac = int(df["arousal"].isna().sum())
    p(f"- 250건 중 arousal 산출 실패: **{n_no_ac}건** (분포 계산에서 제외, 삭제 아님)")
    p("")

    df["group"] = np.where(df["is_target"], "타깃18",
                   np.where(df["is_hit"], "맞춘불만",
                    np.where(df["gold_actual"] == COMPLAINT, "기타불만", "비불만전체")))
    df["r_gold_valid"] = True  # 초반 창은 전량 valid (위 로그 참조)

    # ---------- 3. 그룹별 분포 ----------
    masks = {
        "타깃18": df["is_target"],
        "맞춘불만": df["is_hit"],
        "전체불만": df["gold_actual"] == COMPLAINT,
        "비불만전체": df["gold_actual"] != COMPLAINT,
    }

    p("## 3. 그룹별 분포 (절대값)")
    p("")
    for col in DIST_COLS:
        note = " (참고용, ref=np.max 기반)" if col == "z_energy_db" else ""
        p(f"### {col}{note}")
        p("")
        p("| 그룹 | n | mean | median | std |")
        p("|---|---|---|---|---|")
        for gname in GROUP_ORDER:
            v = df.loc[masks[gname], col].dropna()
            p(f"| {gname} | {len(v)} | {v.mean():.4f} | {v.median():.4f} | {v.std(ddof=1):.4f} |")
        p("")

    p("### Mann-Whitney U (타깃18 vs 맞춘불만)")
    p("")
    p(f"n이 {int(masks['타깃18'].sum())} vs {int(masks['맞춘불만'].sum())}로 작아 참고용.")
    p("")
    p("| 지표 | n(타깃18) | n(맞춘불만) | median(타깃18) | median(맞춘불만) | U | p (양측) |")
    p("|---|---|---|---|---|---|---|")
    for col in DIST_COLS:
        a_ = df.loc[masks["타깃18"], col].dropna()
        b_ = df.loc[masks["맞춘불만"], col].dropna()
        u, pv = mannwhitneyu(a_, b_, alternative="two-sided")
        p(f"| {col} | {len(a_)} | {len(b_)} | {a_.median():.4f} | {b_.median():.4f} | "
          f"{u:.1f} | {pv:.4f} |")
    p("")

    # ---------- 4. 저장 ----------
    out_cols = ["call_id", "gold_actual", "pred_A", "pred_B", "group", "gender",
                "z_pitch", "z_energy", "arousal", "z_energy_db",
                "n_utt_초반", "dur_초반_sec", "r_gold_valid"]
    df[out_cols].to_parquet(PERCALL_PATH, index=False)

    header = ["# 타깃18(A·B 공통 불만→환불) 초반 고객 arousal 점검", "",
              "가설 확인용 진단이며 층화 실험이 아니다. API 호출 0, 전부 로컬 파일. "
              "gold_actual은 그룹을 나누는 데만 쓰였고 arousal 계산에는 쓰이지 않았다.", ""]
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(header + log) + "\n")

    print(f"\n저장: {MD_PATH}")
    print(f"저장: {PERCALL_PATH} ({len(df)}행)")
    print("AROUSAL_TARGET_CHECK_COMPLETE")


if __name__ == "__main__":
    main()
