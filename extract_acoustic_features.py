"""
D04 wav 샘플에 대한 acoustic feature(F0, energy, silence ratio) 추출 프로토타입.

실행:
    source venv/bin/activate
    python scripts/extract_acoustic_features.py
"""
import logging
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import parselmouth

# ---- 경로 설정 ----
PROJECT_DIR = Path("/Users/eunyeongkim/Desktop/skku-ads/1.논문")
DIALOG_INDEX_PATH = PROJECT_DIR / "outputs" / "d04_dialog_index.parquet"
AUDIO_BASE_DIR = Path(
    "/Volumes/AIHub/007.저음질 전화망 음성인식 데이터/01.데이터/1.Training/원천데이터_230316"
)
OUTPUT_CSV_PATH = PROJECT_DIR / "outputs" / "acoustic_features_sample50.csv"

N_SAMPLES = 50
MIN_DURATION_SEC = 1.0
RANDOM_STATE = 42
F0_FLOOR_HZ = 75.0
F0_CEILING_HZ = 500.0
EXPECTED_SR = 8000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("acoustic_features")


# ---------------------------------------------------------------------------
# 0단계: 샘플 선정
# ---------------------------------------------------------------------------
def select_samples() -> pd.DataFrame:
    df = pd.read_parquet(DIALOG_INDEX_PATH)

    filtered = df[(df["r_gold_valid_flag"] == True) & (df["duration"] >= MIN_DURATION_SEC)].copy()  # noqa: E712

    # speaker_type은 상담사1/2/3, 고객1/2/3 등으로 세분화되어 있으므로
    # 상담사 vs 고객 대분류로 묶어서 stratified sampling 수행
    filtered["speaker_group"] = filtered["speaker_type"].str.extract(r"^(상담사|고객)")

    n_groups = filtered["speaker_group"].nunique()
    n_per_group = N_SAMPLES // n_groups

    sampled = (
        filtered.groupby("speaker_group", group_keys=False)
        .apply(lambda g: g.sample(n=min(n_per_group, len(g)), random_state=RANDOM_STATE))
        .reset_index(drop=True)
    )

    logger.info("필터링 후 대상: %d행 / 샘플 추출: %d행", len(filtered), len(sampled))
    logger.info("샘플 speaker_group 분포:\n%s", sampled["speaker_group"].value_counts())
    return sampled


# ---------------------------------------------------------------------------
# 1단계: 오디오 로딩 공통 처리
# ---------------------------------------------------------------------------
def load_audio(wav_path: Path):
    """librosa로 오디오를 로딩하고 sr을 확인. 실패 시 (None, None) 반환."""
    try:
        y, sr = librosa.load(str(wav_path), sr=None)
    except Exception:
        logger.exception("오디오 로딩 실패: %s", wav_path)
        return None, None

    if abs(sr - EXPECTED_SR) > 100:
        logger.warning("예상과 다른 샘플링레이트 감지: %s (sr=%d, 기대값=%dHz 근처)", wav_path, sr, EXPECTED_SR)
    else:
        logger.info("sr 확인 OK: %s (sr=%d)", wav_path.name, sr)

    return y, sr


# ---------------------------------------------------------------------------
# 2단계: Feature 함수
# ---------------------------------------------------------------------------
def extract_f0(wav_path: Path) -> dict:
    """parselmouth(Praat)로 F0(기본주파수)를 추출한다.

    무성음/무음 구간(F0=0 또는 undefined)은 제외하고 유성음 프레임만으로
    mean/std를 계산한다. 전화망 음성(남녀 화자 혼재)을 고려해 pitch floor/
    ceiling을 각각 F0_FLOOR_HZ(75Hz), F0_CEILING_HZ(500Hz)로 명시한다.

    Parameters
    ----------
    wav_path : Path
        분석할 wav 파일 경로.

    Returns
    -------
    dict
        - f0_mean (float): 유성음 프레임 F0 평균(Hz). 유성음이 없으면 NaN.
        - f0_std (float): 유성음 프레임 F0 표준편차(Hz). 유성음이 없으면 NaN.
        - f0_voiced_frame_ratio (float): 전체 프레임 중 유성음 프레임 비율(0~1).
          추출 자체가 실패한 경우 NaN.
    """
    try:
        sound = parselmouth.Sound(str(wav_path))
        pitch = sound.to_pitch(pitch_floor=F0_FLOOR_HZ, pitch_ceiling=F0_CEILING_HZ)
        f0_values = pitch.selected_array["frequency"]
        voiced = f0_values[f0_values > 0]  # 0Hz = unvoiced/undefined 프레임 제외

        if voiced.size == 0:
            return {"f0_mean": np.nan, "f0_std": np.nan, "f0_voiced_frame_ratio": 0.0}

        return {
            "f0_mean": float(np.mean(voiced)),
            "f0_std": float(np.std(voiced)),
            "f0_voiced_frame_ratio": float(voiced.size / f0_values.size),
        }
    except Exception:
        logger.exception("F0 추출 실패: %s", wav_path)
        return {"f0_mean": np.nan, "f0_std": np.nan, "f0_voiced_frame_ratio": np.nan}


def extract_energy(y: np.ndarray) -> dict:
    """librosa RMS(root-mean-square) 기반 energy를 계산한다.

    프레임 단위 RMS를 amplitude 그대로와 dB(librosa.amplitude_to_db, 프레임 내
    최댓값 기준 상대 dB) 두 가지 스케일로 함께 반환해 이후 비교에 쓸 수 있게 한다.

    Parameters
    ----------
    y : np.ndarray
        librosa.load()로 로딩된 오디오 시계열(1차원 float 배열).

    Returns
    -------
    dict
        - e_mean_amp (float): 프레임별 RMS(amplitude)의 평균.
        - e_std_amp (float): 프레임별 RMS(amplitude)의 표준편차.
        - e_mean_db (float): 프레임별 RMS를 dB로 변환한 값의 평균.
        - e_std_db (float): 프레임별 RMS를 dB로 변환한 값의 표준편차.
    """
    rms = librosa.feature.rms(y=y)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    return {
        "e_mean_amp": float(np.mean(rms)),
        "e_std_amp": float(np.std(rms)),
        "e_mean_db": float(np.mean(rms_db)),
        "e_std_db": float(np.std(rms_db)),
    }


def extract_silence_ratio(y: np.ndarray, sr: int, duration: float) -> dict:
    """librosa.effects.split() 기반 무음 비율(silence ratio)을 계산한다.

    non-silent 구간을 top_db 임계값으로 탐지한 뒤, 전체 duration 대비
    non-silent 구간이 차지하지 않는 비율(s = 1 - non_silent/전체)을 구한다.
    top_db가 낮을수록(예: 20) 더 작은 소리도 유효 음성으로 간주해 무음 비율이
    낮게 나오고, 높을수록(예: 30) 엄격해져 무음 비율이 높게 나오는 경향이 있어
    민감도 비교를 위해 두 값을 모두 계산한다.

    Parameters
    ----------
    y : np.ndarray
        librosa.load()로 로딩된 오디오 시계열(1차원 float 배열).
    sr : int
        오디오의 샘플링레이트(Hz). non-silent 구간 길이(초) 환산에 사용.
    duration : float
        전체 오디오 길이(초). 무음 비율 계산의 분모.

    Returns
    -------
    dict
        - s_top_db20 (float): top_db=20 기준 무음 비율(0~1).
        - s_top_db30 (float): top_db=30 기준 무음 비율(0~1).
    """
    result = {}
    for top_db in (20, 30):
        intervals = librosa.effects.split(y, top_db=top_db)
        non_silent_sec = sum((end - start) for start, end in intervals) / sr
        s = 1 - (non_silent_sec / duration) if duration > 0 else np.nan
        result[f"s_top_db{top_db}"] = float(s)
    return result


# ---------------------------------------------------------------------------
# 3단계: 실행 및 결과 취합
# ---------------------------------------------------------------------------
def run_extraction(sampled: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in sampled.iterrows():
        wav_path = AUDIO_BASE_DIR / row["audioPath"]
        y, sr = load_audio(wav_path)

        record = {
            "call_id": row["call_id"],
            "dialog_idx": row["dialog_idx"],
            "speaker_type": row["speaker_type"],
            "speaker_group": row["speaker_group"],
            "duration": row["duration"],
            "audioPath": row["audioPath"],
        }

        if y is None:
            record.update({
                "load_ok": False,
                "sr": np.nan,
                "f0_mean": np.nan, "f0_std": np.nan, "f0_voiced_frame_ratio": np.nan,
                "e_mean_amp": np.nan, "e_std_amp": np.nan,
                "e_mean_db": np.nan, "e_std_db": np.nan,
                "s_top_db20": np.nan, "s_top_db30": np.nan,
            })
            rows.append(record)
            continue

        f0_feats = extract_f0(wav_path)
        energy_feats = extract_energy(y)
        silence_feats = extract_silence_ratio(y, sr, row["duration"])

        record["load_ok"] = True
        record["sr"] = sr
        record.update(f0_feats)
        record.update(energy_feats)
        record.update(silence_feats)
        rows.append(record)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    logger.info("결과 저장 완료: %s (%d행)", OUTPUT_CSV_PATH, len(result_df))
    return result_df


# ---------------------------------------------------------------------------
# 4단계: 품질 체크
# ---------------------------------------------------------------------------
def quality_check(result_df: pd.DataFrame) -> None:
    n_total = len(result_df)
    n_load_fail = (~result_df["load_ok"]).sum()

    n_f0_total_fail = result_df["f0_voiced_frame_ratio"].fillna(0).eq(0).sum()
    f0_fail_rate = n_f0_total_fail / n_total if n_total else np.nan

    valid_f0 = result_df["f0_mean"].dropna()
    out_of_range = valid_f0[(valid_f0 < F0_FLOOR_HZ) | (valid_f0 > F0_CEILING_HZ)]

    print("\n" + "=" * 70)
    print("[품질 체크]")
    print("=" * 70)
    print(f"전체 샘플 수: {n_total}")
    print(f"오디오 로딩 실패: {n_load_fail}건 ({n_load_fail / n_total:.1%})")
    print(f"F0 완전 실패(전체 무성음/무음) 비율: {f0_fail_rate:.1%} ({n_f0_total_fail}/{n_total})")
    print(f"F0_mean이 {F0_FLOOR_HZ}-{F0_CEILING_HZ}Hz 범위를 벗어난 케이스: {len(out_of_range)}건")
    if len(out_of_range):
        print(out_of_range.to_string())

    print("\n--- speaker_group별 F0/Energy sanity check (groupby 통계) ---")
    sanity = result_df.groupby("speaker_group")[
        ["f0_mean", "f0_std", "e_mean_amp", "e_mean_db"]
    ].agg(["mean", "std", "count"])
    print(sanity)

    print("\n--- top_db 20 vs 30 민감도 비교 ---")
    diff = (result_df["s_top_db20"] - result_df["s_top_db30"]).dropna()
    print(f"s_top_db20 - s_top_db30 : mean={diff.mean():.4f}, std={diff.std():.4f}, "
          f"min={diff.min():.4f}, max={diff.max():.4f}")

    print("\n--- 전체 feature 분포 요약 (describe) ---")
    feature_cols = [
        "duration", "f0_mean", "f0_std", "e_mean_amp", "e_std_amp",
        "e_mean_db", "e_std_db", "s_top_db20", "s_top_db30",
    ]
    print(result_df[feature_cols].describe())
    print("=" * 70)


def main():
    sampled = select_samples()
    result_df = run_extraction(sampled)
    quality_check(result_df)


if __name__ == "__main__":
    main()
