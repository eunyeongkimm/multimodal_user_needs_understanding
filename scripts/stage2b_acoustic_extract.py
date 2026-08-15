"""Stage 2-2: Acoustic feature 계산 (발화 단위, raw only - 정규화는 stage2c에서).

stage2_utterance_pool.parquet(135,340건)의 각 발화 오디오에서 F0(mean/std),
Energy(mean/std, dB 기준), silence ratio(top_db=30)를 추출한다.
r_gold는 이미 d04_dialog_index에 있으므로 재계산하지 않고 pool에서 그대로 가져온다.

체크포인트: 청크 단위(기본 5,000행)로 나눠 처리하고, 청크마다
outputs/stage2_acoustic_partial/chunk_{i}.parquet에 저장 + outputs/stage2_progress.json
갱신. 이미 완료된 청크는 재실행 시 건너뛴다 (multiprocessing.Pool 병렬 처리).

실행: python scripts/stage2b_acoustic_extract.py
"""

import json
import time
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import parselmouth

BASE_DIR = Path(__file__).resolve().parent.parent
POOL_PATH = BASE_DIR / "outputs" / "stage2_utterance_pool.parquet"
PARTIAL_DIR = BASE_DIR / "outputs" / "stage2_acoustic_partial"
PROGRESS_PATH = BASE_DIR / "outputs" / "stage2_progress.json"
FINAL_PATH = BASE_DIR / "outputs" / "stage2_acoustic_features.parquet"
AUDIO_BASE_DIR = Path(
    "/Volumes/AIHub/007.저음질 전화망 음성인식 데이터/01.데이터/1.Training/원천데이터_230316"
)

CHUNK_SIZE = 5000
N_WORKERS = 6
F0_FLOOR_HZ = 75.0
F0_CEILING_HZ = 500.0


def load_progress():
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(progress):
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def extract_one(args):
    call_id, dialog_idx, audio_path, duration = args
    wav_path = AUDIO_BASE_DIR / audio_path
    record = {"call_id": call_id, "dialog_idx": dialog_idx}
    try:
        y, sr = librosa.load(str(wav_path), sr=None)
    except Exception as e:
        record.update({
            "load_ok": False, "error": str(e),
            "f0_mean": np.nan, "f0_std": np.nan,
            "e_mean_db": np.nan, "e_std_db": np.nan,
            "e_mean_amp": np.nan, "e_std_amp": np.nan,
            "s_top_db30": np.nan,
        })
        return record

    record["load_ok"] = True
    record["error"] = None

    try:
        sound = parselmouth.Sound(str(wav_path))
        pitch = sound.to_pitch(pitch_floor=F0_FLOOR_HZ, pitch_ceiling=F0_CEILING_HZ)
        f0_values = pitch.selected_array["frequency"]
        voiced = f0_values[f0_values > 0]
        record["f0_mean"] = float(np.mean(voiced)) if voiced.size else np.nan
        record["f0_std"] = float(np.std(voiced)) if voiced.size else np.nan
    except Exception:
        record["f0_mean"] = np.nan
        record["f0_std"] = np.nan

    try:
        rms = librosa.feature.rms(y=y)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        record["e_mean_amp"] = float(np.mean(rms))
        record["e_std_amp"] = float(np.std(rms))
        record["e_mean_db"] = float(np.mean(rms_db))
        record["e_std_db"] = float(np.std(rms_db))
    except Exception:
        record["e_mean_amp"] = np.nan
        record["e_std_amp"] = np.nan
        record["e_mean_db"] = np.nan
        record["e_std_db"] = np.nan

    try:
        intervals = librosa.effects.split(y, top_db=30)
        non_silent_sec = sum((end - start) for start, end in intervals) / sr
        record["s_top_db30"] = float(1 - (non_silent_sec / duration)) if duration > 0 else np.nan
    except Exception:
        record["s_top_db30"] = np.nan

    return record


def main():
    pool = pd.read_parquet(POOL_PATH)
    n_chunks = (len(pool) + CHUNK_SIZE - 1) // CHUNK_SIZE

    progress = load_progress()
    acoustic_progress = progress.get("acoustic_extraction", {"chunks_done": [], "chunks_total": n_chunks})
    done_chunks = set(acoustic_progress.get("chunks_done", []))

    print(f"전체 {len(pool)}행, {n_chunks}개 청크 (청크당 최대 {CHUNK_SIZE}행), "
          f"이미 완료: {len(done_chunks)}/{n_chunks}")

    PARTIAL_DIR.mkdir(exist_ok=True)

    for chunk_idx in range(n_chunks):
        if chunk_idx in done_chunks:
            continue

        chunk_path = PARTIAL_DIR / f"chunk_{chunk_idx}.parquet"
        chunk_df = pool.iloc[chunk_idx * CHUNK_SIZE:(chunk_idx + 1) * CHUNK_SIZE]
        args_list = list(zip(chunk_df["call_id"], chunk_df["dialog_idx"],
                              chunk_df["audioPath"], chunk_df["duration"]))

        t0 = time.time()
        with Pool(N_WORKERS) as p:
            results = p.map(extract_one, args_list)
        elapsed = time.time() - t0

        result_df = pd.DataFrame(results)
        result_df.to_parquet(chunk_path, index=False)

        done_chunks.add(chunk_idx)
        acoustic_progress["chunks_done"] = sorted(done_chunks)
        acoustic_progress["chunks_total"] = n_chunks
        progress["acoustic_extraction"] = acoustic_progress
        save_progress(progress)

        n_fail = (~result_df["load_ok"]).sum()
        print(f"청크 {chunk_idx+1}/{n_chunks} 완료 ({len(chunk_df)}행, {elapsed:.1f}초, "
              f"로딩실패 {n_fail}건) -> {chunk_path.name}")

    if len(done_chunks) == n_chunks:
        parts = [pd.read_parquet(PARTIAL_DIR / f"chunk_{i}.parquet") for i in range(n_chunks)]
        merged = pd.concat(parts, ignore_index=True)
        merged.to_parquet(FINAL_PATH, index=False)
        progress["acoustic_extraction"]["status"] = "done"
        save_progress(progress)
        n_fail_total = (~merged["load_ok"]).sum()
        print(f"\n저장: {FINAL_PATH} ({len(merged)}행)")
        print(f"전체 로딩 실패: {n_fail_total}건 ({n_fail_total/len(merged)*100:.2f}%)")
        print("STAGE2B_COMPLETE")


if __name__ == "__main__":
    main()
