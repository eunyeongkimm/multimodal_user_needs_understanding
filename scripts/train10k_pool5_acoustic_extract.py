"""train10k pool5(N<=5) 발화의 acoustic feature 추출. N=2 때 이미 추출한 발화
(train10k_acoustic_features.parquet, 26,440행)는 재사용하고, 새로 추가된
3~5번째 발화만 신규 추출한다(중복 추출 회피).

출력: outputs/train10k_pool5_acoustic_features.parquet (기존 26,440행 + 신규분 병합)
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
POOL_PATH = BASE_DIR / "outputs" / "train10k_pool5_utterance_pool.parquet"
EXISTING_N2_PATH = BASE_DIR / "outputs" / "train10k_acoustic_features.parquet"
PARTIAL_DIR = BASE_DIR / "outputs" / "train10k_pool5_acoustic_partial"
PROGRESS_PATH = BASE_DIR / "outputs" / "train10k_pool5_acoustic_progress.json"
FINAL_PATH = BASE_DIR / "outputs" / "train10k_pool5_acoustic_features.parquet"
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
    return {"chunks_done": []}


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
    if not AUDIO_BASE_DIR.exists():
        print(f"ERROR: 외장하드 미마운트 - {AUDIO_BASE_DIR} 없음.")
        return

    pool = pd.read_parquet(POOL_PATH)
    existing = pd.read_parquet(EXISTING_N2_PATH)
    existing_keys = set(zip(existing["call_id"], existing["dialog_idx"]))
    print(f"pool5 전체: {len(pool)}행, 기존(N=2) 추출분 재사용 가능: {len(existing_keys)}건")

    pool_keys = pd.Series(list(zip(pool["call_id"], pool["dialog_idx"])))
    is_new = ~pool_keys.isin(existing_keys)
    new_pool = pool[is_new.values].reset_index(drop=True)
    n_reused = len(pool) - len(new_pool)
    print(f"재사용: {n_reused}건, 신규 추출 필요: {len(new_pool)}건")

    n_chunks = (len(new_pool) + CHUNK_SIZE - 1) // CHUNK_SIZE
    progress = load_progress()
    done_chunks = set(progress.get("chunks_done", []))
    print(f"신규분 {n_chunks}개 청크, 이미 완료: {len(done_chunks)}/{n_chunks}")

    PARTIAL_DIR.mkdir(exist_ok=True)

    for chunk_idx in range(n_chunks):
        if chunk_idx in done_chunks:
            continue
        chunk_path = PARTIAL_DIR / f"chunk_{chunk_idx}.parquet"
        chunk_df = new_pool.iloc[chunk_idx * CHUNK_SIZE:(chunk_idx + 1) * CHUNK_SIZE]
        args_list = list(zip(chunk_df["call_id"], chunk_df["dialog_idx"],
                              chunk_df["audioPath"], chunk_df["duration"]))
        t0 = time.time()
        with Pool(N_WORKERS) as p:
            results = p.map(extract_one, args_list)
        elapsed = time.time() - t0
        result_df = pd.DataFrame(results)
        result_df.to_parquet(chunk_path, index=False)
        done_chunks.add(chunk_idx)
        progress["chunks_done"] = sorted(done_chunks)
        progress["chunks_total"] = n_chunks
        save_progress(progress)
        n_fail = (~result_df["load_ok"]).sum()
        print(f"청크 {chunk_idx+1}/{n_chunks} 완료 ({len(chunk_df)}행, {elapsed:.1f}초, "
              f"로딩실패 {n_fail}건) -> {chunk_path.name}")

    if n_chunks == 0 or len(done_chunks) == n_chunks:
        if n_chunks > 0:
            new_parts = [pd.read_parquet(PARTIAL_DIR / f"chunk_{i}.parquet") for i in range(n_chunks)]
            new_merged = pd.concat(new_parts, ignore_index=True)
        else:
            new_merged = pd.DataFrame(columns=existing.columns)
        merged = pd.concat([existing, new_merged], ignore_index=True)
        # pool5 대상 키만 남김 (existing에는 pool5에 없는 키는 없음 - N=2가 pool5의 부분집합)
        merged = merged.merge(pool[["call_id", "dialog_idx"]], on=["call_id", "dialog_idx"], how="inner")
        merged = merged.drop_duplicates(subset=["call_id", "dialog_idx"])
        merged.to_parquet(FINAL_PATH, index=False)
        progress["status"] = "done"
        save_progress(progress)
        n_fail_total = (~merged["load_ok"]).sum()
        print(f"\n저장: {FINAL_PATH} ({len(merged)}행, pool5 전체 {len(pool)}행과 "
              f"{'일치' if len(merged)==len(pool) else '불일치'})")
        print(f"전체 로딩 실패: {n_fail_total}건 ({n_fail_total/len(merged)*100:.2f}%)")
        print("TRAIN10K_POOL5_ACOUSTIC_COMPLETE")


if __name__ == "__main__":
    main()
