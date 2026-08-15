"""Probing 실험용 "콜 전체" 스코프 acoustic feature 추출.

기존 stage2_acoustic_features.parquet(135,340건)은 stage2 N<=5 pool만 커버한다.
콜 전체 스코프 실험을 위해, batch1(19,847콜)의 r_gold_valid_flag==True 전체
dialog(450,659건) 중 아직 추출 안 된 나머지(315,319건)를 추가로 추출한다.
stage2b_acoustic_extract.py와 동일한 F0/Energy/silence 추출 함수를 그대로
재사용하고(로직 변경 없음), 체크포인트 구조도 동일하게 유지한다.

출력:
  - outputs/probing_full_acoustic_partial/chunk_{i}.parquet (신규 추출분 청크)
  - outputs/full_acoustic_features.parquet (기존 135,340 + 신규 315,319 병합, 450,659건)
"""

import json
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2b_acoustic_extract import extract_one  # 동일 F0/Energy/silence 로직 재사용

BASE_DIR = Path(__file__).resolve().parent.parent
D04_PATH = BASE_DIR / "outputs" / "d04_dialog_index.parquet"
GOLD_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_final.parquet"
EXISTING_ACOUSTIC_PATH = BASE_DIR / "outputs" / "stage2_acoustic_features.parquet"
PARTIAL_DIR = BASE_DIR / "outputs" / "probing_full_acoustic_partial"
PROGRESS_PATH = BASE_DIR / "outputs" / "probing_full_acoustic_progress.json"
FINAL_PATH = BASE_DIR / "outputs" / "full_acoustic_features.parquet"

CHUNK_SIZE = 5000
N_WORKERS = 6


def load_progress():
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"chunks_done": []}


def save_progress(progress):
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def main():
    d04 = pd.read_parquet(D04_PATH)
    gold = pd.read_parquet(GOLD_PATH)
    batch1_ids = set(gold["call_id"])
    valid = d04[(d04["call_id"].isin(batch1_ids)) & (d04["r_gold_valid_flag"] == True)]  # noqa: E712
    print(f"batch1 유효 dialog 전체: {len(valid)}건")

    existing = pd.read_parquet(EXISTING_ACOUSTIC_PATH)
    existing_keys = set(zip(existing["call_id"], existing["dialog_idx"]))
    print(f"기존 추출 완료: {len(existing)}건")

    valid = valid.copy()
    valid["_key"] = list(zip(valid["call_id"], valid["dialog_idx"]))
    todo = valid[~valid["_key"].isin(existing_keys)].drop(columns=["_key"])
    print(f"신규 추출 필요: {len(todo)}건")

    n_chunks = (len(todo) + CHUNK_SIZE - 1) // CHUNK_SIZE
    progress = load_progress()
    done_chunks = set(progress.get("chunks_done", []))
    print(f"청크 {n_chunks}개, 이미 완료: {len(done_chunks)}")

    PARTIAL_DIR.mkdir(exist_ok=True)

    for chunk_idx in range(n_chunks):
        if chunk_idx in done_chunks:
            continue
        chunk_path = PARTIAL_DIR / f"chunk_{chunk_idx}.parquet"
        chunk_df = todo.iloc[chunk_idx * CHUNK_SIZE:(chunk_idx + 1) * CHUNK_SIZE]
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
        print(f"청크 {chunk_idx+1}/{n_chunks} 완료 ({len(chunk_df)}행, {elapsed:.1f}초, 로딩실패 {n_fail}건)")

    if len(done_chunks) == n_chunks:
        new_parts = [pd.read_parquet(PARTIAL_DIR / f"chunk_{i}.parquet") for i in range(n_chunks)]
        new_df = pd.concat(new_parts, ignore_index=True) if new_parts else pd.DataFrame()
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged.to_parquet(FINAL_PATH, index=False)
        n_fail_total = (~merged["load_ok"]).sum()
        print(f"\n저장: {FINAL_PATH} ({len(merged)}행, 기존{len(existing)}+신규{len(new_df)})")
        print(f"전체 로딩 실패: {n_fail_total}건")
        print("PROBING_FULL_ACOUSTIC_COMPLETE")


if __name__ == "__main__":
    main()
