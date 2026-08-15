"""prompt_explore_check.py의 parse_response 버그(v3/v4 <카테고리> 태그의 번호prefix
미정규화) 수정 후, 이미 완료된 배치 3개를 API 재호출 없이 재조회(files.content)만
다시 받아 재파싱한다. 새 인퍼런스 비용 없음(output_file_id 재조회만)."""

import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_explore_check import (
    STATE_PATH, REQUEST_INDEX_PATH, PARTIAL_DIR, FINAL_PATH,
    download_chunk_result,
)

BASE_DIR = Path(__file__).resolve().parent.parent


def main():
    load_dotenv(BASE_DIR / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    with open(STATE_PATH, "r", encoding="utf-8") as f:
        state = json.load(f)
    versions = pd.read_parquet(REQUEST_INDEX_PATH).set_index("custom_id")["version"]

    completed_chunks = [c for c in state["chunks"] if c["status"] == "completed"]
    print(f"재조회 대상 청크: {len(completed_chunks)}개")

    for c in completed_chunks:
        df = download_chunk_result(client, c["index"], c["batch_id"], versions)
        n_fail = df["parse_error"].notna().sum()
        print(f"  청크 {c['index']}: {len(df)}건 재파싱, 형식오류/응답없음 {n_fail}건")

    parts = [pd.read_parquet(PARTIAL_DIR / f"chunk_{c['index']}.parquet") for c in completed_chunks]
    preds = pd.concat(parts, ignore_index=True)
    req_idx = pd.read_parquet(REQUEST_INDEX_PATH)[["custom_id", "call_id", "version"]]
    merged = req_idx.merge(preds, on="custom_id", how="left")
    merged.to_parquet(FINAL_PATH, index=False)

    print(f"\n저장: {FINAL_PATH} ({len(merged)}행)")
    print("버전별 파싱 실패/누락 건수:")
    print(merged.groupby("version")["parse_error"].apply(lambda s: s.notna().sum()))
    print("\n버전별 parse_error 상위 값:")
    for v in ["v3", "v4"]:
        sub = merged[merged["version"] == v]
        print(f"--- {v} ---")
        print(sub["parse_error"].value_counts(dropna=False).head(10))

    print("\nREPARSE_COMPLETE")


if __name__ == "__main__":
    main()
