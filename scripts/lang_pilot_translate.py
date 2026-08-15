"""EN-full 조건용 전사 텍스트 MT 캐시 생성. 한 번만 돌리고 이후 재사용.

대상: 250건 샘플의 조건 B 창(window_n=5, window_version=customer_only) 발화 1,119건.
키: (call_id, dialog_idx). 발화 단위 개별 번역이라 정렬 어긋날 위험이 없다.
번역기: gpt-5.4 (저품질 MT 사용 금지 지시에 따름). 음성 description은 번역 대상이
아니며 영어 템플릿으로 별도 생성한다(nl_en_template.py).

출력: outputs/lang_pilot_translation_cache.parquet (call_id, dialog_idx, ko_text, en_text)
"""

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_PATH = BASE_DIR / "outputs" / "model_pilot_sample.csv"
WINDOWS_PATH = BASE_DIR / "outputs" / "stage2_windows_nl_v2.parquet"
CACHE_PATH = BASE_DIR / "outputs" / "lang_pilot_translation_cache.parquet"

WINDOW_N = 5
WINDOW_VERSION = "customer_only"
MT_MODEL = "gpt-5.4"
CONCURRENCY = 8
MAX_RETRIES = 4

MT_INSTRUCTION = (
    "You are a professional Korean-to-English translator for call center transcripts. "
    "Translate the given Korean utterance into natural English. "
    "Preserve the speaker's tone, hedging, disfluency and politeness level. "
    "Do not summarize, do not add or remove information, do not add explanations. "
    "Output only the English translation."
)


def translate_one(client, ko_text: str):
    if not str(ko_text).strip():
        return "", None
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.responses.create(
                model=MT_MODEL, instructions=MT_INSTRUCTION, input=str(ko_text),
            )
            out = (resp.output_text or "").strip()
            if out:
                return out, None
            last_err = "빈 응답"
        except Exception as e:
            last_err = e
        time.sleep(min(2 ** attempt, 16))
    return None, f"번역실패({last_err})"


def main():
    if CACHE_PATH.exists():
        cached = pd.read_parquet(CACHE_PATH)
        print(f"캐시 이미 존재: {CACHE_PATH} ({len(cached)}행). 재번역하지 않습니다.")
        print(f"  번역 실패 잔여: {cached['en_text'].isna().sum()}건")
        print("LANG_PILOT_TRANSLATE_SKIPPED")
        return

    sample = pd.read_csv(SAMPLE_PATH)
    win = pd.read_parquet(WINDOWS_PATH)
    src = win[(win["window_n"] == WINDOW_N) & (win["window_version"] == WINDOW_VERSION)
              & win["call_id"].isin(sample["call_id"])][
        ["call_id", "dialog_idx", "text_clean"]].drop_duplicates(
        subset=["call_id", "dialog_idx"]).reset_index(drop=True)
    src = src.rename(columns={"text_clean": "ko_text"})
    print(f"번역 대상: {len(src)}발화 / {src['call_id'].nunique()}콜 "
          f"(고유 텍스트 {src['ko_text'].nunique()}종)")
    print(f"모델={MT_MODEL}, concurrency={CONCURRENCY}\n")

    load_dotenv(BASE_DIR / ".env")
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key or api_key == "REPLACE_ME":
        print("ERROR: OPENAI_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    done, lock, t0 = [0], threading.Lock(), time.time()

    def work(row):
        en, err = translate_one(client, row.ko_text)
        with lock:
            done[0] += 1
            if done[0] % 100 == 0 or done[0] == len(src):
                print(f"  {done[0]}/{len(src)} ({time.time()-t0:.0f}s)", flush=True)
        return {"call_id": row.call_id, "dialog_idx": row.dialog_idx,
                "ko_text": row.ko_text, "en_text": en, "mt_error": err}

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        rows = list(ex.map(work, src.itertuples()))

    out = pd.DataFrame(rows)
    n_fail = out["en_text"].isna().sum()
    print(f"\n완료: {len(out)}건 / 번역 실패 {n_fail}건 / 소요 {time.time()-t0:.0f}s")
    if n_fail:
        for r in out[out["en_text"].isna()].itertuples():
            print(f"  {r.call_id}#{r.dialog_idx}: {r.mt_error}")

    out.to_parquet(CACHE_PATH, index=False)
    print(f"저장: {CACHE_PATH}")
    print("\n=== 스팟체크 5건 ===")
    for r in out.dropna(subset=["en_text"]).head(5).itertuples():
        print(f"  KO: {r.ko_text}")
        print(f"  EN: {r.en_text}\n")
    print("LANG_PILOT_TRANSLATE_COMPLETE")


if __name__ == "__main__":
    main()
