"""batch1(19,848콜, r_gold_valid_flag 전량 False인 152콜 제외)를
GPT-4.1 mini + v2 프롬프트로 OpenAI Batch API에 제출한다.

완료까지 최대 24시간 걸릴 수 있는 비동기 작업이므로, 제출 후 batch_id를
outputs/batch1_openai_batch_meta.json에 저장해두고 batch1_check.py로 상태를 확인한다.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relabel_v2_and_kappa import make_prompt  # v2 프롬프트 재사용

BASE_DIR = Path(__file__).resolve().parent.parent
CALL_TEXTS_PATH = BASE_DIR / "outputs" / "batch1_call_texts.parquet"
BATCH1_IDS_PATH = BASE_DIR / "outputs" / "batch1_call_ids.csv"
JSONL_PATH = BASE_DIR / "outputs" / "batch1_requests.jsonl"
META_PATH = BASE_DIR / "outputs" / "batch1_openai_batch_meta.json"
NO_TEXT_LOG_PATH = BASE_DIR / "outputs" / "batch1_no_valid_text_call_ids.csv"

MODEL = "gpt-4.1-mini"
PROMPT_VERSION = "v2"


def main():
    if META_PATH.exists():
        print(f"이미 제출된 배치 메타데이터가 있습니다: {META_PATH}")
        print("재제출하려면 이 파일을 먼저 지우세요.")
        return

    load_dotenv(BASE_DIR / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key == "REPLACE_ME":
        print("ERROR: OPENAI_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    call_texts = pd.read_parquet(CALL_TEXTS_PATH)
    batch1_ids = set(pd.read_csv(BATCH1_IDS_PATH)["call_id"])
    no_text_ids = sorted(batch1_ids - set(call_texts["call_id"]))
    pd.DataFrame({"call_id": no_text_ids}).to_csv(NO_TEXT_LOG_PATH, index=False, encoding="utf-8-sig")
    print(f"유효 발화 없어 라벨링 대상에서 제외된 콜: {len(no_text_ids)}건 -> {NO_TEXT_LOG_PATH}")

    print(f"JSONL 작성 중: {len(call_texts)}건")
    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for row in call_texts.itertuples():
            body = {
                "model": MODEL,
                "messages": [{"role": "user", "content": make_prompt(row.call_text)}],
                "temperature": 0,
            }
            line = {
                "custom_id": row.call_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"업로드 중: {JSONL_PATH}")
    with open(JSONL_PATH, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    print(f"배치 생성 중 (input_file_id={uploaded.id})")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"project": "gold_actual_batch1", "prompt_version": PROMPT_VERSION, "model": MODEL},
    )

    meta = {
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "n_requests": len(call_texts),
        "status": batch.status,
        "created_at": batch.created_at,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n배치 제출 완료: batch_id={batch.id}, status={batch.status}")
    print(f"메타데이터 저장: {META_PATH}")
    print("상태 확인은 scripts/batch1_check.py 로 하세요.")


if __name__ == "__main__":
    main()
