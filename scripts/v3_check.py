"""v3_chunk_plan.py로 나눈 청크를 순차적으로 OpenAI Batch API에 제출/확인한다.
batch1_check.py와 동일한 재시도/스킵/재시작-중복방지/일시적오류 처리 로직을
v3(환불요청/배송확인 11,579건 재라벨링) 파이프라인에 맞게 적용한 버전.

동작 방식 (한 번 실행할 때마다):
0. 로컬 state가 유실/오래됐을 가능성에 대비해, pending 청크는 서버 배치 목록과
   대조해 이미 제출된 것이 있으면 그걸 재사용한다 (재시작 시 중복 제출 방지).
1. 상태 파일(outputs/v3_chunks_state.json)이 없으면 초기화하고 청크 0을 제출.
2. "submitted" 상태인 청크가 있으면 상태를 조회:
   - completed: 결과를 outputs/v3_partial/chunk_{i}.parquet로 저장하고,
     다음 pending 청크가 있으면 바로 제출.
   - failed/expired/cancelled: MAX_RETRIES 안에서는 같은 청크를 재제출한다.
     재시도를 모두 소진하면 그 청크만 failed_permanent로 표시하고 건너뛴 뒤
     다음 pending 청크를 제출한다 (전체 파이프라인은 멈추지 않음).
   - 그 외(validating/in_progress/finalizing): 현재 상태만 보고하고 종료.
3. 모든 청크가 completed/failed_permanent(터미널 상태)면 completed된 것만 병합해
   outputs/gold_actual_batch1_v3.parquet로 저장한다. 실패한 청크가 하나도 없으면
   "ALL_CHUNKS_COMPLETE", 일부라도 영구 실패했으면 "ALL_CHUNKS_DONE_WITH_FAILURES"를
   출력한다.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_prompt import make_prompt

BASE_DIR = Path(__file__).resolve().parent.parent
PLAN_PATH = BASE_DIR / "outputs" / "v3_chunks_plan.json"
STATE_PATH = BASE_DIR / "outputs" / "v3_chunks_state.json"
CALL_TEXTS_PATH = BASE_DIR / "outputs" / "batch1_call_texts.parquet"
PARTIAL_DIR = BASE_DIR / "outputs" / "v3_partial"
FINAL_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_v3.parquet"

MODEL = "gpt-4.1-mini"
PROMPT_VERSION = "v3"
PROJECT_TAG = "gold_actual_batch1_v3"  # v2 배치와 구분하기 위한 별도 태그 (reconcile 충돌 방지)
MAX_RETRIES = 2


def normalize_gpt_label(raw: str) -> str:
    return re.sub(r"^\s*\d+[.)]\s*", "", str(raw)).strip()


def load_state():
    if STATE_PATH.exists():
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
        for c in state["chunks"]:
            c.setdefault("retries", 0)
            c.setdefault("last_error", None)
        return state
    with open(PLAN_PATH, "r", encoding="utf-8") as f:
        plan = json.load(f)
    return {
        "chunks": [
            {"index": i, "n_requests": len(ids), "status": "pending", "batch_id": None,
             "submitted_at": None, "completed_at": None, "retries": 0, "last_error": None}
            for i, ids in enumerate(plan)
        ]
    }


def reconcile_with_server(client, state):
    if not any(c["status"] == "pending" for c in state["chunks"]):
        return False

    chunks_by_index = {c["index"]: c for c in state["chunks"]}
    latest_by_index = {}
    for b in client.batches.list(limit=100):
        meta = b.metadata or {}
        if meta.get("project") != PROJECT_TAG or "chunk_index" not in meta:
            continue
        idx = int(meta["chunk_index"])
        prev = latest_by_index.get(idx)
        if prev is None or b.created_at > prev.created_at:
            latest_by_index[idx] = b

    changed = False
    for idx, batch in latest_by_index.items():
        chunk = chunks_by_index.get(idx)
        if chunk is None or chunk["status"] != "pending":
            continue
        if batch.status in ("failed", "expired", "cancelled"):
            continue
        print(f"청크 {idx}: 로컬은 pending이지만 서버에 배치 {batch.id}(status={batch.status})가 "
              f"이미 있어 재사용합니다 (중복 제출 방지).")
        chunk["batch_id"] = batch.id
        chunk["status"] = "submitted"
        chunk["submitted_at"] = batch.created_at
        changed = True
    return changed


def save_state(state):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def submit_chunk(client, chunk_index: int, call_texts: pd.DataFrame):
    with open(PLAN_PATH, "r", encoding="utf-8") as f:
        plan = json.load(f)
    ids = plan[chunk_index]
    subset = call_texts[call_texts["call_id"].isin(set(ids))]

    jsonl_path = BASE_DIR / "outputs" / f"v3_chunk{chunk_index}_requests.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in subset.itertuples():
            body = {
                "model": MODEL,
                "messages": [{"role": "user", "content": make_prompt(row.call_text)}],
                "temperature": 0,
            }
            line = {"custom_id": row.call_id, "method": "POST", "url": "/v1/chat/completions", "body": body}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    with open(jsonl_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"project": PROJECT_TAG, "chunk_index": str(chunk_index), "prompt_version": PROMPT_VERSION},
    )
    print(f"청크 {chunk_index} 제출: batch_id={batch.id}, n={len(subset)}")
    return batch.id


def download_chunk_result(client, chunk_index: int, batch_id: str):
    batch = client.batches.retrieve(batch_id)
    content = client.files.content(batch.output_file_id).text
    rows = []
    for line in content.strip().split("\n"):
        rec = json.loads(line)
        call_id = rec["custom_id"]
        body = rec.get("response", {}).get("body", {})
        try:
            label = normalize_gpt_label(body["choices"][0]["message"]["content"])
        except Exception:
            label = None
        rows.append({"call_id": call_id, "label": label})
    df = pd.DataFrame(rows)
    df["prompt_version"] = PROMPT_VERSION
    df["model"] = MODEL
    PARTIAL_DIR.mkdir(exist_ok=True)
    df.to_parquet(PARTIAL_DIR / f"chunk_{chunk_index}.parquet", index=False)
    return df


def main():
    load_dotenv(BASE_DIR / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key == "REPLACE_ME":
        print("ERROR: OPENAI_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    import openai
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    try:
        _run(client)
    except (openai.APIConnectionError, openai.APITimeoutError,
            openai.RateLimitError, openai.InternalServerError) as e:
        print(f"일시적 API 오류, 다음 폴링에 재시도합니다: {e}")


def _run(client):
    state = load_state()
    chunks = state["chunks"]
    call_texts = pd.read_parquet(CALL_TEXTS_PATH)

    if reconcile_with_server(client, state):
        save_state(state)

    in_flight = next((c for c in chunks if c["status"] == "submitted"), None)

    if in_flight is None:
        pending = next((c for c in chunks if c["status"] == "pending"), None)
        if pending is not None:
            batch_id = submit_chunk(client, pending["index"], call_texts)
            pending["batch_id"] = batch_id
            pending["status"] = "submitted"
            pending["submitted_at"] = time.time()
            save_state(state)
            print(f"청크 {pending['index']}/{len(chunks)-1} 제출 완료, 상태: submitted")
            return
    else:
        batch = client.batches.retrieve(in_flight["batch_id"])
        print(f"청크 {in_flight['index']}: status={batch.status}, "
              f"completed={batch.request_counts.completed}/{batch.request_counts.total}")

        if batch.status == "completed":
            download_chunk_result(client, in_flight["index"], in_flight["batch_id"])
            in_flight["status"] = "completed"
            in_flight["completed_at"] = time.time()
            save_state(state)
            print(f"청크 {in_flight['index']} 완료 및 저장.")

            next_pending = next((c for c in chunks if c["status"] == "pending"), None)
            if next_pending is not None:
                batch_id = submit_chunk(client, next_pending["index"], call_texts)
                next_pending["batch_id"] = batch_id
                next_pending["status"] = "submitted"
                next_pending["submitted_at"] = time.time()
                save_state(state)
                print(f"다음 청크 {next_pending['index']}/{len(chunks)-1} 제출 완료.")
                return
        elif batch.status in ("failed", "expired", "cancelled"):
            err_msg = str(batch.errors) if batch.errors else batch.status
            in_flight["last_error"] = err_msg

            if in_flight["retries"] < MAX_RETRIES:
                in_flight["retries"] += 1
                print(f"청크 {in_flight['index']} {batch.status} "
                      f"(재시도 {in_flight['retries']}/{MAX_RETRIES}). 에러: {err_msg}")
                new_batch_id = submit_chunk(client, in_flight["index"], call_texts)
                in_flight["batch_id"] = new_batch_id
                in_flight["status"] = "submitted"
                in_flight["submitted_at"] = time.time()
                save_state(state)
                return

            in_flight["status"] = "failed_permanent"
            save_state(state)
            print(f"청크 {in_flight['index']} 재시도 {MAX_RETRIES}회 모두 실패, 건너뜁니다. 에러: {err_msg}")

            next_pending = next((c for c in chunks if c["status"] == "pending"), None)
            if next_pending is not None:
                batch_id = submit_chunk(client, next_pending["index"], call_texts)
                next_pending["batch_id"] = batch_id
                next_pending["status"] = "submitted"
                next_pending["submitted_at"] = time.time()
                save_state(state)
                print(f"다음 청크 {next_pending['index']}/{len(chunks)-1} 제출 완료.")
                return
        else:
            return

    terminal = {"completed", "failed_permanent"}
    statuses = [c["status"] for c in chunks]
    if all(s in terminal for s in statuses):
        completed_chunks = [c for c in chunks if c["status"] == "completed"]
        failed_chunks = [c for c in chunks if c["status"] == "failed_permanent"]

        parts = [pd.read_parquet(PARTIAL_DIR / f"chunk_{c['index']}.parquet") for c in completed_chunks]
        final_df = (pd.concat(parts, ignore_index=True) if parts
                    else pd.DataFrame(columns=["call_id", "label", "prompt_version", "model"]))
        final_df.to_parquet(FINAL_PATH, index=False)

        submitted_times = [c["submitted_at"] for c in chunks if c["submitted_at"]]
        completed_times = [c["completed_at"] for c in chunks if c["completed_at"]]
        n_null = final_df["label"].isna().sum() if len(final_df) else 0
        print(f"\n저장: {FINAL_PATH} ({len(final_df)}행)")
        if submitted_times and completed_times:
            elapsed_hours = (max(completed_times) - min(submitted_times)) / 3600
            print(f"전체 소요 시간: {elapsed_hours:.2f}시간")
        print(f"라벨 파싱 실패: {n_null}건")
        if len(final_df):
            print("\n라벨 분포:")
            print(final_df["label"].value_counts(dropna=False).to_string())

        if failed_chunks:
            idxs = [c["index"] for c in failed_chunks]
            print(f"\n영구 실패 청크: {idxs} (재시도 {MAX_RETRIES}회 소진, 이 청크의 콜은 결과에서 빠짐). "
                  f"사람 확인 필요.")
            print("ALL_CHUNKS_DONE_WITH_FAILURES")
        else:
            print("\nALL_CHUNKS_COMPLETE")
    else:
        n_done = sum(1 for s in statuses if s in terminal)
        print(f"\n진행 상황: {n_done}/{len(statuses)} 청크 처리 완료")


if __name__ == "__main__":
    main()
