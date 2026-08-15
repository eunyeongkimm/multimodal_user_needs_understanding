"""v4_chunk_plan.py로 나눈 청크를 순차적으로 OpenAI Batch API에 제출/확인한다.
batch1_check.py/v3_check.py와 동일한 재시도/스킵/재시작-중복방지/일시적오류 처리
로직을, v4(JSON 출력: has_complaint + label) 파이프라인에 맞게 적용한 버전.

모든 청크가 터미널 상태에 도달하면 완료를 알리고, 이어서 자동으로
merge -> self-consistency -> suspect scan을 실행한다 (v4_autorun.sh가 담당).
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v4_prompt import make_prompt, parse_v4_response

BASE_DIR = Path(__file__).resolve().parent.parent
PLAN_PATH = BASE_DIR / "outputs" / "v4_chunks_plan.json"
STATE_PATH = BASE_DIR / "outputs" / "v4_chunks_state.json"
CALL_TEXTS_PATH = BASE_DIR / "outputs" / "batch1_call_texts.parquet"
PARTIAL_DIR = BASE_DIR / "outputs" / "v4_partial"
FINAL_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_v4.parquet"

MODEL = "gpt-4.1-mini"
PROMPT_VERSION = "v4"
PROJECT_TAG = "gold_actual_batch1_v4"
MAX_RETRIES = 2


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

    jsonl_path = BASE_DIR / "outputs" / f"v4_chunk{chunk_index}_requests.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in subset.itertuples():
            body = {
                "model": MODEL,
                "messages": [{"role": "user", "content": make_prompt(row.call_text)}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
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


def retry_failed_requests(client, chunk_index: int, failed_records: list, call_texts: pd.DataFrame,
                           max_retries: int = 3) -> pd.DataFrame:
    """error_file_id에 기록된 개별 요청 레벨 실패를 동기 API로 즉시 재시도한다."""
    rows = []
    for custom_id, err_msg in failed_records:
        text_row = call_texts[call_texts["call_id"] == custom_id]
        if text_row.empty:
            print(f"  경고: {custom_id}의 call_text를 찾을 수 없어 재시도 불가.")
            rows.append({"call_id": custom_id, "has_complaint": None, "label_v4": None})
            continue

        prompt = make_prompt(text_row.iloc[0]["call_text"])
        has_complaint, label, last_err = None, None, None
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0,
                    response_format={"type": "json_object"},
                )
                has_complaint, label = parse_v4_response(resp.choices[0].message.content)
                break
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)
        if label is None and last_err is not None:
            print(f"  경고: {custom_id} 동기 재시도도 실패 (원래 에러: {err_msg}, 재시도 에러: {last_err})")
        rows.append({"call_id": custom_id, "has_complaint": has_complaint, "label_v4": label})
    return pd.DataFrame(rows)


def download_chunk_result(client, chunk_index: int, batch_id: str, call_texts: pd.DataFrame):
    batch = client.batches.retrieve(batch_id)
    content = client.files.content(batch.output_file_id).text
    rows = []
    for line in content.strip().split("\n"):
        rec = json.loads(line)
        call_id = rec["custom_id"]
        body = rec.get("response", {}).get("body", {})
        try:
            raw = body["choices"][0]["message"]["content"]
            has_complaint, label = parse_v4_response(raw)
        except Exception:
            has_complaint, label = None, None
        rows.append({"call_id": call_id, "has_complaint": has_complaint, "label_v4": label})
    df = pd.DataFrame(rows)

    # batch 자체는 completed로 보고돼도, 개별 요청이 API 레벨에서 실패(예: 500)하면
    # output_file_id가 아니라 error_file_id에 기록되고 request_counts.failed에도
    # 반영 안 될 수 있다. error_file_id를 확인해 그런 건을 잡아내고 즉시 동기 재시도한다.
    if batch.error_file_id:
        error_content = client.files.content(batch.error_file_id).text
        failed_records = []
        for line in error_content.strip().split("\n"):
            if not line.strip():
                continue
            err_rec = json.loads(line)
            err_body = err_rec.get("response", {}).get("body", {})
            err_msg = err_body.get("error", {}).get("message", str(err_rec.get("error")))
            failed_records.append((err_rec["custom_id"], err_msg))

        if failed_records:
            print(f"경고: 청크 {chunk_index}에서 개별 요청 레벨 실패 {len(failed_records)}건 감지"
                  f"(batch 자체는 completed로 보고됐지만 error_file_id에 기록된 건): "
                  f"{[c for c, _ in failed_records]}")
            retry_df = retry_failed_requests(client, chunk_index, failed_records, call_texts)
            n_recovered = retry_df["label_v4"].notna().sum()
            print(f"  -> 동기 API 즉시 재시도로 {n_recovered}/{len(failed_records)}건 복구")
            df = pd.concat([df, retry_df], ignore_index=True)

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
            download_chunk_result(client, in_flight["index"], in_flight["batch_id"], call_texts)
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
                    else pd.DataFrame(columns=["call_id", "has_complaint", "label_v4", "prompt_version", "model"]))
        final_df.to_parquet(FINAL_PATH, index=False)

        submitted_times = [c["submitted_at"] for c in chunks if c["submitted_at"]]
        completed_times = [c["completed_at"] for c in chunks if c["completed_at"]]
        n_null = final_df["label_v4"].isna().sum() if len(final_df) else 0
        print(f"\n저장: {FINAL_PATH} ({len(final_df)}행)")
        if submitted_times and completed_times:
            elapsed_hours = (max(completed_times) - min(submitted_times)) / 3600
            print(f"전체 소요 시간: {elapsed_hours:.2f}시간")
        print(f"라벨 파싱 실패: {n_null}건")
        if len(final_df):
            print("\n라벨 분포:")
            print(final_df["label_v4"].value_counts(dropna=False).to_string())
            print("\nhas_complaint 분포:")
            print(final_df["has_complaint"].value_counts(dropna=False).to_string())

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
