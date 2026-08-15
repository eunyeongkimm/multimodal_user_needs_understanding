"""Stage 2-4b: 909개 청크(20만 토큰씩)를 '동시-제출 파이프라인' 방식으로
OpenAI Batch API에 처리한다.

기존 v2~v5 파이프라인은 청크 1개만 in-flight 상태로 유지하고 완료될 때까지
기다렸다가 다음을 제출하는 완전 순차 방식이었다. 이번에는 조직 큐 한도
(~200만 토�큰) 안에서 여러 청크를 동시에 제출하고, 하나가 끝날 때마다 즉시
다음 pending 청크로 그 자리를 채우는 파이프라인 방식으로 바꿔 처리량을
동시성 배수만큼(이론상 최대 ~8-9배) 높인다 (Little's Law: steady-state
처리량 ~= 동시 슬롯 수 / 평균 완료시간).

동작 (한 번 실행할 때마다):
0. pending 청크 중 서버에 이미 제출된 게 있으면 재사용 (재시작 시 중복 제출 방지).
1. 현재 submitted 상태인 모든 청크의 상태를 조회:
   - completed: 결과 다운로드 + 저장.
   - failed/expired/cancelled: MAX_RETRIES 안에서 재제출, 소진되면 failed_permanent.
2. 남은 in-flight 토큰 여유만큼 pending 청크를 순서대로 추가 제출
   (제출 자체가 API 에러로 거부되면 - 큐 한도 초과로 추정 - 그 폴링에서는
   더 이상 제출 시도하지 않고 다음 폴링에 재시도).
3. 모든 청크가 터미널 상태면 결과를 병합해 저장하고 완료를 알린다.
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relabel_v2_and_kappa import normalize_gpt_label

BASE_DIR = Path(__file__).resolve().parent.parent
PLAN_PATH = BASE_DIR / "outputs" / "prompt_pilot_requests_plan.json"
REQUEST_INDEX_PATH = BASE_DIR / "outputs" / "prompt_pilot_request_index.parquet"
STATE_PATH = BASE_DIR / "outputs" / "prompt_pilot_chunks_state.json"
PARTIAL_DIR = BASE_DIR / "outputs" / "prompt_pilot_partial"
FINAL_PATH = BASE_DIR / "outputs" / "prompt_pilot_gpt_predictions.parquet"
PROGRESS_PATH = BASE_DIR / "outputs" / "stage2_progress.json"


def mark_progress(key, value):
    progress = {}
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            progress = json.load(f)
    progress[key] = value
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

MODEL = "gpt-4.1-mini"
PROMPT_VERSION = "prompt_pilot"
PROJECT_TAG = "prompt_pilot_v1"
MAX_RETRIES = 2
TOKEN_CEILING = 1_600_000


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
    idx = pd.read_parquet(REQUEST_INDEX_PATH).set_index("custom_id")["tokens"]
    chunks = []
    for i, custom_ids in enumerate(plan):
        n_tokens = int(idx.loc[custom_ids].sum())
        chunks.append({
            "index": i, "n_requests": len(custom_ids), "n_tokens": n_tokens,
            "status": "pending", "batch_id": None, "submitted_at": None,
            "completed_at": None, "retries": 0, "last_error": None,
        })
    return {"chunks": chunks}


def save_state(state):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


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
        print(f"청크 {idx}: 서버에 이미 배치 {batch.id}(status={batch.status})가 있어 재사용.")
        chunk["batch_id"] = batch.id
        chunk["status"] = "submitted"
        chunk["submitted_at"] = batch.created_at
        changed = True
    return changed


def build_jsonl(chunk_index: int, custom_ids: list, prompts: pd.Series) -> Path:
    jsonl_path = BASE_DIR / "outputs" / f"prompt_pilot_chunk{chunk_index}_requests.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for cid in custom_ids:
            body = {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompts.loc[cid]}],
                "temperature": 0,
            }
            line = {"custom_id": cid, "method": "POST", "url": "/v1/chat/completions", "body": body}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return jsonl_path


def submit_chunk(client, chunk_index: int, plan: list, prompts: pd.Series):
    custom_ids = plan[chunk_index]
    jsonl_path = build_jsonl(chunk_index, custom_ids, prompts)
    with open(jsonl_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"project": PROJECT_TAG, "chunk_index": str(chunk_index), "prompt_version": PROMPT_VERSION},
    )
    return batch.id


def retry_failed_requests(client, chunk_index: int, failed_records: list, max_retries: int = 3) -> pd.DataFrame:
    """error_file_id에 기록된 개별 요청 레벨 실패를 동기 API로 즉시 재시도한다.
    프롬프트는 prompt_pilot_request_index.parquet에서 custom_id로 바로 조회한다."""
    prompts = pd.read_parquet(REQUEST_INDEX_PATH).set_index("custom_id")["prompt"]
    rows = []
    for custom_id, err_msg in failed_records:
        if custom_id not in prompts.index:
            print(f"  경고: {custom_id}의 프롬프트를 찾을 수 없어 재시도 불가.")
            rows.append({"custom_id": custom_id, "predicted_label": None})
            continue

        prompt = prompts.loc[custom_id]
        label, last_err = None, None
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0,
                )
                label = normalize_gpt_label(resp.choices[0].message.content)
                break
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)
        if label is None and last_err is not None:
            print(f"  경고: {custom_id} 동기 재시도도 실패 (원래 에러: {err_msg}, 재시도 에러: {last_err})")
        rows.append({"custom_id": custom_id, "predicted_label": label})
    return pd.DataFrame(rows)


def download_chunk_result(client, chunk_index: int, batch_id: str):
    batch = client.batches.retrieve(batch_id)
    content = client.files.content(batch.output_file_id).text
    rows = []
    for line in content.strip().split("\n"):
        rec = json.loads(line)
        custom_id = rec["custom_id"]
        body = rec.get("response", {}).get("body", {})
        try:
            label = normalize_gpt_label(body["choices"][0]["message"]["content"])
        except Exception:
            label = None
        rows.append({"custom_id": custom_id, "predicted_label": label})
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
            retry_df = retry_failed_requests(client, chunk_index, failed_records)
            n_recovered = retry_df["predicted_label"].notna().sum()
            print(f"  -> 동기 API 즉시 재시도로 {n_recovered}/{len(failed_records)}건 복구")
            df = pd.concat([df, retry_df], ignore_index=True)

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

    if reconcile_with_server(client, state):
        save_state(state)

    submitted = [c for c in chunks if c["status"] == "submitted"]
    changed_any = False

    for chunk in submitted:
        batch = client.batches.retrieve(chunk["batch_id"])
        if batch.status == "completed":
            download_chunk_result(client, chunk["index"], chunk["batch_id"])
            chunk["status"] = "completed"
            chunk["completed_at"] = time.time()
            print(f"청크 {chunk['index']} 완료 및 저장 ({chunk['n_requests']}건).")
            changed_any = True
        elif batch.status in ("failed", "expired", "cancelled"):
            err_msg = str(batch.errors) if batch.errors else batch.status
            chunk["last_error"] = err_msg
            if chunk["retries"] < MAX_RETRIES:
                chunk["retries"] += 1
                chunk["status"] = "pending"  # 다음 backfill 루프에서 재제출
                chunk["batch_id"] = None
                print(f"청크 {chunk['index']} {batch.status} (재시도 {chunk['retries']}/{MAX_RETRIES}). "
                      f"에러: {err_msg}")
            else:
                chunk["status"] = "failed_permanent"
                print(f"청크 {chunk['index']} 재시도 소진, 건너뜁니다. 에러: {err_msg}")
            changed_any = True
        # else: 아직 진행 중, 그대로 둠

    if changed_any:
        save_state(state)

    # === backfill: 여유 토큰만큼 pending 청크를 순서대로 추가 제출 ===
    in_flight_tokens = sum(c["n_tokens"] for c in chunks if c["status"] == "submitted")
    pending = [c for c in chunks if c["status"] == "pending"]

    if pending:
        with open(PLAN_PATH, "r", encoding="utf-8") as f:
            plan = json.load(f)
        idx = pd.read_parquet(REQUEST_INDEX_PATH).set_index("custom_id")["prompt"]

        n_submitted_now = 0
        for chunk in pending:
            if in_flight_tokens + chunk["n_tokens"] > TOKEN_CEILING:
                break
            try:
                batch_id = submit_chunk(client, chunk["index"], plan, idx)
            except Exception as e:
                print(f"청크 {chunk['index']} 제출 실패(큐 한도 추정): {e}. 다음 폴링에 재시도.")
                break
            chunk["batch_id"] = batch_id
            chunk["status"] = "submitted"
            chunk["submitted_at"] = time.time()
            in_flight_tokens += chunk["n_tokens"]
            n_submitted_now += 1

        if n_submitted_now:
            save_state(state)
            print(f"이번 폴링에서 {n_submitted_now}개 청크 신규 제출 "
                  f"(현재 in-flight 토큰: {in_flight_tokens:,}/{TOKEN_CEILING:,})")

    statuses = [c["status"] for c in chunks]
    n_completed = statuses.count("completed")
    n_submitted_s = statuses.count("submitted")
    n_pending_s = statuses.count("pending")
    n_failed = statuses.count("failed_permanent")
    print(f"진행 상황: 완료 {n_completed} / 진행중 {n_submitted_s} / 대기 {n_pending_s} / "
          f"영구실패 {n_failed} (전체 {len(chunks)})")
    mark_progress("prompt_pilot_progress", {
        "status": "in_progress", "completed": n_completed, "submitted": n_submitted_s,
        "pending": n_pending_s, "failed_permanent": n_failed, "total": len(chunks),
    })

    terminal = {"completed", "failed_permanent"}
    if all(s in terminal for s in statuses):
        completed_chunks = [c for c in chunks if c["status"] == "completed"]
        parts = [pd.read_parquet(PARTIAL_DIR / f"chunk_{c['index']}.parquet") for c in completed_chunks]
        preds = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["custom_id", "predicted_label"])

        req_idx = pd.read_parquet(REQUEST_INDEX_PATH)[["custom_id", "call_id", "sample", "version"]]
        merged = req_idx.merge(preds, on="custom_id", how="left")
        merged.to_parquet(FINAL_PATH, index=False)

        n_null = merged["predicted_label"].isna().sum()
        print(f"\n저장: {FINAL_PATH} ({len(merged)}행)")
        print(f"라벨 파싱 실패/누락: {n_null}건")

        if n_failed:
            print(f"\n영구 실패 청크 {n_failed}개 있음 - 사람 확인 필요.")
            mark_progress("prompt_pilot_progress", {"status": "done_with_failures", "failed_permanent": n_failed,
                                              "total": len(chunks)})
            print("ALL_CHUNKS_DONE_WITH_FAILURES")
        else:
            mark_progress("prompt_pilot_progress", {"status": "done", "total": len(chunks)})
            print("\nALL_CHUNKS_COMPLETE")


if __name__ == "__main__":
    main()
