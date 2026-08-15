"""prompt_explore_v3(v3a/v3b/v3c, 600건) Batch 제출/확인.
<카테고리> 태그 파싱, 번호prefix는 normalize_gpt_label로 정규화 후 CATEGORIES 비교
(최근접 매칭 시도하지 않음). v3c는 <불만> 태그도 별도 저장.
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
from relabel_v2_and_kappa import normalize_gpt_label
from stage2d_prompt import CATEGORIES

BASE_DIR = Path(__file__).resolve().parent.parent
PLAN_PATH = BASE_DIR / "outputs" / "prompt_explore_v3_requests_plan.json"
REQUEST_INDEX_PATH = BASE_DIR / "outputs" / "prompt_explore_v3_request_index.parquet"
STATE_PATH = BASE_DIR / "outputs" / "prompt_explore_v3_chunks_state.json"
PARTIAL_DIR = BASE_DIR / "outputs" / "prompt_explore_v3_partial"
FINAL_PATH = BASE_DIR / "outputs" / "prompt_explore_v3_gpt_predictions.parquet"

MODEL = "gpt-4.1-mini"
PROJECT_TAG = "prompt_explore_v3"
MAX_RETRIES = 2
TOKEN_CEILING = 1_600_000

CATEGORY_TAG_RE = re.compile(r"<카테고리>(.*?)</카테고리>", re.DOTALL)
COMPLAINT_TAG_RE = re.compile(r"<불만>(.*?)</불만>", re.DOTALL)
COMPLAINT_TAG_VERSIONS = {"v3c"}


def parse_response(raw: str, version: str):
    """반환: (label_or_None, parse_error_or_None, complaint_tag_or_None)"""
    if raw is None:
        return None, "응답없음", None
    m = CATEGORY_TAG_RE.search(raw)
    if not m:
        return None, "형식오류(카테고리태그누락)", None
    cat = normalize_gpt_label(m.group(1).strip())
    if cat not in CATEGORIES:
        return None, f"형식오류(카테고리불일치:{cat!r})", None
    complaint_tag = None
    if version in COMPLAINT_TAG_VERSIONS:
        m2 = COMPLAINT_TAG_RE.search(raw)
        complaint_tag = m2.group(1).strip() if m2 else None
    return cat, None, complaint_tag


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


def submit_chunk(client, chunk_index: int, plan: list, prompts: pd.Series):
    custom_ids = plan[chunk_index]
    jsonl_path = BASE_DIR / "outputs" / f"prompt_explore_v3_chunk{chunk_index}_requests.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for cid in custom_ids:
            body = {"model": MODEL, "messages": [{"role": "user", "content": prompts.loc[cid]}], "temperature": 0}
            f.write(json.dumps({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions", "body": body},
                                ensure_ascii=False) + "\n")
    with open(jsonl_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id, endpoint="/v1/chat/completions", completion_window="24h",
        metadata={"project": PROJECT_TAG, "chunk_index": str(chunk_index)},
    )
    return batch.id


def retry_failed_requests(client, chunk_index: int, failed_records: list, versions: pd.Series,
                           max_retries: int = 3) -> pd.DataFrame:
    prompts = pd.read_parquet(REQUEST_INDEX_PATH).set_index("custom_id")["prompt"]
    rows = []
    for custom_id, err_msg in failed_records:
        version = versions.loc[custom_id] if custom_id in versions.index else "v3a"
        if custom_id not in prompts.index:
            rows.append({"custom_id": custom_id, "predicted_label": None,
                         "parse_error": "프롬프트없음", "complaint_tag": None})
            continue
        prompt = prompts.loc[custom_id]
        label, perr, ctag, last_err = None, None, None, None
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0,
                )
                label, perr, ctag = parse_response(resp.choices[0].message.content, version)
                break
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)
        if label is None and perr is None and last_err is not None:
            perr = f"API재시도실패({last_err})"
        rows.append({"custom_id": custom_id, "predicted_label": label, "parse_error": perr, "complaint_tag": ctag})
    return pd.DataFrame(rows)


def download_chunk_result(client, chunk_index: int, batch_id: str, versions: pd.Series):
    batch = client.batches.retrieve(batch_id)
    content = client.files.content(batch.output_file_id).text
    rows = []
    for line in content.strip().split("\n"):
        rec = json.loads(line)
        custom_id = rec["custom_id"]
        version = versions.loc[custom_id] if custom_id in versions.index else "v3a"
        body = rec.get("response", {}).get("body", {})
        try:
            raw = body["choices"][0]["message"]["content"]
        except Exception:
            raw = None
        label, perr, ctag = parse_response(raw, version)
        rows.append({"custom_id": custom_id, "predicted_label": label, "parse_error": perr, "complaint_tag": ctag})
    df = pd.DataFrame(rows)

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
            print(f"경고: 청크 {chunk_index}에서 개별 요청 레벨 실패 {len(failed_records)}건 감지: "
                  f"{[c for c, _ in failed_records]}")
            retry_df = retry_failed_requests(client, chunk_index, failed_records, versions)
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
    versions = pd.read_parquet(REQUEST_INDEX_PATH).set_index("custom_id")["version"]

    if reconcile_with_server(client, state):
        save_state(state)

    submitted = [c for c in chunks if c["status"] == "submitted"]
    changed_any = False
    for chunk in submitted:
        batch = client.batches.retrieve(chunk["batch_id"])
        if batch.status == "completed":
            download_chunk_result(client, chunk["index"], chunk["batch_id"], versions)
            chunk["status"] = "completed"
            chunk["completed_at"] = time.time()
            print(f"청크 {chunk['index']} 완료 및 저장 ({chunk['n_requests']}건).")
            changed_any = True
        elif batch.status in ("failed", "expired", "cancelled"):
            err_msg = str(batch.errors) if batch.errors else batch.status
            chunk["last_error"] = err_msg
            if chunk["retries"] < MAX_RETRIES:
                chunk["retries"] += 1
                chunk["status"] = "pending"
                chunk["batch_id"] = None
                print(f"청크 {chunk['index']} {batch.status} (재시도 {chunk['retries']}/{MAX_RETRIES}).")
            else:
                chunk["status"] = "failed_permanent"
                print(f"청크 {chunk['index']} 재시도 소진, 건너뜁니다.")
            changed_any = True

    if changed_any:
        save_state(state)

    in_flight_tokens = sum(c["n_tokens"] for c in chunks if c["status"] == "submitted")
    pending = [c for c in chunks if c["status"] == "pending"]
    if pending:
        with open(PLAN_PATH, "r", encoding="utf-8") as f:
            plan = json.load(f)
        prompts = pd.read_parquet(REQUEST_INDEX_PATH).set_index("custom_id")["prompt"]
        n_submitted_now = 0
        for chunk in pending:
            if in_flight_tokens + chunk["n_tokens"] > TOKEN_CEILING:
                break
            try:
                batch_id = submit_chunk(client, chunk["index"], plan, prompts)
            except Exception as e:
                print(f"청크 {chunk['index']} 제출 실패: {e}. 다음 폴링에 재시도.")
                break
            chunk["batch_id"] = batch_id
            chunk["status"] = "submitted"
            chunk["submitted_at"] = time.time()
            in_flight_tokens += chunk["n_tokens"]
            n_submitted_now += 1
        if n_submitted_now:
            save_state(state)
            print(f"이번 폴링에서 {n_submitted_now}개 청크 신규 제출 (in-flight {in_flight_tokens:,}/{TOKEN_CEILING:,})")

    statuses = [c["status"] for c in chunks]
    n_completed = statuses.count("completed")
    n_submitted_s = statuses.count("submitted")
    n_pending_s = statuses.count("pending")
    n_failed = statuses.count("failed_permanent")
    print(f"진행 상황: 완료 {n_completed} / 진행중 {n_submitted_s} / 대기 {n_pending_s} / "
          f"영구실패 {n_failed} (전체 {len(chunks)})")

    terminal = {"completed", "failed_permanent"}
    if all(s in terminal for s in statuses):
        completed_chunks = [c for c in chunks if c["status"] == "completed"]
        parts = [pd.read_parquet(PARTIAL_DIR / f"chunk_{c['index']}.parquet") for c in completed_chunks]
        preds = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=["custom_id", "predicted_label", "parse_error", "complaint_tag"])
        req_idx = pd.read_parquet(REQUEST_INDEX_PATH)[["custom_id", "call_id", "version"]]
        merged = req_idx.merge(preds, on="custom_id", how="left")
        merged.to_parquet(FINAL_PATH, index=False)
        n_null = merged["predicted_label"].isna().sum()
        print(f"\n저장: {FINAL_PATH} ({len(merged)}행)")
        print(f"파싱 실패/누락: {n_null}건")
        print(merged.groupby("version")["parse_error"].apply(lambda s: s.notna().sum()))
        if n_failed:
            print("ALL_CHUNKS_DONE_WITH_FAILURES")
        else:
            print("\nALL_CHUNKS_COMPLETE")


if __name__ == "__main__":
    main()
