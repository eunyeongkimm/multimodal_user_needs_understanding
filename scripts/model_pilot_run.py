"""모델 버전 파일럿: gpt-5.4 / gpt-5.6-terra 를 reasoning.effort="medium"으로 250건 동기 실행.

조건 B(customer-only + text + acoustic NL description) 고정.
프롬프트는 stage2m(4.1mini-현재)이 쓴 것과 동일하게 stage2d_prompt.make_prompt(
    rows, with_acoustic=True)  # version="base"
로 재구성하며, --verify 로 기존 배치 JSONL과 바이트 동일함을 먼저 확인한다.

출력 형식은 기존과 동일한 평문 카테고리명(JSON schema 미사용).
Batch API 대신 동기 호출 + bounded concurrency + 재시도.

사용:
  python3 scripts/model_pilot_run.py --verify        # 프롬프트 재현 검증만
  python3 scripts/model_pilot_run.py --model gpt-5.4
  python3 scripts/model_pilot_run.py --model gpt-5.6-terra
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relabel_v2_and_kappa import normalize_gpt_label
from stage2d_prompt import CATEGORIES, make_prompt

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_PATH = BASE_DIR / "outputs" / "model_pilot_sample.csv"
WINDOWS_NL_PATH = BASE_DIR / "outputs" / "stage2_windows_nl_v2.parquet"
OUT_TMPL = str(BASE_DIR / "outputs" / "model_pilot_pred_{tag}.parquet")

WINDOW_N = 5
WINDOW_VERSION = "customer_only"
# 조건 -> (acoustic NL description 포함 여부, 바이트동일 검증용 원본 배치 JSONL glob)
CONDITIONS = {
    "B": (True, "stage2m_chunk*_requests.jsonl"),   # customer-only + text + acoustic
    "A": (False, "stage2d_chunk*_requests.jsonl"),  # customer-only + text-only
}
CONCURRENCY = 6
MAX_RETRIES = 4

# (model, effort) -> 출력 태그. 기존 산출물 파일명 호환 위해 명시적으로 매핑.
RUN_TAGS = {
    ("gpt-5.4", "medium"): "54med",
    ("gpt-5.4", "low"): "54low",
    ("gpt-5.6-terra", "medium"): "56terra_med",
    ("gpt-5.6-luna", "low"): "lunalow",
    ("gpt-5.6-luna", "medium"): "lunamed",
}
MODELS = sorted({m for m, _ in RUN_TAGS})
EFFORTS = ["low", "medium"]


def build_prompts(condition: str = "B") -> pd.DataFrame:
    """조건 A(text-only) / B(text+acoustic). 차이는 with_acoustic 하나뿐이며,
    A는 발화별 '  [음성 특징: ...]' 줄만 빠지고 나머지 텍스트는 B와 동일하다."""
    with_acoustic, _ = CONDITIONS[condition]
    sample = pd.read_csv(SAMPLE_PATH)
    windows = pd.read_parquet(WINDOWS_NL_PATH)
    win = windows[(windows["window_n"] == WINDOW_N)
                  & (windows["window_version"] == WINDOW_VERSION)]
    win = win[win["call_id"].isin(sample["call_id"])]

    rows = []
    n_missing = 0
    for cid in sample["call_id"]:
        g = win[win["call_id"] == cid].sort_values("dialog_idx")
        utt = g[["speaker_group", "text_clean", "nl_description"]].to_dict("records")
        if not utt:
            n_missing += 1
            continue
        rows.append({"call_id": cid, "prompt": make_prompt(utt, with_acoustic=with_acoustic)})
    if n_missing:
        print(f"경고: 윈도우 없는 콜 {n_missing}건 (제외)")
    return sample.merge(pd.DataFrame(rows), on="call_id", how="inner")


def verify_prompts(df: pd.DataFrame, condition: str = "B") -> None:
    """해당 조건의 원본 배치 JSONL과 프롬프트가 바이트 동일한지 확인."""
    _, glob_pat = CONDITIONS[condition]
    target = {f"{cid}::{condition}::N{WINDOW_N}": p
              for cid, p in zip(df["call_id"], df["prompt"])}
    found, mismatched = {}, []
    for path in sorted((BASE_DIR / "outputs").glob(glob_pat)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                cid = rec["custom_id"]
                if cid not in target:
                    continue
                submitted = rec["body"]["messages"][0]["content"]
                found[cid] = True
                if submitted != target[cid]:
                    mismatched.append(cid)
        if len(found) == len(target):
            break
    print(f"프롬프트 재현 검증: 대조 가능 {len(found)}/{len(target)}건, 불일치 {len(mismatched)}건")
    if mismatched:
        print(f"  불일치 예시: {mismatched[:3]}")
        cid = mismatched[0]
        print("--- 재구성 ---");  print(repr(target[cid][:600]))
        sys.exit(1)
    if not found:
        print("  경고: 대조할 stage2m JSONL을 찾지 못했습니다.")
        sys.exit(1)
    print("PROMPT_VERIFY_OK")


def parse_response(raw):
    if raw is None or not str(raw).strip():
        return None, "빈응답"
    cat = normalize_gpt_label(str(raw).strip())
    if cat not in CATEGORIES:
        return None, f"형식오류({cat[:60]!r})"
    return cat, None


def call_one(client, model, effort, call_id, prompt):
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.responses.create(
                model=model, input=prompt, reasoning={"effort": effort},
            )
            label, perr = parse_response(resp.output_text)
            return {"call_id": call_id, "predicted_label": label,
                    "parse_error": perr, "n_attempts": attempt + 1}
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 16))
    return {"call_id": call_id, "predicted_label": None,
            "parse_error": f"API실패({type(last_err).__name__}: {last_err})",
            "n_attempts": MAX_RETRIES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--effort", choices=EFFORTS, default="medium")
    ap.add_argument("--condition", choices=list(CONDITIONS), default="B")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    cond = args.condition
    df = build_prompts(cond)
    print(f"프롬프트 생성: {len(df)}건 (조건 {cond}, "
          f"acoustic={'포함' if CONDITIONS[cond][0] else '제외(text-only)'})")

    if args.verify:
        verify_prompts(df, cond)
        return
    if not args.model:
        ap.error("--model 또는 --verify 필요")
    if (args.model, args.effort) not in RUN_TAGS:
        ap.error(f"미정의 조합: {args.model}/{args.effort}")

    load_dotenv(BASE_DIR / ".env")
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key or api_key == "REPLACE_ME":
        print("ERROR: OPENAI_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    model, effort = args.model, args.effort
    tag = RUN_TAGS[(model, effort)]
    if cond != "B":  # B는 기존 파일명 유지, 그 외 조건은 접두사로 구분
        tag = f"{cond}_{tag}"
    print(f"\n=== 조건 {cond} | {model} (reasoning.effort={effort}) "
          f"{len(df)}건, concurrency={CONCURRENCY} ===")

    done, lock, t0 = [0], threading.Lock(), time.time()

    def work(row):
        r = call_one(client, model, effort, row.call_id, row.prompt)
        with lock:
            done[0] += 1
            if done[0] % 25 == 0 or done[0] == len(df):
                print(f"  {done[0]}/{len(df)} ({time.time()-t0:.0f}s)", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        results = list(ex.map(work, df.itertuples()))

    res = pd.DataFrame(results)
    out = df[["call_id", "gold_actual"]].merge(res, on="call_id", how="left")

    n_fail = out["predicted_label"].isna().sum()
    n_retried = (out["n_attempts"] > 1).sum()
    print(f"\n완료: {len(out)}건 / 파싱·API 실패 {n_fail}건 / 재시도 발생 {n_retried}건 "
          f"/ 소요 {time.time()-t0:.0f}s")
    if n_fail:
        print("실패 상세:")
        for r in out[out["predicted_label"].isna()].itertuples():
            print(f"  {r.call_id}: {r.parse_error}")

    path = OUT_TMPL.format(tag=tag)
    out.to_parquet(path, index=False)
    print(f"저장: {path}")
    print(f"MODEL_PILOT_RUN_COMPLETE::{tag}")


if __name__ == "__main__":
    main()
