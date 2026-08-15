"""상담사 대응 품질 GPT-as-judge 실행 (full-call, 상담사 응대만 평가).

텍스트 전용 게이트 — acoustic 미사용.
모델: gpt-5.4 / reasoning.effort=low (라벨링 확정 세팅과 동일), Responses API.
gold_actual / mismatch 는 프롬프트에 일절 들어가지 않는다(누수 차단 자동 검증 포함).

사용:
  python3 scripts/agent_judge_run.py --verify        # 프롬프트/누수 점검만
  python3 scripts/agent_judge_run.py --calls 250     # 게이트용 250건
  python3 scripts/agent_judge_run.py --calls-file <csv>
"""

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_judge_prompt import (JUDGE_INSTRUCTION, TRANSCRIPT_MARK, build_prompt,
                                parse_judge)

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_PATH = BASE_DIR / "outputs" / "model_pilot_sample.csv"
D04_PATH = BASE_DIR / "outputs" / "d04_dialog_index.parquet"
OUT_PATH = BASE_DIR / "outputs" / "agent_judge_scores.parquet"
HUMAN_PATH = BASE_DIR / "outputs" / "agent_judge_human_eval_slots.csv"

MODEL = "gpt-5.4"
EFFORT = "low"
CONCURRENCY = 6
MAX_RETRIES = 4
N_HUMAN_EVAL = 40
SEED = 42
AGENT_PREFIX = "상담사"

PRICE_IN_PER_M, PRICE_OUT_PER_M = 1.25, 10.0  # 추정 단가(참고용)


def build_inputs(call_ids) -> pd.DataFrame:
    d04 = pd.read_parquet(D04_PATH, columns=[
        "call_id", "dialog_idx", "speaker_type", "text_clean", "r_gold_valid_flag"])
    d04 = d04[d04["call_id"].isin(call_ids)].sort_values(["call_id", "dialog_idx"])
    n_before = len(d04)
    d04 = d04[d04["r_gold_valid_flag"] == True]  # noqa: E712
    print(f"전사 발화 {n_before:,} -> r_gold_valid 필터 후 {len(d04):,}건")

    d04["speaker"] = d04["speaker_type"].astype(str).str.replace(r"\d+$", "", regex=True)
    d04["is_agent"] = d04["speaker_type"].astype(str).str.startswith(AGENT_PREFIX)
    print(f"  상담사 발화 {int(d04['is_agent'].sum()):,} / 고객 발화 "
          f"{int((~d04['is_agent']).sum()):,}")

    transcripts = (d04.assign(line=lambda x: "(" + x["speaker"] + ") " + x["text_clean"].fillna(""))
                   .groupby("call_id")["line"].apply(lambda s: "\n".join(s)))

    rows, n_empty = [], 0
    for cid in call_ids:
        tr = transcripts.get(cid)
        if tr is None or not tr.strip():
            n_empty += 1
            continue
        rows.append({"call_id": cid, "n_utt": tr.count("\n") + 1,
                     "prompt": build_prompt(tr)})
    if n_empty:
        print(f"  전사 비어 있는 콜(제외): {n_empty}건")
    return pd.DataFrame(rows)


def verify_no_leak(df: pd.DataFrame) -> dict:
    """누수 차단 검증.

    프롬프트는 전사(d04)와 acoustic level만으로 조립되며 gold_actual 컬럼은 어떤
    경로로도 읽지 않는다(구조적 보장). 검증은 두 가지로 나눈다.

      (a) 스캐폴딩 누수 - 지시문/평가항목/음성블록 등 전사 밖 영역에 카테고리명이
          들어갔는가. 하나라도 있으면 실제 누수이므로 중단.
      (b) 전사 내 우발적 등장 - 고객·상담사가 통화 중 "환불요청" 같은 말을 실제로
          한 경우. 이건 원본 입력이지 정답 주입이 아니므로 카운트만 남긴다.
          gold와 일치하는 경우도 마찬가지(고객이 자기 용건을 말한 것).
    """
    from stage2d_prompt import CATEGORIES
    gold = pd.read_parquet(BASE_DIR / "outputs" / "gold_actual_batch1_final.parquet")[
        ["call_id", "label"]].set_index("call_id")["label"]

    marker = TRANSCRIPT_MARK
    scaffold_hits, incidental, incidental_eq_gold = {}, {}, 0
    for r in df.itertuples():
        head, _, body = r.prompt.partition(marker)
        found_head = [c for c in CATEGORIES if c in head]
        if found_head:
            scaffold_hits[r.call_id] = found_head
        found_body = [c for c in CATEGORIES if c in body]
        if found_body:
            incidental[r.call_id] = found_body
            if gold.get(r.call_id) in found_body:
                incidental_eq_gold += 1

    print("\n[누수 차단 검증]")
    print(f"  (a) 전사 밖(지시문·평가항목·음성블록)에 카테고리명 등장: "
          f"{len(scaffold_hits)}건")
    if scaffold_hits:
        for k, v in list(scaffold_hits.items())[:5]:
            print(f"      {k}: {v}")
        raise SystemExit("스캐폴딩 누수 감지 - 중단")
    print("  (b) gold_actual / mismatch 는 프롬프트 조립 경로에서 전혀 읽지 않음"
          "(전사 텍스트만으로 조립, 구조적 보장)")
    print(f"  (c) 전사 본문에 카테고리명이 우발적으로 등장한 콜: {len(incidental)}건 "
          f"(그중 gold와 같은 단어 {incidental_eq_gold}건)")
    print("      - 화자가 통화 중 실제로 발화한 원본 텍스트이며 정답 주입이 아님. 유지.")
    return {"scaffold": 0, "incidental": len(incidental),
            "incidental_eq_gold": incidental_eq_gold}


def call_one(client, call_id, prompt):
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.responses.create(
                model=MODEL, instructions=JUDGE_INSTRUCTION, input=prompt,
                reasoning={"effort": EFFORT})
            k, a, err = parse_judge(resp.output_text)
            return {"call_id": call_id, "친절도": k, "적절대응": a,
                    "judge_error": err, "n_attempts": attempt + 1}
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 16))
    return {"call_id": call_id, "친절도": None, "적절대응": None,
            "judge_error": f"API실패({type(last_err).__name__}: {last_err})",
            "n_attempts": MAX_RETRIES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=250)
    ap.add_argument("--calls-file")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if args.calls_file:
        call_ids = list(pd.read_csv(args.calls_file)["call_id"])
    else:
        call_ids = list(pd.read_csv(SAMPLE_PATH)["call_id"])[:args.calls]
    print(f"대상 콜: {len(call_ids)}건")

    df = build_inputs(call_ids)
    print(f"프롬프트 생성: {len(df)}건")

    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    tok = df["prompt"].map(lambda p: len(enc.encode(p)))
    est = (tok.sum() / 1e6 * PRICE_IN_PER_M) + (len(df) * 400 / 1e6 * PRICE_OUT_PER_M)
    print(f"input 토큰 합계 {tok.sum():,} (평균 {tok.mean():.0f}, 최대 {tok.max()}) "
          f"-> 추정 비용 약 ${est:.2f}")

    verify_no_leak(df)
    if args.verify:
        print("\n=== 프롬프트 예시 (앞 1건, 앞부분 1800자) ===")
        print(df["prompt"].iat[0][:1800])
        print("\nAGENT_JUDGE_VERIFY_OK")
        return

    load_dotenv(BASE_DIR / ".env")
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key or api_key == "REPLACE_ME":
        print("ERROR: OPENAI_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    print(f"\n=== judge {MODEL} (effort={EFFORT}) {len(df)}건, concurrency={CONCURRENCY} ===")
    done, lock, t0 = [0], threading.Lock(), time.time()

    def work(row):
        r = call_one(client, row.call_id, row.prompt)
        with lock:
            done[0] += 1
            if done[0] % 50 == 0 or done[0] == len(df):
                print(f"  {done[0]}/{len(df)} ({time.time()-t0:.0f}s)", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        results = list(ex.map(work, df.itertuples()))

    res = pd.DataFrame(results).merge(df[["call_id", "n_utt"]], on="call_id", how="left")
    n_fail = res["친절도"].isna().sum()
    print(f"\n완료: {len(res)}건 / 채점 실패 {n_fail}건 / "
          f"재시도 발생 {(res['n_attempts'] > 1).sum()}건 / 소요 {time.time()-t0:.0f}s")
    if n_fail:
        for r in res[res["친절도"].isna()].itertuples():
            print(f"  {r.call_id}: {r.judge_error}")

    res.to_parquet(OUT_PATH, index=False)
    print(f"저장: {OUT_PATH}")

    ok = res.dropna(subset=["친절도"])
    human = ok.sample(n=min(N_HUMAN_EVAL, len(ok)), random_state=SEED)[
        ["call_id", "친절도", "적절대응"]].rename(
        columns={"친절도": "gpt_친절도", "적절대응": "gpt_적절대응"})
    human["human_친절도"] = ""
    human["human_적절대응"] = ""
    human.to_csv(HUMAN_PATH, index=False, encoding="utf-8-sig")
    print(f"저장: {HUMAN_PATH} ({len(human)}건, 수동 채점 슬롯 비어 있음)")
    print("AGENT_JUDGE_RUN_COMPLETE")


if __name__ == "__main__":
    main()
