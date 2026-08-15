"""human_eval_sample.csv의 80콜에 대해 GPT-4.1 mini로 인바운드/아웃바운드 판별을 실행한다.

200콜 전체가 아니라 사람이 대조할 80콜에 대해서만 실행해서, 사람 라벨링 대상과 정확히 일치시킨다.
결과는 outputs/inbound_outbound_gpt.csv (call_id, call_type_gpt, reason_gpt)에 저장한다.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
HUMAN_EVAL_PATH = BASE_DIR / "outputs" / "human_eval_sample.csv"
RESULT_PATH = BASE_DIR / "outputs" / "inbound_outbound_gpt.csv"

MODEL = "gpt-4.1-mini"
PRICING = {"input": 0.40, "output": 1.60}

PROMPT_TEMPLATE = """[지시문]
당신은 전자상거래 고객센터 통화 로그를 분석하는 전문가입니다.
아래 통화 전체 내용을 읽고, 이 통화가 고객이 먼저 전화를 건 것(인바운드)인지,
상담사나 시스템이 먼저 연락한 것(아웃바운드)인지 판단하세요.

판단 근거:
- 인바운드: 고객이 자신의 문의/요청을 먼저 꺼냄, 상담사는 응대 자세로 시작
- 아웃바운드: 상담사가 먼저 연락 사유를 밝힘, 또는 고객이 선행 연락을 언급

[통화 전체 내용]
{call_text}

[질문]
아래 JSON 형식으로만 답하세요:
{{"call_type": "인바운드" 또는 "아웃바운드", "reason": "판단 근거 한 문장"}}
"""


def make_prompt(call_text: str) -> str:
    return PROMPT_TEMPLATE.format(call_text=call_text)


def call_one(client, call_id: str, call_text: str, max_retries: int = 3):
    prompt = make_prompt(call_text)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            parsed = json.loads(content)
            return {
                "call_id": call_id,
                "call_type_gpt": parsed.get("call_type", ""),
                "reason_gpt": parsed.get("reason", ""),
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            }
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    return {
        "call_id": call_id,
        "call_type_gpt": f"ERROR: {last_err}",
        "reason_gpt": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }


def main():
    load_dotenv(BASE_DIR / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key == "REPLACE_ME":
        print("ERROR: .env 파일의 OPENAI_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    df = pd.read_csv(HUMAN_EVAL_PATH)
    results = [None] * len(df)
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            pool.submit(call_one, client, row.call_id, row.call_text): i
            for i, row in enumerate(df.itertuples())
        }
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            if done % 20 == 0 or done == len(df):
                print(f"  {done}/{len(df)} 완료")

    res_df = pd.DataFrame(results)
    res_df[["call_id", "call_type_gpt", "reason_gpt"]].to_csv(
        RESULT_PATH, index=False, encoding="utf-8-sig"
    )
    print(f"저장: {RESULT_PATH}")

    total_in = res_df["prompt_tokens"].sum()
    total_out = res_df["completion_tokens"].sum()
    in_cost = total_in / 1_000_000 * PRICING["input"]
    out_cost = total_out / 1_000_000 * PRICING["output"]
    print(
        f"실제 토큰: input {total_in:,} (${in_cost:.4f}) + output {total_out:,} (${out_cost:.4f}) "
        f"= ${in_cost + out_cost:.4f}"
    )

    n_error = res_df["call_type_gpt"].str.startswith("ERROR").sum()
    if n_error:
        print(f"경고: {n_error}건 실패 (ERROR로 표시됨). 재실행 필요.")
    print(res_df["call_type_gpt"].value_counts().to_string())


if __name__ == "__main__":
    main()
