"""GPT-5.5 vs GPT-4.1 mini gold_actual 라벨링 파일럿.

기본 실행(인자 없음): 200콜 샘플을 만들고 예상 비용만 계산해서 출력한다 (API 호출 없음).
--run 플래그를 주면 실제로 두 모델을 호출해서 라벨링하고 결과/비용/분포를 보고한다.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import tiktoken
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "outputs" / "d04_dialog_index.parquet"
SAMPLE_CACHE_PATH = BASE_DIR / "outputs" / "pilot200_calls.parquet"
RESULT_PATH = BASE_DIR / "outputs" / "pilot200_gpt_labels.csv"

N_CALLS = 200
RANDOM_STATE = 42

CATEGORIES = ["환불요청", "주문취소", "불만제기", "배송확인", "교환반품", "구매진행", "서비스이용"]


def normalize_label(raw: str) -> str:
    """모델이 "7. 서비스이용"처럼 번호를 붙여 답한 경우 카테고리명만 남긴다."""
    import re

    return re.sub(r"^\s*\d+[.)]\s*", "", str(raw)).strip()

# 모델별 단가 ($ / 1M tokens). 실행 시점 공개 가격 기준.
PRICING = {
    "gpt-5.5": {"input": 5.00, "output": 30.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
}

PROMPT_TEMPLATE = """[지시문]
당신은 전자상거래(온라인 교육 플랫폼) 고객센터 상담 통화를 분석하는 전문가입니다.
아래 통화 전체 내용을 읽고, 고객의 실제 니즈(콜 전체를 통해 드러나는 핵심 용건)를
다음 7개 카테고리 중 하나로 분류하세요.

[카테고리 정의 및 예시]
1. 환불요청: 수령/수강 후 금전 반환 요청
   예: "이거 환불 가능할까요?", "그 교재 환불하려고요"
2. 주문취소: 결제 후 배송/수강 전 취소
   예: "교재 구매한 거를 취소하려고 하는데요"
3. 불만제기: 문의처럼 들리나 본질은 항의(약속 불이행, 응대 불만 등)
   예: "그쪽 전화를 안 받으셔서요", "몇 번을 전화했는지 아세요?"
4. 배송확인: 배송 상태·도착 문의
   예: "탭이 아직 안 오고 있어서 언제쯤 오나 싶어서요"
5. 교환반품: 불량·오배송으로 인한 현물 교체
   예: "파본이 돼서 왔어요", "색이 잘못돼서 교환하려고요"
6. 구매진행: 결제 완료를 위한 도움 요청
   예: "결제가 안 돼서요", "쿠폰 적용이 안 돼요"
7. 서비스이용: 로그인·기기·앱 등 이용 관련 문제
   예: "윈도우 환경에서만 재생된다고 떠서요"

[통화 전체 내용]
{call_text}

[질문]
위 통화에서 고객의 실제 니즈를 위 7개 카테고리 중 하나로만 답하세요.
카테고리명만 정확히 출력하고, 다른 설명은 추가하지 마세요.
"""


def build_sample() -> pd.DataFrame:
    """r_gold_valid_flag==True dialog만 사용해 call_id별 전체 텍스트를 만들고 200콜 샘플링한다."""
    if SAMPLE_CACHE_PATH.exists():
        return pd.read_parquet(SAMPLE_CACHE_PATH)

    df = pd.read_parquet(
        DATA_PATH,
        columns=["call_id", "dialog_idx", "speaker_type", "text_clean", "r_gold_valid_flag"],
    )
    df = df[df["r_gold_valid_flag"] == True]  # noqa: E712
    df = df.sort_values(["call_id", "dialog_idx"])
    df["line"] = df["speaker_type"].astype(str) + ": " + df["text_clean"].astype(str)

    call_texts = (
        df.groupby("call_id")["line"]
        .apply(lambda s: "\n".join(s))
        .reset_index()
        .rename(columns={"line": "call_text"})
    )

    sample = call_texts.sample(n=N_CALLS, random_state=RANDOM_STATE).reset_index(drop=True)
    sample.to_parquet(SAMPLE_CACHE_PATH, index=False)
    return sample


def make_prompt(call_text: str) -> str:
    return PROMPT_TEMPLATE.format(call_text=call_text)


def estimate_cost(sample: pd.DataFrame) -> None:
    enc = tiktoken.get_encoding("o200k_base")  # gpt-4.1/4o/5.x 계열 공통 인코딩
    prompt_token_counts = [len(enc.encode(make_prompt(t))) for t in sample["call_text"]]
    total_input_tokens = sum(prompt_token_counts)
    n = len(sample)

    # 출력은 카테고리명 하나 (예: "환불요청") 이므로 여유있게 콜당 10 토큰으로 추정
    est_output_tokens_per_call = 10
    total_output_tokens = est_output_tokens_per_call * n

    print(f"=== 비용 예측 (콜 {n}개, 모델 2개 각각 동일 프롬프트로 호출) ===")
    print(f"콜당 평균 prompt 토큰: {total_input_tokens / n:.0f}")
    print(f"모델당 총 input 토큰: {total_input_tokens:,}")
    print(f"모델당 예상 총 output 토큰 (콜당 ~{est_output_tokens_per_call} 토큰 가정): {total_output_tokens:,}")
    print()

    grand_total = 0.0
    for model, price in PRICING.items():
        in_cost = total_input_tokens / 1_000_000 * price["input"]
        out_cost = total_output_tokens / 1_000_000 * price["output"]
        total = in_cost + out_cost
        grand_total += total
        print(
            f"[{model}] input ${in_cost:.4f} (${price['input']}/1M) "
            f"+ output ${out_cost:.4f} (${price['output']}/1M) = ${total:.4f}"
        )
    print(f"\n두 모델 합계 예상 비용: ${grand_total:.4f}")
    print(
        "\n주의: gpt-5.5는 tiktoken이 아직 모델명을 인식하지 못해 o200k_base 인코딩으로 근사 계산했습니다. "
        "실제 토큰 수와 소폭 차이가 있을 수 있습니다."
    )


def call_one(client, model: str, call_id: str, call_text: str, max_retries: int = 3):
    prompt = make_prompt(call_text)
    kwargs = dict(model=model, messages=[{"role": "user", "content": prompt}], temperature=0)

    last_err = None
    for attempt in range(max_retries):
        try:
            try:
                resp = client.chat.completions.create(**kwargs)
            except Exception as e:
                # 일부 reasoning 계열 모델은 temperature 커스텀 값을 거부함 -> 기본값으로 재시도
                if "temperature" in str(e).lower() and "temperature" in kwargs:
                    kwargs.pop("temperature")
                    resp = client.chat.completions.create(**kwargs)
                else:
                    raise
            label = normalize_label(resp.choices[0].message.content)
            return {
                "call_id": call_id,
                "label": label,
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            }
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    return {
        "call_id": call_id,
        "label": f"ERROR: {last_err}",
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }


def run_model(client, model: str, sample: pd.DataFrame, max_workers: int = 5) -> pd.DataFrame:
    results = [None] * len(sample)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(call_one, client, model, row.call_id, row.call_text): i
            for i, row in enumerate(sample.itertuples())
        }
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            if done % 20 == 0 or done == len(sample):
                print(f"  [{model}] {done}/{len(sample)} 완료")
    return pd.DataFrame(results)


def run_pilot(sample: pd.DataFrame) -> None:
    load_dotenv(BASE_DIR / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key == "REPLACE_ME":
        print("ERROR: .env 파일의 OPENAI_API_KEY가 설정되지 않았습니다. 실제 키를 채운 뒤 다시 실행하세요.")
        sys.exit(1)

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    all_usage = {}
    label_cols = {}
    for model in PRICING:
        print(f"\n=== {model} 호출 시작 ({len(sample)}콜) ===")
        res_df = run_model(client, model, sample)
        all_usage[model] = {
            "prompt_tokens": res_df["prompt_tokens"].sum(),
            "completion_tokens": res_df["completion_tokens"].sum(),
        }
        label_cols[model] = res_df.set_index("call_id")["label"]

    out = sample[["call_id"]].copy()
    out["gpt55_label"] = out["call_id"].map(label_cols["gpt-5.5"])
    out["gpt41mini_label"] = out["call_id"].map(label_cols["gpt-4.1-mini"])
    out.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장: {RESULT_PATH}")

    print("\n=== 실제 토큰 사용량 및 비용 ===")
    grand_total = 0.0
    for model, usage in all_usage.items():
        price = PRICING[model]
        in_cost = usage["prompt_tokens"] / 1_000_000 * price["input"]
        out_cost = usage["completion_tokens"] / 1_000_000 * price["output"]
        total = in_cost + out_cost
        grand_total += total
        print(
            f"[{model}] input {usage['prompt_tokens']:,} tokens (${in_cost:.4f}) + "
            f"output {usage['completion_tokens']:,} tokens (${out_cost:.4f}) = ${total:.4f}"
        )
    print(f"합계: ${grand_total:.4f}")

    print("\n=== 라벨 분포 비교 (GPT-5.5 vs GPT-4.1 mini) ===")
    dist55 = out["gpt55_label"].value_counts()
    dist41 = out["gpt41mini_label"].value_counts()
    dist_table = pd.DataFrame(
        {
            "gpt-5.5": [dist55.get(c, 0) for c in CATEGORIES],
            "gpt-4.1-mini": [dist41.get(c, 0) for c in CATEGORIES],
        },
        index=CATEGORIES,
    )
    # 카테고리 외 응답(파싱 실패 등)이 있으면 별도로 표시
    other55 = out.loc[~out["gpt55_label"].isin(CATEGORIES), "gpt55_label"]
    other41 = out.loc[~out["gpt41mini_label"].isin(CATEGORIES), "gpt41mini_label"]
    if len(other55) or len(other41):
        dist_table.loc["기타/오류"] = [len(other55), len(other41)]
    print(dist_table.to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="실제로 API를 호출해서 라벨링을 실행한다")
    parser.add_argument("--max-workers", type=int, default=5)
    args = parser.parse_args()

    sample = build_sample()

    if not args.run:
        estimate_cost(sample)
        print("\n실행하려면 --run 플래그를 붙여서 다시 실행하세요 (사용자 승인 후).")
        return

    run_pilot(sample)


if __name__ == "__main__":
    main()
