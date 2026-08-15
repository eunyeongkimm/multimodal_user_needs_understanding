"""수정된 프롬프트(v2)로 80콜 전체를 GPT-5.5 / GPT-4.1 mini로 재라벨링하고,
사람 라벨(정규화)과 비교해 v1(기존 프롬프트) vs v2(수정 프롬프트) Cohen's Kappa를 비교한다.

입력: outputs/human_eval_final_fin.csv
  - call_id, call_text, gpt55_label(v1), gpt41mini_label(v1), human_label 등 포함

출력:
  - outputs/pilot80_gpt_labels_v2.csv (call_id, gpt55_label_v2, gpt41mini_label_v2)
  - 콘솔에 v1 vs v2 Kappa 비교표 + 오분류(혼동) 패턴 출력
"""

import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import tiktoken
from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score

BASE_DIR = Path(__file__).resolve().parent.parent
FINAL_FIN_PATH = BASE_DIR / "outputs" / "human_eval_final_fin.csv"
V2_RESULT_PATH = BASE_DIR / "outputs" / "pilot80_gpt_labels_v2.csv"

CATEGORIES = ["환불요청", "주문취소", "불만제기", "배송확인", "교환반품", "구매진행", "서비스이용"]

PRICING = {
    "gpt-5.5": {"input": 5.00, "output": 30.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
}

PROMPT_TEMPLATE_V2 = """[지시문]
당신은 전자상거래(온라인 교육 플랫폼) 고객센터 상담 통화를 분석하는 전문가입니다.
아래 통화 전체 내용을 읽고, 고객의 실제 니즈(콜 전체를 통해 드러나는 핵심 용건)를
다음 7개 카테고리 중 하나로 분류하세요.

[카테고리 정의 및 예시]
1. 환불요청: 결제한 금액을 돌려받는 것이 최종 목적인 경우
   - "취소", "반품", "반송"이라는 단어가 나와도, 그 절차의 최종 목적이
     금전 반환(현금/캐시)이면 환불요청으로 분류할 것
   - 예: "책 취소하고 환불받고 싶어요", "반송했으니 캐시로 환불해주세요"
   - 예: "재결제하고 전체 카드취소 해주세요" (재결제는 절차일 뿐,
     목적이 기존 결제건 환불이면 환불요청)

2. 주문취소: 아직 배송/수강 전 상태에서, 환불 절차 없이 주문 자체를
   취소하는 경우 (결제 후 물건을 받기 전, 단순 주문 취소)
   - 환불요청과의 핵심 차이: 이미 수령한 물건에 대한 반송·환불 절차가
     없고, 순수하게 "주문을 무르는" 경우만 해당
   - 예: "어제 주문한 거 아직 배송 전인데 취소하고 싶어요"

3. 불만제기: 문의처럼 들리나 본질은 항의(약속 불이행, 응대 불만 등)
   예: "그쪽 전화를 안 받으셔서요", "몇 번을 전화했는지 아세요?"

4. 배송확인: 배송 상태·도착 문의
   예: "탭이 아직 안 오고 있어서 언제쯤 오나 싶어서요"

5. 교환반품: 불량·오배송으로 물건을 다른 물건으로 교체하는 경우
   (금전 반환이 아니라 물건↔물건 교체가 목적일 때만 해당)
   예: "파본이 돼서 왔어요", "색이 잘못돼서 교환하려고요"

6. 구매진행: 결제 완료를 위한 도움 요청
   예: "결제가 안 돼서요", "쿠폰 적용이 안 돼요"

7. 서비스이용: 로그인·기기·앱·시스템 이용 관련 문제
   (재구매 제약, 시스템 오류 확인 등 결제 자체와 무관한 절차 문제,
   실질적 용건 없이 통화가 종료되는 경우 포함)
   - 결제/재구매가 실제로 진행되지 않고, 시스템 문제 해결이
     최종 목적인 경우 구매진행이 아닌 서비스이용으로 분류
   예: "윈도우 환경에서만 재생된다고 떠서요"

[판단 원칙 - 공통]
- 통화 중 여러 절차(취소→재결제 등)가 언급되어도, "이 통화에서
  고객이 최종적으로 원하는 결과"를 기준으로 분류할 것
- 통화 초반 표현과 후반 표현이 다르면, 통화 전체 맥락에서
  더 우세하게 실행/확정된 액션을 기준으로 판단할 것
- 통화 내용이 지나치게 짧거나 실질적 용건이 드러나지 않는 경우
  서비스이용으로 분류할 것

[통화 전체 내용]
{call_text}

[질문]
위 통화에서 고객의 실제 니즈를 위 7개 카테고리 중 하나로만 답하세요.
카테고리명만 정확히 출력하고, 다른 설명은 추가하지 마세요.
"""


def normalize_gpt_label(raw: str) -> str:
    return re.sub(r"^\s*\d+[.)]\s*", "", str(raw)).strip()


def normalize_human_label(raw: str) -> str:
    raw = str(raw).strip()
    return "서비스이용" if raw.startswith("기타-") else raw


def make_prompt(call_text: str) -> str:
    return PROMPT_TEMPLATE_V2.format(call_text=call_text)


def estimate_cost(df: pd.DataFrame) -> None:
    enc = tiktoken.get_encoding("o200k_base")
    tokens = [len(enc.encode(make_prompt(t))) for t in df["call_text"]]
    total_input = sum(tokens)
    n = len(df)
    total_output = n * 10
    print(f"=== v2 프롬프트 비용 예측 (콜 {n}개 x 모델 2개) ===")
    grand_total = 0.0
    for model, price in PRICING.items():
        in_cost = total_input / 1_000_000 * price["input"]
        out_cost = total_output / 1_000_000 * price["output"]
        total = in_cost + out_cost
        grand_total += total
        print(f"[{model}] input ${in_cost:.4f} + output ${out_cost:.4f} = ${total:.4f}")
    print(f"합계 예상 비용: ${grand_total:.4f}\n")


def call_one(client, model: str, call_id: str, call_text: str, max_retries: int = 3):
    prompt = make_prompt(call_text)
    kwargs = dict(model=model, messages=[{"role": "user", "content": prompt}], temperature=0)
    last_err = None
    for attempt in range(max_retries):
        try:
            try:
                resp = client.chat.completions.create(**kwargs)
            except Exception as e:
                if "temperature" in str(e).lower() and "temperature" in kwargs:
                    kwargs.pop("temperature")
                    resp = client.chat.completions.create(**kwargs)
                else:
                    raise
            label = normalize_gpt_label(resp.choices[0].message.content)
            return {
                "call_id": call_id,
                "label": label,
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            }
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    return {"call_id": call_id, "label": f"ERROR: {last_err}", "prompt_tokens": 0, "completion_tokens": 0}


def run_model(client, model: str, df: pd.DataFrame, max_workers: int = 5) -> pd.DataFrame:
    results = [None] * len(df)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(call_one, client, model, row.call_id, row.call_text): i
            for i, row in enumerate(df.itertuples())
        }
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            if done % 20 == 0 or done == len(df):
                print(f"  [{model}] {done}/{len(df)} 완료")
    return pd.DataFrame(results)


def run_v2(df: pd.DataFrame) -> pd.DataFrame:
    load_dotenv(BASE_DIR / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key == "REPLACE_ME":
        print("ERROR: OPENAI_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    usage = {}
    label_cols = {}
    for model in PRICING:
        print(f"\n=== {model} (v2 프롬프트) 호출 시작 ({len(df)}콜) ===")
        res_df = run_model(client, model, df)
        usage[model] = {
            "prompt_tokens": res_df["prompt_tokens"].sum(),
            "completion_tokens": res_df["completion_tokens"].sum(),
        }
        label_cols[model] = res_df.set_index("call_id")["label"]

    out = df[["call_id"]].copy()
    out["gpt55_label_v2"] = out["call_id"].map(label_cols["gpt-5.5"])
    out["gpt41mini_label_v2"] = out["call_id"].map(label_cols["gpt-4.1-mini"])
    out.to_csv(V2_RESULT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n저장: {V2_RESULT_PATH}")

    print("\n=== v2 실제 토큰 사용량 및 비용 ===")
    grand_total = 0.0
    for model, u in usage.items():
        price = PRICING[model]
        in_cost = u["prompt_tokens"] / 1_000_000 * price["input"]
        out_cost = u["completion_tokens"] / 1_000_000 * price["output"]
        total = in_cost + out_cost
        grand_total += total
        print(
            f"[{model}] input {u['prompt_tokens']:,} (${in_cost:.4f}) + "
            f"output {u['completion_tokens']:,} (${out_cost:.4f}) = ${total:.4f}"
        )
    print(f"합계: ${grand_total:.4f}")
    return out


def print_kappa_comparison(merged: pd.DataFrame) -> None:
    v1_55 = merged.dropna(subset=["gpt55_label", "human_label_norm"])
    v1_41 = merged.dropna(subset=["gpt41mini_label", "human_label_norm"])
    v2_55 = merged.dropna(subset=["gpt55_label_v2", "human_label_norm"])
    v2_41 = merged.dropna(subset=["gpt41mini_label_v2", "human_label_norm"])

    k_v1_55 = cohen_kappa_score(v1_55["human_label_norm"], v1_55["gpt55_label"])
    k_v1_41 = cohen_kappa_score(v1_41["human_label_norm"], v1_41["gpt41mini_label"])
    k_v2_55 = cohen_kappa_score(v2_55["human_label_norm"], v2_55["gpt55_label_v2"])
    k_v2_41 = cohen_kappa_score(v2_41["human_label_norm"], v2_41["gpt41mini_label_v2"])

    print("\n=== v1(기존 프롬프트) vs v2(수정 프롬프트) Cohen's Kappa 비교 ===")
    print("(사람 라벨은 human_label_norm으로 양쪽 동일하게 정규화한 값 사용)")
    table = pd.DataFrame(
        {
            "v1 Kappa": [f"{k_v1_55:.4f} (n={len(v1_55)})", f"{k_v1_41:.4f} (n={len(v1_41)})"],
            "v2 Kappa": [f"{k_v2_55:.4f} (n={len(v2_55)})", f"{k_v2_41:.4f} (n={len(v2_41)})"],
        },
        index=["GPT-5.5", "GPT-4.1 mini"],
    )
    print(table.to_string())
    if len(v1_55) != len(v2_55) or len(v1_41) != len(v2_41):
        print("\n※ 각주: v1/v2 건수(n)가 다르면 결측 라벨(모델 응답 파싱 실패 등)이 있었다는 뜻입니다.")
    else:
        print("\n※ 각주: v1/v2 모두 80건 전체(제외 없음) 기준으로 계산했습니다.")


def print_confusion(merged: pd.DataFrame, label_col: str, model_name: str) -> None:
    print(f"\n--- {model_name} 오분류 패턴 (행: 사람 라벨, 열: 모델 라벨) ---")
    sub = merged.dropna(subset=[label_col, "human_label_norm"])
    ct = pd.crosstab(sub["human_label_norm"], sub[label_col])
    ct = ct.reindex(index=CATEGORIES, columns=CATEGORIES, fill_value=0)
    print(ct.to_string())

    refund_as_cancel = ct.loc["환불요청", "주문취소"] if "환불요청" in ct.index and "주문취소" in ct.columns else 0
    cancel_as_refund = ct.loc["주문취소", "환불요청"] if "주문취소" in ct.index and "환불요청" in ct.columns else 0
    print(f"환불요청(사람)→주문취소(모델) 혼동: {refund_as_cancel}건")
    print(f"주문취소(사람)→환불요청(모델) 혼동: {cancel_as_refund}건")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="실제로 v2 프롬프트로 API를 호출한다")
    args = parser.parse_args()

    df = pd.read_csv(FINAL_FIN_PATH)
    df["human_label_norm"] = df["human_label"].apply(normalize_human_label)

    if not args.run:
        estimate_cost(df)
        print("실행하려면 --run 플래그를 붙여 다시 실행하세요.")
        return

    v2 = run_v2(df)
    merged = df.merge(v2, on="call_id", how="left")

    print("\n=== human_label 정규화 결과 ===")
    print(merged["human_label_norm"].value_counts().to_string())

    print_kappa_comparison(merged)
    print_confusion(merged, "gpt55_label", "GPT-5.5 v1")
    print_confusion(merged, "gpt55_label_v2", "GPT-5.5 v2")
    print_confusion(merged, "gpt41mini_label", "GPT-4.1 mini v1")
    print_confusion(merged, "gpt41mini_label_v2", "GPT-4.1 mini v2")


if __name__ == "__main__":
    main()
