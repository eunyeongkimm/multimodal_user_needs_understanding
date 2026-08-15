"""v4로 라벨링된 11,579건 중 무작위 200건(random_state=42)을 동일 v4 프롬프트로
한 번 더 실행해서, 1차 결과와 2차 결과의 일치율/Cohen's Kappa를 계산한다.
사람 라벨 없이도 모델 자체의 판정 안정성(특히 has_complaint 게이트)을 확인하기 위함.

입력: outputs/gold_actual_batch1_v4.parquet (1차 결과)
출력:
  - outputs/v4_self_consistency_pass2.csv (call_id, has_complaint_pass2, label_pass2)
  - 콘솔에 일치율 + Kappa (label 전체, has_complaint 게이트 별도)
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v4_prompt import make_prompt, parse_v4_response

BASE_DIR = Path(__file__).resolve().parent.parent
V4_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_v4.parquet"
CALL_TEXTS_PATH = BASE_DIR / "outputs" / "batch1_call_texts.parquet"
PASS2_PATH = BASE_DIR / "outputs" / "v4_self_consistency_pass2.csv"

MODEL = "gpt-4.1-mini"
N_SAMPLE = 200
SEED = 42


def call_one(client, call_id: str, call_text: str, max_retries: int = 3):
    prompt = make_prompt(call_text)
    kwargs = dict(model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0,
                  response_format={"type": "json_object"})
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            has_complaint, label = parse_v4_response(resp.choices[0].message.content)
            return {"call_id": call_id, "has_complaint_pass2": has_complaint, "label_pass2": label}
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    return {"call_id": call_id, "has_complaint_pass2": None, "label_pass2": f"ERROR: {last_err}"}


def safe_kappa(a, b):
    try:
        k = cohen_kappa_score(a, b)
        return k if k == k else float("nan")  # NaN 체크
    except Exception:
        return float("nan")


def main():
    load_dotenv(BASE_DIR / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key == "REPLACE_ME":
        print("ERROR: OPENAI_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    v4 = pd.read_parquet(V4_PATH)
    call_texts = pd.read_parquet(CALL_TEXTS_PATH)
    v4 = v4.merge(call_texts, on="call_id", how="left")

    sample = v4.sample(n=min(N_SAMPLE, len(v4)), random_state=SEED)
    print(f"자기일관성 검증 샘플: {len(sample)}건 (전체 {len(v4)}건 중 random_state={SEED})")

    results = [None] * len(sample)
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            pool.submit(call_one, client, row.call_id, row.call_text): i
            for i, row in enumerate(sample.itertuples())
        }
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            if done % 20 == 0 or done == len(sample):
                print(f"  [pass2] {done}/{len(sample)} 완료")

    pass2 = pd.DataFrame(results)
    pass2.to_csv(PASS2_PATH, index=False, encoding="utf-8-sig")
    print(f"\n저장: {PASS2_PATH}")

    merged = sample[["call_id", "has_complaint", "label_v4"]].merge(pass2, on="call_id", how="left")
    merged = merged.rename(columns={"has_complaint": "has_complaint_pass1", "label_v4": "label_pass1"})

    valid = merged.dropna(subset=["label_pass1", "label_pass2"])
    valid = valid[~valid["label_pass2"].astype(str).str.startswith("ERROR")]

    n_agree_label = (valid["label_pass1"] == valid["label_pass2"]).sum()
    agree_rate_label = n_agree_label / len(valid) if len(valid) else float("nan")
    kappa_label = safe_kappa(valid["label_pass1"], valid["label_pass2"])

    hc_valid = valid.dropna(subset=["has_complaint_pass1", "has_complaint_pass2"])
    n_agree_hc = (hc_valid["has_complaint_pass1"] == hc_valid["has_complaint_pass2"]).sum()
    agree_rate_hc = n_agree_hc / len(hc_valid) if len(hc_valid) else float("nan")
    kappa_hc = safe_kappa(hc_valid["has_complaint_pass1"], hc_valid["has_complaint_pass2"])

    print(f"\n=== 자기일관성 결과 (n={len(valid)}, 파싱 실패/에러 {len(sample)-len(valid)}건 제외) ===")
    print(f"라벨(7개 카테고리) 일치율: {agree_rate_label*100:.1f}% ({n_agree_label}/{len(valid)})")
    print(f"라벨 Cohen's Kappa: {kappa_label:.4f}")
    print(f"\nhas_complaint(불만제기 게이트) 일치율: {agree_rate_hc*100:.1f}% ({n_agree_hc}/{len(hc_valid)})")
    print(f"has_complaint Cohen's Kappa: {kappa_hc:.4f}")

    disagree = valid[valid["label_pass1"] != valid["label_pass2"]]
    if len(disagree):
        print(f"\n--- 불일치 사례 ({len(disagree)}건) ---")
        print(disagree[["call_id", "has_complaint_pass1", "label_pass1",
                         "has_complaint_pass2", "label_pass2"]].to_string(index=False))


if __name__ == "__main__":
    main()
