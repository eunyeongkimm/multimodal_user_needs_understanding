"""human_eval_final_fin.csv(80건) 중 v2(gpt-4.1-mini) 라벨이 환불요청/배송확인이었던
콜만 골라 v3 프롬프트(불만제기 우선판단 규칙)로 재실행하고, human_label과 비교해
v2 대비 Kappa가 어떻게 바뀌는지, 불만제기로 재분류된 콜이 실제로 타당한지 검증한다.

왜 gpt41mini_label_v2 기준으로 필터링하는가:
  gold_actual_batch1(운영 배치)도 v2 프롬프트 + gpt-4.1-mini로 만들어졌으므로,
  "v2가 환불요청/배송확인으로 분류했던 콜"이 실제 파이프라인에서 v3 재라벨링 대상이
  되는 모집단과 동일한 성격을 가진다.

입력:
  - outputs/human_eval_final_fin.csv (call_id, call_text, human_label 등)
  - outputs/pilot80_gpt_labels_v2.csv (call_id, gpt41mini_label_v2)
출력:
  - outputs/human_eval_v3_labels.csv (call_id, gpt41mini_label_v3)
  - 콘솔에 v2 vs v3 Kappa 비교, 혼동행렬, v3가 불만제기로 재분류한 콜 목록
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relabel_v2_and_kappa import CATEGORIES, normalize_gpt_label, normalize_human_label
from v3_prompt import make_prompt

BASE_DIR = Path(__file__).resolve().parent.parent
FINAL_FIN_PATH = BASE_DIR / "outputs" / "human_eval_final_fin.csv"
V2_RESULT_PATH = BASE_DIR / "outputs" / "pilot80_gpt_labels_v2.csv"
V3_RESULT_PATH = BASE_DIR / "outputs" / "human_eval_v3_labels.csv"

MODEL = "gpt-4.1-mini"
TARGET_V2_LABELS = ["환불요청", "배송확인"]


def call_one(client, call_id: str, call_text: str, max_retries: int = 3):
    prompt = make_prompt(call_text)
    kwargs = dict(model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            label = normalize_gpt_label(resp.choices[0].message.content)
            return {"call_id": call_id, "label": label}
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    return {"call_id": call_id, "label": f"ERROR: {last_err}"}


def run_v3(df: pd.DataFrame) -> pd.DataFrame:
    load_dotenv(BASE_DIR / ".env")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key == "REPLACE_ME":
        print("ERROR: OPENAI_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

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
            print(f"  [{MODEL} v3] {done}/{len(df)} 완료")

    res_df = pd.DataFrame(results).rename(columns={"label": "gpt41mini_label_v3"})
    res_df.to_csv(V3_RESULT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n저장: {V3_RESULT_PATH}")
    return res_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="실제로 v3 프롬프트로 API를 호출한다")
    args = parser.parse_args()

    human = pd.read_csv(FINAL_FIN_PATH)
    human["human_label_norm"] = human["human_label"].apply(normalize_human_label)
    v2 = pd.read_csv(V2_RESULT_PATH)

    merged = human.merge(v2, on="call_id", how="left")
    subset = merged[merged["gpt41mini_label_v2"].isin(TARGET_V2_LABELS)].copy()
    print(f"v2(gpt-4.1-mini)가 환불요청/배송확인으로 분류했던 콜: {len(subset)}건 "
          f"(환불요청 {(subset['gpt41mini_label_v2']=='환불요청').sum()}, "
          f"배송확인 {(subset['gpt41mini_label_v2']=='배송확인').sum()})")

    if not args.run:
        print("실행하려면 --run 플래그를 붙여 다시 실행하세요.")
        return

    v3_res = run_v3(subset[["call_id", "call_text"]])
    subset = subset.merge(v3_res, on="call_id", how="left")

    print("\n=== v3 재분류 결과 (v2 라벨 -> v3 라벨 변경 건수) ===")
    changed = subset[subset["gpt41mini_label_v2"] != subset["gpt41mini_label_v3"]]
    print(f"변경됨: {len(changed)}/{len(subset)}건")
    if len(changed):
        print(changed[["call_id", "human_label_norm", "gpt41mini_label_v2", "gpt41mini_label_v3"]]
              .to_string(index=False))

    to_complaint = subset[subset["gpt41mini_label_v3"] == "불만제기"]
    print(f"\nv3가 '불만제기'로 새로 분류한 콜: {len(to_complaint)}건")
    if len(to_complaint):
        print(to_complaint[["call_id", "human_label_norm", "gpt41mini_label_v2"]].to_string(index=False))

    v2_eval = subset.dropna(subset=["gpt41mini_label_v2", "human_label_norm"])
    v3_eval = subset.dropna(subset=["gpt41mini_label_v3", "human_label_norm"])
    k_v2 = cohen_kappa_score(v2_eval["human_label_norm"], v2_eval["gpt41mini_label_v2"])
    k_v3 = cohen_kappa_score(v3_eval["human_label_norm"], v3_eval["gpt41mini_label_v3"])

    print("\n=== v2 vs v3 Cohen's Kappa (환불요청/배송확인 서브셋, human_label_norm 대비) ===")
    print(f"v2: {k_v2:.4f} (n={len(v2_eval)})")
    print(f"v3: {k_v3:.4f} (n={len(v3_eval)})")

    print("\n--- v2 혼동행렬 (행: human_label_norm, 열: gpt41mini_label_v2) ---")
    ct_v2 = pd.crosstab(v2_eval["human_label_norm"], v2_eval["gpt41mini_label_v2"])
    ct_v2 = ct_v2.reindex(index=sorted(set(ct_v2.index) | set(CATEGORIES)), fill_value=0)
    print(ct_v2.loc[(ct_v2 != 0).any(axis=1)].to_string())

    print("\n--- v3 혼동행렬 (행: human_label_norm, 열: gpt41mini_label_v3) ---")
    ct_v3 = pd.crosstab(v3_eval["human_label_norm"], v3_eval["gpt41mini_label_v3"])
    ct_v3 = ct_v3.reindex(index=sorted(set(ct_v3.index) | set(CATEGORIES)), fill_value=0)
    print(ct_v3.loc[(ct_v3 != 0).any(axis=1)].to_string())


if __name__ == "__main__":
    main()
