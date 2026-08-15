"""v_unified 검증용 층화 샘플 2,000건 추출.
gold_actual_batch1_final(19,847콜) 기준:
  - 불만제기: 250건(오버샘플, gold 전체 484건 중)
  - 나머지 6클래스: 1,750건을 gold 내 상대 비율대로 배분

출력: outputs/v_unified_sample_call_ids.csv (call_id, gold_actual, stratum)
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
GOLD_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_final.parquet"
OUT_PATH = BASE_DIR / "outputs" / "v_unified_sample_call_ids.csv"

COMPLAINT_N = 250
REST_TOTAL = 1750
RANDOM_STATE = 42
NON_COMPLAINT_CLASSES = ["환불요청", "주문취소", "배송확인", "교환반품", "구매진행", "서비스이용"]


def main():
    gold = pd.read_parquet(GOLD_PATH)[["call_id", "label"]].rename(columns={"label": "gold_actual"})
    dist = gold["gold_actual"].value_counts()
    print("=== gold_actual_batch1_final 전체 분포 ===")
    print(dist.to_string())

    complaint_pool = gold[gold["gold_actual"] == "불만제기"]
    print(f"\n불만제기 pool: {len(complaint_pool)}건 중 {COMPLAINT_N}건 샘플링")
    complaint_sample = complaint_pool.sample(n=min(COMPLAINT_N, len(complaint_pool)), random_state=RANDOM_STATE)

    rest_pool = gold[gold["gold_actual"].isin(NON_COMPLAINT_CLASSES)]
    rest_dist = rest_pool["gold_actual"].value_counts()
    raw_alloc = (rest_dist / rest_dist.sum() * REST_TOTAL)
    alloc = raw_alloc.round().astype(int)
    diff = REST_TOTAL - alloc.sum()
    if diff != 0:
        largest_class = alloc.idxmax()
        alloc[largest_class] += diff
    print(f"\n=== 나머지 6클래스 배분 계획 (목표 {REST_TOTAL}건) ===")
    print(pd.DataFrame({"gold_비율%": (rest_dist/rest_dist.sum()*100).round(2), "배분건수": alloc}).to_string())

    rest_samples = []
    for cls, n in alloc.items():
        pool_cls = rest_pool[rest_pool["gold_actual"] == cls]
        rest_samples.append(pool_cls.sample(n=n, random_state=RANDOM_STATE))
    rest_sample = pd.concat(rest_samples, ignore_index=True)

    sample = pd.concat([complaint_sample, rest_sample], ignore_index=True)
    sample = sample.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)  # 셔플

    print(f"\n=== 실제 층별 추출 수 (전체 {len(sample)}건, seed={RANDOM_STATE}) ===")
    print(sample["gold_actual"].value_counts().to_string())

    sample.to_csv(OUT_PATH, index=False)
    print(f"\n저장: {OUT_PATH}")
    print("\nV_UNIFIED_SAMPLE_COMPLETE")


if __name__ == "__main__":
    main()
