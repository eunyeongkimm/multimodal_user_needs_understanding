"""v4로 재라벨링된 11,579건(환불요청/배송확인 대상) 중, 최종 라벨이 '불만제기'가
아닌 콜들을 대상으로 항의 신호 키워드를 전수 스캔한다. v4의 has_complaint 게이트가
놓쳤을 수 있는 콜(위양성 아닌 위음성 후보)을 사람이 나중에 검토할 수 있도록
후보 목록을 뽑아두는 용도.

입력: outputs/gold_actual_batch1_v4.parquet, outputs/batch1_call_texts.parquet
출력: outputs/suspect_calls_for_review.csv (call_id, 현재_label, 매칭된_키워드, call_text)
"""

import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
V4_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_v4.parquet"
CALL_TEXTS_PATH = BASE_DIR / "outputs" / "batch1_call_texts.parquet"
OUT_PATH = BASE_DIR / "outputs" / "suspect_calls_for_review.csv"

KEYWORDS = ["몇 번", "지난번", "저번에", "계속", "한 달", "왜 이렇게", "일방적으로", "당연히"]
N_SAMPLE = 50
SEED = 42


def matched_keywords(text: str) -> list:
    return [kw for kw in KEYWORDS if kw in text]


def main():
    v4 = pd.read_parquet(V4_PATH)
    call_texts = pd.read_parquet(CALL_TEXTS_PATH)
    df = v4.merge(call_texts, on="call_id", how="left")

    non_complaint = df[df["label_v4"].notna() & (df["label_v4"] != "불만제기")].copy()
    non_complaint["매칭된_키워드"] = non_complaint["call_text"].apply(matched_keywords)
    hits = non_complaint[non_complaint["매칭된_키워드"].apply(len) > 0].copy()

    print(f"v4 대상 전체: {len(df)}건")
    print(f"'불만제기'가 아닌 콜: {len(non_complaint)}건")
    print(f"그 중 항의 신호 키워드 히트: {len(hits)}건 ({len(hits)/len(non_complaint)*100:.2f}%)")

    kw_counts = {}
    for kws in hits["매칭된_키워드"]:
        for kw in kws:
            kw_counts[kw] = kw_counts.get(kw, 0) + 1
    print("\n키워드별 히트 건수:")
    for kw, cnt in sorted(kw_counts.items(), key=lambda x: -x[1]):
        print(f"  {kw}: {cnt}건")

    print("\n현재 라벨 분포 (히트된 콜 기준):")
    print(hits["label_v4"].value_counts().to_string())

    sample = hits.sample(n=min(N_SAMPLE, len(hits)), random_state=SEED).copy()
    sample["매칭된_키워드"] = sample["매칭된_키워드"].apply(lambda ks: ", ".join(ks))
    out = sample.rename(columns={"label_v4": "현재_label"})[
        ["call_id", "현재_label", "매칭된_키워드", "call_text"]
    ]
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT_PATH} ({len(out)}건 샘플)")


if __name__ == "__main__":
    main()
