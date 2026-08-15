"""환불요청 vs 주문취소 gold_actual 경계 분석 (전체 라벨된 콜 기준, 파일럿 샘플 아님).
API 호출 없음 - 키워드 등장률만 집계해서 표로 출력.

출력: outputs/refund_cancel_boundary.xlsx (표만, 해석/결론 없음)
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
GOLD_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_final.parquet"
CALL_TEXTS_PATH = BASE_DIR / "outputs" / "batch1_call_texts.parquet"
OUT_XLSX = BASE_DIR / "outputs" / "refund_cancel_boundary.xlsx"

KEYWORDS = ["취소", "반품", "반송", "환불", "환급", "캐시", "돌려"]
GROUPS = ["환불요청", "주문취소"]


def main():
    gold = pd.read_parquet(GOLD_PATH)[["call_id", "label"]]
    texts = pd.read_parquet(CALL_TEXTS_PATH)[["call_id", "call_text"]]
    df = gold.merge(texts, on="call_id", how="left")

    n_missing_text = df["call_text"].isna().sum()
    print(f"전체 라벨 콜: {len(df)}, call_text 결측: {n_missing_text}")

    df = df[df["label"].isin(GROUPS)].copy()
    df["call_text"] = df["call_text"].fillna("")

    for kw in KEYWORDS:
        df[f"has_{kw}"] = df["call_text"].str.contains(kw, regex=False)

    # === 표 1: 그룹별 키워드 등장률 ===
    rate_rows = []
    for label in GROUPS:
        sub = df[df["label"] == label]
        row = {"gold_actual": label, "n_calls": len(sub)}
        for kw in KEYWORDS:
            row[kw] = round(sub[f"has_{kw}"].mean() * 100, 2)
        rate_rows.append(row)
    rate_table = pd.DataFrame(rate_rows)

    print("\n=== 표1: 그룹별 키워드 등장률(%) ===")
    print(rate_table.to_string(index=False))

    # === 표 2: 환불요청 중 '취소'는 있으나 '환불'/'환급' 없는 비율 ===
    refund = df[df["label"] == "환불요청"]
    cond2 = refund["has_취소"] & ~(refund["has_환불"] | refund["has_환급"])
    n_cond2 = cond2.sum()
    pct_cond2 = n_cond2 / len(refund) * 100 if len(refund) else float("nan")

    # === 표 3: 주문취소 중 '환불'/'환급'/'캐시' 언급 비율 ===
    cancel = df[df["label"] == "주문취소"]
    cond3 = cancel["has_환불"] | cancel["has_환급"] | cancel["has_캐시"]
    n_cond3 = cond3.sum()
    pct_cond3 = n_cond3 / len(cancel) * 100 if len(cancel) else float("nan")

    special_table = pd.DataFrame([
        {
            "항목": "환불요청 중 '취소' 있으나 '환불/환급' 없는 콜",
            "분모(그룹 전체)": len(refund), "해당 건수": int(n_cond2), "비율(%)": round(pct_cond2, 2),
        },
        {
            "항목": "주문취소 중 '환불/환급/캐시' 언급 있는 콜",
            "분모(그룹 전체)": len(cancel), "해당 건수": int(n_cond3), "비율(%)": round(pct_cond3, 2),
        },
    ])

    print("\n=== 표2: 경계 케이스 비율 ===")
    print(special_table.to_string(index=False))

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        rate_table.to_excel(writer, sheet_name="키워드_등장률", index=False)
        special_table.to_excel(writer, sheet_name="경계_케이스_비율", index=False)

    print(f"\n저장: {OUT_XLSX}")


if __name__ == "__main__":
    main()
