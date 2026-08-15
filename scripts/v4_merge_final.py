"""v4 재라벨링 결과(outputs/gold_actual_batch1_v4.parquet, 환불요청/배송확인 11,579건)를
gold_actual_batch1.parquet(v2, 19,847건 전체)에 병합해 최종본을 만든다.

병합 규칙:
- v4 대상이었던 call_id(원래 환불요청/배송확인) -> label_v4로 교체
  (단, label_v4 파싱 실패(NaN)면 원래 v2 라벨로 폴백하고 별도 로그)
- 나머지(교환반품/구매진행/서비스이용/주문취소/이미 불만제기였던 32건) -> v2 라벨 유지

출력: outputs/gold_actual_batch1_final.parquet
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
V2_PATH = BASE_DIR / "outputs" / "gold_actual_batch1.parquet"
V4_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_v4.parquet"
FINAL_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_final.parquet"


def main():
    v2 = pd.read_parquet(V2_PATH)
    v4 = pd.read_parquet(V4_PATH)

    v4_ids = set(v4["call_id"])
    kept = v2[~v2["call_id"].isin(v4_ids)].copy()
    kept["source"] = "v2"

    v4_merged = v4.merge(v2[["call_id", "label"]].rename(columns={"label": "label_v2_fallback"}),
                          on="call_id", how="left")
    n_fallback = v4_merged["label_v4"].isna().sum()
    if n_fallback:
        print(f"경고: label_v4 파싱 실패로 v2 라벨로 폴백한 콜 {n_fallback}건 "
              f"(call_id 예시: {v4_merged.loc[v4_merged['label_v4'].isna(), 'call_id'].head(5).tolist()})")
    v4_merged["label"] = v4_merged["label_v4"].fillna(v4_merged["label_v2_fallback"])
    v4_merged["source"] = "v4"

    final = pd.concat(
        [kept, v4_merged[["call_id", "label", "prompt_version", "model", "has_complaint", "source"]]],
        ignore_index=True,
    )
    assert len(final) == len(v2), f"행 수 불일치: v2={len(v2)}, final={len(final)}"
    assert final["call_id"].is_unique, "call_id 중복 발생"

    final.to_parquet(FINAL_PATH, index=False)
    print(f"\n저장: {FINAL_PATH} ({len(final)}행)")
    print(f"v4로 교체된 콜: {len(v4_merged)}건 (그 중 폴백 {n_fallback}건), v2 유지된 콜: {len(kept)}건")

    print("\n=== 최종 라벨 분포 (7개 카테고리) ===")
    dist = final["label"].value_counts(dropna=False)
    print(dist.to_string())
    print(f"\n비율:")
    print((dist / len(final) * 100).round(2).astype(str).add("%").to_string())

    print("\n=== v2 -> v4 재분류 매트릭스 (환불요청/배송확인 대상) ===")
    comp = v2[v2["call_id"].isin(v4_ids)][["call_id", "label"]].rename(columns={"label": "label_v2"})
    comp = comp.merge(v4_merged[["call_id", "label"]].rename(columns={"label": "label_v4_final"}), on="call_id")
    ct = pd.crosstab(comp["label_v2"], comp["label_v4_final"])
    print(ct.to_string())

    n_changed = (comp["label_v2"] != comp["label_v4_final"]).sum()
    n_to_complaint = (comp["label_v4_final"] == "불만제기").sum()
    print(f"\nv2->v4 라벨이 바뀐 콜: {n_changed}/{len(comp)}건")
    print(f"그 중 '불만제기'로 새로 분류된 콜: {n_to_complaint}건")

    n_complaint_final = (final["label"] == "불만제기").sum()
    print(f"\n최종 불만제기 건수: {n_complaint_final}건 ({n_complaint_final/len(final)*100:.2f}%) "
          f"(v2 32건 -> 이번 재라벨링 후)")


if __name__ == "__main__":
    main()
