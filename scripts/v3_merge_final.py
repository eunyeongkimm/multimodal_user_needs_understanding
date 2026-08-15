"""v3 재라벨링 결과(outputs/gold_actual_batch1_v3.parquet, 환불요청/배송확인 11,579건)를
gold_actual_batch1.parquet(v2, 19,847건 전체)에 병합해 최종본을 만든다.

병합 규칙: v3 대상이었던 call_id(환불요청/배송확인)는 v3 라벨로 교체하고,
나머지(교환반품/구매진행/서비스이용/주문취소/이미 불만제기였던 32건)는 v2 라벨 유지.

출력: outputs/gold_actual_batch1_final.parquet
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
V2_PATH = BASE_DIR / "outputs" / "gold_actual_batch1.parquet"
V3_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_v3.parquet"
FINAL_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_final.parquet"


def main():
    v2 = pd.read_parquet(V2_PATH)
    v3 = pd.read_parquet(V3_PATH)

    v3_ids = set(v3["call_id"])
    kept = v2[~v2["call_id"].isin(v3_ids)].copy()
    kept["source"] = "v2"

    replaced = v3.copy()
    replaced["source"] = "v3"

    final = pd.concat([kept, replaced], ignore_index=True)
    assert len(final) == len(v2), f"행 수 불일치: v2={len(v2)}, final={len(final)}"
    assert final["call_id"].is_unique, "call_id 중복 발생"

    final.to_parquet(FINAL_PATH, index=False)
    print(f"저장: {FINAL_PATH} ({len(final)}행)")
    print(f"v3로 교체된 콜: {len(replaced)}건, v2 유지된 콜: {len(kept)}건")

    n_null = final["label"].isna().sum()
    print(f"라벨 파싱 실패: {n_null}건")

    print("\n=== 최종 라벨 분포 ===")
    dist = final["label"].value_counts(dropna=False)
    print(dist.to_string())

    print("\n=== v2 대비 변화 (환불요청/배송확인 -> 어디로 재분류됐는지) ===")
    v2_sub = v2[v2["call_id"].isin(v3_ids)][["call_id", "label"]].rename(columns={"label": "label_v2"})
    v3_sub = v3[["call_id", "label"]].rename(columns={"label": "label_v3"})
    comp = v2_sub.merge(v3_sub, on="call_id", how="inner")
    ct = pd.crosstab(comp["label_v2"], comp["label_v3"])
    print(ct.to_string())

    n_changed = (comp["label_v2"] != comp["label_v3"]).sum()
    n_to_complaint = (comp["label_v3"] == "불만제기").sum()
    print(f"\nv2->v3 라벨이 바뀐 콜: {n_changed}/{len(comp)}건")
    print(f"그 중 '불만제기'로 새로 분류된 콜: {n_to_complaint}건")

    print(f"\n최종 불만제기 건수: {(final['label']=='불만제기').sum()}건 "
          f"({(final['label']=='불만제기').sum()/len(final)*100:.2f}%) "
          f"(v2 32건 대비 변화)")


if __name__ == "__main__":
    main()
