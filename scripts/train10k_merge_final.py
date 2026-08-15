"""train10k v2/v4/v5 결과를 병합해 최종 gold_actual과 동일 기준의 라벨을 만든다.
v5_merge_final.py와 동일 규칙:
1. v4에서 has_complaint=True인 콜 -> label='불만제기'로 확정
2. v4에서 has_complaint=False인 콜 -> v5 재분류 결과(label_v5)로 교체
   (파싱 실패(NaN)면 v4의 label_v4로 폴백)
3. 나머지(원래부터 환불요청/배송확인이 아니었던 콜) -> v2(gold_actual) 라벨 유지

출력: outputs/train_10k_final_labeled.parquet (call_id, gold_actual)
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
V2_PATH = BASE_DIR / "outputs" / "train10k_gpt_labels.parquet"
V4_PATH = BASE_DIR / "outputs" / "train10k_gpt_labels_v4.parquet"
V5_PATH = BASE_DIR / "outputs" / "train10k_gpt_labels_v5.parquet"
OUT_PATH = BASE_DIR / "outputs" / "train_10k_final_labeled.parquet"

CATEGORY_ORDER = ["환불요청", "주문취소", "불만제기", "배송확인", "교환반품", "구매진행", "서비스이용"]


def main():
    v2 = pd.read_parquet(V2_PATH).rename(columns={"gold_actual": "label"})[["call_id", "label"]]
    v4 = pd.read_parquet(V4_PATH)
    v5 = pd.read_parquet(V5_PATH)

    print(f"v2(전체): {len(v2)}건, v4 대상: {len(v4)}건, v5 대상: {len(v5)}건")

    n_v4_parse_fail = v4["has_complaint"].isna().sum()
    if n_v4_parse_fail:
        print(f"경고: v4 자체가 파싱 실패(has_complaint=None)인 콜 {n_v4_parse_fail}건 -> v2 라벨로 폴백: "
              f"{v4.loc[v4['has_complaint'].isna(), 'call_id'].tolist()}")
    v4_parse_fail_ids = set(v4.loc[v4["has_complaint"].isna(), "call_id"])

    complaint_rows = v4[v4["has_complaint"] == True][["call_id"]].copy()  # noqa: E712
    complaint_rows["label"] = "불만제기"

    non_complaint = v4[v4["has_complaint"] == False][["call_id", "label_v4"]].copy()  # noqa: E712
    non_complaint = non_complaint.merge(v5[["call_id", "label_v5"]], on="call_id", how="left")
    n_fallback = non_complaint["label_v5"].isna().sum()
    if n_fallback:
        print(f"경고: label_v5 파싱 실패로 v4 라벨로 폴백한 콜 {n_fallback}건")
    non_complaint["label"] = non_complaint["label_v5"].fillna(non_complaint["label_v4"])

    v4_target = pd.concat(
        [complaint_rows[["call_id", "label"]], non_complaint[["call_id", "label"]]],
        ignore_index=True,
    )
    assert len(v4_target) == len(v4) - n_v4_parse_fail, \
        f"대상 건수 불일치: v4={len(v4)}(파싱실패 {n_v4_parse_fail}건 제외), v4_target={len(v4_target)}"

    v4_ids = set(v4_target["call_id"])
    kept = v2[~v2["call_id"].isin(v4_ids)][["call_id", "label"]].copy()
    kept["source"] = "v2"

    v4_target["source"] = v4_target["call_id"].isin(complaint_rows["call_id"]).map(
        {True: "v4(has_complaint)", False: "v5"}
    )

    final = pd.concat([kept, v4_target[["call_id", "label", "source"]]], ignore_index=True)
    assert len(final) == len(v2), f"행 수 불일치: v2={len(v2)}, final={len(final)}"
    assert final["call_id"].is_unique, "call_id 중복 발생"

    final = final.rename(columns={"label": "gold_actual"})
    final[["call_id", "gold_actual"]].to_parquet(OUT_PATH, index=False)
    print(f"\n저장: {OUT_PATH} ({len(final)}행)")
    print(f"소스 구성: {final['source'].value_counts().to_dict()}")

    print("\n=== 최종 라벨 분포 ===")
    dist = final["gold_actual"].value_counts().reindex(CATEGORY_ORDER, fill_value=0)
    print(dist.to_string())
    print("\n비율:")
    print((dist / len(final) * 100).round(2).astype(str).add("%").to_string())

    n_complaint = dist["불만제기"]
    print(f"\n불만제기 최종 건수: {n_complaint}건 ({n_complaint/len(final)*100:.2f}%)")
    if n_complaint < 200:
        print("경고: 200건 미만 - 추가 샘플링 고려 필요할 수 있음(보고 대상)")
    else:
        print(f"200건 기준 충족 (예상 ~244건 대비 {'정상 범위' if 150 <= n_complaint <= 400 else '예상과 차이 있음, 확인 필요'})")

    print("\n=== v2 -> v4/v5 재분류 매트릭스 (v4 대상 5,784건) ===")
    v2_target_labels = v2[v2["call_id"].isin(v4_ids)][["call_id", "label"]].rename(columns={"label": "label_v2"})
    comp = v2_target_labels.merge(v4_target[["call_id", "label"]].rename(columns={"label": "label_final"}),
                                   on="call_id")
    ct = pd.crosstab(comp["label_v2"], comp["label_final"])
    print(ct.to_string())

    print("\nTRAIN10K_MERGE_COMPLETE")


if __name__ == "__main__":
    main()
