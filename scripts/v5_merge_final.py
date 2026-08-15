"""v5 결과를 병합해 최종본을 만든다.

병합 규칙:
1. v4에서 has_complaint=True였던 452건 -> label='불만제기'로 확정
2. v4에서 has_complaint=False였던 11,127건 -> v5 재분류 결과(label_v5)로 교체
   (파싱 실패(NaN)면 v4의 label_v4로 폴백)
3. 나머지(원래부터 환불요청/배송확인이 아니었던 8,268건) -> v2 라벨 유지

출력: outputs/gold_actual_batch1_final.parquet (v4 기반 이전 버전을 덮어씀)
추가로 v2 / v4-final(참고용, 파일로 남기지 않고 메모리상 재구성) / v5-final
3버전 라벨 분포를 비교 출력한다.
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
V2_PATH = BASE_DIR / "outputs" / "gold_actual_batch1.parquet"
V4_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_v4.parquet"
V5_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_v5.parquet"
FINAL_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_final.parquet"

CATEGORY_ORDER = ["환불요청", "주문취소", "불만제기", "배송확인", "교환반품", "구매진행", "서비스이용"]


def build_v4_final_labels(v2: pd.DataFrame, v4: pd.DataFrame) -> pd.Series:
    """비교표 용도로만 v4 기반 최종 라벨을 메모리상에서 재구성 (파일 재생성 없음)."""
    v4_ids = set(v4["call_id"])
    kept = v2[~v2["call_id"].isin(v4_ids)][["call_id", "label"]]
    v4_labels = v4[["call_id", "label_v4"]].rename(columns={"label_v4": "label"})
    v4_labels["label"] = v4_labels["label"].fillna(
        v4_labels["call_id"].map(v2.set_index("call_id")["label"])
    )
    combined = pd.concat([kept, v4_labels], ignore_index=True)
    return combined.set_index("call_id")["label"]


def main():
    v2 = pd.read_parquet(V2_PATH)
    v4 = pd.read_parquet(V4_PATH)
    v5 = pd.read_parquet(V5_PATH)

    complaint_rows = v4[v4["has_complaint"] == True][["call_id"]].copy()  # noqa: E712
    complaint_rows["label"] = "불만제기"

    non_complaint = v4[v4["has_complaint"] == False][["call_id", "label_v4"]].copy()  # noqa: E712
    non_complaint = non_complaint.merge(v5[["call_id", "label_v5"]], on="call_id", how="left")
    n_fallback = non_complaint["label_v5"].isna().sum()
    if n_fallback:
        print(f"경고: label_v5 파싱 실패로 v4 라벨로 폴백한 콜 {n_fallback}건")
    non_complaint["label"] = non_complaint["label_v5"].fillna(non_complaint["label_v4"])

    v5_target = pd.concat(
        [complaint_rows[["call_id", "label"]], non_complaint[["call_id", "label"]]],
        ignore_index=True,
    )
    assert len(v5_target) == len(v4), f"대상 건수 불일치: v4={len(v4)}, v5_target={len(v5_target)}"

    v5_ids = set(v5_target["call_id"])
    kept = v2[~v2["call_id"].isin(v5_ids)][["call_id", "label", "prompt_version", "model"]].copy()
    kept["source"] = "v2"

    v5_target["source"] = v5_target["call_id"].isin(complaint_rows["call_id"]).map(
        {True: "v4(has_complaint)", False: "v5"}
    )
    v5_target["prompt_version"] = "v4+v5"
    v5_target["model"] = "gpt-4.1-mini"

    final = pd.concat(
        [kept, v5_target[["call_id", "label", "prompt_version", "model", "source"]]],
        ignore_index=True,
    )
    assert len(final) == len(v2), f"행 수 불일치: v2={len(v2)}, final={len(final)}"
    assert final["call_id"].is_unique, "call_id 중복 발생"

    final.to_parquet(FINAL_PATH, index=False)
    print(f"저장: {FINAL_PATH} ({len(final)}행) - v4 기반 이전 최종본을 덮어씀")
    print(f"불만제기 확정: {len(complaint_rows)}건, v5 재분류: {len(non_complaint)}건, "
          f"v2 유지: {len(kept)}건")

    # === 3버전 비교표 ===
    v4_final_labels = build_v4_final_labels(v2, v4)

    def dist(labels: pd.Series) -> pd.Series:
        return labels.value_counts().reindex(CATEGORY_ORDER, fill_value=0)

    v2_dist = dist(v2.set_index("call_id")["label"])
    v4_dist = dist(v4_final_labels)
    v5_dist = dist(final.set_index("call_id")["label"])

    table = pd.DataFrame({"v2": v2_dist, "v4-final": v4_dist, "v5-final": v5_dist})
    table["v2 비율%"] = (table["v2"] / table["v2"].sum() * 100).round(2)
    table["v4 비율%"] = (table["v4-final"] / table["v4-final"].sum() * 100).round(2)
    table["v5 비율%"] = (table["v5-final"] / table["v5-final"].sum() * 100).round(2)

    print("\n=== v2 vs v4-final vs v5-final 라벨 분포 비교 (7개 카테고리) ===")
    print(table.to_string())

    print(f"\n불만제기 추이: v2 {v2_dist['불만제기']}건 -> "
          f"v4-final {v4_dist['불만제기']}건 -> v5-final {v5_dist['불만제기']}건 (동일, v5는 재작업 안 함)")

    print("\n=== v4 -> v5 재분류 매트릭스 (has_complaint=False 11,127건 대상) ===")
    ct = pd.crosstab(non_complaint["label_v4"], non_complaint["label"])
    print(ct.to_string())
    n_changed = (non_complaint["label_v4"] != non_complaint["label"]).sum()
    print(f"\nv4->v5 라벨이 바뀐 콜: {n_changed}/{len(non_complaint)}건")


if __name__ == "__main__":
    main()
