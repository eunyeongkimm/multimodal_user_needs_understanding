"""프롬프트 파일럿용 샘플 A(150, 층화)/B(200, 자연분포 mismatch) 추출.

- N=2, customer_only 윈도우, Track A(성별 이원화 재정규화, stage2_windows_nl_v2.parquet)
  음성서술 사용.
- 윈도우가 정확히 2개 발화로 채워진 콜만 후보(과거 발견된 버그 재발 방지).
- 샘플 A/B call_id 중복 없음.
- seed 고정(42), 콘솔에 출력.

출력: outputs/prompt_pilot_sample_a.csv, outputs/prompt_pilot_sample_b.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
GOLD_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_final.parquet"
WINDOWS_NL_PATH = BASE_DIR / "outputs" / "stage2_windows_nl_v2.parquet"
FEATURES_PATH = BASE_DIR / "outputs" / "probing_call_features.parquet"

OUT_A = BASE_DIR / "outputs" / "prompt_pilot_sample_a.csv"
OUT_B = BASE_DIR / "outputs" / "prompt_pilot_sample_b.csv"

SEED = 42
WINDOW_N = 2
WINDOW_VERSION = "customer_only"

STRATA_A = [
    ("불만제기", ["불만제기"], 50),
    ("환불요청_주문취소", ["환불요청", "주문취소"], 40),
    ("중립(배송/구매/서비스/교환)", ["배송확인", "구매진행", "서비스이용", "교환반품"], 60),
]
N_B = 200


def diverse_or_random_sample(candidate_ids, n_needed, seed):
    rng = np.random.default_rng(seed)
    ids = sorted(candidate_ids)
    rng.shuffle(ids)
    return ids[:n_needed]


def main():
    gold = pd.read_parquet(GOLD_PATH)[["call_id", "label"]].rename(columns={"label": "gold_actual"})
    windows = pd.read_parquet(WINDOWS_NL_PATH)
    feats = pd.read_parquet(FEATURES_PATH)[["call_id", "A_N2_pred"]]

    win_sub = windows[(windows["window_n"] == WINDOW_N) & (windows["window_version"] == WINDOW_VERSION)]
    group_sizes = win_sub.groupby("call_id").size()
    full_ids = set(group_sizes[group_sizes == WINDOW_N].index)
    nl_ok = win_sub.groupby("call_id")["nl_description"].apply(lambda s: s.notna().all())
    nl_ok_ids = set(nl_ok[nl_ok].index)
    valid_ids = full_ids & nl_ok_ids
    print(f"윈도우 완전 + nl_description 유효 콜: {len(valid_ids)} / 전체 {len(gold)}")

    gold_valid = gold[gold["call_id"].isin(valid_ids)]

    # === 샘플 A: 층화 ===
    selected_a = []
    a_counts = {}
    for stratum_name, labels, n_needed in STRATA_A:
        candidates = gold_valid[gold_valid["gold_actual"].isin(labels)]["call_id"].tolist()
        chosen = diverse_or_random_sample(candidates, n_needed, SEED)
        a_counts[stratum_name] = len(chosen)
        for cid in chosen:
            selected_a.append({"call_id": cid, "층": stratum_name})

    order_rng = np.random.default_rng(SEED + 1)
    order = np.arange(len(selected_a))
    order_rng.shuffle(order)
    selected_a = [selected_a[i] for i in order]
    sample_a = pd.DataFrame(selected_a).merge(gold_valid, on="call_id", how="left")

    print(f"\n=== 샘플 A (seed={SEED}, 순서셔플 seed={SEED+1}) ===")
    for stratum_name, _, n_needed in STRATA_A:
        print(f"  {stratum_name}: 요청 {n_needed} / 추출 {a_counts[stratum_name]}")
    print(f"  총 {len(sample_a)}건")

    # === 샘플 B: mismatch subset(A_N2_pred notna & != gold_actual), 샘플A 제외, natural random ===
    gold_valid_b = gold_valid.merge(feats, on="call_id", how="left")
    mismatch_pool = gold_valid_b[
        gold_valid_b["A_N2_pred"].notna()
        & (gold_valid_b["A_N2_pred"] != gold_valid_b["gold_actual"])
        & (~gold_valid_b["call_id"].isin(sample_a["call_id"]))
    ]
    print(f"\nmismatch subset(A_N2 결측 제외, 샘플A 제외) 후보: {len(mismatch_pool)}건")

    chosen_b_ids = diverse_or_random_sample(mismatch_pool["call_id"].tolist(), N_B, SEED + 2)
    sample_b = mismatch_pool[mismatch_pool["call_id"].isin(chosen_b_ids)].copy()
    sample_b = sample_b.set_index("call_id").loc[chosen_b_ids].reset_index()
    sample_b["is_mismatch"] = True  # 정의상 전부 True

    print(f"\n=== 샘플 B (seed={SEED+2}) ===")
    print(f"  총 {len(sample_b)}건")
    print("  gold_actual 자연분포:")
    print(sample_b["gold_actual"].value_counts().to_string())

    overlap = set(sample_a["call_id"]) & set(sample_b["call_id"])
    print(f"\n샘플 A/B call_id 중복: {len(overlap)}건 (0이어야 정상)")

    sample_a.to_csv(OUT_A, index=False, encoding="utf-8-sig")
    sample_b.to_csv(OUT_B, index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT_A}")
    print(f"저장: {OUT_B}")


if __name__ == "__main__":
    main()
