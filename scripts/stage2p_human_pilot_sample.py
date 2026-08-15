"""Human pilot(Hailey + 협업자)용 음성 프롬프트 설계 샘플 25건 추출.

대상: mismatch subset(gold_surface != gold_actual), N=2, customer_only 버전
(조건 A/B와 매칭 - 답지에 A/B 모델 예측을 같이 남기므로 문제지 발화 내용도
customer_only 윈도우로 통일).

gold_surface 정의(기존 관례 그대로): 조건 A(고객only+Text only, N=1) 예측.
nl_description 소스: stage2_windows_nl_v2.parquet (성별 기반 이원화 정규화,
Track A - 현재까지 검증을 통과한 버전. Track B(shrinkage)는 아직 채택 여부
미정이라 사용하지 않음).

층화: 환불요청 10 / 불만제기 5 / (배송확인+구매진행) 중립 10, 각 층 내부는
random sample(seed=42). 최종 25개는 순번이 층을 노출하지 않도록 seed로 셔플.

출력:
  - outputs/human_pilot_questions.md (문제지, 정답 정보 없음)
  - outputs/human_pilot_answers.md (답지, 별도 파일, 상단 경고 문구)
  - outputs/human_pilot_meta.csv (집계용 메타, 답지 성격이라 미리 안 봄)
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PRED_PATH = BASE_DIR / "outputs" / "stage2d_gpt_predictions.parquet"
GOLD_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_final.parquet"
WINDOWS_NL_PATH = BASE_DIR / "outputs" / "stage2_windows_nl_v2.parquet"
D04_PATH = BASE_DIR / "outputs" / "d04_dialog_index.parquet"

OUT_QUESTIONS = BASE_DIR / "outputs" / "human_pilot_questions.md"
OUT_ANSWERS = BASE_DIR / "outputs" / "human_pilot_answers.md"
OUT_META = BASE_DIR / "outputs" / "human_pilot_meta.csv"

SEED = 42
WINDOW_N = 2
WINDOW_VERSION = "customer_only"

STRATA = [
    ("환불요청", ["환불요청"], 10),
    ("불만제기", ["불만제기"], 5),
    ("중립(배송확인/구매진행)", ["배송확인", "구매진행"], 10),
]


def build_mismatch_ids(pred_gold: pd.DataFrame) -> set:
    surface = pred_gold[(pred_gold["condition"] == "A") & (pred_gold["n"] == 1)]
    return set(surface[surface["predicted_label"] != surface["actual_label"]]["call_id"])


def main():
    preds = pd.read_parquet(PRED_PATH)
    gold = pd.read_parquet(GOLD_PATH)[["call_id", "label"]].rename(columns={"label": "actual_label"})
    pred_gold = preds.merge(gold, on="call_id", how="left")

    mismatch_ids = build_mismatch_ids(pred_gold)

    windows = pd.read_parquet(WINDOWS_NL_PATH)
    win_sub = windows[
        (windows["window_n"] == WINDOW_N)
        & (windows["window_version"] == WINDOW_VERSION)
        & (windows["call_id"].isin(mismatch_ids))
    ].copy()

    # 콜별로 (1) 윈도우가 정확히 WINDOW_N개 발화로 꽉 채워져 있고
    # (2) nl_description이 전부 결측 아닌 경우만 후보로 인정.
    # (고객 유효 발화가 WINDOW_N개 미만인 콜은 윈도우가 덜 채워지므로 제외해야 함 -
    # 이전 버전은 이 체크가 없어서 발화 1개짜리 콜이 섞여 들어간 적이 있었음)
    group_sizes = win_sub.groupby("call_id").size()
    full_window_ids = set(group_sizes[group_sizes == WINDOW_N].index)
    nl_ok = win_sub.groupby("call_id")["nl_description"].apply(lambda s: s.notna().all())
    nl_ok_ids = set(nl_ok[nl_ok].index)
    valid_call_ids = full_window_ids & nl_ok_ids

    call_actual = gold.set_index("call_id")["actual_label"]

    rng = np.random.default_rng(SEED)
    selected = []
    actual_counts = {}

    for stratum_name, actual_labels, n_needed in STRATA:
        candidates = [
            cid for cid in valid_call_ids
            if cid in call_actual.index and call_actual.loc[cid] in actual_labels
        ]
        candidates = sorted(candidates)  # 정렬 후 셔플해야 재현 가능
        rng.shuffle(candidates)
        chosen = candidates[:n_needed]
        actual_counts[stratum_name] = len(chosen)
        for cid in chosen:
            selected.append({"call_id": cid, "stratum": stratum_name, "actual_label": call_actual.loc[cid]})

    order_rng = np.random.default_rng(SEED + 1)
    order = np.arange(len(selected))
    order_rng.shuffle(order)
    selected = [selected[i] for i in order]

    # === subcategory (있으면) ===
    subcat = pd.read_parquet(D04_PATH)[["call_id", "subcategory"]].drop_duplicates("call_id").set_index("call_id")

    # === N=2 조건 A/B 모델 예측 (있으면) ===
    ab_preds = pred_gold[(pred_gold["n"] == 2) & (pred_gold["condition"].isin(["A", "B"]))]
    ab_pivot = ab_preds.pivot_table(index="call_id", columns="condition", values="predicted_label", aggfunc="first")

    q_lines = []
    a_lines = ["⚠️ 판단 완료 전 열람 금지\n"]
    meta_rows = []

    for idx, item in enumerate(selected, start=1):
        cid = item["call_id"]
        g = win_sub[win_sub["call_id"] == cid].sort_values("dialog_idx")

        q_lines.append("-" * 32)
        q_lines.append(f"### 콜 {idx} (call_id: {cid}, N=2)")
        for i, row in enumerate(g.itertuples(), start=1):
            q_lines.append(f"발화{i}({row.speaker_group}): {row.text_clean}")
            q_lines.append(f"  [음성: {row.nl_description}]")
        q_lines.append("")
        q_lines.append("내 판단(음성 전, 텍스트만): ")
        q_lines.append("내 판단(음성 후): ")
        q_lines.append("-" * 32)
        q_lines.append("")

        surface_row = pred_gold[(pred_gold["call_id"] == cid) & (pred_gold["condition"] == "A") & (pred_gold["n"] == 1)]
        gold_surface = surface_row["predicted_label"].iloc[0] if not surface_row.empty else None
        gold_actual = item["actual_label"]
        subcat_val = subcat.loc[cid, "subcategory"] if cid in subcat.index else None
        a_pred = ab_pivot.loc[cid, "A"] if cid in ab_pivot.index and "A" in ab_pivot.columns else None
        b_pred = ab_pivot.loc[cid, "B"] if cid in ab_pivot.index and "B" in ab_pivot.columns else None

        a_lines.append("-" * 32)
        a_lines.append(f"### 콜 {idx} (call_id: {cid})")
        a_lines.append(f"gold_surface(조건A,N=1): {gold_surface}")
        a_lines.append(f"gold_actual: {gold_actual}")
        a_lines.append(f"subcategory: {subcat_val}")
        a_lines.append(f"N=2 조건A 예측: {a_pred}")
        a_lines.append(f"N=2 조건B 예측: {b_pred}")
        a_lines.append("-" * 32)
        a_lines.append("")

        meta_rows.append({
            "순번": idx, "call_id": cid, "gold_actual": gold_actual,
            "층": item["stratum"], "seed": SEED,
        })

    OUT_QUESTIONS.write_text("\n".join(q_lines), encoding="utf-8")
    OUT_ANSWERS.write_text("\n".join(a_lines), encoding="utf-8")
    pd.DataFrame(meta_rows).to_csv(OUT_META, index=False, encoding="utf-8-sig")

    print(f"저장: {OUT_QUESTIONS}")
    print(f"저장: {OUT_ANSWERS}")
    print(f"저장: {OUT_META}")
    print(f"\nseed = {SEED} (순서 셔플용 seed = {SEED + 1})")
    print("층별 실제 추출 개수:")
    for stratum_name, _, n_needed in STRATA:
        print(f"  {stratum_name}: 요청 {n_needed} / 추출 {actual_counts[stratum_name]}")


if __name__ == "__main__":
    main()
