"""게이트 on/off 트리거 6종 비교 (전부 gold 불필요, 배포 가능).
후보:
  1. shift: A_N2_pred != A_N5_pred
  2. trajectory: N in {1,2,3,5}(condition A) 예측 중 서로 다른 라벨 개수 >= 2
  3. text_entropy: train10k 텍스트로 학습한 LogReg 7-way 분류기의 predict_proba
     entropy가 train 상위 33%(train 기준 고정 컷) 이상
  4. combo: trajectory AND text_entropy (둘 다 불안정)
  5. shift_notshort: shift AND NOT short
  6. trajectory_notshort: trajectory AND NOT short

평가:
  A. mismatch(A_N2_pred != gold_actual) 대리 precision/recall - 전체 유효 추론셋(19,548건)
  B. 레이어 선택성(정보충분/정보부족/N-A) - 전체 유효 추론셋
  C/D. rescue/harm/net + 불만 recall/precision/F1/macro-F1 - gate_prompt 샘플(1,973건,
       이미 실행된 base/gate 예측 재사용, API 0)

출력: outputs/trigger_comparison.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
SHIFT_MISMATCH_PATH = BASE_DIR / "outputs" / "shift_mismatch_full.parquet"
INFOPOOR_PATH = BASE_DIR / "outputs" / "infopoor_tagged.csv"
STAGE2D_PRED_PATH = BASE_DIR / "outputs" / "stage2d_gpt_predictions.parquet"
PROBING_FEATURES_PATH = BASE_DIR / "outputs" / "probing_call_features.parquet"
TRAIN10K_LABELED_PATH = BASE_DIR / "outputs" / "train_10k_final_labeled.parquet"
TRAIN10K_POOL_PATH = BASE_DIR / "outputs" / "train10k_utterance_pool.parquet"
WINDOWS_NL_V2_PATH = BASE_DIR / "outputs" / "stage2_windows_nl_v2.parquet"
GATE_PRED_PATH = BASE_DIR / "outputs" / "gate_prompt_gpt_predictions.parquet"

OUT_PATH = BASE_DIR / "outputs" / "trigger_comparison.csv"
CATEGORY_ORDER = ["환불요청", "주문취소", "불만제기", "배송확인", "교환반품", "구매진행", "서비스이용"]
SEED = 42


# ============================================================
# 1) 전체 유효 추론셋(19,548건) 트리거 피처 구성
# ============================================================
def build_full_triggers():
    base = pd.read_parquet(SHIFT_MISMATCH_PATH)  # call_id, gold_actual, A_N2_pred, A_N5_pred, mismatch, shift
    print(f"기준셋(shift_mismatch_full): {len(base)}건")

    # --- trajectory: A조건 N=1,2,3,5 예측 pivot ---
    preds = pd.read_parquet(STAGE2D_PRED_PATH)
    a_pred = preds[preds["condition"] == "A"]
    piv = a_pred.pivot_table(index="call_id", columns="n", values="predicted_label", aggfunc="first")
    piv.columns = [f"A_N{c}_pred" for c in piv.columns]
    n_cols = [c for c in ["A_N1_pred", "A_N2_pred", "A_N3_pred", "A_N5_pred"] if c in piv.columns]
    print(f"trajectory용 N 컬럼: {n_cols}")
    traj_nunique = piv[n_cols].nunique(axis=1)
    piv["trajectory"] = traj_nunique >= 2
    base = base.merge(piv[["trajectory"]], on="call_id", how="left")
    n_no_traj = base["trajectory"].isna().sum()
    print(f"trajectory 결측(N=1/3 예측 없는 콜): {n_no_traj}건")

    # --- short: probing_call_features의 n2_total_tokens, 유효셋 기준 quantile(0.2) 컷 ---
    probing = pd.read_parquet(PROBING_FEATURES_PATH)[["call_id", "n2_total_tokens"]]
    base = base.merge(probing, on="call_id", how="left")
    short_threshold = base["n2_total_tokens"].quantile(0.2)
    base["short"] = base["n2_total_tokens"] <= short_threshold
    print(f"short 임계값(유효셋 n2_total_tokens 하위 20%): {short_threshold:.1f} 토큰")

    # --- infopoor 레이어(정보충분/정보부족/N-A), 기존 infopoor_tagged.csv 재사용 ---
    infopoor = pd.read_csv(INFOPOOR_PATH)[["call_id", "is_infopoor", "short"]].rename(
        columns={"short": "short_infopoor_orig"})
    base = base.merge(infopoor[["call_id", "is_infopoor"]], on="call_id", how="left")

    def layer_of(row):
        if not row["mismatch"]:
            return "N-A(이미정답)"
        return "정보부족" if row["is_infopoor"] else "정보충분"
    base["layer"] = base.apply(layer_of, axis=1)
    print(f"\n레이어 분포(전체 유효셋):\n{base['layer'].value_counts().to_string()}")

    # --- text_entropy: train10k 텍스트로 7-way LogReg 학습 -> 추론셋 텍스트에 적용 ---
    train_labeled = pd.read_parquet(TRAIN10K_LABELED_PATH)[["call_id", "gold_actual"]]
    train_pool = pd.read_parquet(TRAIN10K_POOL_PATH)
    train_cust_n2 = train_pool[train_pool["is_customer"] & train_pool["customer_rank"].between(1, 2)]
    train_cust_n2 = train_cust_n2.sort_values(["call_id", "dialog_idx"])
    train_text = train_cust_n2.groupby("call_id")["text_clean"].apply(
        lambda s: " ".join(str(t) for t in s)).reset_index().rename(columns={"text_clean": "text"})
    train_df = train_labeled.merge(train_text, on="call_id", how="inner")
    n_train_no_text = len(train_labeled) - len(train_df)
    print(f"\ntrain10k 텍스트 분류기 학습 데이터: {len(train_df)}건 (텍스트 없음 {n_train_no_text}건 제외)")

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, min_df=2)),
        ("logreg", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)),
    ])
    clf.fit(train_df["text"], train_df["gold_actual"])

    train_proba = clf.predict_proba(train_df["text"])
    train_entropy = -(train_proba * np.log(np.clip(train_proba, 1e-12, 1))).sum(axis=1)
    entropy_cut = np.quantile(train_entropy, 2 / 3)  # train 기준 상위 33% 컷, 추론셋엔 고정 적용
    print(f"text_entropy 임계값(train10k 자체 예측확률 entropy 상위 33% 컷, 고정): {entropy_cut:.4f}")

    windows = pd.read_parquet(WINDOWS_NL_V2_PATH)
    win2 = windows[(windows["window_n"] == 2) & (windows["window_version"] == "customer_only")]
    win2 = win2.sort_values(["call_id", "dialog_idx"])
    infer_text = win2.groupby("call_id")["text_clean"].apply(
        lambda s: " ".join(str(t) for t in s)).reset_index().rename(columns={"text_clean": "text"})

    base = base.merge(infer_text, on="call_id", how="left")
    n_no_text = base["text"].isna().sum()
    print(f"추론셋 중 N=2 customer_only 텍스트 없는 콜: {n_no_text}건 (text_entropy 트리거는 False 처리)")
    base["text"] = base["text"].fillna("")

    infer_proba = clf.predict_proba(base["text"])
    infer_entropy = -(infer_proba * np.log(np.clip(infer_proba, 1e-12, 1))).sum(axis=1)
    base["text_entropy_value"] = infer_entropy
    base["text_entropy"] = (infer_entropy >= entropy_cut) & (base["text"] != "")

    # --- 최종 6개 트리거 ---
    base["trigger_1_shift"] = base["shift"]
    base["trigger_2_trajectory"] = base["trajectory"].fillna(False)
    base["trigger_3_text_entropy"] = base["text_entropy"]
    base["trigger_4_combo"] = base["trigger_2_trajectory"] & base["trigger_3_text_entropy"]
    base["trigger_5_shift_notshort"] = base["trigger_1_shift"] & (~base["short"])
    base["trigger_6_trajectory_notshort"] = base["trigger_2_trajectory"] & (~base["short"])

    return base


TRIGGER_NAMES = {
    "trigger_1_shift": "1.shift",
    "trigger_2_trajectory": "2.trajectory(N궤적)",
    "trigger_3_text_entropy": "3.text_entropy",
    "trigger_4_combo": "4.combo(2&3)",
    "trigger_5_shift_notshort": "5.shift&~short",
    "trigger_6_trajectory_notshort": "6.trajectory&~short",
}


# ============================================================
# 2) A) mismatch 대리 성능, B) 레이어 선택성 - 전체 유효셋 기준
# ============================================================
def evaluate_full(full_df, trigger_col):
    y_mismatch = full_df["mismatch"]
    y_trigger = full_df[trigger_col]
    tp = (y_trigger & y_mismatch).sum()
    fp = (y_trigger & ~y_mismatch).sum()
    fn = (~y_trigger & y_mismatch).sum()
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    trigger_rate = y_trigger.mean()

    triggered = full_df[y_trigger]
    layer_dist = triggered["layer"].value_counts(normalize=True).to_dict()
    pct_jeongbo_chungbun = layer_dist.get("정보충분", 0.0)
    pct_jeongbo_buzok = layer_dist.get("정보부족", 0.0)
    pct_na = layer_dist.get("N-A(이미정답)", 0.0)

    na_total = (full_df["layer"] == "N-A(이미정답)").sum()
    na_triggered = ((full_df["layer"] == "N-A(이미정답)") & y_trigger).sum()
    na_exclusion_rate = 1 - (na_triggered / na_total) if na_total else float("nan")

    chungbun_mismatch_total = (full_df["layer"] == "정보충분").sum()
    chungbun_mismatch_covered = ((full_df["layer"] == "정보충분") & y_trigger).sum()
    chungbun_coverage = chungbun_mismatch_covered / chungbun_mismatch_total if chungbun_mismatch_total else float("nan")

    return {
        "mismatch_precision": precision, "mismatch_recall": recall,
        "trigger_rate_full": trigger_rate, "n_triggered_full": int(y_trigger.sum()),
        "pct_triggered_in_jeongbochungbun": pct_jeongbo_chungbun,
        "pct_triggered_in_jeongbobuzok": pct_jeongbo_buzok,
        "pct_triggered_in_NA": pct_na,
        "NA_exclusion_rate": na_exclusion_rate,
        "jeongbochungbun_coverage": chungbun_coverage,
    }


# ============================================================
# 3) C/D) rescue/harm/net + 불만지표 - gate_prompt 샘플(1,973건) 재사용
# ============================================================
def evaluate_sample(full_df, trigger_col):
    gate_pred = pd.read_parquet(GATE_PRED_PATH)
    base_p = gate_pred[gate_pred.variant == "base"][["call_id", "gold_actual", "predicted_label"]].rename(
        columns={"predicted_label": "pred_base"})
    gate_p = gate_pred[gate_pred.variant == "gate"][["call_id", "predicted_label"]].rename(
        columns={"predicted_label": "pred_gate"})
    merged = base_p.merge(gate_p, on="call_id").merge(
        full_df[["call_id", "layer", trigger_col]], on="call_id", how="left")
    merged[trigger_col] = merged[trigger_col].fillna(False)
    merged["pred_base"] = merged["pred_base"].fillna("PARSE_FAIL")
    merged["pred_gate"] = merged["pred_gate"].fillna("PARSE_FAIL")
    merged["pred_hybrid"] = np.where(merged[trigger_col], merged["pred_gate"], merged["pred_base"])

    base_correct = merged["pred_base"] == merged["gold_actual"]
    hybrid_correct = merged["pred_hybrid"] == merged["gold_actual"]
    rescue = ((~base_correct) & hybrid_correct)
    harm = (base_correct & (~hybrid_correct))

    net = int(rescue.sum()) - int(harm.sum())

    sub_chungbun = merged[merged["layer"] == "정보충분"]
    rescue_c = ((merged.loc[sub_chungbun.index, "pred_base"] != merged.loc[sub_chungbun.index, "gold_actual"]) &
                (merged.loc[sub_chungbun.index, "pred_hybrid"] == merged.loc[sub_chungbun.index, "gold_actual"]))
    harm_c = ((merged.loc[sub_chungbun.index, "pred_base"] == merged.loc[sub_chungbun.index, "gold_actual"]) &
              (merged.loc[sub_chungbun.index, "pred_hybrid"] != merged.loc[sub_chungbun.index, "gold_actual"]))
    net_chungbun = int(rescue_c.sum()) - int(harm_c.sum())

    sub_na = merged[merged["layer"] == "N-A(이미정답)"]
    harm_na = ((merged.loc[sub_na.index, "pred_base"] == merged.loc[sub_na.index, "gold_actual"]) &
               (merged.loc[sub_na.index, "pred_hybrid"] != merged.loc[sub_na.index, "gold_actual"])).sum()

    yt = (merged["gold_actual"] == "불만제기").astype(int)
    yp = (merged["pred_hybrid"] == "불만제기").astype(int)
    complaint_recall = recall_score(yt, yp, zero_division=0)
    complaint_precision = precision_score(yt, yp, zero_division=0)
    complaint_f1 = f1_score(yt, yp, zero_division=0)
    macro_f1 = f1_score(merged["gold_actual"], merged["pred_hybrid"], labels=CATEGORY_ORDER,
                         average="macro", zero_division=0)

    return {
        "sample_n": len(merged), "rescue": int(rescue.sum()), "harm": int(harm.sum()), "net": net,
        "net_jeongbochungbun_layer": net_chungbun, "harm_NA_layer": int(harm_na),
        "complaint_recall": complaint_recall, "complaint_precision": complaint_precision,
        "complaint_f1": complaint_f1, "macro_f1": macro_f1,
    }


def main():
    full_df = build_full_triggers()

    rows = []
    for col, name in TRIGGER_NAMES.items():
        row = {"trigger": name}
        row.update(evaluate_full(full_df, col))
        row.update(evaluate_sample(full_df, col))
        rows.append(row)

    res = pd.DataFrame(rows)
    res.to_csv(OUT_PATH, index=False)

    print("\n" + "=" * 100)
    print("트리거 비교 결과")
    print("=" * 100)
    display_cols = ["trigger", "mismatch_precision", "mismatch_recall", "trigger_rate_full",
                     "pct_triggered_in_jeongbochungbun", "pct_triggered_in_jeongbobuzok",
                     "pct_triggered_in_NA", "NA_exclusion_rate", "jeongbochungbun_coverage"]
    print(res[display_cols].to_string(index=False))
    print()
    display_cols2 = ["trigger", "rescue", "harm", "net", "net_jeongbochungbun_layer", "harm_NA_layer",
                      "complaint_recall", "complaint_precision", "complaint_f1", "macro_f1"]
    print(res[display_cols2].to_string(index=False))

    print(f"\n저장: {OUT_PATH}")

    # === shift(1) 기준 벤치마크 ===
    shift_row = res[res["trigger"] == "1.shift"].iloc[0]
    print("\n" + "=" * 60)
    print("요약")
    print("=" * 60)
    print(f"1. shift 기준: mismatch precision={shift_row['mismatch_precision']:.4f}, "
          f"recall={shift_row['mismatch_recall']:.4f}, N-A제외율={shift_row['NA_exclusion_rate']:.4f}, "
          f"net(sample)={shift_row['net']:.0f}, harm_NA={shift_row['harm_NA_layer']:.0f}")

    best_net = res.loc[res["net"].idxmax()]
    print(f"2. 최고 net(sample) 트리거: {best_net['trigger']} (net={best_net['net']:.0f}, "
          f"shift 대비 {best_net['net']-shift_row['net']:+.0f})")

    beats_shift = res[(res["trigger"] != "1.shift") & (res["net"] > shift_row["net"]) &
                       (res["harm_NA_layer"] <= shift_row["harm_NA_layer"] * 1.1) &
                       (res["mismatch_precision"] >= shift_row["mismatch_precision"] - 0.05)]
    if len(beats_shift):
        print(f"3. shift를 넘는 트리거 있음: {beats_shift['trigger'].tolist()} "
              f"(net 개선 + N-A차단 유지 + precision 유지)")
    else:
        print("3. shift를 넘는 트리거 없음 - net/N-A차단/precision을 동시에 개선하는 rule 트리거 미발견")

    print(f"4. 레이어 선택성 최고(정보충분 집중도): " +
          res.loc[res["pct_triggered_in_jeongbochungbun"].idxmax(), "trigger"] +
          f" ({res['pct_triggered_in_jeongbochungbun'].max()*100:.1f}%)")

    if len(beats_shift) == 0:
        print("5. 결론: rule 기반 트리거는 shift가 상한선 - 정보충분 mismatch 서브셋을 더 정교하게 "
              "식별하려면 rule이 아니라 학습된(gold 활용) 분류기가 필요함")
    else:
        print(f"5. 결론: {beats_shift.iloc[0]['trigger']}가 배포 가능한 shift 대체 후보")

    print("\nTRIGGER_COMPARISON_COMPLETE")


if __name__ == "__main__":
    main()
