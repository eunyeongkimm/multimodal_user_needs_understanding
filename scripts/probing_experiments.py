"""음성 6피처 probing 실험 A~F. 텍스트 미사용, 음성 raw 피처(콜 단위 mean/std)만 사용.
모든 판정은 5-fold StratifiedKFold CV 기준 (train 정확도 사용 금지).
스케일링은 파이프라인 내부에서 fold별로 fit (데이터 누수 방지).

스코프 2개(둘 다 실행, 핵심 판정은 scope_n2 우선):
  - n2: 첫 2발화(overall_rank<=2) - 게이트 실작동 시점
  - pool5: stage2_acoustic_features.parquet 전체(N<=5 pool, "콜 전체" 아님)

mismatch subset 정의(이 실험 전체에서 일관 사용): A_N2_예측 != gold_actual.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE_DIR / "outputs" / "probing_call_features.parquet"

OUT_SUMMARY = BASE_DIR / "outputs" / "probing_summary.txt"
OUT_IMPORTANCE = BASE_DIR / "outputs" / "probing_feature_importance.csv"
OUT_THRESHOLD_SWEEP = BASE_DIR / "outputs" / "probing_gate_threshold_sweep.csv"
OUT_INFOPOOR_TAGGED = BASE_DIR / "outputs" / "infopoor_tagged.csv"
OUT_LAYER_DELTA = BASE_DIR / "outputs" / "probing_layer_acoustic_delta.csv"

PRIMARY_METRICS = ["f0_mean", "f0_std", "e_mean_db", "e_std_db", "r_gold", "s_top_db30"]
SCOPES = ["n2", "pool5"]
SEED = 42
N_SPLITS = 5
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

log_lines = []


def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


def scope_cols(scope):
    return [f"{scope}_{m}_{stat}" for m in PRIMARY_METRICS for stat in ["mean", "std"]]


def get_models(kind: str, y_for_weight=None):
    """kind: 'multiclass' or 'binary'. XGB는 sample_weight로 균형 처리."""
    models = {
        "Dummy(most_frequent)": DummyClassifier(strategy="most_frequent"),
        "Dummy(stratified)": DummyClassifier(strategy="stratified", random_state=SEED),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED)),
        ]),
    }
    if kind == "binary":
        n_pos = (y_for_weight == 1).sum()
        n_neg = (y_for_weight == 0).sum()
        spw = n_neg / max(n_pos, 1)
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=4, eval_metric="logloss",
            scale_pos_weight=spw, random_state=SEED, verbosity=0,
        )
    else:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=4, eval_metric="mlogloss",
            random_state=SEED, verbosity=0,
        )
    return models


def fit_params_for(name, model, X, y):
    if name == "XGBoost":
        return {"sample_weight": compute_sample_weight("balanced", y)}
    return {}


# ============================================================
# 실험 A: 7-way 음성 단독 예측
# ============================================================
def experiment_A(df):
    log("\n" + "=" * 70)
    log("실험 A: 7-way 음성 단독 예측 (신호 존재 여부)")
    log("=" * 70)
    y_raw = df["gold_actual"]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    results = {}
    for scope in SCOPES:
        X = df[scope_cols(scope)].values
        log(f"\n--- scope_{scope} ---")
        models = get_models("multiclass")
        scope_res = {}
        for name, model in models.items():
            fp = fit_params_for(name, model, X, y)
            scores = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "f1_macro"],
                                     params=fp if fp else None)
            acc_m, acc_s = scores["test_accuracy"].mean(), scores["test_accuracy"].std()
            f1_m, f1_s = scores["test_f1_macro"].mean(), scores["test_f1_macro"].std()
            scope_res[name] = {"acc_mean": acc_m, "acc_std": acc_s, "f1_mean": f1_m, "f1_std": f1_s}
            log(f"  {name:22s}: accuracy={acc_m:.4f}±{acc_s:.4f}  macro-F1={f1_m:.4f}±{f1_s:.4f}")

        dummy_max = max(scope_res["Dummy(most_frequent)"]["f1_mean"], scope_res["Dummy(stratified)"]["f1_mean"])
        logreg_gain = scope_res["LogisticRegression"]["f1_mean"] - dummy_max
        xgb_gain = scope_res["XGBoost"]["f1_mean"] - dummy_max
        log(f"  -> LogReg macro-F1 - Dummy최댓값 = {logreg_gain:+.4f} "
            f"({'신호 있음(>=0.05)' if logreg_gain >= 0.05 else '신호 약함(<0.05)'})")
        log(f"  -> XGB macro-F1 - Dummy최댓값 = {xgb_gain:+.4f} "
            f"({'신호 있음(>=0.05)' if xgb_gain >= 0.05 else '신호 약함(<0.05)'})")
        scope_res["_dummy_max_f1"] = dummy_max
        scope_res["_logreg_gain"] = logreg_gain
        scope_res["_xgb_gain"] = xgb_gain
        results[scope] = scope_res
    return results


# ============================================================
# 실험 B: 불만제기 이진분류
# ============================================================
def experiment_B(df):
    log("\n" + "=" * 70)
    log("실험 B (핵심): 불만제기 이진분류")
    log("=" * 70)
    y = df["is_complaint"].values
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    results = {}
    for scope in SCOPES:
        X = df[scope_cols(scope)].values
        log(f"\n--- scope_{scope} ---")
        models = get_models("binary", y)
        scope_res = {}
        for name, model in models.items():
            fp = fit_params_for(name, model, X, y)
            scores = cross_validate(model, X, y, cv=cv,
                                     scoring=["roc_auc", "f1", "recall", "precision"],
                                     params=fp if fp else None)
            auc_m, auc_s = scores["test_roc_auc"].mean(), scores["test_roc_auc"].std()
            f1_m = scores["test_f1"].mean()
            rec_m = scores["test_recall"].mean()
            prec_m = scores["test_precision"].mean()
            scope_res[name] = {"auc_mean": auc_m, "auc_std": auc_s, "f1_mean": f1_m,
                                "recall_mean": rec_m, "precision_mean": prec_m}
            log(f"  {name:22s}: AUC={auc_m:.4f}±{auc_s:.4f}  F1={f1_m:.4f}  "
                f"recall={rec_m:.4f}  precision={prec_m:.4f}")

        for name in ["LogisticRegression", "XGBoost"]:
            auc = scope_res[name]["auc_mean"]
            level = ("강함(0.65+)" if auc >= 0.65 else "뚜렷(0.6+)" if auc >= 0.6
                     else "약함(<0.55)" if auc < 0.55 else "경계(0.55~0.6)")
            log(f"  -> {name} AUC={auc:.4f} -> {level}")
        results[scope] = scope_res
    return results


# ============================================================
# 실험 C: 피처 중요도
# ============================================================
def experiment_C(df):
    log("\n" + "=" * 70)
    log("실험 C: 피처 중요도 (XGB importance + LogReg 계수)")
    log("=" * 70)
    rows = []
    for scope in SCOPES:
        cols = scope_cols(scope)
        X = df[cols].values

        # 7-way
        y_multi = LabelEncoder().fit_transform(df["gold_actual"])
        xgb_multi = XGBClassifier(n_estimators=200, max_depth=4, eval_metric="mlogloss",
                                   random_state=SEED, verbosity=0)
        xgb_multi.fit(X, y_multi, sample_weight=compute_sample_weight("balanced", y_multi))
        scaler = StandardScaler().fit(X)
        lr_multi = LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED)
        lr_multi.fit(scaler.transform(X), y_multi)

        for i, col in enumerate(cols):
            rows.append({
                "scope": scope, "task": "7way", "feature": col,
                "xgb_importance": xgb_multi.feature_importances_[i],
                "logreg_coef_abs_mean": np.abs(lr_multi.coef_[:, i]).mean(),
            })

        # 불만 이진
        y_bin = df["is_complaint"].values
        n_pos, n_neg = (y_bin == 1).sum(), (y_bin == 0).sum()
        xgb_bin = XGBClassifier(n_estimators=200, max_depth=4, eval_metric="logloss",
                                 scale_pos_weight=n_neg / max(n_pos, 1), random_state=SEED, verbosity=0)
        xgb_bin.fit(X, y_bin)
        lr_bin = LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED)
        lr_bin.fit(scaler.transform(X), y_bin)

        for i, col in enumerate(cols):
            rows.append({
                "scope": scope, "task": "complaint_binary", "feature": col,
                "xgb_importance": xgb_bin.feature_importances_[i],
                "logreg_coef_signed": lr_bin.coef_[0, i],
            })

        top_bin = sorted(zip(cols, xgb_bin.feature_importances_), key=lambda x: -x[1])[:3]
        log(f"\n--- scope_{scope}, 불만이진 XGB 중요도 top3 ---")
        for col, imp in top_bin:
            log(f"  {col}: {imp:.4f}")

    imp_df = pd.DataFrame(rows)
    imp_df.to_csv(OUT_IMPORTANCE, index=False, encoding="utf-8-sig")
    log(f"\n저장: {OUT_IMPORTANCE}")
    return imp_df


# ============================================================
# 실험 D: 게이트 유효성 (text_correct vs text_wrong)
# ============================================================
def experiment_D(df):
    log("\n" + "=" * 70)
    log("실험 D: 게이트 유효성 - 텍스트 잔차(text_wrong)에서의 음성 신호")
    log("=" * 70)
    text_correct = (df["A_N2_pred"] == df["gold_actual"]).values
    text_wrong = ~text_correct
    y = df["is_complaint"].values
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    results = {}
    for scope in SCOPES:
        X = df[scope_cols(scope)].values
        log(f"\n--- scope_{scope} (text_correct={text_correct.sum()}건, text_wrong={text_wrong.sum()}건) ---")
        scope_res = {}
        for name, model in [
            ("LogisticRegression", Pipeline([("scaler", StandardScaler()),
                                              ("clf", LogisticRegression(class_weight="balanced",
                                                                          max_iter=5000, random_state=SEED))])),
            ("XGBoost", XGBClassifier(n_estimators=200, max_depth=4, eval_metric="logloss",
                                       scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
                                       random_state=SEED, verbosity=0)),
        ]:
            proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
            auc_full = roc_auc_score(y, proba)
            auc_wrong = (roc_auc_score(y[text_wrong], proba[text_wrong])
                         if text_wrong.sum() > 1 and len(np.unique(y[text_wrong])) > 1 else float("nan"))
            auc_correct = (roc_auc_score(y[text_correct], proba[text_correct])
                           if text_correct.sum() > 1 and len(np.unique(y[text_correct])) > 1 else float("nan"))
            retained_pct = (auc_wrong / auc_full * 100) if auc_full else float("nan")
            scope_res[name] = {"auc_full": auc_full, "auc_text_wrong": auc_wrong,
                                "auc_text_correct": auc_correct, "retained_pct": retained_pct}
            gate_verdict = ("게이트 강함(유지)" if retained_pct >= 80 else
                            "게이트 약함(급락)" if retained_pct < 50 else "게이트 보통")
            log(f"  {name:22s}: AUC 전체={auc_full:.4f}  text_wrong={auc_wrong:.4f}  "
                f"text_correct={auc_correct:.4f}  (wrong/전체={retained_pct:.1f}%) -> {gate_verdict}")
        results[scope] = scope_res

    # 텍스트가 놓친 불만(gold=불만, A_N2!=불만)의 음성 프로파일
    log("\n--- 텍스트가 놓친 불만 콜의 음성 프로파일 (scope_n2, 6개 평균값) ---")
    missed = df[(df["gold_actual"] == "불만제기") & (df["A_N2_pred"] != "불만제기")]
    caught = df[(df["gold_actual"] == "불만제기") & (df["A_N2_pred"] == "불만제기")]
    log(f"텍스트가 놓친 불만(missed): {len(missed)}건, 텍스트가 잡은 불만(caught): {len(caught)}건")
    profile_rows = []
    for m in PRIMARY_METRICS:
        col = f"n2_{m}_mean"
        profile_rows.append({
            "feature": col,
            "missed_mean": missed[col].mean() if len(missed) else float("nan"),
            "caught_mean": caught[col].mean() if len(caught) else float("nan"),
            "전체_mean": df[col].mean(),
        })
    profile_df = pd.DataFrame(profile_rows)
    log(profile_df.to_string(index=False))

    return results, profile_df


# ============================================================
# 실험 E: 게이트 임계값 스윕
# ============================================================
def experiment_E(df):
    log("\n" + "=" * 70)
    log("실험 E: 게이트 임계값(threshold) 시뮬레이션 (mismatch subset)")
    log("=" * 70)
    mismatch = df[df["A_N2_pred"] != df["gold_actual"]].copy()
    log(f"mismatch subset(A_N2!=gold_actual): {len(mismatch)}건")

    y_full = df["is_complaint"].values
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    rows = []
    for scope in SCOPES:
        X_full = df[scope_cols(scope)].values
        model = Pipeline([("scaler", StandardScaler()),
                           ("clf", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED))])
        proba_full = cross_val_predict(model, X_full, y_full, cv=cv, method="predict_proba")[:, 1]
        df_proba = df[["call_id"]].copy()
        df_proba["voice_proba"] = proba_full
        mm = mismatch.merge(df_proba, on="call_id", how="left")

        for t in THRESHOLDS:
            gated_pred = np.where(mm["voice_proba"].values > t, "불만제기", mm["A_N2_pred"].values)
            acc = accuracy_score(mm["gold_actual"], gated_pred)
            f1_macro = f1_score(mm["gold_actual"], gated_pred, average="macro", zero_division=0)
            is_gold_complaint = (mm["gold_actual"] == "불만제기").values
            is_pred_complaint = (gated_pred == "불만제기")
            rec = recall_score(is_gold_complaint, is_pred_complaint, zero_division=0)
            prec = precision_score(is_gold_complaint, is_pred_complaint, zero_division=0)
            rows.append({"scope": scope, "threshold": t, "accuracy": acc, "macro_f1": f1_macro,
                         "complaint_recall": rec, "complaint_precision": prec})

    sweep_df = pd.DataFrame(rows)
    log(sweep_df.to_string(index=False))
    sweep_df.to_csv(OUT_THRESHOLD_SWEEP, index=False, encoding="utf-8-sig")
    log(f"\n저장: {OUT_THRESHOLD_SWEEP}")

    # 순이득 구간 판정: baseline(t 미적용, 즉 gate 없이 A_N2 그대로) 대비
    verdicts = {}
    for scope in SCOPES:
        base_recall = 0.0  # mismatch subset은 정의상 A_N2가 전부 틀렸으므로 gate 없으면 불만 recall=0
        sub = sweep_df[sweep_df["scope"] == scope]
        base_precision_ref = sub["complaint_precision"].max()  # 참고용 상한
        gains = sub[(sub["complaint_recall"] > base_recall)]
        good_t = gains[gains["complaint_precision"] >= gains["complaint_precision"].max() * 0.8]
        verdicts[scope] = good_t["threshold"].tolist()
        log(f"scope_{scope}: recall 개선되면서 precision 크게 안 깨지는 임계값 구간 = {good_t['threshold'].tolist()}")

    return sweep_df, verdicts


# ============================================================
# 실험 F: 정보부족형 식별 + 층화
# ============================================================
def experiment_F(df):
    log("\n" + "=" * 70)
    log("실험 F: 정보부족형 식별 + 층화 (조건부 음향 효과)")
    log("=" * 70)

    has_n5 = "A_N5_pred" in df.columns and df["A_N5_pred"].notna().any()
    if not has_n5:
        log("경고: A_N5_pred 없음 -> unlock 신호 skip")
        unlock = pd.Series(False, index=df.index)
    else:
        unlock = (df["A_N2_pred"] != df["gold_actual"]) & (df["A_N5_pred"] == df["gold_actual"])

    shift = (df["A_N2_pred"] != df["A_N5_pred"]) if has_n5 else pd.Series(False, index=df.index)
    short_threshold = df["n2_total_tokens"].quantile(0.2)
    short = df["n2_total_tokens"] <= short_threshold
    log(f"short 판정 기준(전체 콜 n2_total_tokens 하위 20% 분위값): {short_threshold:.1f} 토큰")

    is_infopoor = unlock | (shift & short)

    mismatch_mask = df["A_N2_pred"] != df["gold_actual"]
    mm = df[mismatch_mask].copy()
    mm["is_infopoor"] = is_infopoor[mismatch_mask].values
    mm["unlock"] = unlock[mismatch_mask].values
    mm["shift"] = shift[mismatch_mask].values
    mm["short"] = short[mismatch_mask].values
    mm["layer"] = np.where(mm["is_infopoor"], "정보부족(infopoor)", "정보충분")

    log(f"\nmismatch subset: {len(mm)}건")
    log(mm["layer"].value_counts().to_string())
    log("\n층별 gold_actual 분포:")
    log(pd.crosstab(mm["layer"], mm["gold_actual"]).to_string())

    tagged_cols = ["call_id", "gold_actual", "A_N2_pred", "A_N5_pred" if has_n5 else "A_N2_pred",
                   "unlock", "shift", "short", "is_infopoor", "layer"]
    tagged_cols = list(dict.fromkeys(tagged_cols))
    mm[tagged_cols].to_csv(OUT_INFOPOOR_TAGGED, index=False, encoding="utf-8-sig")
    log(f"\n저장: {OUT_INFOPOOR_TAGGED}")

    # === 층별 음향 델타: (A_N2==gold) vs (B_N2==gold) ===
    log("\n--- 층별 음향 델타 (주의: mismatch subset 내에서는 A_N2==gold이 정의상 항상 0%) ---")
    delta_rows = []
    for layer_name, sub in [("전체(mismatch 전체)", mm), ("정보충분", mm[~mm["is_infopoor"]]),
                            ("정보부족(infopoor)", mm[mm["is_infopoor"]])]:
        if len(sub) == 0:
            continue
        a_rate = (sub["A_N2_pred"] == sub["gold_actual"]).mean()
        b_rate = (sub["B_N2_pred"] == sub["gold_actual"]).mean()
        delta_rows.append({"layer": layer_name, "n": len(sub), "A_N2_match_rate": a_rate,
                            "B_N2_match_rate": b_rate, "delta(B-A)": b_rate - a_rate})
    delta_df = pd.DataFrame(delta_rows)
    log(delta_df.to_string(index=False))
    delta_df.to_csv(OUT_LAYER_DELTA, index=False, encoding="utf-8-sig")
    log(f"\n저장: {OUT_LAYER_DELTA}")

    sufficient_delta = delta_df[delta_df["layer"] == "정보충분"]["delta(B-A)"].iloc[0] if len(delta_df) > 1 else None
    overall_delta = delta_df[delta_df["layer"] == "전체(mismatch 전체)"]["delta(B-A)"].iloc[0]
    infopoor_delta = delta_df[delta_df["layer"] == "정보부족(infopoor)"]["delta(B-A)"].iloc[0] if len(delta_df) > 2 else None
    if sufficient_delta is not None and infopoor_delta is not None:
        ordering_ok = sufficient_delta > overall_delta > infopoor_delta
        log(f"\n정보충분({sufficient_delta:.4f}) > 전체({overall_delta:.4f}) > infopoor({infopoor_delta:.4f}) "
            f"= {ordering_ok}")

    # === 정보충분층에서 실험 B(불만이진) 재실행 ===
    log("\n--- 정보충분층에서 불만이진 AUC 재측정 (vs 전체 population AUC) ---")
    sufficient_ids = set(mm[~mm["is_infopoor"]]["call_id"])
    layer_auc = {}
    for scope in SCOPES:
        sub_df = df[df["call_id"].isin(sufficient_ids)]
        if sub_df["is_complaint"].nunique() < 2 or len(sub_df) < 20:
            log(f"scope_{scope}: 정보충분층 표본 부족/단일클래스 -> skip")
            continue
        X = sub_df[scope_cols(scope)].values
        y = sub_df["is_complaint"].values
        cv_local = StratifiedKFold(n_splits=min(N_SPLITS, y.sum(), len(y) - y.sum()), shuffle=True, random_state=SEED)
        model = Pipeline([("scaler", StandardScaler()),
                           ("clf", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED))])
        try:
            scores = cross_validate(model, X, y, cv=cv_local, scoring=["roc_auc"])
            auc = scores["test_roc_auc"].mean()
        except Exception as e:
            log(f"scope_{scope}: 재실행 실패 ({e})")
            continue
        layer_auc[scope] = auc
        log(f"scope_{scope}: 정보충분층 AUC(n={len(sub_df)}) = {auc:.4f}")

    return {"delta_df": delta_df, "layer_auc": layer_auc, "tagged": mm}


def main():
    df = pd.read_parquet(FEATURES_PATH)
    log(f"probing_call_features.parquet 로드: {len(df)}행, 컬럼={df.columns.tolist()}")

    # A_N2_pred(및 B_N2/A_N5)가 결측인 299건은 "고객 유효 발화가 아예 없어
    # customer_only 조건(A/B) 예측 자체가 존재하지 않는 콜"이다(이전 stage2에서
    # 이미 확인된 케이스). 실험 A/B는 예측 컬럼을 안 쓰므로 전체 19,847건을
    # 그대로 쓰고, 실험 D/E/F는 A_N2_pred가 있어야 text_correct/wrong,
    # mismatch를 정의할 수 있으므로 이 299건을 제외한다.
    n_no_pred = df["A_N2_pred"].isna().sum()
    dropna_cols = [c for c in ["A_N2_pred", "B_N2_pred", "A_N5_pred"] if c in df.columns]
    df_pred_valid = df.dropna(subset=dropna_cols).copy()
    log(f"\nA_N2_pred/B_N2_pred 결측(고객 발화 없는 콜) {n_no_pred}건 -> "
        f"실험 D/E/F에서는 제외 (실험 A/B는 전체 {len(df)}건 그대로 사용)")

    res_a = experiment_A(df)
    res_b = experiment_B(df)
    res_c = experiment_C(df)
    res_d, profile_d = experiment_D(df_pred_valid)
    sweep_e, verdicts_e = experiment_E(df_pred_valid)
    res_f = experiment_F(df_pred_valid)

    # === 맨 위 요약 (파일에는 나중에 prepend) ===
    b_n2_auc_lr = res_b["n2"]["LogisticRegression"]["auc_mean"]
    b_n2_auc_xgb = res_b["n2"]["XGBoost"]["auc_mean"]
    a_n2_logreg_gain = res_a["n2"]["_logreg_gain"]
    a_n2_xgb_gain = res_a["n2"]["_xgb_gain"]
    d_n2_retained = res_d["n2"]["LogisticRegression"]["retained_pct"]
    gate_t_range = verdicts_e.get("n2", [])
    suff_auc = res_f["layer_auc"].get("n2")

    summary_lines = [
        "=" * 70,
        "핵심 요약 (scope_n2 기준)",
        "=" * 70,
        f"1. 7-way 신호: LogReg macro-F1 gain={a_n2_logreg_gain:+.4f}, XGB gain={a_n2_xgb_gain:+.4f} "
        f"({'신호 있음' if max(a_n2_logreg_gain, a_n2_xgb_gain) >= 0.05 else '신호 약함'})",
        f"2. 불만제기 이진 AUC: LogReg={b_n2_auc_lr:.4f}, XGB={b_n2_auc_xgb:.4f}",
        f"3. 게이트 유효성(text_wrong AUC 유지율): LogReg={d_n2_retained:.1f}% "
        f"({'게이트 강함' if d_n2_retained >= 80 else '게이트 약함' if d_n2_retained < 50 else '게이트 보통'})",
        f"4. 게이트 순이득 임계값 구간: {gate_t_range if gate_t_range else '없음'}",
        f"5. 정보충분층 불만이진 AUC: {suff_auc:.4f}" if suff_auc else "5. 정보충분층 AUC: 계산 불가",
        "=" * 70,
        "",
    ]

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
        f.write("\n".join(log_lines))

    print("\n".join(summary_lines))
    print(f"저장: {OUT_SUMMARY}")
    print("\nPROBING_ALL_COMPLETE")


if __name__ == "__main__":
    main()
