"""train10k pool5(N<=5) 음성->불만제기 이진 분류기 학습/검증.
train10k_classifier.py(scope_n2)와 동일 방법론, 피처만 pool5_*로 교체.

출력: outputs/train10k_pool5_classifier_summary.txt
"""

import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE_DIR / "outputs" / "train_10k_pool5_labeled.parquet"
OUT_PATH = BASE_DIR / "outputs" / "train10k_pool5_classifier_summary.txt"

PRIMARY_METRICS = ["f0_mean", "f0_std", "e_mean_db", "e_std_db", "r_gold", "s_top_db30"]
SEED = 42
N_SPLITS = 5
PROBING_BATCH1_N2_AUC = 0.5693
PROBING_BATCH1_POOL5_AUC = 0.6218
TRAIN10K_N2_AUC = 0.5147  # 직전 실행 결과(train10k_classifier.py)


def scope_cols():
    return [f"pool5_{m}_{stat}" for m in PRIMARY_METRICS for stat in ["mean", "std"]]


def get_models(y):
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    spw = n_neg / max(n_pos, 1)
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED)),
        ]),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=4, eval_metric="logloss",
            scale_pos_weight=spw, random_state=SEED, verbosity=0,
        ),
    }


def fit_params_for(name, y):
    if name == "XGBoost":
        return {"sample_weight": compute_sample_weight("balanced", y)}
    return {}


def main():
    df = pd.read_parquet(FEATURES_PATH)
    cols = scope_cols()
    n_before = len(df)
    df = df.dropna(subset=cols)
    n_after = len(df)
    print(f"입력: {n_before}행 -> 피처 결측 제외 {n_after}행")

    y = df["is_complaint"].values
    X = df[cols].values
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(f"불만제기(양성): {n_pos}건, 비불만(음성): {n_neg}건, 양성비율={n_pos/len(y)*100:.2f}%")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    lines = []
    lines.append(f"train10k pool5(N<=5) 게이트 재구성 음성->불만 이진분류 결과")
    lines.append(f"표본: {n_after}건 (양성 {n_pos}건, 음성 {n_neg}건), scope=pool5, "
                 f"{N_SPLITS}-fold StratifiedKFold(seed={SEED})")
    lines.append("")

    models = get_models(y)
    results = {}
    for name, model in models.items():
        fp = fit_params_for(name, y)
        scores = cross_validate(model, X, y, cv=cv,
                                 scoring=["roc_auc", "f1", "recall", "precision"],
                                 params=fp if fp else None)
        auc_folds = scores["test_roc_auc"]
        auc_m, auc_s = auc_folds.mean(), auc_folds.std()
        f1_m = scores["test_f1"].mean()
        rec_m = scores["test_recall"].mean()
        prec_m = scores["test_precision"].mean()
        results[name] = {"auc_folds": auc_folds, "auc_mean": auc_m, "auc_std": auc_s,
                          "f1_mean": f1_m, "recall_mean": rec_m, "precision_mean": prec_m}

        lines.append(f"=== {name} ===")
        lines.append(f"AUC = {auc_m:.4f} ± {auc_s:.4f}")
        lines.append(f"fold별 AUC: " + ", ".join(f"{a:.4f}" for a in auc_folds))
        lines.append(f"F1={f1_m:.4f}  recall={rec_m:.4f}  precision={prec_m:.4f}")
        cv_ratio = auc_s / auc_m if auc_m else float("nan")
        lines.append(f"fold 변동계수(std/mean) = {cv_ratio:.4f} "
                     f"({'편차 큼(불안정)' if cv_ratio > 0.15 else '편차 보통' if cv_ratio > 0.08 else '편차 작음(안정적)'})")
        lines.append("")

    lr_auc = results["LogisticRegression"]["auc_mean"]
    lr_std = results["LogisticRegression"]["auc_std"]
    xgb_auc = results["XGBoost"]["auc_mean"]

    lines.append("=== 비교 (probing batch1 / train10k N=2) ===")
    lines.append(f"probing scope_n2(batch1)    = {PROBING_BATCH1_N2_AUC:.4f}")
    lines.append(f"probing scope_pool5(batch1) = {PROBING_BATCH1_POOL5_AUC:.4f}")
    lines.append(f"train10k scope_n2(신규)      = {TRAIN10K_N2_AUC:.4f}")
    lines.append(f"train10k scope_pool5(신규, 이번 결과) = {lr_auc:.4f}")

    output = "\n".join(lines)
    print("\n" + output)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(output + "\n")
    print(f"\n저장: {OUT_PATH}")

    print("\n" + "=" * 60)
    print("요약")
    print("=" * 60)
    print(f"1. pool5 AUC(LogReg) = {lr_auc:.4f} ± {lr_std:.4f} (5-fold), XGB = {xgb_auc:.4f} ± "
          f"{results['XGBoost']['auc_std']:.4f}")
    print(f"2. N=2({TRAIN10K_N2_AUC:.4f}) 대비: {lr_auc:.4f} (차이 {lr_auc-TRAIN10K_N2_AUC:+.4f}), "
          f"probing pool5({PROBING_BATCH1_POOL5_AUC:.4f}) 대비: 차이 {lr_auc-PROBING_BATCH1_POOL5_AUC:+.4f}")
    recovered = lr_auc >= 0.58
    print(f"3. 0.62 근처로 회복 여부: {'회복됨(0.58+) - 게이트를 pool5로 세울 근거 있음' if recovered else '회복 안 됨 - N=2처럼 신호 미미, 게이트 재고 필요'}")

    print("\nTRAIN10K_POOL5_CLASSIFIER_COMPLETE")


if __name__ == "__main__":
    main()
