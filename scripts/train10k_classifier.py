"""train10k(게이트 프로토타입 B) 음성->불만제기 이진 분류기 학습/검증.
probing_experiments.py 실험 B와 동일 방법론(LogReg balanced 주력 + XGB 비교,
5-fold StratifiedKFold, 스케일링은 Pipeline 내부에서 fold별 fit - 누수 방지).
불만 226건 소수라 fold별 AUC 편차도 함께 출력.

출력: outputs/train10k_classifier_summary.txt
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
FEATURES_PATH = BASE_DIR / "outputs" / "train_10k_labeled.parquet"
OUT_PATH = BASE_DIR / "outputs" / "train10k_classifier_summary.txt"

PRIMARY_METRICS = ["f0_mean", "f0_std", "e_mean_db", "e_std_db", "r_gold", "s_top_db30"]
SEED = 42
N_SPLITS = 5
PROBING_BATCH1_N2_AUC = 0.5693  # 기존 probing 실험 B, scope_n2, LogReg 기준 참고값 (outputs/probing_summary.txt 확인)
# 주의: probing_summary.txt에는 scope_pool5(N<=5)의 LogReg AUC=0.6218도 있으나,
# train10k는 N=2만 존재하므로 scope_n2(0.5693)와 비교하는 것이 올바른 비교임.


def scope_cols():
    return [f"n2_{m}_{stat}" for m in PRIMARY_METRICS for stat in ["mean", "std"]]


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
    lines.append(f"train10k 게이트 프로토타입(B) 음성->불만 이진분류 결과")
    lines.append(f"표본: {n_after}건 (양성 {n_pos}건, 음성 {n_neg}건), scope=n2, {N_SPLITS}-fold StratifiedKFold(seed={SEED})")
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

    lines.append("=== probing(batch1, scope_n2, LogReg) 대비 비교 ===")
    lines.append(f"probing 당시 AUC = {PROBING_BATCH1_N2_AUC:.4f} (참고값)")
    lines.append(f"이번 train10k AUC(LogReg) = {lr_auc:.4f} (차이 {lr_auc - PROBING_BATCH1_N2_AUC:+.4f})")

    output = "\n".join(lines)
    print("\n" + output)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(output + "\n")
    print(f"\n저장: {OUT_PATH}")

    print("\n" + "=" * 60)
    print("요약")
    print("=" * 60)
    print(f"1. 불만 이진 AUC(LogReg) = {lr_auc:.4f} ± {lr_std:.4f} (5-fold), XGB = {xgb_auc:.4f} ± "
          f"{results['XGBoost']['auc_std']:.4f}")
    cv_ratio_lr = lr_std / lr_auc if lr_auc else float("nan")
    print(f"2. fold별 편차: LogReg 변동계수(std/mean)={cv_ratio_lr:.4f} "
          f"({'226건 소수 표본치고 편차 큼 - 안정성 주의' if cv_ratio_lr > 0.15 else '허용 범위 내로 비교적 안정적'})")
    print(f"3. probing(batch1, scope_n2, AUC={PROBING_BATCH1_N2_AUC:.4f}) 대비: {lr_auc:.4f} "
          f"({'개선' if lr_auc > PROBING_BATCH1_N2_AUC else '유사' if abs(lr_auc-PROBING_BATCH1_N2_AUC)<0.02 else '하락'}, "
          f"차이 {lr_auc - PROBING_BATCH1_N2_AUC:+.4f})")

    print("\nTRAIN10K_CLASSIFIER_COMPLETE")


if __name__ == "__main__":
    main()
