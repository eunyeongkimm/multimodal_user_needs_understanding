"""pool5 음성->불만 분류기(train10k로 학습, AUC~0.59)를 추론셋(19,847콜) 전체에
적용해 불만확률을 산출하고, train10k 확률 분포 기준 분위수 경계로 강/중/약 구간을
부여한다.

- 학습: train_10k_pool5_labeled.parquet 전체(10,000건)로 StandardScaler+LogReg 최종 적합
  (fold 없이 전량 fit - 이 모델이 곧 배포될 게이트).
- 추론: probing_call_features.parquet의 pool5_* 12피처에 그 fit된 scaler/모델
  그대로 적용(추론셋으로 재fit 금지).
- 구간 경계: train10k 예측확률의 33/67 분위수를 그대로 추론셋에 적용(추론셋
  자체 분포로 재계산 금지 - 누수 방지).

출력: outputs/gate_signal_inference.parquet (call_id, complaint_proba, gate_level)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "outputs" / "train_10k_pool5_labeled.parquet"
INFER_PATH = BASE_DIR / "outputs" / "probing_call_features.parquet"
OUT_PATH = BASE_DIR / "outputs" / "gate_signal_inference.parquet"
OUT_TRAIN_PATH = BASE_DIR / "outputs" / "gate_signal_train10k.parquet"

PRIMARY_METRICS = ["f0_mean", "f0_std", "e_mean_db", "e_std_db", "r_gold", "s_top_db30"]
SEED = 42


def scope_cols():
    return [f"pool5_{m}_{stat}" for m in PRIMARY_METRICS for stat in ["mean", "std"]]


def main():
    train = pd.read_parquet(TRAIN_PATH)
    infer = pd.read_parquet(INFER_PATH)
    cols = scope_cols()

    n_train_before = len(train)
    train = train.dropna(subset=cols)
    print(f"train10k: {n_train_before} -> 피처 결측 제외 {len(train)}건")

    n_infer_before = len(infer)
    infer_valid = infer.dropna(subset=cols).copy()
    n_infer_missing = n_infer_before - len(infer_valid)
    print(f"추론셋: {n_infer_before} -> 피처 결측 제외 {len(infer_valid)}건 (결측 {n_infer_missing}건)")

    X_train = train[cols].values
    y_train = train["is_complaint"].values

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED)),
    ])
    model.fit(X_train, y_train)
    print(f"\n최종 모델 적합 완료 (train10k 전체 {len(train)}건, 양성 {y_train.sum()}건)")

    train_proba = model.predict_proba(X_train)[:, 1]
    q33, q67 = np.quantile(train_proba, [1/3, 2/3])
    print(f"\ntrain10k 예측확률 분위수 경계: q33={q33:.4f}, q67={q67:.4f}")

    def bucket(p):
        if p >= q67:
            return "강"
        elif p >= q33:
            return "중"
        else:
            return "약"

    train_out = train[["call_id", "gold_actual", "is_complaint"]].copy()
    train_out["complaint_proba"] = train_proba
    train_out["gate_level"] = [bucket(p) for p in train_proba]
    train_out.to_parquet(OUT_TRAIN_PATH, index=False)
    print(f"\n=== train10k 자체 강/중/약 분포 (검산용, 정확히 1/3씩이어야 함) ===")
    print(train_out["gate_level"].value_counts().to_string())

    X_infer = infer_valid[cols].values
    infer_proba = model.predict_proba(X_infer)[:, 1]
    infer_valid["complaint_proba"] = infer_proba
    infer_valid["gate_level"] = [bucket(p) for p in infer_proba]

    gate_out = infer_valid[["call_id", "gold_actual", "complaint_proba", "gate_level"]].copy()
    gate_out.to_parquet(OUT_PATH, index=False)
    print(f"\n저장: {OUT_PATH} ({len(gate_out)}행)")

    print(f"\n=== 추론셋 강/중/약 분포 (train 경계 고정 적용, 재계산 안 함) ===")
    print(gate_out["gate_level"].value_counts().to_string())

    print(f"\n=== 게이트 신호 타당성: 강/중/약별 실제 불만제기 비율 (추론셋, gold_actual 기준) ===")
    for level in ["강", "중", "약"]:
        sub = gate_out[gate_out["gate_level"] == level]
        rate = (sub["gold_actual"] == "불만제기").mean()
        n_complaint = (sub["gold_actual"] == "불만제기").sum()
        print(f"  {level}: {rate*100:.2f}% ({n_complaint}/{len(sub)}건)")

    print("\nGATE_SIGNAL_BUILD_COMPLETE")


if __name__ == "__main__":
    main()
