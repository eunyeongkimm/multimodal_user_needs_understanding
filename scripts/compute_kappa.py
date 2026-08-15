"""GPT-5.5 vs 사람, GPT-4.1 mini vs 사람의 Cohen's Kappa를 계산한다.

전제:
- outputs/pilot200_gpt_labels.csv : pilot_label_llm.py --run 결과 (call_id, gpt55_label, gpt41mini_label)
- outputs/human_eval_sample.csv   : prepare_human_eval.py 결과에 사람이 human_label을 채워 넣은 파일

human_label이 비어있는 행이 있으면 라벨링이 끝나지 않은 것으로 보고 안내만 하고 종료한다.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

BASE_DIR = Path(__file__).resolve().parent.parent
GPT_LABELS_PATH = BASE_DIR / "outputs" / "pilot200_gpt_labels.csv"
HUMAN_EVAL_PATH = BASE_DIR / "outputs" / "human_eval_sample.csv"


def main():
    if not GPT_LABELS_PATH.exists():
        raise FileNotFoundError(f"{GPT_LABELS_PATH} 가 없습니다. 먼저 pilot_label_llm.py --run 을 실행하세요.")
    if not HUMAN_EVAL_PATH.exists():
        raise FileNotFoundError(f"{HUMAN_EVAL_PATH} 가 없습니다. 먼저 prepare_human_eval.py 를 실행하세요.")

    human_df = pd.read_csv(HUMAN_EVAL_PATH, dtype=str)
    if human_df["human_label"].isna().any() or (human_df["human_label"].str.strip() == "").any():
        n_missing = (human_df["human_label"].isna() | (human_df["human_label"].str.strip() == "")).sum()
        print(
            f"human_label이 아직 비어있는 행이 {n_missing}개 있습니다. "
            f"{HUMAN_EVAL_PATH}에서 전부 채운 뒤 다시 실행하세요."
        )
        return

    gpt_df = pd.read_csv(GPT_LABELS_PATH, dtype=str)
    merged = human_df.merge(gpt_df, on="call_id", how="left")

    if merged["gpt55_label"].isna().any() or merged["gpt41mini_label"].isna().any():
        print("경고: human_eval_sample.csv의 일부 call_id가 pilot200_gpt_labels.csv에 없습니다.")
        merged = merged.dropna(subset=["gpt55_label", "gpt41mini_label"])

    n = len(merged)
    kappa_55 = cohen_kappa_score(merged["human_label"], merged["gpt55_label"])
    kappa_41 = cohen_kappa_score(merged["human_label"], merged["gpt41mini_label"])

    print(f"=== Cohen's Kappa (n={n}) ===")
    print(f"GPT-5.5      vs 사람: {kappa_55:.4f}")
    print(f"GPT-4.1 mini vs 사람: {kappa_41:.4f}")


if __name__ == "__main__":
    main()
