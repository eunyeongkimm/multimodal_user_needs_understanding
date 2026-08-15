"""파일럿 200콜 중 일부를 사람이 직접 라벨링할 수 있는 검증용 CSV로 뽑는다.

pilot_label_llm.py가 만든 outputs/pilot200_calls.parquet(200콜 샘플)을 그대로 사용하고,
그중 N_HUMAN개를 다시 random_state=42로 샘플링해 (call_id, 전체통화텍스트, human_label) 형태로 저장한다.
human_label 컬럼은 사람이 채워 넣도록 비워둔다.
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_CACHE_PATH = BASE_DIR / "outputs" / "pilot200_calls.parquet"
HUMAN_EVAL_PATH = BASE_DIR / "outputs" / "human_eval_sample.csv"

N_HUMAN = 80  # 50~100 범위 내
RANDOM_STATE = 42


def main():
    if not SAMPLE_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"{SAMPLE_CACHE_PATH} 가 없습니다. 먼저 pilot_label_llm.py를 실행해 200콜 샘플을 생성하세요."
        )

    sample = pd.read_parquet(SAMPLE_CACHE_PATH)
    human_sample = sample.sample(n=N_HUMAN, random_state=RANDOM_STATE).reset_index(drop=True)
    human_sample["human_label"] = ""

    human_sample[["call_id", "call_text", "human_label"]].to_csv(
        HUMAN_EVAL_PATH, index=False, encoding="utf-8-sig"
    )

    print(f"검증용 CSV 저장 완료: {HUMAN_EVAL_PATH} ({N_HUMAN}콜)")
    print("human_label 컬럼에 사람이 직접 카테고리를 채워 넣은 뒤 compute_kappa.py를 실행하세요.")


if __name__ == "__main__":
    main()
