"""전체 74,123콜을 random_state=42로 shuffle해 20,000 / 54,123 두 배치로 고정 분할한다.

한 번 저장된 이후에는 재실행해도 기존 파일이 있으면 덮어쓰지 않고 그대로 재사용한다
(계속 재사용할 기준 파일이므로 실수로 다시 shuffle되는 것을 방지).
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "outputs" / "d04_dialog_index.parquet"
BATCH1_PATH = BASE_DIR / "outputs" / "batch1_call_ids.csv"
BATCH2_PATH = BASE_DIR / "outputs" / "batch2_call_ids.csv"

N_BATCH1 = 20000
RANDOM_STATE = 42


def main():
    if BATCH1_PATH.exists() or BATCH2_PATH.exists():
        print("이미 배치 분할 파일이 존재합니다. 재shuffle을 방지하기 위해 아무 것도 하지 않습니다.")
        print(f"  {'존재' if BATCH1_PATH.exists() else '없음'}: {BATCH1_PATH}")
        print(f"  {'존재' if BATCH2_PATH.exists() else '없음'}: {BATCH2_PATH}")
        return

    df = pd.read_parquet(DATA_PATH, columns=["call_id"])
    call_ids = df["call_id"].unique()
    print(f"전체 고유 call_id 수: {len(call_ids)}")

    rng = np.random.RandomState(RANDOM_STATE)
    shuffled = call_ids.copy()
    rng.shuffle(shuffled)

    batch1_ids = shuffled[:N_BATCH1]
    batch2_ids = shuffled[N_BATCH1:]

    assert len(set(batch1_ids) & set(batch2_ids)) == 0, "batch1/batch2가 겹칩니다!"
    assert set(batch1_ids) | set(batch2_ids) == set(call_ids), "batch1+batch2가 전체 call_id와 일치하지 않습니다!"

    pd.DataFrame({"call_id": batch1_ids}).to_csv(BATCH1_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame({"call_id": batch2_ids}).to_csv(BATCH2_PATH, index=False, encoding="utf-8-sig")

    print(f"batch1 저장: {BATCH1_PATH} ({len(batch1_ids)}개)")
    print(f"batch2 저장: {BATCH2_PATH} ({len(batch2_ids)}개)")
    print("검증: 교집합 0건, 합집합 = 전체 74,123건 -> OK")


if __name__ == "__main__":
    main()
