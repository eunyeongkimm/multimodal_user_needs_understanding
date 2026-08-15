"""모델 버전 파일럿(4.1mini-현재 / 5.4-med / 5.6terra-med)용 250건 층화 샘플.

모집단 = acoustically-valid subset ∩ B조건 기존 예측 보유 콜
  - B조건: stage2m_gpt_predictions_BD.parquet에서 condition=='B', n==5,
    version=='customer_only' (customer-only + text + acoustic)
  - 정보충분: infopoor_tagged.csv의 is_infopoor==False
    (infopoor_tagged는 mismatch 콜만 담고 있으므로, 미등재 콜은 정보충분으로 간주)

소수 클래스(불만제기) >= MIN_MINORITY를 보장하기 위해 해당 stratum만 오버샘플하고
구성/오버샘플 여부를 로그로 남긴다.

출력: outputs/model_pilot_sample.csv (3조건 공통 call_id 리스트)
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
BD_PATH = BASE_DIR / "outputs" / "stage2m_gpt_predictions_BD.parquet"
GOLD_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_final.parquet"
INFOPOOR_PATH = BASE_DIR / "outputs" / "infopoor_tagged.csv"
OUT_PATH = BASE_DIR / "outputs" / "model_pilot_sample.csv"

CATEGORIES = ["환불요청", "주문취소", "불만제기", "배송확인", "교환반품", "구매진행", "서비스이용"]
MINORITY = "불만제기"
N_SAMPLE = 250
MIN_MINORITY = 50
SEED = 42

WINDOW_N = 5
WINDOW_VERSION = "customer_only"
CONDITION = "B"


def main():
    bd = pd.read_parquet(BD_PATH)
    b = bd[(bd["condition"] == CONDITION) & (bd["n"] == WINDOW_N)
           & (bd["version"] == WINDOW_VERSION)].copy()
    print(f"B/N={WINDOW_N}/{WINDOW_VERSION} 기존 예측: {len(b)}행, 고유콜 {b['call_id'].nunique()}건")
    assert len(b) == b["call_id"].nunique(), "call_id 중복"

    n_pred_null = b["predicted_label"].isna().sum()
    print(f"  예측 결측: {n_pred_null}건 (모집단에서 제외)")
    b = b.dropna(subset=["predicted_label"])

    gold = pd.read_parquet(GOLD_PATH).rename(columns={"label": "gold_actual"})
    pop = b[["call_id", "predicted_label"]].merge(
        gold[["call_id", "gold_actual"]], on="call_id", how="left")
    n_no_gold = pop["gold_actual"].isna().sum()
    print(f"  gold 없는 콜: {n_no_gold}건 (제외)")
    pop = pop.dropna(subset=["gold_actual"])

    infopoor = pd.read_csv(INFOPOOR_PATH)[["call_id", "is_infopoor"]]
    pop = pop.merge(infopoor, on="call_id", how="left")
    pop["is_infopoor"] = pop["is_infopoor"].fillna(False).astype(bool)
    n_poor = pop["is_infopoor"].sum()
    pop = pop[~pop["is_infopoor"]].drop(columns=["is_infopoor"])
    print(f"  정보부족 제외: {n_poor}건 -> 최종 모집단 {len(pop)}건")

    dist = pop["gold_actual"].value_counts()
    print("\n=== 모집단 gold 분포 ===")
    for c in CATEGORIES:
        print(f"  {c}: {dist.get(c, 0)}")

    n_minority_pop = int(dist.get(MINORITY, 0))
    prop_minority = round(N_SAMPLE * n_minority_pop / len(pop))
    oversampled = prop_minority < MIN_MINORITY
    n_minority_take = max(prop_minority, MIN_MINORITY)
    print(f"\n소수클래스 '{MINORITY}': 비례추출 시 {prop_minority}건 -> "
          f"{'오버샘플 적용' if oversampled else '오버샘플 불필요'}, 실제 {n_minority_take}건 배정")
    assert n_minority_take <= n_minority_pop, \
        f"모집단 {MINORITY} {n_minority_pop}건 < 요구 {n_minority_take}건"

    rest_pop = pop[pop["gold_actual"] != MINORITY]
    n_rest = N_SAMPLE - n_minority_take
    rest_dist = rest_pop["gold_actual"].value_counts()
    alloc = {c: int(round(n_rest * rest_dist.get(c, 0) / len(rest_pop))) for c in rest_dist.index}
    # 반올림 오차 보정: 가장 큰 stratum에서 가감
    diff = n_rest - sum(alloc.values())
    if diff:
        biggest = rest_dist.index[0]
        alloc[biggest] += diff
    alloc[MINORITY] = n_minority_take

    parts = []
    for cat, k in alloc.items():
        if k <= 0:
            continue
        sub = pop[pop["gold_actual"] == cat]
        parts.append(sub.sample(n=k, random_state=SEED))
    sample = pd.concat(parts, ignore_index=True).sort_values("call_id").reset_index(drop=True)
    sample = sample.rename(columns={"predicted_label": "pred_41mini_cur"})

    print(f"\n=== 샘플 구성 (n={len(sample)}, seed={SEED}) ===")
    print(f"{'클래스':<10} {'모집단':>8} {'모집단%':>8} {'샘플':>6} {'샘플%':>8} {'오버샘플':>8}")
    for c in CATEGORIES:
        n_pop_c = int(dist.get(c, 0))
        n_s = int((sample["gold_actual"] == c).sum())
        flag = "O" if (c == MINORITY and oversampled) else "-"
        print(f"{c:<10} {n_pop_c:>8} {n_pop_c/len(pop)*100:>7.2f}% {n_s:>6} "
              f"{n_s/len(sample)*100:>7.2f}% {flag:>8}")

    assert len(sample) == N_SAMPLE, f"샘플 수 불일치: {len(sample)}"
    assert sample["call_id"].nunique() == N_SAMPLE, "call_id 중복"
    assert (sample["gold_actual"] == MINORITY).sum() >= MIN_MINORITY, "소수클래스 하한 미달"

    sample.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n저장: {OUT_PATH}")
    print("MODEL_PILOT_SAMPLE_COMPLETE")


if __name__ == "__main__":
    main()
