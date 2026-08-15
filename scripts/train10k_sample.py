"""게이트 프로토타입(B) train 데이터: batch2 풀(54,123콜, batch1과 call_id 중복 0)에서
1만 콜을 샘플링한다.

필터:
  - call-level r_gold_valid_flag: 해당 콜에 r_gold_valid_flag==True인 dialog가
    1개 이상 존재(=call_text가 비지 않음). batch1에서 이 조건으로 152건이
    제외됐던 것과 동일한 기준(outputs/batch1_no_valid_text_call_ids.csv 참고).
  - "N=2 윈도우 음성서술 존재" 필터는 이번 실행 시점에 적용 불가능함이 확인됨:
    outputs/stage2_windows_nl_v2.parquet, stage2_acoustic_features.parquet,
    stage2_utterance_pool.parquet 전부 batch2 콜을 0건 포함(batch2는 아직
    음향 파이프라인이 전혀 실행된 적 없음). 이 필터는 스킵하고 텍스트
    유효성만으로 샘플링한 뒤, 결과를 사람에게 보고한다(추측/임의 처리 금지).
  - batch1(추론셋, 20,000)과 call_id 중복 없음을 샘플링 후 재확인.

출력: outputs/train10k_call_ids.csv, outputs/train10k_call_texts.parquet
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DIALOG_PATH = BASE_DIR / "outputs" / "d04_dialog_index.parquet"
BATCH1_IDS_PATH = BASE_DIR / "outputs" / "batch1_call_ids.csv"
BATCH2_IDS_PATH = BASE_DIR / "outputs" / "batch2_call_ids.csv"
WINDOWS_NL_V2_PATH = BASE_DIR / "outputs" / "stage2_windows_nl_v2.parquet"
ACOUSTIC_FEATURES_PATH = BASE_DIR / "outputs" / "stage2_acoustic_features.parquet"
UTTERANCE_POOL_PATH = BASE_DIR / "outputs" / "stage2_utterance_pool.parquet"

OUT_IDS_PATH = BASE_DIR / "outputs" / "train10k_call_ids.csv"
OUT_TEXTS_PATH = BASE_DIR / "outputs" / "train10k_call_texts.parquet"

N_SAMPLE = 10_000
RANDOM_STATE = 42


def main():
    batch1_ids = set(pd.read_csv(BATCH1_IDS_PATH)["call_id"])
    batch2_ids = set(pd.read_csv(BATCH2_IDS_PATH)["call_id"])
    print(f"batch1(추론셋, 고정): {len(batch1_ids)}건, batch2(train 풀): {len(batch2_ids)}건, "
          f"중복: {len(batch1_ids & batch2_ids)}건")

    # === 음향 파이프라인 커버리지 확인 (추측 금지, 실측) ===
    for name, path in [("stage2_windows_nl_v2", WINDOWS_NL_V2_PATH),
                        ("stage2_acoustic_features", ACOUSTIC_FEATURES_PATH),
                        ("stage2_utterance_pool", UTTERANCE_POOL_PATH)]:
        ids = set(pd.read_parquet(path, columns=["call_id"])["call_id"].unique())
        n_overlap = len(ids & batch2_ids)
        print(f"  {name}: 전체 {len(ids)}건 중 batch2 포함 {n_overlap}건")
    print("=> batch2는 음향 파이프라인 실행 이력이 전혀 없음(N=2 윈도우 음성서술 필터 적용 불가). "
          "텍스트 유효성 기준으로만 샘플링.")

    df = pd.read_parquet(DIALOG_PATH, columns=["call_id", "dialog_idx", "speaker_type",
                                                "text_clean", "r_gold_valid_flag"])
    df2 = df[df["call_id"].isin(batch2_ids)]
    n_calls_in_pool = df2["call_id"].nunique()
    print(f"\nbatch2 풀 중 dialog_index에 존재하는 콜: {n_calls_in_pool}/{len(batch2_ids)}건")

    valid = df2[df2["r_gold_valid_flag"] == True]  # noqa: E712
    valid_call_ids = set(valid["call_id"].unique())
    n_no_valid_text = len(batch2_ids) - len(valid_call_ids)
    print(f"call-level r_gold_valid_flag 유효(콜 내 유효 dialog >=1): {len(valid_call_ids)}건, "
          f"무효(텍스트 없음): {n_no_valid_text}건")

    if len(valid_call_ids) < N_SAMPLE:
        print(f"경고: 유효 콜({len(valid_call_ids)}건)이 목표 샘플({N_SAMPLE}건)보다 적습니다.")

    valid_call_ids_sorted = sorted(valid_call_ids)
    sample_ids = pd.Series(valid_call_ids_sorted).sample(n=N_SAMPLE, random_state=RANDOM_STATE).tolist()
    sample_ids_set = set(sample_ids)

    overlap_with_batch1 = sample_ids_set & batch1_ids
    print(f"\n샘플 {len(sample_ids)}건, batch1(추론셋)과 중복: {len(overlap_with_batch1)}건 "
          f"({'정상' if len(overlap_with_batch1) == 0 else '경고: 중복 발견!'})")

    valid_sorted = valid[valid["call_id"].isin(sample_ids_set)].sort_values(["call_id", "dialog_idx"])
    line = valid_sorted["speaker_type"].astype(str) + ": " + valid_sorted["text_clean"].astype(str)
    valid_sorted = valid_sorted.assign(line=line)
    call_texts = (
        valid_sorted.groupby("call_id")["line"]
        .apply(lambda s: "\n".join(s))
        .reset_index()
        .rename(columns={"line": "call_text"})
    )
    print(f"call_text 생성: {len(call_texts)}건 (목표 {N_SAMPLE}건과 {'일치' if len(call_texts)==N_SAMPLE else '불일치'})")

    pd.DataFrame({"call_id": sorted(sample_ids_set)}).to_csv(OUT_IDS_PATH, index=False)
    call_texts.to_parquet(OUT_TEXTS_PATH, index=False)
    print(f"\n저장: {OUT_IDS_PATH}, {OUT_TEXTS_PATH}")
    print("\nTRAIN10K_SAMPLE_COMPLETE")


if __name__ == "__main__":
    main()
