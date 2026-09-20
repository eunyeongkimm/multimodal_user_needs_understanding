"""250콜 probe 라벨을 **저장소 수록본만으로** 생성한다. outputs/ 불필요.

`probe_audio_prep.py --calls pilot250 --labels-only` 와 **같은 파일**을 만들지만,
161MB짜리 `outputs/d04_dialog_index.parquet` 대신 이미 저장소에 커밋된
`results/06_pilot_250/arousal_target_percall.parquet` 에서 성별을 가져온다.
맥의 outputs/ 가 없는 환경(CI, 새 클론, 원격 컨테이너)에서도 돌릴 수 있다.

--------------------------------------------------------------------------
성별을 재사용해도 되는 근거
--------------------------------------------------------------------------
`arousal_target_percall.parquet` 의 gender 는 `arousal_target_check.py` 가
d04 에서 뽑은 값이고, 산출 정의가 `probe_audio_prep.py` 와 글자 그대로 같다.

  arousal_target_check.py:
      gender=("speaker_gender", lambda s: s.mode().iat[0] if len(s.mode()) else None)
      대상 = stage2_windows_nl_v2, window_n=5, window_version="customer_only"

  probe_audio_prep.py:
      gender = seg.groupby("call_id")["speaker_gender"]
                  .agg(lambda s: s.mode().iat[0] if len(s.mode()) else None)
      대상 = 고객 발화 & r_gold_valid_flag==True & 앞 5개

즉 (1) 같은 집계(최빈 speaker_gender), (2) 같은 발화 창(앞 5개 고객 발화)이다.
call_id 집합도 `model_pilot_sample.csv` 250콜과 완전히 일치하며 결측이 없다.

gold_actual 은 `results/01_gold_labeling/gold_actual_batch1_final.parquet`
(최종본)에서 다시 붙이고, arousal 쪽 gold 와 250/250 일치하는지 assert 한다.

--------------------------------------------------------------------------
출력
--------------------------------------------------------------------------
  outputs/probe_set_pilot250/probe_labels.parquet
      call_id, gender, n_utt, gold_actual, is_complaint
      (probe_audio_prep.py 산출물과 동일 스키마)

이 파일 하나를 Drive 의 MyDrive/audio_seg_2/ 에 올리면 노트북 셀 2가 찾아
붙이고 positive control 이 활성화된다.

실행: python scripts/probe_labels_from_repo.py
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
AROUSAL_PATH = BASE_DIR / "results" / "06_pilot_250" / "arousal_target_percall.parquet"
SAMPLE_PATH = BASE_DIR / "results" / "06_pilot_250" / "model_pilot_sample.csv"
GOLD_PATH = BASE_DIR / "results" / "01_gold_labeling" / "gold_actual_batch1_final.parquet"
MANIFEST_PATH = BASE_DIR / "results" / "08_audio_seg" / "audio_seg_manifest.parquet"
OUT_PATH = BASE_DIR / "outputs" / "probe_set_pilot250" / "probe_labels.parquet"

COMPLAINT = "불만제기"


def main():
    ar = pd.read_parquet(AROUSAL_PATH, columns=["call_id", "gender", "gold_actual"])
    sample = pd.read_csv(SAMPLE_PATH)
    gold = pd.read_parquet(GOLD_PATH)
    man = pd.read_parquet(MANIFEST_PATH, columns=["call_id", "utt_idx"])

    # ---------- 검증 ----------
    assert set(ar["call_id"]) == set(sample["call_id"]), \
        "arousal_target_percall 과 model_pilot_sample 의 call_id 집합 불일치"
    assert ar["gender"].isna().sum() == 0, "성별 결측 존재"

    gcol = "label" if "label" in gold.columns else "gold_actual"
    gold = gold[["call_id", gcol]].rename(columns={gcol: "gold_final"})
    chk = ar.merge(gold, on="call_id", how="left")
    assert chk["gold_final"].isna().sum() == 0, "gold 최종본에 없는 call_id 존재"
    n_match = int((chk["gold_actual"] == chk["gold_final"]).sum())
    assert n_match == len(chk), \
        f"gold 불일치 {len(chk) - n_match}건 - arousal 쪽 gold 가 최종본과 다르다"

    # n_utt = 앞 5개 고객 발화 중 실제 개수 (audio_seg 매니페스트 행수)
    n_utt = man.groupby("call_id").size().rename("n_utt")

    lab = (ar[["call_id", "gender"]]
           .merge(n_utt, on="call_id", how="left")
           .merge(chk[["call_id", "gold_final"]].rename(columns={"gold_final": "gold_actual"}),
                  on="call_id", how="left"))
    lab["is_complaint"] = (lab["gold_actual"] == COMPLAINT).astype(int)
    assert lab["n_utt"].isna().sum() == 0, "n_utt 결측 (audio_seg 매니페스트에 없는 콜)"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lab.to_parquet(OUT_PATH, index=False)

    print(f"저장: {OUT_PATH}")
    print(f"  {len(lab)}콜 / 컬럼 {list(lab.columns)}")
    print(f"  성별: {lab['gender'].value_counts().to_dict()}")
    print(f"  불만제기: {int(lab['is_complaint'].sum())}건 "
          f"({lab['is_complaint'].mean()*100:.1f}%)")
    print(f"  콜당 고객 발화 수: {lab['n_utt'].value_counts().sort_index().to_dict()}")
    print()
    print("Drive 업로드: MyDrive/audio_seg_2/probe_labels.parquet")
    print("PROBE_LABELS_FROM_REPO_COMPLETE")


if __name__ == "__main__":
    main()
