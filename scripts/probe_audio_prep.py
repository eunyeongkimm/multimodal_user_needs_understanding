"""층별 probe(1단계)용 오디오 + 라벨 세트 준비. 맥 + AIHub 외장하드에서 실행.

`audio_seg_extract.py`(250콜 전용)를 임의의 call_id 목록으로 일반화한 것이다.
발화 선택 규칙은 §3.1 파이프라인과 동일하게 유지한다.
  - 고객 발화만 (speaker_type이 '고객'으로 시작)
  - 전사 품질 통과분만 (r_gold_valid_flag == True)
  - dialog_idx 오름차순 앞 N개 고객 발화 (기본 N=5)

원천 WAV는 AIHub가 이미 발화 단위 개별 파일로 주므로 **그대로 복사**한다.
리샘플·정규화·재인코딩 없음(원음 바이트 동일). 8kHz -> 16kHz 리샘플은
Colab 노트북의 로더가 담당한다(모델 입력 규격이 16kHz이므로).

--------------------------------------------------------------------------
대상 call_id 세트 (--calls 로 선택)
--------------------------------------------------------------------------
  pilot250  : results/06_pilot_250/model_pilot_sample.csv (250콜, 불만 50 오버샘플)
              -> 이미 outputs/audio_seg/ 에 추출돼 있고 Drive에도 올라가 있다.
                 다시 뽑을 필요 없음. 기본값이 아닌 이유.
  unified2k : results/01_gold_labeling/v_unified_sample_call_ids.csv
              (2,000콜, 불만제기 250 오버샘플)  <- 기본값
  <경로>    : call_id 컬럼을 가진 임의의 csv/parquet

--------------------------------------------------------------------------
출력 (전부 outputs/probe_set_{name}/ 아래)
--------------------------------------------------------------------------
  audio/{call_id}/c{i}.wav     발화 단위 WAV (원음 복사)
  probe_manifest.parquet       call_id, utt_idx, wav_path, dialog_idx, duration,
                               wav_duration, sample_rate, n_channels, error
  probe_labels.parquet         call_id, gold_actual, is_complaint, gender, n_utt
                               (gender = 고객 발화 최빈 성별. sanity check의
                                positive control 라벨로 쓴다)
  probe_prep_summary.md        추출 로그

전사 텍스트는 **일부러 넣지 않는다**. Colab/Drive로 올라가는 산출물이라
전사 원문이 저장소 밖으로 나가지 않게 한다. 저음질 sanity check에서 전사
대조가 필요하면 --with-text 로 별도 파일(probe_text_spotcheck.parquet,
기본 5콜)만 뽑는다.

실행:
  python scripts/probe_audio_prep.py --calls unified2k
  python scripts/probe_audio_prep.py --calls pilot250 --with-text
"""

import argparse
import shutil
import sys
import wave
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_ROOT = BASE_DIR / "outputs"
D04_PATH = OUT_ROOT / "d04_dialog_index.parquet"
GOLD_PATH = OUT_ROOT / "gold_actual_batch1_final.parquet"

CALL_SETS = {
    "pilot250": BASE_DIR / "results" / "06_pilot_250" / "model_pilot_sample.csv",
    "unified2k": BASE_DIR / "results" / "01_gold_labeling" / "v_unified_sample_call_ids.csv",
}

AUDIO_BASE_DIR = Path(
    "/Volumes/AIHub/007.저음질 전화망 음성인식 데이터/01.데이터/1.Training/원천데이터_230316"
)

MAX_CUSTOMER_UTT = 5
CUSTOMER_PREFIX = "고객"
COMPLAINT = "불만제기"
N_TEXT_SPOTCHECK = 5

L = []


def p(msg=""):
    print(msg)
    L.append(msg)


def wav_info(path: Path):
    """(duration_sec, sample_rate, n_channels) 또는 예외 시 (None, None, None)."""
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / w.getframerate(), w.getframerate(), w.getnchannels()
    except Exception:
        return None, None, None


def load_call_ids(spec: str) -> list:
    path = CALL_SETS.get(spec, Path(spec))
    if not Path(path).exists():
        sys.exit(f"ERROR: call 세트를 찾을 수 없음 - {path}")
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    if "call_id" not in df.columns:
        sys.exit(f"ERROR: call_id 컬럼 없음 - {path}")
    return list(dict.fromkeys(df["call_id"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", default="unified2k",
                    help="pilot250 | unified2k | call_id 컬럼을 가진 csv/parquet 경로")
    ap.add_argument("--n", type=int, default=MAX_CUSTOMER_UTT, help="콜당 앞 고객 발화 수")
    ap.add_argument("--with-text", action="store_true",
                    help="저음질 sanity check용 전사 대조 파일을 별도로 뽑는다")
    args = ap.parse_args()

    if not AUDIO_BASE_DIR.exists():
        sys.exit(f"ERROR: 외장하드 미마운트 - {AUDIO_BASE_DIR}")
    for f in (D04_PATH, GOLD_PATH):
        if not f.exists():
            sys.exit(f"ERROR: 필요한 파일 없음 - {f}")

    name = args.calls if args.calls in CALL_SETS else Path(args.calls).stem
    out_dir = OUT_ROOT / f"probe_set_{name}"
    audio_dir = out_dir / "audio"

    call_ids = load_call_ids(args.calls)
    d04 = pd.read_parquet(D04_PATH, columns=[
        "call_id", "dialog_idx", "speaker_type", "speaker_gender",
        "r_gold_valid_flag", "duration", "audioPath", "text_clean"])

    p(f"# 층별 probe용 오디오 세트 준비 — `{name}`")
    p("")
    p(f"대상 call_id: **{len(call_ids):,}콜** (`{CALL_SETS.get(args.calls, args.calls)}`)")
    p("")
    p("발화 선택은 §3.1 파이프라인과 동일 규칙:")
    p("")

    sub = d04[d04["call_id"].isin(call_ids)]
    n0 = len(sub)
    sub = sub[sub["speaker_type"].astype(str).str.startswith(CUSTOMER_PREFIX)]
    n_cust = len(sub)
    sub = sub[sub["r_gold_valid_flag"] == True]  # noqa: E712
    n_valid = len(sub)
    sub = sub.sort_values(["call_id", "dialog_idx"])
    sub["utt_idx"] = sub.groupby("call_id").cumcount() + 1
    seg = sub[sub["utt_idx"] <= args.n].copy()

    p(f"- 대상 콜의 전체 dialog: {n0:,}행")
    p(f"- 고객 발화만: -> {n_cust:,}행")
    p(f"- `r_gold_valid_flag == True`: -> {n_valid:,}행")
    p(f"- dialog_idx 오름차순 앞 {args.n}개: -> **{len(seg):,}발화 / "
      f"{seg['call_id'].nunique():,}콜**")
    p("")
    missing_calls = set(call_ids) - set(seg["call_id"])
    if missing_calls:
        p(f"- 유효 고객 발화가 하나도 없어 제외된 콜: **{len(missing_calls)}건**")
        p("")

    p("**오디오**: AIHub 원천이 이미 발화 단위 개별 WAV라 슬라이싱 없이 그대로 복사한다. "
      "리샘플·정규화·재인코딩이 개입하지 않는다(원음 바이트 동일). "
      "모델 입력 규격인 16kHz 리샘플은 Colab 로더가 담당한다.")
    p("")

    audio_dir.mkdir(parents=True, exist_ok=True)
    rows, n_missing, n_broken = [], 0, 0
    for r in seg.itertuples():
        src = AUDIO_BASE_DIR / r.audioPath
        dst = audio_dir / r.call_id / f"c{r.utt_idx}.wav"
        rec = {"call_id": r.call_id, "utt_idx": int(r.utt_idx),
               "dialog_idx": r.dialog_idx, "duration": r.duration}
        if not src.exists():
            n_missing += 1
            rows.append({**rec, "wav_path": None, "wav_duration": None,
                         "sample_rate": None, "n_channels": None, "error": "원본없음"})
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        dur, sr, ch = wav_info(dst)
        if dur is None:
            n_broken += 1
        rows.append({**rec, "wav_path": str(dst.relative_to(out_dir)),
                     "wav_duration": dur, "sample_rate": sr, "n_channels": ch,
                     "error": None if dur is not None else "WAV읽기실패"})

    man = pd.DataFrame(rows)
    man.to_parquet(out_dir / "probe_manifest.parquet", index=False)

    # ---------- 라벨 ----------
    gold = pd.read_parquet(GOLD_PATH)
    gcol = "label" if "label" in gold.columns else "gold_actual"
    gold = gold[["call_id", gcol]].rename(columns={gcol: "gold_actual"})

    gender = (seg.groupby("call_id")["speaker_gender"]
                 .agg(lambda s: s.mode().iat[0] if len(s.mode()) else None))
    n_utt = seg.groupby("call_id").size().rename("n_utt")
    lab = (pd.DataFrame({"gender": gender}).join(n_utt).reset_index()
             .merge(gold, on="call_id", how="left"))
    lab["is_complaint"] = (lab["gold_actual"] == COMPLAINT).astype(int)
    lab.to_parquet(out_dir / "probe_labels.parquet", index=False)

    p("## 추출 결과")
    p("")
    p(f"- 원본 WAV 없음: **{n_missing}건**")
    p(f"- WAV 읽기 실패: **{n_broken}건**")
    p(f"- 정상 추출: **{len(man) - n_missing - n_broken:,}발화**")
    p("")
    sr_counts = man["sample_rate"].value_counts(dropna=True).to_dict()
    p(f"- 샘플레이트 분포: {sr_counts} (원본 유지, 리샘플 없음)")
    p(f"- 채널 분포: {man['n_channels'].value_counts(dropna=True).to_dict()}")
    p("")
    p("### 콜당 고객 발화 수")
    p("")
    p("| 발화 수 | 콜 수 |")
    p("|---|---|")
    for k, v in n_utt.value_counts().sort_index().items():
        p(f"| {int(k)} | {int(v):,} |")
    p("")
    p("### 라벨 분포")
    p("")
    p("| gold_actual | 콜 수 |")
    p("|---|---|")
    for k, v in lab["gold_actual"].value_counts().items():
        p(f"| {k} | {int(v):,} |")
    p("")
    n_c = int(lab["is_complaint"].sum())
    p(f"이진 라벨: 불만제기 **{n_c:,}건 ({n_c/len(lab)*100:.1f}%)** / "
      f"비불만 {len(lab)-n_c:,}건.")
    p("")
    p("| 성별(고객 최빈) | 콜 수 |")
    p("|---|---|")
    for k, v in lab["gender"].value_counts(dropna=False).items():
        p(f"| {k} | {int(v):,} |")
    p("")
    p("성별은 sanity check의 **positive control** 라벨이다. 오디오 인코더가 우리 "
      "8kHz 한국어를 제대로 표현한다면 성별은 선형으로 쉽게 갈려야 한다. "
      "성별 AUC도 chance 근처면 '불만이 안 갈린다'가 아니라 "
      "'모델이 이 오디오를 못 읽는다'가 된다.")
    p("")

    if args.with_text:
        spot = (seg[seg["call_id"].isin(lab["call_id"].head(N_TEXT_SPOTCHECK))]
                [["call_id", "utt_idx", "text_clean"]])
        spot.to_parquet(out_dir / "probe_text_spotcheck.parquet", index=False)
        p(f"- 전사 대조용 `probe_text_spotcheck.parquet` 생성 "
          f"({spot['call_id'].nunique()}콜 / {len(spot)}발화). "
          f"저음질 sanity check에서 모델 전사와 대조하는 용도이며, "
          f"**저장소에는 커밋하지 않는다**.")
        p("")

    p("## Colab 업로드")
    p("")
    p("```")
    p(f"outputs/probe_set_{name}/")
    p("  audio/                     -> Drive: MyDrive/probe_set/audio/")
    p("  probe_manifest.parquet     -> Drive: MyDrive/probe_set/")
    p("  probe_labels.parquet       -> Drive: MyDrive/probe_set/")
    p("```")
    p("")
    p("노트북의 `PROBE_ROOT`를 그 경로로 맞추면 된다.")
    p("")

    with open(out_dir / "probe_prep_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    print(f"\n저장: {out_dir}")
    print("PROBE_AUDIO_PREP_COMPLETE")


if __name__ == "__main__":
    main()
