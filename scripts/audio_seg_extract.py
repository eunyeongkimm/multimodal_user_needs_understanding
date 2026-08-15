"""audio 실험용 초반 5발화 고객 세그먼트 WAV 준비.

GPT A/B(조건 B, window_n=5, window_version=customer_only)와 **정확히 같은**
250콜 / 1,119발화를 쓴다. 발화 선택을 새로 계산하지 않고
outputs/stage2_windows_nl_v2.parquet 의 기존 창 정의를 그대로 재사용한다.

오디오는 AIHub 원천이 이미 발화 단위 개별 WAV(audioPath가 행마다 고유)이므로
timestamp 슬라이싱 없이 그대로 복사한다 -> 리샘플/정규화가 개입할 여지 자체가 없다.
외장하드 원본은 읽기 전용으로만 접근한다.

출력:
  outputs/audio_seg/{call_id}/c{i}.wav   (i = 1..n_customer_utt, 고객 발화 순번)
  outputs/audio_seg_manifest.parquet
  outputs/audio_seg_summary.md
"""

import shutil
import sys
import wave
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_PATH = BASE_DIR / "outputs" / "model_pilot_sample.csv"
WINDOWS_PATH = BASE_DIR / "outputs" / "stage2_windows_nl_v2.parquet"
D04_PATH = BASE_DIR / "outputs" / "d04_dialog_index.parquet"
OUT_DIR = BASE_DIR / "outputs" / "audio_seg"
MANIFEST_PATH = BASE_DIR / "outputs" / "audio_seg_manifest.parquet"
MD_PATH = BASE_DIR / "outputs" / "audio_seg_summary.md"

AUDIO_BASE_DIR = Path(
    "/Volumes/AIHub/007.저음질 전화망 음성인식 데이터/01.데이터/1.Training/원천데이터_230316"
)

WINDOW_N = 5
WINDOW_VERSION = "customer_only"
CUSTOMER_PREFIX = "고객"
N_SPOTCHECK = 3

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


def main():
    if not AUDIO_BASE_DIR.exists():
        print(f"ERROR: 외장하드 미마운트 - {AUDIO_BASE_DIR}")
        sys.exit(1)

    sample = pd.read_csv(SAMPLE_PATH)
    win = pd.read_parquet(WINDOWS_PATH, columns=[
        "call_id", "window_n", "window_version", "dialog_idx",
        "speaker_type", "customer_rank", "text_clean"])
    win = win[(win["window_n"] == WINDOW_N) & (win["window_version"] == WINDOW_VERSION)
              & win["call_id"].isin(sample["call_id"])]

    assert set(win["call_id"]) == set(sample["call_id"]), "250 call_id 집합 불일치"
    assert win["speaker_type"].astype(str).str.startswith(CUSTOMER_PREFIX).all(), \
        "고객 이외 화자 포함"

    d04 = pd.read_parquet(D04_PATH, columns=["call_id", "dialog_idx", "audioPath", "duration"])
    seg = win.merge(d04, on=["call_id", "dialog_idx"], how="left")
    assert seg["audioPath"].isna().sum() == 0, "audioPath 결측"
    seg = seg.sort_values(["call_id", "customer_rank"]).reset_index(drop=True)
    seg["utt_idx"] = seg["customer_rank"].astype(int)
    seg["n_customer_utt"] = seg.groupby("call_id")["call_id"].transform("size")

    p("# 초반 5발화 고객 세그먼트 WAV 추출")
    p("")
    p("GPT A/B와 **동일한 발화 선택**을 재사용했다: "
      f"`stage2_windows_nl_v2.parquet`의 `window_n={WINDOW_N}` & "
      f"`window_version={WINDOW_VERSION}` 창을 그대로 가져왔고 새로 계산하지 않았다. "
      f"250 call_id 집합이 `model_pilot_sample.csv`와 완전 일치함을 assert로 확인.")
    p("")
    p("**오디오 저장 형태**: AIHub 원천은 이미 **발화 단위 개별 WAV**다"
      "(`audioPath`가 1,672,062행 모두 고유, 예: `D04/J16/S000001/0001.wav`). "
      "따라서 full-call 슬라이싱 없이 **그대로 복사**했고, 리샘플·정규화·재인코딩이 "
      "개입하지 않는다(원음 바이트 동일). 외장하드 원본은 읽기 전용 접근.")
    p("")
    p(f"**화자 식별**: `speaker_type` 접두 `'{CUSTOMER_PREFIX}'`만 사용(id 미사용). "
      f"상담사 발화는 건너뛴 customer_only 선택이며 시간 통구간 자르기가 아니다.")
    p("")
    p(f"대상: {len(seg)}발화 / {seg['call_id'].nunique()}콜")
    p("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, n_missing, n_broken = [], 0, 0
    for r in seg.itertuples():
        src = AUDIO_BASE_DIR / r.audioPath
        dst_dir = OUT_DIR / r.call_id
        dst = dst_dir / f"c{r.utt_idx}.wav"
        if not src.exists():
            n_missing += 1
            rows.append({"call_id": r.call_id, "utt_idx": r.utt_idx, "wav_path": None,
                         "dialog_idx": r.dialog_idx, "duration": r.duration,
                         "text": r.text_clean, "n_customer_utt": r.n_customer_utt,
                         "wav_duration": None, "sample_rate": None, "error": "원본없음"})
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        dur, sr, ch = wav_info(dst)
        err = None
        if dur is None:
            n_broken += 1
            err = "WAV읽기실패"
        rows.append({"call_id": r.call_id, "utt_idx": r.utt_idx,
                     "wav_path": str(dst.relative_to(BASE_DIR)),
                     "dialog_idx": r.dialog_idx, "duration": r.duration,
                     "text": r.text_clean, "n_customer_utt": r.n_customer_utt,
                     "wav_duration": dur, "sample_rate": sr, "n_channels": ch,
                     "error": err})

    man = pd.DataFrame(rows)
    man.to_parquet(MANIFEST_PATH, index=False)

    p("## 검증")
    p("")
    nc = seg.groupby("call_id")["utt_idx"].max()
    dist = nc.value_counts().sort_index()
    p("| 콜당 고객발화 수 | 콜 수 |")
    p("|---|---|")
    for k, v in dist.items():
        p(f"| {int(k)} | {int(v)} |")
    p("")
    n_short = int((nc < WINDOW_N).sum())
    p(f"- 5발화 미만 콜: **{n_short}건** ({n_short/len(nc)*100:.1f}%) — "
      f"있는 만큼만 추출하고 `n_customer_utt`에 실제 개수 기록. 제외하지 않았다.")
    p(f"- 원본 WAV 없음: **{n_missing}건**")
    p(f"- WAV 읽기 실패(깨진 파일): **{n_broken}건**")
    p("")

    ok = man[man["wav_path"].notna()]
    total_bytes = sum((BASE_DIR / w).stat().st_size for w in ok["wav_path"])
    p(f"- 총 파일 수: **{len(ok):,}개** / 콜 디렉터리 {ok['call_id'].nunique()}개")
    p(f"- 총 용량: **{total_bytes/1024/1024:.1f} MB** ({total_bytes:,} bytes)")
    srs = ok["sample_rate"].dropna().unique()
    p(f"- 샘플레이트: {sorted(int(x) for x in srs)} Hz (원본 유지, 리샘플 없음)")
    p(f"- 채널: {sorted(int(x) for x in ok['n_channels'].dropna().unique())}")
    p("")

    p("## 스팟체크 (manifest text vs 실제 WAV 길이)")
    p("")
    spot_calls = list(ok["call_id"].drop_duplicates().head(N_SPOTCHECK))
    for cid in spot_calls:
        g = ok[ok["call_id"] == cid].sort_values("utt_idx")
        p(f"**{cid}** (n_customer_utt={int(g['n_customer_utt'].iat[0])})")
        p("")
        p("| utt_idx | dialog_idx | index duration | 실제 WAV duration | 차이 | text |")
        p("|---|---|---|---|---|---|")
        for r in g.itertuples():
            diff = abs(r.wav_duration - r.duration) if r.wav_duration else float("nan")
            txt = str(r.text)[:40] + ("…" if len(str(r.text)) > 40 else "")
            p(f"| c{r.utt_idx} | {r.dialog_idx} | {r.duration:.3f}s | "
              f"{r.wav_duration:.3f}s | {diff:.3f}s | {txt} |")
        p("")

    d = (ok["wav_duration"] - ok["duration"]).abs()
    p(f"전체 {len(ok)}개 파일에서 index duration과 실제 WAV 길이 차이: "
      f"최대 {d.max():.4f}s, 평균 {d.mean():.4f}s, 0.01s 초과 {int((d > 0.01).sum())}건")
    p("")

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n저장: {MANIFEST_PATH} ({len(man)}행)")
    print(f"저장: {MD_PATH}")
    print(f"저장: {OUT_DIR}/")
    print("AUDIO_SEG_EXTRACT_COMPLETE")


if __name__ == "__main__":
    main()
