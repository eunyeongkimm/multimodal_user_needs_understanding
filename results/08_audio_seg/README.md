# 08. 오디오 세그먼트 준비 (audio 모델용)

**질문**: GPT A/B와 정확히 같은 250콜·같은 발화로 audio 모델 입력을 만들 수 있는가?

## 스크립트

| 스크립트 | 역할 |
|---|---|
| `scripts/audio_seg_extract.py` | 초반 5발화 고객 세그먼트 WAV 추출 + manifest + 검증 |

## 산출물

| 파일 | 크기 |
|---|---|
| `audio_seg_summary.md` | 3 KB |
| `audio_seg_manifest.parquet` | 29 KB |

## 결론

AIHub 원천은 이미 **발화 단위 개별 WAV**다(`audioPath`가 1,672,062행 모두 고유). 따라서 timestamp 슬라이싱이 필요 없고 `shutil.copy2`로 바이트 동일 복사만 하면 되므로 리샘플·정규화가 개입할 여지가 구조적으로 없다.

결과: 1,119개 / 250콜 / 78.6MB / 8kHz mono. 원본 결측 0건, 깨진 파일 0건. index duration 대비 실제 WAV 길이 차이는 최대 0.059초, 0.01초 초과 1건.
5발화 미만 콜 67건(26.8%)은 제외하지 않고 `n_customer_utt`에 실제 개수를 기록했다.

> **WAV 파일 자체는 이 repo에 없다** (AIHub 재배포 금지). `audio_seg_extract.py`를 외장하드 마운트 상태에서 돌리면 로컬에 재생성된다. manifest도 전사(`text`) 컬럼을 제거한 상태로 수록했다.
