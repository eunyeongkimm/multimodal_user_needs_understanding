# 06. 250건 paired 파일럿 — 모델 / effort / 모달리티 / 언어

**질문**: 'gold=불만제기인데 환불요청으로 예측'하는 오류를, 모델·추론량·모달리티·추론언어 중 무엇으로든 줄일 수 있는가?

## 스크립트

| 스크립트 | 역할 |
|---|---|
| `scripts/model_pilot_sample.py` | 층화 250건 추출 (seed=42, 불만 오버샘플 7→50). **이후 전 실험 공통 표본** |
| `scripts/model_pilot_run.py` | Responses API 동기 호출 + concurrency 6 + 재시도. 프롬프트 바이트 동일성 검증 포함 |
| `scripts/model_pilot_eval.py` | 5조건(4.1현재 / 5.4-low / 5.4-med / luna-low / luna-med) 절대값 평가 |
| `scripts/modality_ab_eval.py` | 조건 A(text-only) vs B(text+acoustic), 둘 다 5.4-low |
| `scripts/nl_en_template.py` | 음성 description 영어 템플릿 (MT 아님 — level 컬럼에서 직접 생성) |
| `scripts/lang_pilot_translate.py` | 전사 MT 1회 캐시 (재번역 금지) |
| `scripts/lang_pilot_run.py / lang_pilot_eval.py` | KO / EN-rsn / EN-full 3조건 |
| `scripts/arousal_target_check.py` | 공통 오답 콜의 초반 고객 arousal 진단 (API 0, 전부 로컬) |

## 산출물

| 파일 | 크기 |
|---|---|
| `model_pilot_summary.md` | 6 KB |
| `model_pilot_sample.csv` | 10 KB |
| `model_pilot_percall.parquet` | 10 KB |
| `model_pilot_pred_54low.parquet` | 6 KB |
| `model_pilot_pred_54med.parquet` | 6 KB |
| `model_pilot_pred_56terra_med.parquet` | 6 KB |
| `model_pilot_pred_lunalow.parquet` | 6 KB |
| `model_pilot_pred_lunamed.parquet` | 6 KB |
| `model_pilot_pred_A_54low.parquet` | 6 KB |
| `modality_ab_summary.md` | 4 KB |
| `modality_ab_percall.parquet` | 6 KB |
| `lang_pilot_summary.md` | 6 KB |
| `lang_pilot_percall.parquet` | 8 KB |
| `lang_pilot_pred_ko.parquet` | 6 KB |
| `lang_pilot_pred_en_rsn.parquet` | 6 KB |
| `lang_pilot_pred_en_full.parquet` | 6 KB |
| `arousal_target_summary.md` | 5 KB |
| `arousal_target_percall.parquet` | 21 KB |

## 결론

**이 프로젝트의 중심 발견.** 같은 250콜에서 다섯 개 축을 흔들었는데 `gold=불만제기 → 예측=환불요청` 셀이 거의 움직이지 않았다.

| 흔든 축 | 조건 | 불만→환불 셀 |
|---|---|---|
| 모델 버전 | 4.1-mini / 5.4 / luna / terra | 18~20 |
| reasoning effort | low / medium | 18~20 |
| 모달리티 | A(text) / B(text+acoustic) | 19 / 19 |
| 추론 언어 | 한국어 / 영어 | 19 / 19 |
| 입력 언어 | 한국어 / 영어(MT) | 19 / 18 |

성능 자체는 개선된다(macro-F1 0.476→0.538, 불만 F1 0.036→0.364). 그런데도 **A·B 양쪽에서 동시에 틀리는 18콜**은 그대로 남는다.

arousal 진단 결과 그 18콜의 초반 고객 arousal은 오히려 맞춘 불만군보다 **높았고**(0.407 vs 0.104), Mann-Whitney 전 지표 p>0.22로 유의하지 않다. 즉 '음성이 밋밋해서 놓친다'는 가설은 지지되지 않는다.
