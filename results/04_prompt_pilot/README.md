# 04. 프롬프트 변형 파일럿

**질문**: 음성 정보를 '어떻게 설명해 주느냐'에 따라 모델이 불만제기를 더 잘 잡는가?

## 스크립트

| 스크립트 | 역할 |
|---|---|
| `scripts/prompt_pilot_*.py / prompt_pilot_autorun.sh` | v_base vs 음성 안내문 유무(kr_full/kr_slim) |
| `scripts/prompt_explore_*.py` | v2(억제문) / v3(분석 태그) / v4(2단계 게이트) 비교 |
| `scripts/prompt_explore_v3_*.py` | v3 축 변형 4종 (v3_orig/a/b/c) |

## 산출물

| 파일 | 크기 |
|---|---|
| `prompt_explore.xlsx` | 13 KB |
| `prompt_explore_base_reused.parquet` | 3 KB |
| `prompt_explore_gpt_predictions.parquet` | 10 KB |
| `prompt_explore_v3.xlsx` | 13 KB |
| `prompt_explore_v3_gpt_predictions.parquet` | 10 KB |
| `prompt_explore_v3_reused.parquet` | 5 KB |
| `prompt_pilot.xlsx` | 17 KB |
| `prompt_pilot_gpt_predictions.parquet` | 13 KB |
| `prompt_pilot_sample_a.csv` | 8 KB |
| `prompt_pilot_sample_b.csv` | 9 KB |

## 결론

v3(먼저 분석 → 태그 출력)이 v_base 대비 순효과가 가장 컸다. 다만 음성 안내문(`VOICE_GUIDE_*`)은 **이 파일럿 계열에서만** 사용됐고, 이후 250건 파일럿(06)은 전부 `version="base"`로 돌았다 — 즉 06의 조건 B 프롬프트에는 음성 해석 안내가 한 줄도 들어가지 않고, 발화별 `[음성 특징: ...]` 라인만 인터리브된다. 06과 대조 실험을 설계할 때 반드시 확인해야 하는 지점이다.
