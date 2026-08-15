# 02. 음향 피처 추출 · 정규화

**질문**: 발화 단위 F0/에너지/무음을 어떤 기준으로 정규화해야 '이 화자가 평소보다 격앙됐다'가 제대로 표현되는가?

## 스크립트

| 스크립트 | 역할 |
|---|---|
| `scripts/stage2a_window_pool.py` | N=1,2,3,5 × 고객only/고객&상담사 윈도우 pool |
| `scripts/stage2b_acoustic_extract.py` | 발화 단위 F0·Energy·silence 추출 (8코어 병렬) |
| `scripts/stage2c_normalize_nl.py` | Track A: 콜/윈도우 내 z-score + 5분위 자연어 변환 |
| `scripts/stage2c_v2_normalize_nl_gender.py` | **성별 population 기준 정규화 (채택본)** |
| `scripts/stage2c_v3_normalize_nl_shrinkage.py` | Track B: shrinkage 정규화 (진단용) |
| `scripts/stage2l_gender_population_stats.py` | 성별 population 통계 산출 |

## 산출물

| 파일 | 크기 |
|---|---|
| `stage2_gender_population_stats.csv` | 1 KB |
| `stage2_renorm_application_log.csv` | 5 KB |
| `stage2_trackA_vs_trackB_diagnosis.csv` | 1 KB |

## 결론

콜 내부 z-score(Track A)는 화자가 통화 내내 격앙된 콜에서 그 격앙을 '평균'으로 흡수해 버린다. **같은 성별 화자 집단 대비 정규화**로 바꾼 것이 채택본이며, 성별을 섞으면 남녀 pitch 차이(남 138.5Hz / 여 223.1Hz)가 arousal로 오염된다.

주의: `e_mean_db`는 `amplitude_to_db(rms, ref=np.max)`로 계산돼 **발화 자신의 피크 대비** 값이다. 절대 음량이 아니라 dynamic range이므로 arousal 용도로는 `e_mean_amp`(선형 RMS)를 써야 한다.
