# 00. 원천 인덱스 구축 · 전처리

**질문**: 74,123콜의 D04 라벨링 JSON을 발화 단위 인덱스로 만들고, 분석에 쓸 수 있는 품질로 정제할 수 있는가?

## 스크립트

| 스크립트 | 역할 |
|---|---|
| `scripts/build_index.py` | D04 라벨링 JSON → dialog 단위 인덱스 |
| `scripts/clean_text.py` | disfluency 태그 제거, r_gold(발화속도)·r_gold_valid_flag 계산 |
| `scripts/split_batches.py` | batch1(20,000) / batch2(54,123) 고정 분할 (seed=42) |
| `scripts/label_inbound_outbound.py` | 인바운드/아웃바운드 판별 (사람 대조 80건) |
| `scripts/extract_acoustic_features.py` | acoustic 추출 프로토타입 (50건, F0/Energy/silence 검증) |

## 산출물

| 파일 | 크기 |
|---|---|
| `acoustic_features_sample50.csv` | 12 KB |
| `batch1_call_ids.csv` | 234 KB |
| `batch1_no_valid_text_call_ids.csv` | 2 KB |
| `batch2_call_ids.csv` | 634 KB |
| `d04_dialog_index_column_dictionary.md` | 4 KB |
| `data_cleaning_report.md` | 4 KB |
| `inbound_outbound_gpt.csv` | 2 KB |

## 결론

발화 단위 인덱스 1,672,062행 확보. `r_gold_valid_flag`로 발화속도 이상치를 걸러 이후 모든 acoustic 분석의 유효 구간을 정의했다. batch1/batch2 분할은 seed=42로 고정되어 이후 전 단계가 같은 콜 집합을 참조한다.
