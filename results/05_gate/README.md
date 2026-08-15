# 05. 정보부족 게이트 (probing · train10k 분류기)

**질문**: '초반 텍스트만으로는 판단이 안 되는 콜'을 미리 골라낼 수 있는가? 그런 콜에만 음성을 쓰면 이득이 있는가?

## 스크립트

| 스크립트 | 역할 |
|---|---|
| `scripts/probing_build_features.py / probing_experiments.py` | 콜 단위 피처 구축 + 실험 A~F |
| `scripts/probing_full_acoustic_extract.py` | 전체 콜 acoustic 확장 추출 |
| `scripts/gate_signal_build.py` | 게이트 학습 신호 구성 |
| `scripts/gate_prompt_*.py` | 게이트 확률을 v3 프롬프트에 주입한 실전 검증 |
| `scripts/train10k_*.py` | train 1만 라벨링 → 피처 집계 → 분류기 학습 (pool5 재구성 포함) |
| `scripts/trigger_comparison.py` | 트리거 기준 비교 |

## 산출물

| 파일 | 크기 |
|---|---|
| `gate_infopoor_layer_comparison.csv` | 0 KB |
| `gate_prompt_gpt_predictions.parquet` | 56 KB |
| `gate_signal_inference.parquet` | 339 KB |
| `gate_signal_train10k.parquet` | 172 KB |
| `infopoor_tagged.csv` | 670 KB |
| `probing_feature_importance.csv` | 3 KB |
| `probing_gate_threshold_sweep.csv` | 1 KB |
| `probing_layer_acoustic_delta.csv` | 0 KB |
| `rescue_harm_analysis.csv` | 1 KB |
| `rescue_harm_raw.parquet` | 30 KB |
| `shift_trigger_validation.csv` | 0 KB |
| `train10k_acoustic_features.parquet` | 1,586 KB |
| `train10k_call_ids.csv` | 117 KB |
| `train10k_gpt_labels.parquet` | 140 KB |
| `train10k_gpt_labels_v4.parquet` | 83 KB |
| `train10k_gpt_labels_v5.parquet` | 79 KB |
| `train10k_v4_target_call_ids.csv` | 141 KB |
| `train10k_v5_target_call_ids.csv` | 65 KB |
| `train_10k_final_labeled.parquet` | 73 KB |
| `train_10k_labeled.parquet` | 1,524 KB |
| `train_10k_pool5_labeled.parquet` | 1,544 KB |
| `trigger_comparison.csv` | 2 KB |

## 결론

`infopoor_tagged.csv`가 '정보부족' 태그의 산출물이며, 이후 **acoustically-valid subset** 정의(조건 B / N=5 / customer_only ∩ `is_infopoor==False` = 16,777건)의 근거가 된다. 06의 250건 표본은 전부 이 모집단에서 뽑았다.

게이트 임계값은 train에서 구해 추론에 **고정 적용**했고, 모든 판단은 CV 기반, 스케일링은 fold 내부에서만 적합시켰다.
