# 03. 조건 실험 — 통화 초반 N발화로 니즈 예측 (A/B/C/D × N)

**질문**: 통화 초반 일부(N=1,2,3,5)만 보고 니즈를 맞출 때, 텍스트만(A/C) 쓸 때와 음성 특징을 함께(B/D) 줄 때 정확도가 달라지는가?

## 스크립트

| 스크립트 | 역할 |
|---|---|
| `scripts/stage2d_prompt.py` | **공용 프롬프트 조립** (`CATEGORY_DEFINITIONS` 원본) |
| `scripts/stage2d_chunk_plan.py / stage2d_check.py / stage2d_autorun.sh` | 16조건 909청크 Batch 실행 |
| `scripts/stage2e~stage2k` | 조건별 정확도 · per-label 효과 · mismatch 부분집합 · confusion |
| `scripts/stage2h/2i_mcnemar_*.py, mcnemar_v3.py` | 쌍대 유의성 검정 |
| `scripts/stage2m_*.py, stage2_bd_*.py` | 성별 재정규화 후 B/D 재실행 및 before/after |
| `scripts/stage2p/2q_human_pilot_*.py` | 사람 파일럿 (텍스트만 → 음성 후 판단 변화) |
| `scripts/stage2r_refund_cancel_boundary.py` | 환불↔취소 경계 진단 |

## 산출물

| 파일 | 크기 |
|---|---|
| `human_pilot_answers.md` | 7 KB |
| `human_pilot_meta.csv` | 1 KB |
| `shift_mismatch_full.parquet` | 160 KB |
| `stage2_agent_vs_acoustic_effect.csv` | 3 KB |
| `stage2_bd_before_after.csv` | 8 KB |
| `stage2_bd_renorm_metrics.csv` | 13 KB |
| `stage2_bd_v3_gpt_predictions.parquet` | 1,038 KB |
| `stage2_bd_v3_metrics.csv` | 13 KB |
| `stage2_bd_v3_vs_base.csv` | 22 KB |
| `stage2_before_after_comparison.csv` | 2 KB |
| `stage2_condition_summary.csv` | 1 KB |
| `stage2_mcnemar_full.csv` | 2 KB |
| `stage2_mcnemar_results.csv` | 1 KB |
| `stage2_mismatch_acoustic_delta.csv` | 12 KB |
| `stage2_mismatch_acoustic_delta_after.csv` | 10 KB |
| `stage2_mismatch_confusion_metrics.csv` | 16 KB |
| `stage2_mismatch_confusion_metrics_after.csv` | 13 KB |
| `stage2_mismatch_subset_accuracy.csv` | 0 KB |
| `stage2_per_label_acoustic_effect.csv` | 4 KB |
| `stage2d_gpt_predictions.parquet` | 2,354 KB |
| `stage2m_gpt_predictions_BD.parquet` | 1,041 KB |

## 결론

핵심 지표 정의가 여기서 나온다. **mismatch = 조건 A / N=1 예측 ≠ gold_actual** — 즉 '통화 첫 발화의 표면 니즈'와 '통화 전체의 실제 니즈'가 어긋난 콜. 유효 19,548건 중 9,658건(49.4%)이 mismatch다.

acoustic을 얹은 효과는 전체 평균에서는 작다. 성별 재정규화 후에도 방향이 크게 달라지지 않았고, 이 때문에 '어디에서' 효과가 나는지를 좁히는 후속 단계(04~07)로 넘어갔다.
