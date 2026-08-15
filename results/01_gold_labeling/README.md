# 01. gold_actual 라벨링 (v1 → v5)

**질문**: 통화 전체를 보고 매긴 '실제 니즈' 7-카테고리 라벨을 사람 수준으로 신뢰할 만하게 만들 수 있는가? 특히 항의성 콜(불만제기)을 놓치지 않는가?

## 스크립트

| 스크립트 | 역할 |
|---|---|
| `scripts/prepare_human_eval.py / build_human_eval_final.py / compute_kappa.py` | 사람 평가셋 구성 + Cohen's Kappa |
| `scripts/pilot_label_llm.py` | GPT-5.5 vs 4.1-mini 파일럿 200건 (v1) |
| `scripts/relabel_v2_and_kappa.py` | v2 프롬프트 80건 재검증 |
| `scripts/batch1_chunk_plan.py / batch1_check.py / batch1_autorun.sh` | batch1 전체 19,847건 라벨링 (v2) |
| `scripts/v3_*.py / v3_autorun.sh` | v3: 불만 우선판단 규칙 삽입 → Kappa 하락으로 폐기 |
| `scripts/v4_*.py / v4_autorun.sh` | v4: 항의 게이트 → 카테고리 2단계 분리 |
| `scripts/v5_*.py / v5_autorun.sh` | v5: v4 부작용 복원, 최종본 확정 |
| `scripts/v_unified_*.py` | 통합 프롬프트 검증 (층화 2,000건) |

## 산출물

| 파일 | 크기 |
|---|---|
| `gold_actual_batch1.parquet` | 142 KB |
| `gold_actual_batch1_final.parquet` | 143 KB |
| `gold_actual_batch1_v4.parquet` | 85 KB |
| `gold_actual_batch1_v5.parquet` | 80 KB |
| `human_eval_v3_labels.csv` | 1 KB |
| `pilot200_gpt_labels.csv` | 8 KB |
| `pilot80_gpt_labels_v2.csv` | 3 KB |
| `refund_cancel_boundary.xlsx` | 6 KB |
| `v3_target_call_ids.csv` | 283 KB |
| `v4_self_consistency_pass2.csv` | 6 KB |
| `v4_target_call_ids.csv` | 283 KB |
| `v5_target_call_ids.csv` | 130 KB |
| `v_unified_sample_call_ids.csv` | 50 KB |
| `v_unified_sample_labels.parquet` | 36 KB |

## 결론

**`gold_actual_batch1_final.parquet`(19,847건)이 이후 모든 분석의 기준 라벨.**

v2는 불만제기를 32건밖에 못 잡았다. v3(정의 안에 규칙 삽입)은 환불요청↔주문취소 경계를 흔들어 Kappa 0.82→0.71로 떨어져 폐기했다. v4(항의 게이트 분리)가 불만제기 452건을 검출했으나 환불요청을 다른 범주로 과다 이동시켰고, v5가 `has_complaint=False` 건만 재분류해 분포를 복원했다. 최종 불만제기 484건(2.44%). v4 자기일관성 Kappa 0.96.

교훈: 카테고리 정의를 건드리는 대신 **판단 단계를 분리**하는 편이 안전했다. 이후 `CATEGORY_DEFINITIONS`는 수정 금지 상수로 고정된다.
