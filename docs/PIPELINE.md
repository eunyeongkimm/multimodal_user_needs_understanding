# 파이프라인 지도

`scripts/`는 평면 구조다. 하위 폴더로 옮기면 102개 스크립트의 `outputs/` 경로 상수와 상호 import가 깨지기 때문에 재현성을 위해 그대로 뒀다. 대신 **어느 단계에서 어떤 스크립트를 보면 되는지는 이 문서가 책임진다.**

결과물은 `results/<단계>/`에 단계별로 정리돼 있고, 각 폴더의 `README.md`에 그 단계의 질문·스크립트·산출물·결론이 적혀 있다.

---

## 흐름

```
00 전처리 ── 01 gold 라벨링 ── 02 음향 정규화 ── 03 조건 실험(A/B/C/D × N)
                                                        │
                              ┌─────────────────────────┴──────────────┐
                              │                                        │
                     04 프롬프트 변형                          05 정보부족 게이트
                              │                                        │
                              └─────────────┬──────────────────────────┘
                                            │
                              06 250건 paired 파일럿 (모델·effort·모달리티·언어)
                                            │
                              ┌─────────────┴─────────────┐
                              │                           │
                     07 상담사 품질 층화          08 오디오 세그먼트 준비
                                                          │
                                            09 audio-native 모델 (Qwen-Omni)
                                              zero-shot + 프롬프트 스윕
                                                          │
                                            10 fine-tuning 타당성 확인
```

09·10단계는 Colab에서 돌아가므로 스크립트가 `scripts/`가 아니라 `results/<단계>/*.ipynb`에 있다. 규약은 [`WORKFLOW.md`](WORKFLOW.md) 참고.

---

## 단계별 지도

| 단계 | 무엇을 물었나 | 시작점 스크립트 | 결과 |
|---|---|---|---|
| [00](../results/00_data_prep/) 원천 인덱스 구축 · 전처리 | 74,123콜의 D04 라벨링 JSON을 발화 단위 인덱스로 만들고, 분석에 쓸 수 있는 품질로 정제할 수 있는가? | `scripts/build_index.py` | [`results/00_data_prep/`](../results/00_data_prep/) |
| [01](../results/01_gold_labeling/) gold_actual 라벨링 (v1 → v5) | 통화 전체를 보고 매긴 '실제 니즈' 7-카테고리 라벨을 사람 수준으로 신뢰할 만하게 만들 수 있는가? 특히 항의성 콜(불만제기)을 놓치지 않는가? | `scripts/prepare_human_eval.py` | [`results/01_gold_labeling/`](../results/01_gold_labeling/) |
| [02](../results/02_acoustic/) 음향 피처 추출 · 정규화 | 발화 단위 F0/에너지/무음을 어떤 기준으로 정규화해야 '이 화자가 평소보다 격앙됐다'가 제대로 표현되는가? | `scripts/stage2a_window_pool.py` | [`results/02_acoustic/`](../results/02_acoustic/) |
| [03](../results/03_condition_exp/) 조건 실험 — 통화 초반 N발화로 니즈 예측 (A/B/C/D × N) | 통화 초반 일부(N=1,2,3,5)만 보고 니즈를 맞출 때, 텍스트만(A/C) 쓸 때와 음성 특징을 함께(B/D) 줄 때 정확도가 달라지는가? | `scripts/stage2d_prompt.py` | [`results/03_condition_exp/`](../results/03_condition_exp/) |
| [04](../results/04_prompt_pilot/) 프롬프트 변형 파일럿 | 음성 정보를 '어떻게 설명해 주느냐'에 따라 모델이 불만제기를 더 잘 잡는가? | `scripts/prompt_pilot_*.py` | [`results/04_prompt_pilot/`](../results/04_prompt_pilot/) |
| [05](../results/05_gate/) 정보부족 게이트 (probing · train10k 분류기) | '초반 텍스트만으로는 판단이 안 되는 콜'을 미리 골라낼 수 있는가? 그런 콜에만 음성을 쓰면 이득이 있는가? | `scripts/probing_build_features.py` | [`results/05_gate/`](../results/05_gate/) |
| [06](../results/06_pilot_250/) 250건 paired 파일럿 — 모델 / effort / 모달리티 / 언어 | 'gold=불만제기인데 환불요청으로 예측'하는 오류를, 모델·추론량·모달리티·추론언어 중 무엇으로든 줄일 수 있는가? | `scripts/model_pilot_sample.py` | [`results/06_pilot_250/`](../results/06_pilot_250/) |
| [07](../results/07_agent_quality/) 상담사 응대 품질 — 층화 변수 탐색 | 상담사가 잘 대응했는지가 mismatch를 설명하는가? 설명한다면 층화 변수로 쓸 수 있는가? | `scripts/agent_judge_prompt.py` | [`results/07_agent_quality/`](../results/07_agent_quality/) |
| [08](../results/08_audio_seg/) 오디오 세그먼트 준비 (audio 모델용) | GPT A/B와 정확히 같은 250콜·같은 발화로 audio 모델 입력을 만들 수 있는가? | `scripts/audio_seg_extract.py` | [`results/08_audio_seg/`](../results/08_audio_seg/) |
| [09](../results/09_audio_native_model_test/) audio-native 모델 테스트 (Qwen-Omni) | 텍스트 전사를 거치지 않고 오디오를 직접 먹는 모델은, 08에서 만든 같은 250콜에서 GPT 텍스트 파이프라인만큼 할 수 있는가? | `results/09_audio_native_model_test/qwen2_5_omni_test.ipynb` | [`results/09_audio_native_model_test/`](../results/09_audio_native_model_test/) |
| [10](../results/10_fine_tuning/) fine-tuning 파이프라인 타당성 확인 | 09에서 zero-shot 한계(macro-F1 0.493)를 본 Qwen3-Omni-30B를, Colab A100 40GB에서 QLoRA로 실제 학습시킬 수 있는가? | `results/10_fine_tuning/fine_tuning_v0_pipeline_test.ipynb` | [`results/10_fine_tuning/`](../results/10_fine_tuning/) |

---

## 스크립트 → 단계 역인덱스

`scripts/` 전체를 단계별로 묶은 것. 파일을 열기 전에 여기서 소속 단계를 찾고, 그 단계의 `results/<단계>/README.md`를 먼저 읽으면 맥락이 잡힌다.

### 00. 원천 인덱스 구축 · 전처리 (5개)

`build_index.py` · `clean_text.py` · `extract_acoustic_features.py` · `label_inbound_outbound.py` · `split_batches.py`

### 01. gold_actual 라벨링 (v1 → v5) (33개)

`batch1_autorun.sh` · `batch1_check.py` · `batch1_chunk_plan.py` · `batch1_submit.py` · `build_human_eval_final.py` · `compute_kappa.py` · `pilot_label_llm.py` · `prepare_human_eval.py` · `relabel_v2_and_kappa.py` · `v3_autorun.sh` · `v3_check.py` · `v3_chunk_plan.py` · `v3_human_eval_kappa.py` · `v3_merge_final.py` · `v3_prompt.py` · `v4_autorun.sh` · `v4_check.py` · `v4_chunk_plan.py` · `v4_merge_final.py` · `v4_prompt.py` · `v4_self_consistency.py` · `v4_suspect_scan.py` · `v5_autorun.sh` · `v5_check.py` · `v5_chunk_plan.py` · `v5_merge_final.py` · `v5_prompt.py` · `v_unified_autorun.sh` · `v_unified_check.py` · `v_unified_chunk_plan.py` · `v_unified_prompt.py` · `v_unified_sample.py` · `v_unified_validate.py`

### 02. 음향 피처 추출 · 정규화 (6개)

`stage2a_window_pool.py` · `stage2b_acoustic_extract.py` · `stage2c_normalize_nl.py` · `stage2c_v2_normalize_nl_gender.py` · `stage2c_v3_normalize_nl_shrinkage.py` · `stage2l_gender_population_stats.py`

### 03. 조건 실험 — 통화 초반 N발화로 니즈 예측 (A/B/C/D × N) (25개)

`mcnemar_v3.py` · `stage2_bd_renorm_finalize.py` · `stage2_bd_v3_autorun.sh` · `stage2_bd_v3_check.py` · `stage2_bd_v3_chunk_plan.py` · `stage2_bd_v3_finalize.py` · `stage2d_autorun.sh` · `stage2d_check.py` · `stage2d_chunk_plan.py` · `stage2d_prompt.py` · `stage2e_analysis.py` · `stage2f_per_label_analysis.py` · `stage2g_mismatch_subset_analysis.py` · `stage2h_mcnemar_test.py` · `stage2i_mcnemar_full.py` · `stage2j_agent_vs_acoustic_effect.py` · `stage2k_mismatch_confusion_metrics.py` · `stage2m_autorun.sh` · `stage2m_check.py` · `stage2m_chunk_plan_v2.py` · `stage2n_before_after_analysis.py` · `stage2o_trackA_vs_trackB_diagnosis.py` · `stage2p_human_pilot_sample.py` · `stage2q_pilot2_sample_excel.py` · `stage2r_refund_cancel_boundary.py`

### 04. 프롬프트 변형 파일럿 (14개)

`prompt_explore_autorun.sh` · `prompt_explore_build_excel.py` · `prompt_explore_check.py` · `prompt_explore_chunk_plan.py` · `prompt_explore_reparse.py` · `prompt_explore_v3_autorun.sh` · `prompt_explore_v3_build_excel.py` · `prompt_explore_v3_check.py` · `prompt_explore_v3_chunk_plan.py` · `prompt_pilot_autorun.sh` · `prompt_pilot_build_excel.py` · `prompt_pilot_check.py` · `prompt_pilot_chunk_plan.py` · `prompt_pilot_sample.py`

### 05. 정보부족 게이트 (probing · train10k 분류기) (28개)

`gate_prompt_autorun.sh` · `gate_prompt_check.py` · `gate_prompt_chunk_plan.py` · `gate_prompt_comparison.py` · `gate_signal_build.py` · `probing_build_features.py` · `probing_experiments.py` · `probing_full_acoustic_extract.py` · `train10k_acoustic_extract.py` · `train10k_autorun.sh` · `train10k_build_features.py` · `train10k_check.py` · `train10k_chunk_plan.py` · `train10k_classifier.py` · `train10k_merge_final.py` · `train10k_pool5_acoustic_extract.py` · `train10k_pool5_build_features.py` · `train10k_pool5_classifier.py` · `train10k_pool5_window_pool.py` · `train10k_sample.py` · `train10k_v4_autorun.sh` · `train10k_v4_check.py` · `train10k_v4_chunk_plan.py` · `train10k_v5_autorun.sh` · `train10k_v5_check.py` · `train10k_v5_chunk_plan.py` · `train10k_window_pool.py` · `trigger_comparison.py`

### 06. 250건 paired 파일럿 — 모델 / effort / 모달리티 / 언어 (9개)

`arousal_target_check.py` · `lang_pilot_eval.py` · `lang_pilot_run.py` · `lang_pilot_translate.py` · `modality_ab_eval.py` · `model_pilot_eval.py` · `model_pilot_run.py` · `model_pilot_sample.py` · `nl_en_template.py`

### 07. 상담사 응대 품질 — 층화 변수 탐색 (4개)

`agent_judge_gate.py` · `agent_judge_prompt.py` · `agent_judge_run.py` · `agent_strat_analysis.py`

### 08. 오디오 세그먼트 준비 (audio 모델용) (1개)

`audio_seg_extract.py`

### 저장소 관리 (1개)

`repo_organize.py`

---

## 반복되는 규칙 (전 단계 공통)

실험을 다시 돌리거나 확장할 때 지켜야 하는 것들.

- `stage2d_prompt.py`의 `CATEGORY_DEFINITIONS`와 gold_actual taxonomy는 **수정 금지**. 01단계 v3 실패가 여기서 나왔다.
- `version="base"` 프롬프트의 출력은 바이트 동일성을 유지해야 한다. `stage2d_prompt.py`를 건드리면 회귀 검증 필수.
- 라벨 정규화는 기존 숫자 접두어 방식만. 유사도·근접 카테고리 매칭 금지.
- 화자 식별은 `speaker_type`(접두어)만. `speaker_id`는 쓰지 않는다.
- `gold_actual`은 평가에만. 채점·judge 프롬프트에 절대 넣지 않는다. `subcategory`는 ground truth가 아니다.
- 결측·이상치를 조용히 지우거나 채우지 않는다. 항상 세어서 보고한다.
- 모든 ML 판단은 CV 기반. 스케일링은 fold 내부에서만 적합. 임계값은 train에서 구해 추론에 **고정** 적용.
- 외장하드 원본(AIHub)은 읽기 전용. 오디오 리샘플·정규화 금지(원음 그대로).
- API 제출 전 비용 추정을 먼저 보고한다. Batch 제출 스크립트에는 `error_file_id` 패치가 들어가야 한다.
