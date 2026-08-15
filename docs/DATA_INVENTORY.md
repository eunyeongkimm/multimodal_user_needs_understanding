# 로컬 데이터 인벤토리

`outputs/`는 git에 올리지 않는다(2.5GB + AIHub 재배포 금지). 하지만 **무엇이 있는지**는 저장소만 봐도 알 수 있어야 하므로, 목록·크기·행수·컬럼을 여기 남긴다.

- 전체 2,334개 항목 / 약 2.6 GB
- 저장소 수록 103개 (`results/`)
- batch 요청 JSONL 2,102개는 표에서 접어 둠(아래 요약만)

재생성: `python3 scripts/repo_organize.py` (이 문서도 함께 갱신됨)

---

## 파일 목록

| 파일 | MB | 행 × 열 | 상태 | 비고 |
|---|---|---|---|---|
| `acoustic_features_sample50.csv` | 0.0 | 50 × 17 | **수록** |  |
| `agent_judge_gate_summary.md` | 0.0 |  | **수록** |  |
| `agent_judge_human_eval_slots.csv` | 0.0 | 40 × 5 | **수록** |  |
| `agent_judge_percall.parquet` | 0.0 | 250 × 7 | **수록** |  |
| `agent_judge_scores.parquet` | 0.0 | 250 × 6 | **수록** |  |
| `agent_strat_percall.parquet` | 0.0 | 250 × 9 | **수록** |  |
| `agent_strat_summary.md` | 0.0 |  | **수록** |  |
| `arousal_target_percall.parquet` | 0.0 | 250 × 13 | **수록** |  |
| `arousal_target_summary.md` | 0.0 |  | **수록** |  |
| `audio_seg/` | 82.4 | 1,119개 파일 | 제외 | 원본 오디오(AIHub 재배포 금지) |
| `audio_seg_manifest.parquet` | 0.1 | 1,119 × 11 | **수록** | `text` 컬럼 제거 후 수록 |
| `audio_seg_summary.md` | 0.0 |  | **수록** |  |
| `batch1_autorun.log` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `batch1_autorun_stdout.log` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `batch1_call_ids.csv` | 0.2 | 5,000 × 1 | **수록** |  |
| `batch1_call_texts.parquet` | 18.0 | 19,848 × 2 | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `batch1_chunks_plan.json` | 0.3 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `batch1_chunks_state.json` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `batch1_no_valid_text_call_ids.csv` | 0.0 | 152 × 1 | **수록** |  |
| `batch1_partial/` | 0.2 | 19개 파일 | 제외 | 원본 오디오(AIHub 재배포 금지) |
| `batch2_call_ids.csv` | 0.6 | 5,000 × 1 | **수록** |  |
| `d04_dialog_index.parquet` | 161.9 |  | 제외 | 대용량 |
| `d04_dialog_index_column_dictionary.md` | 0.0 |  | **수록** |  |
| `d04_dialog_index_old_no_flag.parquet` | 85.7 |  | 제외 | 대용량 |
| `d04_dialog_index_sample.csv` | 0.0 | 5 × 26 | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `data_cleaning_report.md` | 0.0 |  | **수록** |  |
| `exclusion_log_r_gold_gt15.csv` | 4.3 | 5,000 × 14 | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `gate_infopoor_layer_comparison.csv` | 0.0 | 4 × 8 | **수록** |  |
| `gate_prompt_autorun.log` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `gate_prompt_chunks_state.json` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `gate_prompt_comparison.txt` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `gate_prompt_gpt_predictions.parquet` | 0.1 | 3,946 × 7 | **수록** |  |
| `gate_prompt_partial/` | 0.1 | 17개 파일 | 제외 | 원본 오디오(AIHub 재배포 금지) |
| `gate_prompt_request_index.parquet` | 1.3 | 3,946 × 7 | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `gate_prompt_requests_plan.json` | 0.1 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `gate_signal_inference.parquet` | 0.3 | 19,847 × 4 | **수록** |  |
| `gate_signal_train10k.parquet` | 0.2 | 10,000 × 5 | **수록** |  |
| `gold_actual_batch1.parquet` | 0.1 | 19,847 × 4 | **수록** |  |
| `gold_actual_batch1_final.parquet` | 0.1 | 19,847 × 5 | **수록** |  |
| `gold_actual_batch1_v4.parquet` | 0.1 | 11,579 × 5 | **수록** |  |
| `gold_actual_batch1_v5.parquet` | 0.1 | 11,127 × 4 | **수록** |  |
| `human_eval_final.csv` | 0.2 | 80 × 8 | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `human_eval_final.xlsx` | 0.1 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `human_eval_final_fin.csv` | 0.2 | 80 × 8 | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `human_eval_sample.csv` | 0.2 | 80 × 3 | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `human_eval_v3_labels.csv` | 0.0 | 49 × 2 | **수록** |  |
| `human_pilot_answers.md` | 0.0 |  | **수록** |  |
| `human_pilot_meta.csv` | 0.0 | 25 × 5 | **수록** |  |
| `human_pilot_questions.md` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `inbound_outbound_gpt.csv` | 0.0 | 80 × 3 | **수록** | `reason_gpt` 컬럼 제거 후 수록 |
| `infopoor_tagged.csv` | 0.7 | 5,000 × 9 | **수록** |  |
| `lang_pilot_percall.parquet` | 0.0 | 250 × 8 | **수록** |  |
| `lang_pilot_pred_en_full.parquet` | 0.0 | 250 × 5 | **수록** |  |
| `lang_pilot_pred_en_rsn.parquet` | 0.0 | 250 × 5 | **수록** |  |
| `lang_pilot_pred_ko.parquet` | 0.0 | 250 × 5 | **수록** |  |
| `lang_pilot_summary.md` | 0.0 |  | **수록** |  |
| `lang_pilot_translation_cache.parquet` | 0.1 | 1,119 × 5 | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `mcnemar_results.txt` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `modality_ab_percall.parquet` | 0.0 | 250 × 6 | **수록** |  |
| `modality_ab_summary.md` | 0.0 |  | **수록** |  |
| `model_pilot_percall.parquet` | 0.0 | 250 × 12 | **수록** |  |
| `model_pilot_pred_54low.parquet` | 0.0 | 250 × 5 | **수록** |  |
| `model_pilot_pred_54med.parquet` | 0.0 | 250 × 5 | **수록** |  |
| `model_pilot_pred_56terra_med.parquet` | 0.0 | 250 × 5 | **수록** |  |
| `model_pilot_pred_A_54low.parquet` | 0.0 | 250 × 5 | **수록** |  |
| `model_pilot_pred_lunalow.parquet` | 0.0 | 250 × 5 | **수록** |  |
| `model_pilot_pred_lunamed.parquet` | 0.0 | 250 × 5 | **수록** |  |
| `model_pilot_sample.csv` | 0.0 | 250 × 3 | **수록** |  |
| `model_pilot_summary.md` | 0.0 |  | **수록** |  |
| `opening_speaker_audit_sample.csv` | 0.0 | 20 × 6 | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `pilot2.numbers` | 0.9 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `pilot2.xlsx` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `pilot200_calls.parquet` | 0.2 | 200 × 2 | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `pilot200_gpt_labels.csv` | 0.0 | 200 × 3 | **수록** |  |
| `pilot2_260725.xlsx` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `pilot80_gpt_labels_v2.csv` | 0.0 | 80 × 3 | **수록** |  |
| `pilot_run_log.txt` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `probing_call_features.parquet` | 6.2 | 19,847 × 41 | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `probing_feature_importance.csv` | 0.0 | 48 × 6 | **수록** |  |
| `probing_full_acoustic_partial/` | 1.5 | 5개 파일 | 제외 | 원본 오디오(AIHub 재배포 금지) |
| `probing_full_acoustic_progress.json` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `probing_full_acoustic_run.log` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `probing_gate_threshold_sweep.csv` | 0.0 | 18 × 6 | **수록** |  |
| `probing_layer_acoustic_delta.csv` | 0.0 | 3 × 5 | **수록** |  |
| `probing_summary.txt` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `prompt_explore.xlsx` | 0.0 |  | **수록** |  |
| `prompt_explore_autorun.log` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `prompt_explore_autorun_stdout.log` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `prompt_explore_base_reused.parquet` | 0.0 | 200 × 2 | **수록** |  |
| `prompt_explore_chunks_state.json` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `prompt_explore_gpt_predictions.parquet` | 0.0 | 600 × 6 | **수록** |  |
| `prompt_explore_partial/` | 0.0 | 3개 파일 | 제외 | 원본 오디오(AIHub 재배포 금지) |
| `prompt_explore_request_index.parquet` | 0.2 | 600 × 5 | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `prompt_explore_requests_plan.json` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `prompt_explore_v3.xlsx` | 0.0 |  | **수록** |  |
| `prompt_explore_v3_autorun.log` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `prompt_explore_v3_chunks_state.json` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `prompt_explore_v3_gpt_predictions.parquet` | 0.0 | 600 × 6 | **수록** |  |
| `prompt_explore_v3_partial/` | 0.0 | 3개 파일 | 제외 | 원본 오디오(AIHub 재배포 금지) |
| `prompt_explore_v3_request_index.parquet` | 0.2 | 600 × 5 | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `prompt_explore_v3_requests_plan.json` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `prompt_explore_v3_reused.parquet` | 0.0 | 200 × 4 | **수록** |  |
| `prompt_pilot.xlsx` | 0.0 |  | **수록** |  |
| `prompt_pilot_autorun.log` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `prompt_pilot_autorun_stdout.log` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `prompt_pilot_chunks_state.json` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `prompt_pilot_gpt_predictions.parquet` | 0.0 | 700 × 5 | **수록** |  |
| `prompt_pilot_partial/` | 0.0 | 3개 파일 | 제외 | 원본 오디오(AIHub 재배포 금지) |
| `prompt_pilot_request_index.parquet` | 0.2 | 700 × 6 | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `prompt_pilot_requests_plan.json` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `prompt_pilot_sample_a.csv` | 0.0 | 150 × 3 | **수록** |  |
| `prompt_pilot_sample_b.csv` | 0.0 | 200 × 4 | **수록** |  |
| `refund_cancel_boundary.xlsx` | 0.0 |  | **수록** |  |
| `relabel_v2_run_log.txt` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `rescue_harm_analysis.csv` | 0.0 | 8 × 8 | **수록** |  |
| `rescue_harm_raw.parquet` | 0.0 | 1,973 × 16 | **수록** |  |
| `shift_mismatch_full.parquet` | 0.2 | 19,548 × 6 | **수록** |  |
| `shift_trigger_validation.csv` | 0.0 | 4 × 6 | **수록** |  |
| `stage2_acoustic_features.parquet` | 8.1 | 135,340 × 11 | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `stage2_acoustic_partial/` | 8.2 | 28개 파일 | 제외 | 원본 오디오(AIHub 재배포 금지) |
| `stage2_agent_vs_acoustic_effect.csv` | 0.0 | 12 × 18 | **수록** |  |
| `stage2_bd_before_after.csv` | 0.0 | 60 × 10 | **수록** |  |
| `stage2_bd_renorm_metrics.csv` | 0.0 | 84 × 13 | **수록** |  |
| `stage2_bd_v3_autorun.log` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `stage2_bd_v3_chunks_state.json` | 0.2 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `stage2_bd_v3_gpt_predictions.parquet` | 1.1 | 118,185 × 7 | **수록** |  |
| `stage2_bd_v3_metrics.csv` | 0.0 | 84 × 13 | **수록** |  |
| `stage2_bd_v3_partial/` | 2.1 | 565개 파일 | 제외 | 원본 오디오(AIHub 재배포 금지) |
| `stage2_bd_v3_request_index.parquet` | 57.2 |  | 제외 | 대용량 |
| `stage2_bd_v3_requests_plan.json` | 2.6 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `stage2_bd_v3_vs_base.csv` | 0.0 | 186 × 9 | **수록** |  |
| `stage2_before_after_comparison.csv` | 0.0 | 18 × 10 | **수록** |  |
| `stage2_condition_summary.csv` | 0.0 | 16 × 7 | **수록** |  |
| `stage2_gender_population_stats.csv` | 0.0 | 18 × 5 | **수록** |  |
| `stage2_mcnemar_full.csv` | 0.0 | 8 × 18 | **수록** |  |
| `stage2_mcnemar_results.csv` | 0.0 | 4 × 17 | **수록** |  |
| `stage2_mismatch_acoustic_delta.csv` | 0.0 | 56 × 18 | **수록** |  |
| `stage2_mismatch_acoustic_delta_after.csv` | 0.0 | 42 × 18 | **수록** |  |
| `stage2_mismatch_confusion_metrics.csv` | 0.0 | 112 × 13 | **수록** |  |
| `stage2_mismatch_confusion_metrics_after.csv` | 0.0 | 84 × 13 | **수록** |  |
| `stage2_mismatch_subset_accuracy.csv` | 0.0 | 16 × 4 | **수록** |  |
| `stage2_per_label_acoustic_effect.csv` | 0.0 | 42 × 9 | **수록** |  |
| `stage2_progress.json` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `stage2_renorm_application_log.csv` | 0.0 | 90 × 11 | **수록** |  |
| `stage2_trackA_vs_trackB_diagnosis.csv` | 0.0 | 24 × 8 | **수록** |  |
| `stage2_utterance_pool.parquet` | 12.8 | 135,340 × 11 | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `stage2_windows_nl.parquet` | 39.9 |  | 제외 | 대용량 |
| `stage2_windows_nl_v2.parquet` | 37.6 |  | 제외 | 대용량 |
| `stage2_windows_nl_v3_shrinkage.parquet` | 38.0 |  | 제외 | 대용량 |
| `stage2b_run.log` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `stage2d_autorun.log` | 0.1 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `stage2d_autorun_stdout.log` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `stage2d_chunks_state.json` | 0.3 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `stage2d_gpt_predictions.parquet` | 2.4 | 315,160 × 7 | **수록** |  |
| `stage2d_partial/` | 3.7 | 909개 파일 | 제외 | 원본 오디오(AIHub 재배포 금지) |
| `stage2d_request_index.parquet` | 80.2 |  | 제외 | 대용량 |
| `stage2d_requests_plan.json` | 6.9 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `stage2m_autorun.log` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `stage2m_chunks_state.json` | 0.1 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `stage2m_gpt_predictions_BD.parquet` | 1.1 | 118,185 × 7 | **수록** |  |
| `stage2m_partial/` | 1.6 | 417개 파일 | 제외 | 원본 오디오(AIHub 재배포 금지) |
| `stage2m_request_index.parquet` | 46.1 |  | 제외 | 대용량 |
| `stage2m_requests_plan.json` | 2.6 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `suspect_calls_for_review.csv` | 0.1 | 50 × 4 | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `train10k_acoustic_features.parquet` | 1.6 | 26,440 × 11 | **수록** |  |
| `train10k_acoustic_partial/` | 1.7 | 6개 파일 | 제외 | 원본 오디오(AIHub 재배포 금지) |
| `train10k_acoustic_progress.json` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `train10k_autorun.log` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `train10k_call_ids.csv` | 0.1 | 5,000 × 1 | **수록** |  |
| `train10k_call_texts.parquet` | 9.0 | 10,000 × 2 | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `train10k_classifier_summary.txt` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `train10k_full_chunks_state.json` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `train10k_full_partial/` | 0.2 | 71개 파일 | 제외 | 원본 오디오(AIHub 재배포 금지) |
| `train10k_gpt_labels.parquet` | 0.1 | 10,000 × 4 | **수록** |  |
| `train10k_gpt_labels_v4.parquet` | 0.1 | 5,784 × 4 | **수록** |  |
| `train10k_gpt_labels_v5.parquet` | 0.1 | 5,568 × 3 | **수록** |  |
| `train10k_pool5_acoustic_features.parquet` | 4.1 | 67,820 × 11 | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `train10k_pool5_acoustic_partial/` | 2.6 | 9개 파일 | 제외 | 원본 오디오(AIHub 재배포 금지) |
| `train10k_pool5_acoustic_progress.json` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `train10k_pool5_classifier_summary.txt` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `train10k_pool5_utterance_pool.parquet` | 6.5 | 67,820 × 11 | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `train10k_request_index.parquet` | 12.8 | 10,000 × 4 | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `train10k_requests_plan.json` | 0.2 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `train10k_test10_chunks_state.json` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `train10k_test10_gpt_labels.parquet` | 0.0 | 10 × 4 | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `train10k_test10_partial/` | 0.0 | 1개 파일 | 제외 | 원본 오디오(AIHub 재배포 금지) |
| `train10k_test10_request_index.parquet` | 0.0 | 10 × 4 | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `train10k_test10_requests_plan.json` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `train10k_utterance_pool.parquet` | 2.6 | 26,440 × 11 | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `train10k_v4_autorun.log` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `train10k_v4_chunks_state.json` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `train10k_v4_partial/` | 0.1 | 32개 파일 | 제외 | 원본 오디오(AIHub 재배포 금지) |
| `train10k_v4_request_index.parquet` | 6.7 | 5,784 × 4 | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `train10k_v4_requests_plan.json` | 0.1 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `train10k_v4_target_call_ids.csv` | 0.1 | 5,000 × 2 | **수록** |  |
| `train10k_v5_autorun.log` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `train10k_v5_chunks_state.json` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `train10k_v5_partial/` | 0.1 | 26개 파일 | 제외 | 원본 오디오(AIHub 재배포 금지) |
| `train10k_v5_request_index.parquet` | 5.8 | 5,568 × 4 | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `train10k_v5_requests_plan.json` | 0.1 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `train10k_v5_target_call_ids.csv` | 0.1 | 5,000 × 1 | **수록** |  |
| `train_10k_final_labeled.parquet` | 0.1 | 10,000 × 2 | **수록** |  |
| `train_10k_labeled.parquet` | 1.6 | 10,000 × 20 | **수록** |  |
| `train_10k_pool5_labeled.parquet` | 1.6 | 10,000 × 20 | **수록** |  |
| `trigger_comparison.csv` | 0.0 | 6 × 20 | **수록** |  |
| `v3_autorun.log` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `v3_autorun_stdout.log` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `v3_chunks_plan.json` | 0.2 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `v3_chunks_state.json` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `v3_partial/` | 0.0 | 3개 파일 | 제외 | 원본 오디오(AIHub 재배포 금지) |
| `v3_target_call_ids.csv` | 0.3 | 5,000 × 2 | **수록** |  |
| `v4_autorun.log` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `v4_autorun_stdout.log` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `v4_chunks_plan.json` | 0.2 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `v4_chunks_state.json` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `v4_partial/` | 0.1 | 9개 파일 | 제외 | 원본 오디오(AIHub 재배포 금지) |
| `v4_self_consistency_pass2.csv` | 0.0 | 200 × 3 | **수록** |  |
| `v4_target_call_ids.csv` | 0.3 | 5,000 × 2 | **수록** |  |
| `v5_autorun.log` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `v5_autorun_stdout.log` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `v5_chunks_plan.json` | 0.2 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `v5_chunks_state.json` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `v5_partial/` | 0.1 | 7개 파일 | 제외 | 원본 오디오(AIHub 재배포 금지) |
| `v5_target_call_ids.csv` | 0.1 | 5,000 × 1 | **수록** |  |
| `v_unified_autorun.log` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `v_unified_chunks_state.json` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `v_unified_partial/` | 0.0 | 16개 파일 | 제외 | 원본 오디오(AIHub 재배포 금지) |
| `v_unified_request_index.parquet` | 2.9 | 2,000 × 5 | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `v_unified_requests_plan.json` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |
| `v_unified_sample_call_ids.csv` | 0.1 | 2,000 × 2 | **수록** |  |
| `v_unified_sample_labels.parquet` | 0.0 | 2,000 × 4 | **수록** |  |
| `validation_vs_gold.txt` | 0.0 |  | 제외 | 전사 원문 포함 또는 중간 산출물 |

### batch 요청 JSONL (2,102개, 1.9 GB)

전부 제외. 프롬프트 안에 전사 원문이 그대로 들어 있다. 각 단계의 `*_chunk_plan.py`를 다시 돌리면 재생성된다.

---

## 컬럼 사전 (수록 파일)

| 파일 | 컬럼 |
|---|---|
| `acoustic_features_sample50.csv` | call_id, dialog_idx, speaker_type, speaker_group, duration, audioPath, load_ok, sr, … (+9) |
| `agent_judge_human_eval_slots.csv` | call_id, gpt_친절도, gpt_적절대응, human_친절도, human_적절대응 |
| `agent_judge_percall.parquet` | call_id, gold_mismatch, gold_label, 친절도, 적절대응, 합산, is_불만 |
| `agent_judge_scores.parquet` | call_id, 친절도, 적절대응, judge_error, n_attempts, n_utt |
| `agent_strat_percall.parquet` | call_id, 적절대응, 친절도, 그룹, gold_label, gold_mismatch, is_불만, pred_A, … (+1) |
| `arousal_target_percall.parquet` | call_id, gold_actual, pred_A, pred_B, group, gender, z_pitch, z_energy, … (+5) |
| `audio_seg_manifest.parquet` | call_id, utt_idx, wav_path, dialog_idx, duration, text, n_customer_utt, wav_duration, … (+3) |
| `batch1_call_ids.csv` | call_id |
| `batch1_no_valid_text_call_ids.csv` | call_id |
| `batch2_call_ids.csv` | call_id |
| `gate_infopoor_layer_comparison.csv` | layer, variant, n, n_gold_complaint, recall, precision, f1, n_pred_complaint |
| `gate_prompt_gpt_predictions.parquet` | custom_id, call_id, variant, gate_level, gold_actual, predicted_label, parse_error |
| `gate_signal_inference.parquet` | call_id, gold_actual, complaint_proba, gate_level |
| `gate_signal_train10k.parquet` | call_id, gold_actual, is_complaint, complaint_proba, gate_level |
| `gold_actual_batch1.parquet` | call_id, label, prompt_version, model |
| `gold_actual_batch1_final.parquet` | call_id, label, prompt_version, model, source |
| `gold_actual_batch1_v4.parquet` | call_id, has_complaint, label_v4, prompt_version, model |
| `gold_actual_batch1_v5.parquet` | call_id, label_v5, prompt_version, model |
| `human_eval_v3_labels.csv` | call_id, gpt41mini_label_v3 |
| `human_pilot_meta.csv` | 순번, call_id, gold_actual, 층, seed |
| `inbound_outbound_gpt.csv` | call_id, call_type_gpt, reason_gpt |
| `infopoor_tagged.csv` | call_id, gold_actual, A_N2_pred, A_N5_pred, unlock, shift, short, is_infopoor, … (+1) |
| `lang_pilot_percall.parquet` | call_id, gold_actual, pred_ko, pred_enrsn, pred_enfull, correct_pred_ko, correct_pred_enrsn, correct_pred_enfull |
| `lang_pilot_pred_en_full.parquet` | call_id, gold_actual, predicted_label, parse_error, n_attempts |
| `lang_pilot_pred_en_rsn.parquet` | call_id, gold_actual, predicted_label, parse_error, n_attempts |
| `lang_pilot_pred_ko.parquet` | call_id, gold_actual, predicted_label, parse_error, n_attempts |
| `modality_ab_percall.parquet` | call_id, gold_actual, pred_A, pred_B, correct_pred_A, correct_pred_B |
| `model_pilot_percall.parquet` | call_id, gold_actual, pred_41cur, pred_54low, pred_54med, pred_lunalow, pred_lunamed, correct_pred_41cur, … (+4) |
| `model_pilot_pred_54low.parquet` | call_id, gold_actual, predicted_label, parse_error, n_attempts |
| `model_pilot_pred_54med.parquet` | call_id, gold_actual, predicted_label, parse_error, n_attempts |
| `model_pilot_pred_56terra_med.parquet` | call_id, gold_actual, predicted_label, parse_error, n_attempts |
| `model_pilot_pred_A_54low.parquet` | call_id, gold_actual, predicted_label, parse_error, n_attempts |
| `model_pilot_pred_lunalow.parquet` | call_id, gold_actual, predicted_label, parse_error, n_attempts |
| `model_pilot_pred_lunamed.parquet` | call_id, gold_actual, predicted_label, parse_error, n_attempts |
| `model_pilot_sample.csv` | call_id, pred_41mini_cur, gold_actual |
| `pilot200_gpt_labels.csv` | call_id, gpt55_label, gpt41mini_label |
| `pilot80_gpt_labels_v2.csv` | call_id, gpt55_label_v2, gpt41mini_label_v2 |
| `probing_feature_importance.csv` | scope, task, feature, xgb_importance, logreg_coef_abs_mean, logreg_coef_signed |
| `probing_gate_threshold_sweep.csv` | scope, threshold, accuracy, macro_f1, complaint_recall, complaint_precision |
| `probing_layer_acoustic_delta.csv` | layer, n, A_N2_match_rate, B_N2_match_rate, delta(B-A) |
| `prompt_explore_base_reused.parquet` | call_id, base |
| `prompt_explore_gpt_predictions.parquet` | custom_id, call_id, version, predicted_label, parse_error, complaint_tag |
| `prompt_explore_v3_gpt_predictions.parquet` | custom_id, call_id, version, predicted_label, parse_error, complaint_tag |
| `prompt_explore_v3_reused.parquet` | call_id, base, v3_orig, v3_orig_parse_error |
| `prompt_pilot_gpt_predictions.parquet` | custom_id, call_id, sample, version, predicted_label |
| `prompt_pilot_sample_a.csv` | call_id, 층, gold_actual |
| `prompt_pilot_sample_b.csv` | call_id, gold_actual, A_N2_pred, is_mismatch |
| `rescue_harm_analysis.csv` | info_layer, policy, n, rescue, harm, net_rescue, rescue_rate, harm_rate |
| `rescue_harm_raw.parquet` | call_id, gold_actual, pred_base, pred_gate, mismatch, shift, is_infopoor, pred_hybrid, … (+8) |
| `shift_mismatch_full.parquet` | call_id, gold_actual, A_N2_pred, A_N5_pred, mismatch, shift |
| `shift_trigger_validation.csv` | variant, recall, precision, f1, n_pred_complaint, macro_f1 |
| `stage2_agent_vs_acoustic_effect.csv` | n, effect, cond1, cond2, n_total, both_correct, only_cond1_correct, only_cond2_correct, … (+10) |
| `stage2_bd_before_after.csv` | table, n, condition, pair, class, metric, before, after, … (+2) |
| `stage2_bd_renorm_metrics.csv` | n, condition, class, precision, recall, f1, support, n_total, … (+5) |
| `stage2_bd_v3_gpt_predictions.parquet` | custom_id, call_id, condition, n, window_version, predicted_label, parse_error |
| `stage2_bd_v3_metrics.csv` | n, condition, class, precision, recall, f1, support, n_total, … (+5) |
| `stage2_bd_v3_vs_base.csv` | section, n, condition, pair, metric, v_base, v3_orig, delta, … (+1) |
| `stage2_before_after_comparison.csv` | n, pair, class, delta_recall_before, delta_recall_after, delta_precision_before, delta_precision_after, trigger_happy_before, … (+2) |
| `stage2_condition_summary.csv` | condition, condition_label, n, n_total, n_valid, accuracy, mismatch_rate |
| `stage2_gender_population_stats.csv` | gender, metric, n, mean, std |
| `stage2_mcnemar_full.csv` | n, version, cond1, cond2, n_total, both_correct, only_cond1_correct, only_cond2_correct, … (+10) |
| `stage2_mcnemar_results.csv` | n, cond1, cond2, n_total, both_correct, only_cond1_correct, only_cond2_correct, both_wrong, … (+9) |
| `stage2_mismatch_acoustic_delta.csv` | n, pair, cond_no_acoustic, cond_acoustic, class, is_negative_destination, precision_no_ac, precision_ac, … (+10) |
| `stage2_mismatch_acoustic_delta_after.csv` | n, pair, cond_no_acoustic, cond_acoustic, class, is_negative_destination, precision_no_ac, precision_ac, … (+10) |
| `stage2_mismatch_confusion_metrics.csv` | n, condition, class, precision, recall, f1, support, low_support_flag, … (+5) |
| `stage2_mismatch_confusion_metrics_after.csv` | n, condition, class, precision, recall, f1, support, low_support_flag, … (+5) |
| `stage2_mismatch_subset_accuracy.csv` | n, condition, accuracy, n_calls |
| `stage2_per_label_acoustic_effect.csv` | n, pair, cond_no_acoustic, cond_acoustic, label, n_calls, accuracy_no_acoustic, accuracy_acoustic, … (+1) |
| `stage2_renorm_application_log.csv` | window_n, window_version, speaker_group, f0_mean_norm_source, count, metric, f0_std_norm_source, e_mean_db_norm_source, … (+3) |
| `stage2_trackA_vs_trackB_diagnosis.csv` | n, version, feature, n_calls, changed_calls, changed_call_pct, n_rows, changed_rows |
| `stage2d_gpt_predictions.parquet` | custom_id, call_id, condition, n, version, acoustic, predicted_label |
| `stage2m_gpt_predictions_BD.parquet` | custom_id, call_id, condition, n, version, acoustic, predicted_label |
| `train10k_acoustic_features.parquet` | call_id, dialog_idx, load_ok, error, f0_mean, f0_std, e_mean_amp, e_std_amp, … (+3) |
| `train10k_call_ids.csv` | call_id |
| `train10k_gpt_labels.parquet` | custom_id, call_id, gold_actual, error_note |
| `train10k_gpt_labels_v4.parquet` | custom_id, call_id, has_complaint, label_v4 |
| `train10k_gpt_labels_v5.parquet` | custom_id, call_id, label_v5 |
| `train10k_v4_target_call_ids.csv` | call_id, label |
| `train10k_v5_target_call_ids.csv` | call_id |
| `train_10k_final_labeled.parquet` | call_id, gold_actual |
| `train_10k_labeled.parquet` | call_id, gold_actual, n2_f0_mean_mean, n2_f0_mean_std, n2_f0_std_mean, n2_f0_std_std, n2_e_mean_db_mean, n2_e_mean_db_std, … (+12) |
| `train_10k_pool5_labeled.parquet` | call_id, gold_actual, pool5_f0_mean_mean, pool5_f0_mean_std, pool5_f0_std_mean, pool5_f0_std_std, pool5_e_mean_db_mean, pool5_e_mean_db_std, … (+12) |
| `trigger_comparison.csv` | trigger, mismatch_precision, mismatch_recall, trigger_rate_full, n_triggered_full, pct_triggered_in_jeongbochungbun, pct_triggered_in_jeongbobuzok, pct_triggered_in_NA, … (+12) |
| `v3_target_call_ids.csv` | call_id, label |
| `v4_self_consistency_pass2.csv` | call_id, has_complaint_pass2, label_pass2 |
| `v4_target_call_ids.csv` | call_id, label |
| `v5_target_call_ids.csv` | call_id |
| `v_unified_sample_call_ids.csv` | call_id, gold_actual |
| `v_unified_sample_labels.parquet` | custom_id, call_id, gold_actual, predicted_label |
