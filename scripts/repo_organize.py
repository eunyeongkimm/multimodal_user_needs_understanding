"""공개 repo용 산출물 큐레이션: outputs/(평면, 2.5GB) -> results/(단계별, ~15MB).

왜 복사인가
  scripts/ 102개가 `outputs/<평면 파일명>`을 하드코딩하고 있어서 outputs/를 실제로
  재배치하면 재현이 깨진다. outputs/는 로컬 작업 캐시로 두고(.gitignore),
  git에는 단계별로 큐레이션한 사본만 올린다.

무엇을 제외하는가 (공개 repo 기준)
  AIHub D04는 재배포 금지 라이선스다. 따라서 아래는 전부 제외한다.
    - 전사 원문(text / call_text / 대화(텍스트) / ko_text)
    - 프롬프트에 전사가 통째로 들어간 것(*_request_index.parquet, *.jsonl)
    - 원본 오디오(audio_seg/*.wav)
  여기서는 "포함할 파일을 명시적으로 나열"하는 화이트리스트 방식만 쓴다.
  마지막에 results/ 전체를 다시 스캔해 원문이 새어들어갔는지 재검증한다.

사용:
  python3 scripts/repo_organize.py --check   # 복사 없이 존재/안전성만 점검
  python3 scripts/repo_organize.py
"""

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUT = BASE_DIR / "outputs"
RESULTS = BASE_DIR / "results"
DOCS = BASE_DIR / "docs"

KO = re.compile(r"[가-힣]")
LONG_TEXT_CHARS = 60  # 이보다 긴 한글 셀 = 전사 의심
SCAN_ROWS = 500

# 원문/장문 컬럼만 떨어뜨리고 나머지는 살리는 파일.
# 값 자체는 GPT가 쓴 서술이지만 통화 내용을 부분적으로 풀어쓸 수 있어 공개본에서는 제거한다.
STRIP_COLS = {
    "inbound_outbound_gpt.csv": ["reason_gpt"],
    "audio_seg_manifest.parquet": ["text"],
}

STAGES = [
    dict(
        dir="00_data_prep",
        title="원천 인덱스 구축 · 전처리",
        question="74,123콜의 D04 라벨링 JSON을 발화 단위 인덱스로 만들고, 분석에 쓸 수 있는 품질로 정제할 수 있는가?",
        scripts=[
            ("build_index.py", "D04 라벨링 JSON → dialog 단위 인덱스"),
            ("clean_text.py", "disfluency 태그 제거, r_gold(발화속도)·r_gold_valid_flag 계산"),
            ("split_batches.py", "batch1(20,000) / batch2(54,123) 고정 분할 (seed=42)"),
            ("label_inbound_outbound.py", "인바운드/아웃바운드 판별 (사람 대조 80건)"),
            ("extract_acoustic_features.py", "acoustic 추출 프로토타입 (50건, F0/Energy/silence 검증)"),
        ],
        files=[
            "data_cleaning_report.md",
            "d04_dialog_index_column_dictionary.md",
            "batch1_call_ids.csv",
            "batch2_call_ids.csv",
            "batch1_no_valid_text_call_ids.csv",
            "inbound_outbound_gpt.csv",
            "acoustic_features_sample50.csv",
        ],
        conclusion=(
            "발화 단위 인덱스 1,672,062행 확보. `r_gold_valid_flag`로 발화속도 이상치를 걸러 "
            "이후 모든 acoustic 분석의 유효 구간을 정의했다. batch1/batch2 분할은 seed=42로 "
            "고정되어 이후 전 단계가 같은 콜 집합을 참조한다."
        ),
    ),
    dict(
        dir="01_gold_labeling",
        title="gold_actual 라벨링 (v1 → v5)",
        question="통화 전체를 보고 매긴 '실제 니즈' 7-카테고리 라벨을 사람 수준으로 신뢰할 만하게 만들 수 있는가? 특히 항의성 콜(불만제기)을 놓치지 않는가?",
        scripts=[
            ("prepare_human_eval.py / build_human_eval_final.py / compute_kappa.py", "사람 평가셋 구성 + Cohen's Kappa"),
            ("pilot_label_llm.py", "GPT-5.5 vs 4.1-mini 파일럿 200건 (v1)"),
            ("relabel_v2_and_kappa.py", "v2 프롬프트 80건 재검증"),
            ("batch1_chunk_plan.py / batch1_check.py / batch1_autorun.sh", "batch1 전체 19,847건 라벨링 (v2)"),
            ("v3_*.py / v3_autorun.sh", "v3: 불만 우선판단 규칙 삽입 → Kappa 하락으로 폐기"),
            ("v4_*.py / v4_autorun.sh", "v4: 항의 게이트 → 카테고리 2단계 분리"),
            ("v5_*.py / v5_autorun.sh", "v5: v4 부작용 복원, 최종본 확정"),
            ("v_unified_*.py", "통합 프롬프트 검증 (층화 2,000건)"),
        ],
        files=[
            "pilot200_gpt_labels.csv",
            "pilot80_gpt_labels_v2.csv",
            "human_eval_v3_labels.csv",
            "gold_actual_batch1.parquet",
            "gold_actual_batch1_v4.parquet",
            "gold_actual_batch1_v5.parquet",
            "gold_actual_batch1_final.parquet",
            "v3_target_call_ids.csv",
            "v4_target_call_ids.csv",
            "v5_target_call_ids.csv",
            "v4_self_consistency_pass2.csv",
            "v_unified_sample_call_ids.csv",
            "v_unified_sample_labels.parquet",
            "refund_cancel_boundary.xlsx",
        ],
        conclusion=(
            "**`gold_actual_batch1_final.parquet`(19,847건)이 이후 모든 분석의 기준 라벨.**\n\n"
            "v2는 불만제기를 32건밖에 못 잡았다. v3(정의 안에 규칙 삽입)은 환불요청↔주문취소 "
            "경계를 흔들어 Kappa 0.82→0.71로 떨어져 폐기했다. v4(항의 게이트 분리)가 불만제기 "
            "452건을 검출했으나 환불요청을 다른 범주로 과다 이동시켰고, v5가 `has_complaint=False` "
            "건만 재분류해 분포를 복원했다. 최종 불만제기 484건(2.44%). v4 자기일관성 Kappa 0.96.\n\n"
            "교훈: 카테고리 정의를 건드리는 대신 **판단 단계를 분리**하는 편이 안전했다. "
            "이후 `CATEGORY_DEFINITIONS`는 수정 금지 상수로 고정된다."
        ),
    ),
    dict(
        dir="02_acoustic",
        title="음향 피처 추출 · 정규화",
        question="발화 단위 F0/에너지/무음을 어떤 기준으로 정규화해야 '이 화자가 평소보다 격앙됐다'가 제대로 표현되는가?",
        scripts=[
            ("stage2a_window_pool.py", "N=1,2,3,5 × 고객only/고객&상담사 윈도우 pool"),
            ("stage2b_acoustic_extract.py", "발화 단위 F0·Energy·silence 추출 (8코어 병렬)"),
            ("stage2c_normalize_nl.py", "Track A: 콜/윈도우 내 z-score + 5분위 자연어 변환"),
            ("stage2c_v2_normalize_nl_gender.py", "**성별 population 기준 정규화 (채택본)**"),
            ("stage2c_v3_normalize_nl_shrinkage.py", "Track B: shrinkage 정규화 (진단용)"),
            ("stage2l_gender_population_stats.py", "성별 population 통계 산출"),
        ],
        files=[
            "stage2_gender_population_stats.csv",
            "stage2_renorm_application_log.csv",
            "stage2_trackA_vs_trackB_diagnosis.csv",
        ],
        conclusion=(
            "콜 내부 z-score(Track A)는 화자가 통화 내내 격앙된 콜에서 그 격앙을 '평균'으로 "
            "흡수해 버린다. **같은 성별 화자 집단 대비 정규화**로 바꾼 것이 채택본이며, "
            "성별을 섞으면 남녀 pitch 차이(남 138.5Hz / 여 223.1Hz)가 arousal로 오염된다.\n\n"
            "주의: `e_mean_db`는 `amplitude_to_db(rms, ref=np.max)`로 계산돼 **발화 자신의 피크 대비** "
            "값이다. 절대 음량이 아니라 dynamic range이므로 arousal 용도로는 `e_mean_amp`(선형 RMS)를 써야 한다."
        ),
    ),
    dict(
        dir="03_condition_exp",
        title="조건 실험 — 통화 초반 N발화로 니즈 예측 (A/B/C/D × N)",
        question="통화 초반 일부(N=1,2,3,5)만 보고 니즈를 맞출 때, 텍스트만(A/C) 쓸 때와 음성 특징을 함께(B/D) 줄 때 정확도가 달라지는가?",
        scripts=[
            ("stage2d_prompt.py", "**공용 프롬프트 조립** (`CATEGORY_DEFINITIONS` 원본)"),
            ("stage2d_chunk_plan.py / stage2d_check.py / stage2d_autorun.sh", "16조건 909청크 Batch 실행"),
            ("stage2e~stage2k", "조건별 정확도 · per-label 효과 · mismatch 부분집합 · confusion"),
            ("stage2h/2i_mcnemar_*.py, mcnemar_v3.py", "쌍대 유의성 검정"),
            ("stage2m_*.py, stage2_bd_*.py", "성별 재정규화 후 B/D 재실행 및 before/after"),
            ("stage2p/2q_human_pilot_*.py", "사람 파일럿 (텍스트만 → 음성 후 판단 변화)"),
            ("stage2r_refund_cancel_boundary.py", "환불↔취소 경계 진단"),
        ],
        files=[
            "stage2_condition_summary.csv",
            "stage2d_gpt_predictions.parquet",
            "stage2_per_label_acoustic_effect.csv",
            "stage2_mismatch_subset_accuracy.csv",
            "stage2_mismatch_acoustic_delta.csv",
            "stage2_mismatch_acoustic_delta_after.csv",
            "stage2_mismatch_confusion_metrics.csv",
            "stage2_mismatch_confusion_metrics_after.csv",
            "stage2_mcnemar_results.csv",
            "stage2_mcnemar_full.csv",
            "stage2_before_after_comparison.csv",
            "stage2_bd_before_after.csv",
            "stage2_bd_renorm_metrics.csv",
            "stage2_bd_v3_metrics.csv",
            "stage2_bd_v3_vs_base.csv",
            "stage2_agent_vs_acoustic_effect.csv",
            "stage2m_gpt_predictions_BD.parquet",
            "stage2_bd_v3_gpt_predictions.parquet",
            "shift_mismatch_full.parquet",
            "human_pilot_answers.md",
            "human_pilot_meta.csv",
        ],
        conclusion=(
            "핵심 지표 정의가 여기서 나온다. **mismatch = 조건 A / N=1 예측 ≠ gold_actual** — "
            "즉 '통화 첫 발화의 표면 니즈'와 '통화 전체의 실제 니즈'가 어긋난 콜. "
            "유효 19,548건 중 9,658건(49.4%)이 mismatch다.\n\n"
            "acoustic을 얹은 효과는 전체 평균에서는 작다. 성별 재정규화 후에도 방향이 크게 "
            "달라지지 않았고, 이 때문에 '어디에서' 효과가 나는지를 좁히는 후속 단계(04~07)로 넘어갔다."
        ),
    ),
    dict(
        dir="04_prompt_pilot",
        title="프롬프트 변형 파일럿",
        question="음성 정보를 '어떻게 설명해 주느냐'에 따라 모델이 불만제기를 더 잘 잡는가?",
        scripts=[
            ("prompt_pilot_*.py / prompt_pilot_autorun.sh", "v_base vs 음성 안내문 유무(kr_full/kr_slim)"),
            ("prompt_explore_*.py", "v2(억제문) / v3(분석 태그) / v4(2단계 게이트) 비교"),
            ("prompt_explore_v3_*.py", "v3 축 변형 4종 (v3_orig/a/b/c)"),
        ],
        files=[
            "prompt_pilot.xlsx",
            "prompt_explore.xlsx",
            "prompt_explore_v3.xlsx",
            "prompt_pilot_sample_a.csv",
            "prompt_pilot_sample_b.csv",
            "prompt_pilot_gpt_predictions.parquet",
            "prompt_explore_gpt_predictions.parquet",
            "prompt_explore_base_reused.parquet",
            "prompt_explore_v3_gpt_predictions.parquet",
            "prompt_explore_v3_reused.parquet",
        ],
        conclusion=(
            "v3(먼저 분석 → 태그 출력)이 v_base 대비 순효과가 가장 컸다. "
            "다만 음성 안내문(`VOICE_GUIDE_*`)은 **이 파일럿 계열에서만** 사용됐고, "
            "이후 250건 파일럿(06)은 전부 `version=\"base\"`로 돌았다 — 즉 06의 조건 B 프롬프트에는 "
            "음성 해석 안내가 한 줄도 들어가지 않고, 발화별 `[음성 특징: ...]` 라인만 인터리브된다. "
            "06과 대조 실험을 설계할 때 반드시 확인해야 하는 지점이다."
        ),
    ),
    dict(
        dir="05_gate",
        title="정보부족 게이트 (probing · train10k 분류기)",
        question="'초반 텍스트만으로는 판단이 안 되는 콜'을 미리 골라낼 수 있는가? 그런 콜에만 음성을 쓰면 이득이 있는가?",
        scripts=[
            ("probing_build_features.py / probing_experiments.py", "콜 단위 피처 구축 + 실험 A~F"),
            ("probing_full_acoustic_extract.py", "전체 콜 acoustic 확장 추출"),
            ("gate_signal_build.py", "게이트 학습 신호 구성"),
            ("gate_prompt_*.py", "게이트 확률을 v3 프롬프트에 주입한 실전 검증"),
            ("train10k_*.py", "train 1만 라벨링 → 피처 집계 → 분류기 학습 (pool5 재구성 포함)"),
            ("trigger_comparison.py", "트리거 기준 비교"),
        ],
        files=[
            "probing_feature_importance.csv",
            "probing_gate_threshold_sweep.csv",
            "probing_layer_acoustic_delta.csv",
            "gate_infopoor_layer_comparison.csv",
            "infopoor_tagged.csv",
            "gate_signal_train10k.parquet",
            "gate_signal_inference.parquet",
            "gate_prompt_gpt_predictions.parquet",
            "train10k_call_ids.csv",
            "train10k_v4_target_call_ids.csv",
            "train10k_v5_target_call_ids.csv",
            "train10k_gpt_labels.parquet",
            "train10k_gpt_labels_v4.parquet",
            "train10k_gpt_labels_v5.parquet",
            "train_10k_final_labeled.parquet",
            "train_10k_labeled.parquet",
            "train_10k_pool5_labeled.parquet",
            "train10k_acoustic_features.parquet",
            "rescue_harm_analysis.csv",
            "rescue_harm_raw.parquet",
            "shift_trigger_validation.csv",
            "trigger_comparison.csv",
        ],
        conclusion=(
            "`infopoor_tagged.csv`가 '정보부족' 태그의 산출물이며, 이후 **acoustically-valid subset** "
            "정의(조건 B / N=5 / customer_only ∩ `is_infopoor==False` = 16,777건)의 근거가 된다. "
            "06의 250건 표본은 전부 이 모집단에서 뽑았다.\n\n"
            "게이트 임계값은 train에서 구해 추론에 **고정 적용**했고, 모든 판단은 CV 기반, "
            "스케일링은 fold 내부에서만 적합시켰다."
        ),
    ),
    dict(
        dir="06_pilot_250",
        title="250건 paired 파일럿 — 모델 / effort / 모달리티 / 언어",
        question="'gold=불만제기인데 환불요청으로 예측'하는 오류를, 모델·추론량·모달리티·추론언어 중 무엇으로든 줄일 수 있는가?",
        scripts=[
            ("model_pilot_sample.py", "층화 250건 추출 (seed=42, 불만 오버샘플 7→50). **이후 전 실험 공통 표본**"),
            ("model_pilot_run.py", "Responses API 동기 호출 + concurrency 6 + 재시도. 프롬프트 바이트 동일성 검증 포함"),
            ("model_pilot_eval.py", "5조건(4.1현재 / 5.4-low / 5.4-med / luna-low / luna-med) 절대값 평가"),
            ("modality_ab_eval.py", "조건 A(text-only) vs B(text+acoustic), 둘 다 5.4-low"),
            ("nl_en_template.py", "음성 description 영어 템플릿 (MT 아님 — level 컬럼에서 직접 생성)"),
            ("lang_pilot_translate.py", "전사 MT 1회 캐시 (재번역 금지)"),
            ("lang_pilot_run.py / lang_pilot_eval.py", "KO / EN-rsn / EN-full 3조건"),
            ("arousal_target_check.py", "공통 오답 콜의 초반 고객 arousal 진단 (API 0, 전부 로컬)"),
        ],
        files=[
            "model_pilot_summary.md",
            "model_pilot_sample.csv",
            "model_pilot_percall.parquet",
            "model_pilot_pred_54low.parquet",
            "model_pilot_pred_54med.parquet",
            "model_pilot_pred_56terra_med.parquet",
            "model_pilot_pred_lunalow.parquet",
            "model_pilot_pred_lunamed.parquet",
            "model_pilot_pred_A_54low.parquet",
            "modality_ab_summary.md",
            "modality_ab_percall.parquet",
            "lang_pilot_summary.md",
            "lang_pilot_percall.parquet",
            "lang_pilot_pred_ko.parquet",
            "lang_pilot_pred_en_rsn.parquet",
            "lang_pilot_pred_en_full.parquet",
            "arousal_target_summary.md",
            "arousal_target_percall.parquet",
        ],
        conclusion=(
            "**이 프로젝트의 중심 발견.** 같은 250콜에서 다섯 개 축을 흔들었는데 "
            "`gold=불만제기 → 예측=환불요청` 셀이 거의 움직이지 않았다.\n\n"
            "| 흔든 축 | 조건 | 불만→환불 셀 |\n"
            "|---|---|---|\n"
            "| 모델 버전 | 4.1-mini / 5.4 / luna / terra | 18~20 |\n"
            "| reasoning effort | low / medium | 18~20 |\n"
            "| 모달리티 | A(text) / B(text+acoustic) | 19 / 19 |\n"
            "| 추론 언어 | 한국어 / 영어 | 19 / 19 |\n"
            "| 입력 언어 | 한국어 / 영어(MT) | 19 / 18 |\n\n"
            "성능 자체는 개선된다(macro-F1 0.476→0.538, 불만 F1 0.036→0.364). "
            "그런데도 **A·B 양쪽에서 동시에 틀리는 18콜**은 그대로 남는다.\n\n"
            "arousal 진단 결과 그 18콜의 초반 고객 arousal은 오히려 맞춘 불만군보다 **높았고"
            "**(0.407 vs 0.104), Mann-Whitney 전 지표 p>0.22로 유의하지 않다. "
            "즉 '음성이 밋밋해서 놓친다'는 가설은 지지되지 않는다."
        ),
    ),
    dict(
        dir="07_agent_quality",
        title="상담사 응대 품질 — 층화 변수 탐색",
        question="상담사가 잘 대응했는지가 mismatch를 설명하는가? 설명한다면 층화 변수로 쓸 수 있는가?",
        scripts=[
            ("agent_judge_prompt.py", "GPT-as-judge 프롬프트. **gold_actual/카테고리 taxonomy 미포함** (누수 차단)"),
            ("agent_judge_run.py", "250건 채점 + 스캐폴딩 누수 자동 검증(발견 시 중단)"),
            ("agent_judge_gate.py", "층별 AUC 게이트 — 판정 기준은 gold≠불만 층"),
            ("agent_strat_analysis.py", "적절대응 층화 × 불만/mismatch × acoustic 효과"),
        ],
        files=[
            "agent_judge_gate_summary.md",
            "agent_strat_summary.md",
            "agent_judge_scores.parquet",
            "agent_judge_percall.parquet",
            "agent_strat_percall.parquet",
            "agent_judge_human_eval_slots.csv",
        ],
        conclusion=(
            "**게이트 미통과 → 워크스트림 종료.**\n\n"
            "전체 250건에서는 적절대응이 mismatch와 상관있어 보인다(AUC 0.417, r=-0.146, p=0.021). "
            "그런데 250건 표본은 불만을 7→50건으로 오버샘플했고 그 50건 중 45건(90%)이 mismatch다. "
            "즉 전체 층의 신호는 '불만을 탐지한 것'일 수 있다.\n\n"
            "불만 교란을 분리한 **gold≠불만 층(n=200)에서 최고 AUC 0.506** — 판별력 없음. "
            "사전에 정한 기준(≥0.6 통과 / ~0.5 종료)에 따라 표본 확대와 acoustic 재추출을 하지 않고 종료했다.\n\n"
            "층화 탐색(연관만, 인과 아님): 적절대응 '못 대응' 그룹은 불만비율 38.5% vs '잘 대응' 11.6% "
            "(p<0.001), mismatch 66.7% vs 47.7% (p=0.008). 다만 '불만이라 대응이 어려웠다'와 "
            "'대응을 못해 불만이 됐다'는 이 데이터로 구분 불가.\n\n"
            "> `agent_judge_human_eval_slots.csv`의 40건 수동 채점란은 비어 있다. "
            "채워 넣고 `agent_judge_gate.py`를 다시 돌리면 quadratic-weighted Kappa가 자동 산출된다."
        ),
    ),
    dict(
        dir="08_audio_seg",
        title="오디오 세그먼트 준비 (audio 모델용)",
        question="GPT A/B와 정확히 같은 250콜·같은 발화로 audio 모델 입력을 만들 수 있는가?",
        scripts=[
            ("audio_seg_extract.py", "초반 5발화 고객 세그먼트 WAV 추출 + manifest + 검증"),
        ],
        files=["audio_seg_summary.md", "audio_seg_manifest.parquet"],
        conclusion=(
            "AIHub 원천은 이미 **발화 단위 개별 WAV**다(`audioPath`가 1,672,062행 모두 고유). "
            "따라서 timestamp 슬라이싱이 필요 없고 `shutil.copy2`로 바이트 동일 복사만 하면 되므로 "
            "리샘플·정규화가 개입할 여지가 구조적으로 없다.\n\n"
            "결과: 1,119개 / 250콜 / 78.6MB / 8kHz mono. 원본 결측 0건, 깨진 파일 0건. "
            "index duration 대비 실제 WAV 길이 차이는 최대 0.059초, 0.01초 초과 1건.\n"
            "5발화 미만 콜 67건(26.8%)은 제외하지 않고 `n_customer_utt`에 실제 개수를 기록했다.\n\n"
            "> **WAV 파일 자체는 이 repo에 없다** (AIHub 재배포 금지). `audio_seg_extract.py`를 "
            "외장하드 마운트 상태에서 돌리면 로컬에 재생성된다. manifest도 전사 컬럼을 제거한 "
            "`audio_seg_manifest_notext.parquet`만 포함한다."
        ),
    ),
]


def load_stripped(path: Path) -> pd.DataFrame:
    """STRIP_COLS에 등록된 파일을 해당 컬럼 제거 상태로 읽는다."""
    d = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    drop = [c for c in STRIP_COLS[path.name] if c in d.columns]
    return d.drop(columns=drop)


def write_stripped(src: Path, dst: Path) -> None:
    d = load_stripped(src)
    if dst.suffix == ".parquet":
        d.to_parquet(dst, index=False)
    else:
        d.to_csv(dst, index=False, encoding="utf-8-sig")


def scan_file(path: Path):
    """전사 원문 의심 컬럼 목록 반환."""
    bad = []
    try:
        if path.name in STRIP_COLS and path.suffix in (".parquet", ".csv"):
            d = load_stripped(path)
            return [str(c) for c in d.columns if _col_risky(d[c])]
        if path.suffix == ".parquet":
            d = pd.read_parquet(path)
        elif path.suffix == ".csv":
            d = pd.read_csv(path, nrows=SCAN_ROWS, dtype=str)
        elif path.suffix == ".xlsx":
            xl = pd.ExcelFile(path)
            for sh in xl.sheet_names:
                dd = xl.parse(sh, nrows=SCAN_ROWS, dtype=str)
                bad += [f"{sh}:{c}" for c in dd.columns if _col_risky(dd[c])]
            return bad
        elif path.suffix == ".md":
            txt = path.read_text(encoding="utf-8")
            if re.search(r"발화\d+\(", txt):
                bad.append("본문에 '발화N(화자):' 패턴")
            return bad
        else:
            return bad
    except Exception as e:
        return [f"READ_ERR:{type(e).__name__}"]
    bad += [str(c) for c in d.columns if _col_risky(d[c])]
    return bad


def _col_risky(s: pd.Series) -> bool:
    s = s.dropna().astype(str).head(SCAN_ROWS)
    if len(s) == 0:
        return False
    return bool(((s.str.len() > LONG_TEXT_CHARS) & s.str.contains(KO)).any())


def write_stage_readme(st, copied):
    lines = [f"# {st['dir'][:2]}. {st['title']}", "", f"**질문**: {st['question']}", "",
             "## 스크립트", "", "| 스크립트 | 역할 |", "|---|---|"]
    for name, role in st["scripts"]:
        lines.append(f"| `scripts/{name}` | {role} |")
    lines += ["", "## 산출물", ""]
    if copied:
        lines += ["| 파일 | 크기 |", "|---|---|"]
        for f in copied:
            kb = (RESULTS / st["dir"] / f).stat().st_size / 1024
            lines.append(f"| `{f}` | {kb:,.0f} KB |")
    else:
        lines.append("(이 단계의 산출물은 원문 포함으로 공개 repo에서 제외됨)")
    lines += ["", "## 결론", "", st["conclusion"], ""]
    (RESULTS / st["dir"] / "README.md").write_text("\n".join(lines), encoding="utf-8")


# 스크립트 파일명 -> 단계. 위에서부터 먼저 맞는 규칙을 쓴다(긴 접두어 우선).
SCRIPT_RULES = [
    ("repo_organize", None),
    ("build_index", "00"), ("clean_text", "00"), ("split_batches", "00"),
    ("label_inbound_outbound", "00"), ("extract_acoustic_features", "00"),
    ("prepare_human_eval", "01"), ("pilot_label_llm", "01"),
    ("build_human_eval_final", "01"), ("compute_kappa", "01"),
    ("relabel_v2_and_kappa", "01"), ("batch1_", "01"),
    ("v_unified_", "01"), ("v3_", "01"), ("v4_", "01"), ("v5_", "01"),
    ("stage2a_", "02"), ("stage2b_", "02"), ("stage2c_", "02"), ("stage2l_", "02"),
    ("stage2_bd_", "03"), ("stage2d_", "03"), ("stage2e_", "03"), ("stage2f_", "03"),
    ("stage2g_", "03"), ("stage2h_", "03"), ("stage2i_", "03"), ("stage2j_", "03"),
    ("stage2k_", "03"), ("stage2m_", "03"), ("stage2n_", "03"), ("stage2o_", "03"),
    ("stage2p_", "03"), ("stage2q_", "03"), ("stage2r_", "03"), ("mcnemar_v3", "03"),
    ("prompt_pilot_", "04"), ("prompt_explore_", "04"),
    ("probing_", "05"), ("gate_", "05"), ("train10k_", "05"), ("trigger_comparison", "05"),
    ("model_pilot_", "06"), ("modality_ab_", "06"), ("lang_pilot_", "06"),
    ("nl_en_template", "06"), ("arousal_target_", "06"),
    ("agent_judge_", "07"), ("agent_strat_", "07"),
    ("audio_seg_", "08"),
]


def classify_script(name: str):
    for prefix, stage in SCRIPT_RULES:
        if name.startswith(prefix):
            return stage
    return "?"


def write_pipeline_doc():
    """docs/PIPELINE.md — 단계 지도 + 스크립트 역인덱스.

    scripts/가 평면이라 '어느 단계에서 무엇을 보면 되는지'를 이 문서가 책임진다.
    역인덱스는 실제 scripts/ 디렉터리를 읽어 생성하므로 누락이 생기지 않는다.
    """
    by_stage = {st["dir"][:2]: st for st in STAGES}
    L = [
        "# 파이프라인 지도", "",
        "`scripts/`는 평면 구조다. 하위 폴더로 옮기면 102개 스크립트의 "
        "`outputs/` 경로 상수와 상호 import가 깨지기 때문에 재현성을 위해 그대로 뒀다. "
        "대신 **어느 단계에서 어떤 스크립트를 보면 되는지는 이 문서가 책임진다.**", "",
        "결과물은 `results/<단계>/`에 단계별로 정리돼 있고, 각 폴더의 `README.md`에 "
        "그 단계의 질문·스크립트·산출물·결론이 적혀 있다.", "",
        "---", "", "## 흐름", "",
        "```",
        "00 전처리 ── 01 gold 라벨링 ── 02 음향 정규화 ── 03 조건 실험(A/B/C/D × N)",
        "                                                        │",
        "                              ┌─────────────────────────┴──────────────┐",
        "                              │                                        │",
        "                     04 프롬프트 변형                          05 정보부족 게이트",
        "                              │                                        │",
        "                              └─────────────┬──────────────────────────┘",
        "                                            │",
        "                              06 250건 paired 파일럿 (모델·effort·모달리티·언어)",
        "                                            │",
        "                              ┌─────────────┴─────────────┐",
        "                              │                           │",
        "                     07 상담사 품질 층화          08 오디오 세그먼트 준비",
        "```", "",
        "---", "", "## 단계별 지도", "",
        "| 단계 | 무엇을 물었나 | 시작점 스크립트 | 결과 |",
        "|---|---|---|---|",
    ]
    for st in STAGES:
        num = st["dir"][:2]
        entry = st["scripts"][0][0].split(" / ")[0]
        L.append(f"| [{num}]({'../results/' + st['dir']}/) {st['title']} | {st['question']} "
                 f"| `scripts/{entry}` | [`results/{st['dir']}/`](../results/{st['dir']}/) |")
    L += ["", "---", "", "## 스크립트 → 단계 역인덱스", "",
          "`scripts/` 전체를 단계별로 묶은 것. 파일을 열기 전에 여기서 소속 단계를 찾고, "
          "그 단계의 `results/<단계>/README.md`를 먼저 읽으면 맥락이 잡힌다.", ""]

    scripts = sorted(p.name for p in (BASE_DIR / "scripts").iterdir()
                     if p.suffix in (".py", ".sh"))
    grouped = {}
    for s in scripts:
        grouped.setdefault(classify_script(s), []).append(s)

    for num in [st["dir"][:2] for st in STAGES]:
        items = grouped.get(num, [])
        if not items:
            continue
        st = by_stage[num]
        L += [f"### {num}. {st['title']} ({len(items)}개)", "",
              " · ".join(f"`{s}`" for s in items), ""]
    for num, label in [(None, "저장소 관리"), ("?", "미분류")]:
        items = grouped.get(num, [])
        if items:
            L += [f"### {label} ({len(items)}개)", "",
                  " · ".join(f"`{s}`" for s in items), ""]

    L += ["---", "", "## 반복되는 규칙 (전 단계 공통)", "",
          "실험을 다시 돌리거나 확장할 때 지켜야 하는 것들.", "",
          "- `stage2d_prompt.py`의 `CATEGORY_DEFINITIONS`와 gold_actual taxonomy는 **수정 금지**. "
          "01단계 v3 실패가 여기서 나왔다.",
          "- `version=\"base\"` 프롬프트의 출력은 바이트 동일성을 유지해야 한다. "
          "`stage2d_prompt.py`를 건드리면 회귀 검증 필수.",
          "- 라벨 정규화는 기존 숫자 접두어 방식만. 유사도·근접 카테고리 매칭 금지.",
          "- 화자 식별은 `speaker_type`(접두어)만. `speaker_id`는 쓰지 않는다.",
          "- `gold_actual`은 평가에만. 채점·judge 프롬프트에 절대 넣지 않는다. "
          "`subcategory`는 ground truth가 아니다.",
          "- 결측·이상치를 조용히 지우거나 채우지 않는다. 항상 세어서 보고한다.",
          "- 모든 ML 판단은 CV 기반. 스케일링은 fold 내부에서만 적합. "
          "임계값은 train에서 구해 추론에 **고정** 적용.",
          "- 외장하드 원본(AIHub)은 읽기 전용. 오디오 리샘플·정규화 금지(원음 그대로).",
          "- API 제출 전 비용 추정을 먼저 보고한다. Batch 제출 스크립트에는 "
          "`error_file_id` 패치가 들어가야 한다.", ""]

    (DOCS / "PIPELINE.md").write_text("\n".join(L), encoding="utf-8")
    unmapped = grouped.get("?", [])
    print(f"  docs/PIPELINE.md — 스크립트 {len(scripts)}개 분류, 미분류 {len(unmapped)}개")
    for s in unmapped:
        print(f"    [미분류] {s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="복사 없이 점검만")
    args = ap.parse_args()

    missing, risky, total_bytes = [], [], 0
    print("=== 화이트리스트 점검 ===")
    for st in STAGES:
        for f in st["files"]:
            src = OUT / f
            if not src.exists():
                missing.append(f)
                continue
            bad = scan_file(src)
            if bad:
                risky.append((f, bad))
            total_bytes += src.stat().st_size

    print(f"  대상 {sum(len(s['files']) for s in STAGES)}개 / 결측 {len(missing)}개 / "
          f"원문의심 {len(risky)}개 / 합계 {total_bytes/1e6:.1f} MB")
    for f in missing:
        print(f"    [결측] {f}")
    for f, bad in risky:
        print(f"    [원문의심] {f} -> {bad}")
    if risky:
        raise SystemExit("원문 의심 파일이 화이트리스트에 있습니다. 목록에서 빼고 다시 실행하세요.")
    if args.check:
        print("\ncheck 모드 종료 (복사 안 함)")
        return

    if RESULTS.exists():
        shutil.rmtree(RESULTS)
    DOCS.mkdir(exist_ok=True)

    print("\n=== 복사 ===")
    n_stripped = 0
    for st in STAGES:
        d = RESULTS / st["dir"]
        d.mkdir(parents=True, exist_ok=True)
        copied = []
        for f in st["files"]:
            src = OUT / f
            if not src.exists():
                continue
            if f in STRIP_COLS:
                write_stripped(src, d / f)
                n_stripped += 1
            else:
                shutil.copy2(src, d / f)
            copied.append(f)
        write_stage_readme(st, copied)
        print(f"  {st['dir']:20s} {len(copied):2d}개")

    print("\n=== results/ 재검증 (복사된 전체 다시 스캔) ===")
    leaks = []
    for p in sorted(RESULTS.rglob("*")):
        if p.is_file() and p.name != "README.md":
            bad = scan_file(p)
            if bad:
                leaks.append((str(p.relative_to(BASE_DIR)), bad))
    if leaks:
        for f, bad in leaks:
            print(f"  [누수] {f} -> {bad}")
        raise SystemExit("results/에 원문이 남아 있습니다.")
    n_files = sum(1 for p in RESULTS.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in RESULTS.rglob("*") if p.is_file())
    print(f"  누수 0건 / results/ {n_files}개 파일 / {size/1e6:.1f} MB")
    print(f"  컬럼 제거 후 수록: {n_stripped}개 "
          f"({', '.join(f'{k}→{v}' for k, v in STRIP_COLS.items())})")

    print("\n=== 문서 생성 ===")
    write_pipeline_doc()
    print("REPO_ORGANIZE_COMPLETE")


if __name__ == "__main__":
    main()
