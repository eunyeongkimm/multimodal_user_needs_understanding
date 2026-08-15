# 프로젝트 상태 정리 (2026-07-19 기준)

콜센터 상담 통화(D04, 저음질 전화망 음성인식 데이터)를 대상으로 (1) 실제 니즈(gold_actual) 7-카테고리 라벨을 GPT로 구축하고, (2) 통화 초반 일부 발화(N=1,2,3,5)만으로 "surface" 니즈를 예측할 때 텍스트만 쓸 때와 acoustic(음성 특징)을 함께 쓸 때 정확도가 어떻게 달라지는지 검증하는 프로젝트.

---

## 1. 파이프라인 전체 흐름 (스크립트 → 산출물)

### 1-1. 원천 인덱스 구축

| 단계 | 스크립트 | 산출물 |
|---|---|---|
| D04 라벨링 JSON → dialog 단위 인덱스 생성 | `scripts/build_index.py` | `outputs/d04_dialog_index_old_no_flag.parquet` (r_gold 계산 전 버전) |
| 텍스트 정제(disfluency 태그 제거 등) + `r_gold`(발화속도)/`r_gold_valid_flag`(품질필터) 계산 | `scripts/clean_text.py` | `outputs/d04_dialog_index.parquet` **(현재 사용 중인 최신본)**, `outputs/data_cleaning_report.md`, `outputs/exclusion_log_r_gold_gt15.csv` |
| 컬럼 설명 문서 | (수기 작성) | `outputs/d04_dialog_index_column_dictionary.md` |
| 전체 74,123콜을 20,000(batch1) / 54,123(batch2)로 고정 분할 (`random_state=42`) | `scripts/split_batches.py` | `outputs/batch1_call_ids.csv`, `outputs/batch2_call_ids.csv` |
| 인바운드/아웃바운드 판별(사람 대조 80건) | `scripts/label_inbound_outbound.py` | `outputs/inbound_outbound_gpt.csv`, `outputs/opening_speaker_audit_sample.csv` |
| Acoustic feature 추출 프로토타입(50건 샘플, F0/Energy/silence 검증용) | `scripts/extract_acoustic_features.py` | `outputs/acoustic_features_sample50.csv` |

### 1-2. 사람 평가셋 준비 + Pilot 라벨링(v1)

| 단계 | 스크립트 | 산출물 |
|---|---|---|
| 사람 라벨링용 샘플 추출(`random_state=42`) | `scripts/prepare_human_eval.py` | `outputs/human_eval_sample.csv` |
| GPT-5.5 vs GPT-4.1-mini 파일럿(200건, v1 프롬프트) | `scripts/pilot_label_llm.py` | `outputs/pilot200_calls.parquet`, `outputs/pilot200_gpt_labels.csv`, `outputs/pilot_run_log.txt` |
| 사람 라벨 병합 + 최종 평가셋(80건) 구성 | `scripts/build_human_eval_final.py` | `outputs/human_eval_final.csv` → **`outputs/human_eval_final_fin.csv`(human_label까지 채워진 최종본)**, `outputs/human_eval_final.xlsx` |
| Cohen's Kappa 계산(v1) | `scripts/compute_kappa.py` | (콘솔 출력) |

### 1-3. Gold_actual 라벨링 v2 (Batch1 전체 19,847건)

| 단계 | 스크립트 | 산출물 |
|---|---|---|
| v2 프롬프트로 80건 재검증 (v1 대비 Kappa 개선 확인) | `scripts/relabel_v2_and_kappa.py` | `outputs/pilot80_gpt_labels_v2.csv`, `outputs/relabel_v2_run_log.txt` |
| Batch1 청크 계획(토큰 예산 기준 분할) | `scripts/batch1_chunk_plan.py` | `outputs/batch1_chunks_plan.json` |
| Batch1 최초 제출(1회성) | `scripts/batch1_submit.py` | (최초 batch_id 기록, 이후 아래로 대체) |
| 청크 순차 제출/재시도/재시작 시 중복제출 방지 자동화 | `scripts/batch1_check.py` + `scripts/batch1_autorun.sh` | **`outputs/gold_actual_batch1.parquet`(v2, 19,847건, 7개 라벨이지만 불만제기는 32건뿐)** |

→ 이후 샘플 점검(대화 중 인라인 분석, 스크립트로 저장되지 않음)에서 "환불요청/배송확인으로 분류된 콜 중 상당수가 실제로는 항의성 콜인데 불만제기로 안 잡힘"을 확인.

### 1-4. Gold_actual 라벨링 v3 (폐기됨) → v4 → v5

| 단계 | 스크립트 | 결과 |
|---|---|---|
| v3: 카테고리 정의 사이에 "불만제기 우선판단 규칙"을 끼워넣는 방식 | `scripts/v3_prompt.py`, `v3_chunk_plan.py`, `v3_check.py`, `v3_autorun.sh`, `v3_merge_final.py`, `v3_human_eval_kappa.py` | human_eval 49건 검증 결과 **Kappa 0.82→0.71로 하락** → **중단/폐기** (환불요청↔주문취소 경계가 흔들리는 부작용 확인). 청크 0~2만 처리된 채 중단(`outputs/v3_chunks_state.json` 참고), `gold_actual_batch1_v3.parquet`는 생성되지 않음. |
| v4: 판단을 "1단계 항의 게이트(JSON has_complaint) → 2단계 카테고리 분류"로 완전 분리 | `scripts/v4_prompt.py`, `v4_chunk_plan.py`, `v4_check.py`, `v4_autorun.sh`, `v4_merge_final.py`, `v4_self_consistency.py`, `v4_suspect_scan.py` | `outputs/gold_actual_batch1_v4.parquet`(환불요청/배송확인 11,579건 재라벨), 불만제기 452건 검출. `outputs/suspect_calls_for_review.csv`(사람 검토 후보 50건), `outputs/v4_self_consistency_pass2.csv`(자기일관성 Kappa 0.96). 다만 v4가 환불요청→교환반품/주문취소/서비스이용으로도 과다 이동시키는 부작용 발견(공통 판단원칙 누락 때문으로 추정). |
| v5: v4의 `has_complaint=False` 11,127건만 게이트 없이 v2식 판단원칙으로 재분류 (452건 불만제기는 그대로 유지) | `scripts/v5_prompt.py`, `v5_chunk_plan.py`, `v5_check.py`, `v5_autorun.sh`, `v5_merge_final.py` | `outputs/gold_actual_batch1_v5.parquet`(11,127건). **`outputs/gold_actual_batch1_final.parquet`를 v5 기준으로 덮어씀 → 이게 최종본.** v2 대비 카테고리 분포가 거의 복원되면서 불만제기(484건, 2.44%)만 유지됨. |

**→ `outputs/gold_actual_batch1_final.parquet`가 이후 모든 2단계 분석의 gold_actual 기준 파일.**

### 1-5. Stage 2: Surface 윈도우 + Acoustic + A/B/C/D 실험

| 단계 | 스크립트 | 산출물 |
|---|---|---|
| N=1,2,3,5 × 고객only/고객&상담사 윈도우(발화 pool) 추출 | `scripts/stage2a_window_pool.py` | `outputs/stage2_utterance_pool.parquet`(135,340건) |
| 발화 단위 acoustic feature 추출 (F0, Energy, silence, 8코어 병렬) | `scripts/stage2b_acoustic_extract.py` | `outputs/stage2_acoustic_features.parquet` |
| 화자별 정규화(콜/윈도우 내 z-score, 표본부족 시 전체분포 fallback) + 자연어 변환(5단계 분위수) | `scripts/stage2c_normalize_nl.py` | `outputs/stage2_windows_nl.parquet`(414,948행) |
| surface 예측 프롬프트(A/B/C/D 공용) | `scripts/stage2d_prompt.py` | - |
| 16조건(N4×버전2×acoustic2) 요청 생성 + 청크 계획 + 비용 산출 | `scripts/stage2d_chunk_plan.py` | `outputs/stage2d_request_index.parquet`, `outputs/stage2d_requests_plan.json` |
| **동시-제출 파이프라인**(최대 8개 청크 동시 in-flight, 완료 즉시 backfill) 방식으로 909개 청크 실행 | `scripts/stage2d_check.py` + `scripts/stage2d_autorun.sh` | **`outputs/stage2d_gpt_predictions.parquet`(315,160건, 원본 예측)** |
| N×조건별 정확도/mismatch 비율 집계 | `scripts/stage2e_analysis.py` | `outputs/stage2_condition_summary.csv` |
| gold_actual 카테고리별 acoustic 효과 분해 | `scripts/stage2f_per_label_analysis.py` | `outputs/stage2_per_label_acoustic_effect.csv` |
| "surface(A,N=1) ≠ actual" mismatch 서브셋(9,658건)만 재분석 | `scripts/stage2g_mismatch_subset_analysis.py` | `outputs/stage2_mismatch_subset_accuracy.csv` |
| McNemar 유의성 검정(N=1 A/B, N=2·3·5 C/D) | `scripts/stage2h_mcnemar_test.py` | `outputs/stage2_mcnemar_results.csv` |

**핵심 결론(2026-07-19 시점)**: 전체 데이터셋에서는 acoustic 효과가 미미/음수였지만, surface(N=1)와 actual이 어긋나는 어려운 콜(9,658건, 49.4%)만 보면 N=1·2·3에서 acoustic이 통계적으로 유의하게(p<0.001) 정확도를 높였고, N=5에서는 유의성이 사라짐(p=0.073). 카테고리별로는 불만제기가 다른 카테고리보다 acoustic에 더 민감하지는 않음(거의 동일).

---

## 2. 최종본 vs 중간산출물 구분표

### ⚠️ 가장 헷갈리는 것: gold_actual_batch1 계열

| 파일 | 상태 | 설명 |
|---|---|---|
| `gold_actual_batch1.parquet` | 중간산출물 (삭제 금지 — 의존성 있음) | v2 결과, 19,847건. 불만제기 32건뿐이라 그대로 쓰면 안 됨. v3/v4/v5 청크 계획 스크립트들이 이 파일을 베이스로 참조하므로 보존 필요. |
| `gold_actual_batch1_v4.parquet` | 중간산출물 | v4 결과, 11,579건(환불요청/배송확인만). v5의 입력. |
| `gold_actual_batch1_v5.parquet` | 중간산출물 | v5 결과, 11,127건(v4에서 has_complaint=False였던 것만). |
| **`gold_actual_batch1_final.parquet`** | ✅ **최종본** | 19,847건 전체. **주의: 이 파일은 두 번 생성됐음** — 처음엔 `v4_merge_final.py`가 v4 기준으로 만들었고, 이후 `v5_merge_final.py`가 v5 기준으로 **덮어썼음**. 현재 디스크에 있는 건 v5 기준 버전(= v2 카테고리 경계 복원 + v4의 불만제기 게이트 유지)이며, v4 기준 버전은 더 이상 존재하지 않음. **앞으로는 이 파일만 참조.** |

### d04_dialog_index 계열

| 파일 | 상태 | 설명 |
|---|---|---|
| `d04_dialog_index_old_no_flag.parquet` | 구버전(폐기 가능하나 보존 중) | `clean_text.py` 실행 전 백업. `text_clean`/`r_gold`/`r_gold_valid_flag` 없음. |
| **`d04_dialog_index.parquet`** | ✅ **최종/현재 사용본** | `r_gold`, `r_gold_valid_flag` 포함. 모든 stage2 스크립트가 이 파일을 사용. |
| `d04_dialog_index_sample.csv` | 중간산출물 | 구조 확인용 샘플, 폐기 가능. |

### human_eval 계열

| 파일 | 상태 | 설명 |
|---|---|---|
| `human_eval_sample.csv` | 중간산출물 | 라벨링 전 최초 샘플. |
| `human_eval_final.csv` | 중간산출물 | v1 GPT 라벨만 붙은 중간 버전. |
| **`human_eval_final_fin.csv`** | ✅ **최종본** | `human_label`까지 채워진 80건. 이후 모든 Kappa 계산의 기준. |
| `human_eval_final.xlsx` | 원본 입력 파일 (보존 필수) | 사람이 직접 `human_label`을 입력한 작업 파일. |
| `pilot80_gpt_labels_v2.csv`, `human_eval_v3_labels.csv`, `v4_self_consistency_pass2.csv` | 검증용 중간산출물 | 각 프롬프트 버전의 모델 출력. 재현성 참고자료로 보존 권장. |

### Stage2 계열

| 파일 | 상태 | 설명 |
|---|---|---|
| `stage2_utterance_pool.parquet`, `stage2_acoustic_features.parquet`, `stage2_windows_nl.parquet` | 중간산출물(재생성 가능하나 재실행에 시간 소요) | 특히 acoustic 추출은 ~16분 소요되므로 당장은 보존 권장. |
| **`stage2d_gpt_predictions.parquet`** | ✅ **최종 원본 데이터** | 315,160건 전체 예측. 아래 4개 분석표의 원천이므로 반드시 보존. |
| `stage2d_request_index.parquet`(76MB), `stage2d_requests_plan.json`(7MB) | 중간산출물(대용량) | 프롬프트 원문 포함. 예측이 이미 끝났으므로 삭제해도 무방(단 재실행하려면 재생성 필요). |
| **`stage2_condition_summary.csv`, `stage2_per_label_acoustic_effect.csv`, `stage2_mismatch_subset_accuracy.csv`, `stage2_mcnemar_results.csv`** | ✅ **최종 분석 결과** | 논문에 바로 인용 가능한 요약표. |
| `suspect_calls_for_review.csv` | ✅ 최종 산출물이나 **사람 검토 미완료** | TODO 참고. |

### 안전하게 삭제 가능한 것 (재현 시 재생성됨, 디스크 정리용)

| 패턴 | 개수 | 총 용량 | 비고 |
|---|---|---|---|
| `stage2d_chunk{N}_requests.jsonl` | 909개 | ~704MB | 이미 OpenAI에 업로드·처리 완료. **가장 큰 정리 대상.** |
| `batch1_chunk{N}_requests.jsonl` | 19개 | ~103MB | 동일 |
| `v{2,3,4,5}_chunk{N}_requests.jsonl` | 총 20개 | ~107MB | 동일 |
| `*_partial/` 디렉토리 6개 (batch1/v3/v4/v5/stage2_acoustic/stage2d) | - | - | 각 청크 결과 조각. 병합된 최종 파일이 이미 있으므로 디버깅 목적 아니면 불필요. |
| `*_autorun_stdout.log` (대부분 0바이트) | 5개 | - | 사실상 빈 파일. |

**주의**: 위 파일을 지우면 `outputs/*_chunks_state.json`과 불일치가 생겨 해당 파이프라인을 재실행할 수 없게 됩니다(체크포인트 기준 파일이 사라지므로). 완전히 끝난 파이프라인(v2/v4/v5/stage2d 전부 `ALL_CHUNKS_COMPLETE`로 종료됨)만 정리 대상입니다.

### 기타

| 파일 | 상태 |
|---|---|
| `batch2_call_ids.csv` | ✅ 보존 필수 — 다음 단계(54,123건) 대상 목록, 아직 미처리 |
| `inbound_outbound_gpt.csv`, `opening_speaker_audit_sample.csv` | 검증 중(TODO의 "인바운드 필터 검증"과 연결) |
| `acoustic_features_sample50.csv` | 프로토타입 검증용, stage2 파이프라인과 무관, 폐기 가능 |

---

## 3. 미완료/보류 작업 (TODO)

- [ ] **`human_eval_final_fin.csv` 중 환불요청/배송확인 human_label 재검토** — v3 kappa 검증 스크립트 기준으로는 49건(환불요청 36 + 배송확인 13)이었음 (사용자 언급은 48건 — 재확인 필요). v5 규칙(항의 우선판단) 기준으로 사람이 다시 라벨링할 예정. **아직 미착수.**
- [ ] **`suspect_calls_for_review.csv` 50건 사람 검토** — v4의 항의 게이트가 놓쳤을 수 있는 후보(항의 키워드는 있으나 불만제기로 분류 안 된 콜). **아직 미착수.**
- [ ] **Batch2(54,123건) 처리** — `outputs/batch2_call_ids.csv`에 대상만 확정됨. 인바운드/아웃바운드 필터(`inbound_outbound_gpt.csv` 기준) 검증부터 먼저 하고 진행 예정.
- [ ] **Gender/emotion classifier 도입 여부** — 교수님 제안, 아직 방향만 논의되고 미착수.
- [ ] **`J18_S002691` 1건 복구** — batch1(v2) 라벨링 중 개별 요청 레벨 API 에러로 조용히 유실된 콜(아래 "알려진 이슈" 참고). 지금 단당 처리하지 않고, **batch2(54,123건) 처리할 때 같이 포함**시키기로 결정함.

---

## 3-1. 알려진 이슈 및 수정 이력

**[2026-07-19] batch1(v2) 라벨링 중 1건 조용한 유실 발견 + 파이프라인 버그 수정**

- **증상**: `batch1_call_ids.csv`(20,000건) 대비 `gold_actual_batch1_final.parquet`가 19,847건으로 153건 부족. 152건은 `r_gold_valid_flag` 전량 False(유효 텍스트 없음, 정상 제외)였으나, **`J18_S002691` 1건은 정상적인 콜(dialog 65개 전부 valid)인데도 결과에서 빠져 있었음.**
- **근본 원인**: 이 콜이 속한 청크(batch1 청크4, batch_id=`batch_6a5a40b580ec81908bc521dc942985af`)에서 OpenAI가 이 요청 하나만 개별적으로 500 에러(`"BatchAPI failed to execute task in batch"`)를 냈고, 이 에러가 `output_file_id`가 아니라 별도의 **`error_file_id`**에 기록됨. batch 객체 자체는 `request_counts.failed=0`으로 "전부 정상 완료"라고 보고했기 때문에, 기존 `download_chunk_result()` 로직(모든 v2~stage2d 체크 스크립트에 동일하게 복사되어 있던 패턴)이 `output_file_id`만 읽고 `error_file_id`를 전혀 확인하지 않아 **이 1건이 에러도 경고도 없이 조용히 사라짐**.
- **다른 배치 영향 확인**: v4/v5/stage2d(합쳐서 337,866건)는 계획 건수와 실제 결과 건수가 전부 정확히 일치 — 이 버그로 인한 실제 데이터 유실은 batch1(v2)의 이 1건이 유일했음(전체 ~348,000건 중 1건, 매우 드문 개별 API 오류로 추정).
- **수정 완료**: `batch1_check.py`, `v4_check.py`, `v5_check.py`, `stage2d_check.py`의 `download_chunk_result()`에 `error_file_id` 확인 로직을 추가함 — 개별 요청 실패가 감지되면 (1) 콘솔에 경고와 실패한 custom_id 목록을 출력하고, (2) 즉시 동기 API로 자동 재시도해서 복구를 시도함. (`v3_check.py`는 v3 파이프라인 자체가 폐기되어 패치하지 않음.)
- **미해결**: `J18_S002691` 자체의 복구는 지금 하지 않고 **batch2(54,123건) 처리 시점에 함께 처리**하기로 결정 (위 TODO 참고).

---

## 4. 재현을 위한 환경 정보

### 패키지 버전
`requirements.txt`에 `pip freeze` 전체 고정 (venv 기준). 핵심 패키지:

| 패키지 | 버전 |
|---|---|
| librosa | 0.11.0 |
| praat-parselmouth | 0.4.7 |
| pandas | 2.3.3 |
| numpy | 2.0.2 |
| openai | 2.46.0 |
| tiktoken | 0.13.0 |
| scikit-learn | 1.6.1 |
| statsmodels | 0.14.6 (이번 세션에서 McNemar 검정 위해 신규 설치) |
| pyarrow | 21.0.0 |
| python-dotenv | 1.2.1 |

설치: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`

### .env 구조
```
OPENAI_API_KEY=<실제 키 값, 커밋 금지>
```
`.gitignore`에 이미 `.env` 포함됨. 스크립트들은 `load_dotenv(BASE_DIR / ".env")`로 로드.

### 외부 데이터 경로 (하드코딩, 환경 이전 시 수정 필요)
- 라벨링 JSON 원본: `/Volumes/AIHub/007.저음질 전화망 음성인식 데이터/01.데이터/1.Training/라벨링데이터_230316/D04`
- 오디오 원천 데이터: `/Volumes/AIHub/007.저음질 전화망 음성인식 데이터/01.데이터/1.Training/원천데이터_230316`

### random_state / seed 목록 (전부 42로 고정)

| 스크립트 | 용도 |
|---|---|
| `split_batches.py` | batch1(20,000)/batch2(54,123) shuffle 분할 |
| `prepare_human_eval.py` | human_eval 80건 샘플링 |
| `extract_acoustic_features.py` | acoustic 프로토타입 50건 샘플링 |
| `pilot_label_llm.py` | pilot 200건 샘플링 |
| `v4_self_consistency.py` | 자기일관성 검증 200건 샘플링 |
| `v4_suspect_scan.py` | 의심콜 검토용 50건 샘플링 |
| `stage2h_mcnemar_test.py` | paired bootstrap(10,000회) |

**주의**: 대화 중 환불요청/배송확인/교환반품 20건씩 샘플 점검한 인라인 분석(스크립트로 저장 안 됨)도 `random_state=42`를 사용했음 — 재현하려면 `gold_actual_batch1.parquet`에서 해당 라벨로 필터링 후 `.sample(n=20, random_state=42)`로 동일 재현 가능.

---

## 5. 비용 로그 (OpenAI API, 실측/추정 혼합)

| 단계 | 모델 | 방식 | 건수 | 비용 |
|---|---|---|---|---|
| Pilot(v1) | gpt-5.5 + gpt-4.1-mini | 동기 API | 200건×2모델 | $1.7689 |
| v2 human_eval 재검증 | gpt-5.5 + gpt-4.1-mini | 동기 API | 80건×2모델 | $0.8816 |
| **v2 gold_actual_batch1** | gpt-4.1-mini | Batch API | 19,847건 | ~$5.72 (추정치, 당시 비용 로그 미기록) |
| v3 human_eval kappa 검증 | gpt-4.1-mini | 동기 API | 49건 | $0.0354 |
| v3 본배치 (중단, 청크0-3만 처리 후 폐기) | gpt-4.1-mini | Batch API | 3,350건 | ~$1.2258 (**폐기된 sunk cost**) |
| v4 본배치 + 자기일관성 200건 | gpt-4.1-mini | Batch API + 동기 | 11,579건+200건 | $2.7766 |
| v5 본배치 | gpt-4.1-mini | Batch API | 11,127건 | $2.1373 |
| **Stage2d (A/B/C/D×N 16조건)** | gpt-4.1-mini | Batch API(동시-제출) | 315,160건 | $38.3135 |
| **합계** | | | **약 348,000건 API 호출** | **약 $54.85** |

**Batch2(54,123건) 예산 추정 참고**: v2 gold_actual_batch1(19,847건, 전체 통화)이 ~$5.72였으므로, 동일 방식(v5 규칙 적용, 전체 통화 1회 분류)으로 batch2를 처리하면 단순 비례로 **약 $15.6** 예상 (54,123/19,847 × $5.72). 여기에 stage2 실험까지 동일 규모로 반복한다면 추가로 수십 달러 단위(Stage2d 방식 기준 315,160/19,847 × $38.31 ≈ 콜당 비율 적용 시 batch2 규모로는 대략 **$100~150** 수준 예상) — 실제로는 batch2 콜 수·발화 길이 분포에 따라 달라지므로 실행 전 `tiktoken` 기반 재계산 권장.

---

## 6. 폴더 구조 (요약)

```
1.논문/
├── .env                          (OPENAI_API_KEY만 포함, git 제외)
├── .gitignore
├── requirements.txt               (pip freeze, 이번에 생성)
├── README.md                      (본 문서)
├── 1.연구계획서/
│   └── 연구계획서_....pdf
├── venv/                          (Python 3.9 가상환경)
├── scripts/                       (43개 스크립트, 위 1절 표 참고)
└── outputs/
    ├── d04_dialog_index.parquet              (154MB, 최신 dialog 인덱스)
    ├── d04_dialog_index_old_no_flag.parquet   (82MB, 구버전 백업)
    ├── batch1_call_ids.csv / batch2_call_ids.csv
    ├── gold_actual_batch1.parquet             (v2, 중간산출물)
    ├── gold_actual_batch1_v4.parquet          (중간산출물)
    ├── gold_actual_batch1_v5.parquet          (중간산출물)
    ├── gold_actual_batch1_final.parquet       (★ 최종 gold_actual)
    ├── human_eval_final_fin.csv               (★ 최종 사람평가셋 80건)
    ├── human_eval_final.xlsx                  (사람 입력 원본)
    ├── suspect_calls_for_review.csv           (사람 검토 대기 50건)
    ├── stage2_utterance_pool.parquet          (12MB)
    ├── stage2_acoustic_features.parquet       (8MB)
    ├── stage2_windows_nl.parquet              (38MB)
    ├── stage2d_gpt_predictions.parquet        (★ 최종 원본 예측 315,160건)
    ├── stage2_condition_summary.csv           (★ 최종 분석표)
    ├── stage2_per_label_acoustic_effect.csv   (★ 최종 분석표)
    ├── stage2_mismatch_subset_accuracy.csv    (★ 최종 분석표)
    ├── stage2_mcnemar_results.csv             (★ 최종 분석표)
    ├── stage2_progress.json                   (파이프라인 진행상황 체크포인트)
    ├── batch1_chunk{0..18}_requests.jsonl     (19개, 정리 대상)
    ├── stage2d_chunk{0..908}_requests.jsonl   (909개, ~704MB, 정리 대상)
    ├── v{3,4,5}_chunk{N}_requests.jsonl       (20개, 정리 대상)
    ├── batch1_partial/ v3_partial/ v4_partial/ v5_partial/
    │   stage2_acoustic_partial/ stage2d_partial/  (청크 결과 조각, 정리 대상)
    └── *.log, *_chunks_state.json, *_target_call_ids.csv  (실행 로그/체크포인트)
```

전체 파일 목록은 `outputs/` 안에서 `ls -la`로 확인 가능 (총 1,020개 파일, 대부분 위 청크/partial 패턴).
