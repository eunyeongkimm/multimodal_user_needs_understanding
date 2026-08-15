# Multimodal User Needs Understanding via Spoken Queries

콜센터 상담 통화에서 **고객이 말한 것(표면 니즈)**과 **실제로 원한 것(실제 니즈)**이 어긋나는 현상을 다루고, 음성의 물리적 특징(F0·에너지·무음)이 그 간극을 메울 수 있는지 검증한 연구 저장소.

- 데이터: AIHub 「저음질 전화망 음성인식 데이터」 D04 (전자상거래/온라인 교육 플랫폼 상담), 74,123콜
- 라벨: 7개 카테고리 — 환불요청 / 주문취소 / 불만제기 / 배송확인 / 교환반품 / 구매진행 / 서비스이용

---

## 핵심 결과

통화 첫 발화의 표면 니즈와 통화 전체의 실제 니즈는 **유효 19,548콜 중 9,658콜(49.4%)에서 어긋난다.**

가장 두드러진 오류 유형은 *실제로는 항의하려고 건 전화인데 환불 문의로 읽히는* 경우다. 250콜 paired 표본에서 이 오류가 무엇에 반응하는지 다섯 개 축으로 확인했다.

| 흔든 축 | 조건 | `gold=불만제기 → 예측=환불요청` 셀 |
|---|---|---|
| 모델 버전 | 4.1-mini / 5.4 / luna / terra | 18~20 |
| reasoning effort | low / medium | 18~20 |
| 모달리티 | text-only / text+acoustic | 19 / 19 |
| 추론 언어 | 한국어 / 영어 | 19 / 19 |
| 입력 언어 | 한국어 / 영어(MT) | 19 / 18 |

전반 성능은 분명히 개선된다(macro-F1 0.476 → 0.538, 불만 F1 0.036 → 0.364). **그런데도 이 셀은 움직이지 않는다.** 텍스트·음성 양쪽 조건에서 동시에 틀리는 18콜이 남고, 그 콜들의 초반 고객 arousal은 오히려 정답을 맞힌 불만 콜보다 높다(0.407 vs 0.104, Mann-Whitney 전 지표 p>0.22). '음성이 밋밋해서 놓친다'는 설명은 지지되지 않는다.

상담사 응대 품질로 설명되는지도 확인했으나, 불만 교란을 분리한 층(gold≠불만, n=200)에서 AUC 0.506으로 판별력이 없어 사전 기준에 따라 해당 워크스트림을 종료했다.

자세한 근거는 [`results/06_pilot_250/`](results/06_pilot_250/), [`results/07_agent_quality/`](results/07_agent_quality/).

---

## 저장소 구조

```
docs/
  PIPELINE.md        ← 단계 지도 + 스크립트 126개 역인덱스 (여기서 시작)
  PROJECT_STATE.md   프로젝트 상태 스냅샷(2026-07-19): 환경·시드·비용·이슈 이력
scripts/             실행 스크립트 126개 (평면 구조)
results/             단계별 산출물 — 각 폴더 README에 질문·스크립트·결론
outputs/             로컬 작업 캐시 2.5GB (git 제외)
```

| 단계 | 내용 |
|---|---|
| [00_data_prep](results/00_data_prep/) | 원천 인덱스 구축 · 전처리 |
| [01_gold_labeling](results/01_gold_labeling/) | gold_actual 라벨링 (v1 → v5) |
| [02_acoustic](results/02_acoustic/) | 음향 피처 추출 · 성별 기준 정규화 |
| [03_condition_exp](results/03_condition_exp/) | 조건 실험 A/B/C/D × N |
| [04_prompt_pilot](results/04_prompt_pilot/) | 프롬프트 변형 파일럿 |
| [05_gate](results/05_gate/) | 정보부족 게이트 |
| [06_pilot_250](results/06_pilot_250/) | 250건 paired 파일럿 (모델·effort·모달리티·언어) |
| [07_agent_quality](results/07_agent_quality/) | 상담사 응대 품질 층화 |
| [08_audio_seg](results/08_audio_seg/) | 오디오 세그먼트 준비 |

### `scripts/`가 평면인 이유

102개 스크립트가 `outputs/<파일명>`을 하드코딩하고 서로 import하고 있어, 하위 폴더로 옮기면 경로 상수와 import가 동시에 깨진다. 재현성을 위해 평면을 유지하고 **탐색은 [`docs/PIPELINE.md`](docs/PIPELINE.md)의 역인덱스가 담당한다.**

---

## 데이터 정책 (중요)

AIHub D04는 **재배포 금지** 라이선스다. 따라서 이 저장소에는 다음이 **포함되어 있지 않다**.

- 전사 원문 (`text` / `call_text` / `대화(텍스트)` / 번역 캐시)
- 프롬프트에 전사가 통째로 들어간 파일 (`*_request_index.parquet`, batch 요청 `*.jsonl`)
- 원본 오디오 및 추출 세그먼트 WAV

저장소에는 **예측·라벨·지표·요약만** 들어 있다. 큐레이션은 화이트리스트 방식이며, `scripts/repo_organize.py`가 복사 전과 후 두 번 원문 스캔(60자 초과 한글 셀 탐지)을 수행하고 하나라도 걸리면 중단한다.

원문이 필요하면 [AIHub](https://aihub.or.kr)에서 직접 내려받은 뒤 `scripts/build_index.py`부터 재현하면 된다. 외부 경로는 `docs/PROJECT_STATE.md`의 "외부 데이터 경로" 참고.

---

## 재현

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=<your-key>" > .env      # .gitignore 처리됨

# 산출물 재정리 (원문 누수 검사 포함)
python3 scripts/repo_organize.py --check     # 점검만
python3 scripts/repo_organize.py             # results/ + docs/PIPELINE.md 재생성
```

시드는 전부 42로 고정되어 있다(목록은 `docs/PROJECT_STATE.md`). 패키지 버전은 `requirements.txt`에 `pip freeze`로 고정.

---

## 남은 작업

- `results/07_agent_quality/agent_judge_human_eval_slots.csv` 40건 수동 채점 → `agent_judge_gate.py` 재실행 시 quadratic-weighted Kappa 자동 산출
- 250건 파일럿의 조건 간 McNemar 검정 (per-call parquet 존재, 추가 API 호출 불필요)
- Batch2(54,123건) 처리 — `J18_S002691` 1건 복구 포함
- Track B(shrinkage) 정규화 검증

상세 이력과 알려진 이슈는 [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md).
