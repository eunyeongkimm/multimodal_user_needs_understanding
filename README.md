# Multimodal User Needs Understanding via Spoken Queries

콜센터 상담 통화에서 **고객이 말한 것(표면 니즈)**과 **실제로 원한 것(실제 니즈)**이 어긋나는 현상을 다루고, 음성의 물리적 특징(F0·에너지·무음)이 그 간극을 메울 수 있는지 검증한 연구 저장소.

- 데이터: AIHub 「저음질 전화망 음성인식 데이터」 D04 (전자상거래/온라인 교육 플랫폼 상담), 74,123콜
- 라벨: 7개 카테고리 — 환불요청 / 주문취소 / 불만제기 / 배송확인 / 교환반품 / 구매진행 / 서비스이용

---

## 레포 구조

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
| [09_audio_native_model_test](results/09_audio_native_model_test/) | audio-native 모델 테스트 (Qwen-Omni, Colab 노트북) |

### `scripts/`가 평면인 이유

102개 스크립트가 `outputs/<파일명>`을 하드코딩하고 서로 import하고 있어, 하위 폴더로 옮기면 경로 상수와 import가 동시에 깨진다. 재현성을 위해 평면을 유지하고 **탐색은 [`docs/PIPELINE.md`](docs/PIPELINE.md)의 역인덱스가 담당한다.**

---

## 데이터 정책 (중요)

AIHub D04는 **재배포 금지** 라이선스다. 따라서 이 저장소에는 다음이 **포함되어 있지 않다**.

- 전사 원문 (`text` / `call_text` / `대화(텍스트)` / 번역 캐시)
- 프롬프트에 전사가 통째로 들어간 파일 (`*_request_index.parquet`, batch 요청 `*.jsonl`)
- 원본 오디오 및 추출 세그먼트 WAV

저장소에는 **예측·라벨·지표·요약만** 들어 있다. 큐레이션은 화이트리스트 방식이며, `scripts/repo_organize.py`가 복사 전과 후 두 번 원문 스캔(60자 초과 한글 셀 탐지)을 수행하고 하나라도 걸리면 중단한다.

예외로 두 곳에 짧은 발화 인용이 남아 있다. 둘 다 방법·근거의 일부라 검토 후 유지하기로 한 범위다.

- `scripts/stage2d_prompt.py` — v3b 변형의 few-shot 예시 2발화
- `results/09_audio_native_model_test/qwen3_omni_test_vllm.ipynb` — 모델 reasoning 출력에 인용된 5발화

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

상세 이력과 알려진 이슈는 [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md).
