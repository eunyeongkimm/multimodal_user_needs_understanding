# 10. fine-tuning 파이프라인 타당성 확인

**질문**: 09에서 zero-shot 한계(macro-F1 0.493)를 본 Qwen3-Omni-30B를, Colab A100 40GB에서 QLoRA로 실제 학습시킬 수 있는가?

## 스크립트

| 스크립트 | 역할 |
|---|---|
| `results/10_fine_tuning/fine_tuning_v0_pipeline_test.ipynb` | ms-swift + 4bit QLoRA 학습 1 step 통과 여부와 peak VRAM 확인 (더미 12건, max_steps=2) |

## 산출물

| 파일 | 크기 |
|---|---|
| `fine_tuning_v0_pipeline_test.ipynb` | 43 KB |

## 결론

성능 실험이 아니라 **환경 타당성 확인**이다. 판정 기준은 노트북에 미리 적어 뒀다 — "40GB 내에서 안정적으로 step이 완료되면 본 학습 진행".

설정: Qwen3-Omni-30B-A3B-Thinking, 4bit QLoRA (r=8, alpha=32), audio encoder·aligner 고정하고 LLM만 학습, batch 1 + grad accum 4 + gradient checkpointing.

**결과: 미달.** 학습 step에 들어가기도 전에 **가중치 로딩 단계에서 CUDA OOM**이 났다 (39.47/39.49 GiB 사용 중 20 MiB 추가 할당 실패). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`와 double quant를 넣어 재시도했지만 같은 지점에서 동일하게 실패했다. 단편화가 아니라 **모델 자체가 40GB에 안 들어간다.**

다음 선택지는 셋이다 — 더 큰 GPU(A100 80GB / H100), 더 작은 백본(Qwen2.5-Omni-7B), 또는 fine-tuning 대신 09단계의 프롬프트·게이트 쪽을 더 파는 것. 09에서 7B가 5콜 스팟체크 0/5였던 걸 감안하면 두 번째는 회의적이다.

> 학습 데이터 포맷은 08단계 `audio_seg_manifest.parquet` + gold를 엮어 발화 5건 단위 messages JSONL로 만든다(노트북 cell 10). 이때 쓴 카테고리 정의는 `stage2d_prompt.py`의 `CATEGORY_DEFINITIONS`가 아니라 한 줄로 축약한 별도 문자열이다 — 03단계와 직접 비교할 때 주의.
