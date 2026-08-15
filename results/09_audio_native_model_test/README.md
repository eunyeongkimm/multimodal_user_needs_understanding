# 09. audio-native 모델 테스트 (Qwen-Omni)

**질문**: 텍스트 전사를 거치지 않고 오디오를 직접 먹는 모델은, 08에서 만든 같은 250콜에서 GPT 텍스트 파이프라인만큼 할 수 있는가?

## 스크립트

| 스크립트 | 역할 |
|---|---|
| `../results/09_audio_native_model_test/qwen2_5_omni_test.ipynb` | Qwen2.5-Omni-7B 스팟체크 (Colab A100, transformers) |
| `../results/09_audio_native_model_test/qwen3_omni_test_vllm.ipynb` | Qwen3-Omni-30B-A3B-Thinking-AWQ-4bit 250콜 전량 (vLLM) |

## 산출물

| 파일 | 크기 |
|---|---|
| `qwen2_5_omni_test.ipynb` | 70 KB |
| `qwen3_omni_test_vllm.ipynb` | 52 KB |

## 결론

08단계의 `audio_seg_manifest.parquet`와 초반 5발화 WAV를 그대로 입력으로 쓴다. 즉 GPT A/B와 **동일한 250콜·동일한 발화 선택**이라 직접 비교가 된다.

**Qwen3-Omni-30B (250콜 전량)**: accuracy **0.328**, 파싱 250/250 성공, 0.69초/콜.

| | 환불요청 | 서비스이용 | 불만제기 | 배송확인 | 교환반품 | 주문취소 | 구매진행 |
|---|---|---|---|---|---|---|---|
| gold | 87 | 51 | 50 | 24 | 19 | 8 | 11 |
| 예측 | 34 | 4 | 101 | 46 | 25 | 24 | 16 |

예측이 불만제기로 심하게 쏠린다(gold 50건 → 예측 101건). 반대로 서비스이용은 gold 51건인데 4건만 예측했다. 06단계 GPT 텍스트 파이프라인의 accuracy 0.596~0.604와 비교하면 격차가 크다.

**Qwen2.5-Omni-7B (5콜 스팟체크)**: gold가 전부 서비스이용인 5콜에서 0/5. 생성된 요약이 오디오 내용과 어긋나 보이는 사례가 있어(배송 언급이 없는 콜에 "배송에 대한 확인을 요청합니다") 7B 규모로는 한국어 8kHz 저음질 전화 음성을 처리하기 어려운 것으로 보인다.

> 노트북은 Colab(A100) 실행본이며 출력 셀을 보존했다. 예측 parquet(`qwen3omni_30b_predictions.parquet`)은 Google Drive에 저장돼 이 저장소에는 없다.
> Qwen3 노트북의 reasoning 출력에 고객 발화 5건이 인용돼 있다(모델이 옮겨 적은 것).
