# 07. 상담사 응대 품질 — 층화 변수 탐색

**질문**: 상담사가 잘 대응했는지가 mismatch를 설명하는가? 설명한다면 층화 변수로 쓸 수 있는가?

## 스크립트

| 스크립트 | 역할 |
|---|---|
| `scripts/agent_judge_prompt.py` | GPT-as-judge 프롬프트. **gold_actual/카테고리 taxonomy 미포함** (누수 차단) |
| `scripts/agent_judge_run.py` | 250건 채점 + 스캐폴딩 누수 자동 검증(발견 시 중단) |
| `scripts/agent_judge_gate.py` | 층별 AUC 게이트 — 판정 기준은 gold≠불만 층 |
| `scripts/agent_strat_analysis.py` | 적절대응 층화 × 불만/mismatch × acoustic 효과 |

## 산출물

| 파일 | 크기 |
|---|---|
| `agent_judge_gate_summary.md` | 3 KB |
| `agent_judge_human_eval_slots.csv` | 1 KB |
| `agent_judge_percall.parquet` | 7 KB |
| `agent_judge_scores.parquet` | 6 KB |
| `agent_strat_percall.parquet` | 8 KB |
| `agent_strat_summary.md` | 4 KB |

## 결론

**게이트 미통과 → 워크스트림 종료.**

전체 250건에서는 적절대응이 mismatch와 상관있어 보인다(AUC 0.417, r=-0.146, p=0.021). 그런데 250건 표본은 불만을 7→50건으로 오버샘플했고 그 50건 중 45건(90%)이 mismatch다. 즉 전체 층의 신호는 '불만을 탐지한 것'일 수 있다.

불만 교란을 분리한 **gold≠불만 층(n=200)에서 최고 AUC 0.506** — 판별력 없음. 사전에 정한 기준(≥0.6 통과 / ~0.5 종료)에 따라 표본 확대와 acoustic 재추출을 하지 않고 종료했다.

층화 탐색(연관만, 인과 아님): 적절대응 '못 대응' 그룹은 불만비율 38.5% vs '잘 대응' 11.6% (p<0.001), mismatch 66.7% vs 47.7% (p=0.008). 다만 '불만이라 대응이 어려웠다'와 '대응을 못해 불만이 됐다'는 이 데이터로 구분 불가.

> `agent_judge_human_eval_slots.csv`의 40건 수동 채점란은 비어 있다. 채워 넣고 `agent_judge_gate.py`를 다시 돌리면 quadratic-weighted Kappa가 자동 산출된다.
