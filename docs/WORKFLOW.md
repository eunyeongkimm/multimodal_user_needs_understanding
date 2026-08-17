# Colab ↔ 로컬 병행 작업 규약

작업이 세 곳에 흩어져 있다. 각자 맡는 역할을 고정해 두지 않으면 어느 것이 최신인지 알 수 없게 된다.

| 장소 | 맡는 것 | 맡지 않는 것 |
|---|---|---|
| **로컬** (`outputs/`) | 전처리·GPT API·통계 분석. 2.5GB 원본 캐시 | GPU 작업 |
| **Google Drive** (`MyDrive/audio_seg_2/`) | WAV 입력, Colab 결과 parquet 임시 보관 | 최종 기록 |
| **GitHub** | 코드·노트북·소용량 결과·문서 = **단일 기록처** | 원문·오디오·대용량 |

원칙 하나만 지키면 된다. **Drive는 운반 수단이고, 기록은 GitHub에 남긴다.** Colab이 만든 예측 parquet이 Drive에만 있으면 몇 주 뒤에 어느 노트북이 만든 건지 알 수 없다.

---

## 가장 조심할 것: 저장소에 쓰는 손이 둘이다

Colab의 "GitHub에 사본 저장"은 **원격에 직접 커밋한다.** 로컬이 그걸 모르는 상태에서 커밋하면 이력이 갈라진다. 실제로 이 저장소를 정리할 때 원격이 3커밋 앞서 있어 `--allow-unrelated-histories` 머지가 필요했다.

그래서 규칙은 이렇다.

```bash
# 로컬에서 무슨 작업이든 시작하기 전에
git pull --rebase
```

Colab에서 저장할 때는 **경로를 반드시 지정한다.** 저장 대화상자의 파일 경로란을 비워 두면 저장소 루트에 떨어진다.

```
results/<단계폴더>/<노트북이름>.ipynb
```

> ⚠️ macOS는 대소문자를 구분하지 않는다. `readme.md`를 만들면 `README.md`와 충돌해 로컬에서 덮어써진다. 실제로 두 번 발생했다(루트, `10_fine_tuning/`). **폴더 README는 손으로 만들지 않는다** — `repo_organize.py`가 `README.md`로 생성한다.

> ⚠️ Colab의 "GitHub에 사본 저장"은 원격에 직접 커밋하므로 **`repo_organize.py`의 원문 스캔을 건너뛴다.** 노트북 출력 셀에 고객 발화가 인용돼 있어도 아무도 막지 않는다. 공개 저장소이고 AIHub는 재배포 금지이므로, 새 노트북을 올린 뒤에는 pull해서 출력 셀을 직접 확인하고 README의 인용 목록을 갱신한다.

---

## Colab 결과를 저장소에 남기는 절차

Colab 노트북이 `MyDrive/audio_seg_2/`에 예측 parquet을 저장한 뒤:

**1. Drive에서 내려받아 `outputs/`에 둔다**

```bash
mv ~/Downloads/qwen3omni_30b_predictions.parquet outputs/
```

**2. `scripts/repo_organize.py`의 해당 단계 `files`에 이름을 추가한다**

```python
dict(
    dir="09_audio_native_model_test",
    wipe=False,
    files=["qwen3omni_30b_predictions.parquet"],   # ← 여기
    ...
)
```

`wipe=False`라서 폴더 안 노트북은 지워지지 않고, `files`만 새로 복사된다.

**3. 돌리고 커밋한다**

```bash
python3 scripts/repo_organize.py     # 원문 스캔 → 복사 → 문서 갱신
git add -A && git commit && git push
```

원문(전사 컬럼 등)이 섞여 있으면 스크립트가 **중단하고 파일명을 알려준다.** 컬럼만 빼고 싶으면 `STRIP_COLS`에 등록하면 된다.

---

## 새 단계를 추가할 때

`scripts/repo_organize.py`의 `STAGES`에 항목 하나를 추가하면 나머지는 자동이다.

```python
dict(
    dir="10_새단계이름",
    title="사람이 읽을 제목",
    question="이 단계에서 무엇을 물었나",
    scripts=[("스크립트.py", "역할")],
    files=["outputs에서_가져올_파일.parquet"],
    conclusion="무엇이 나왔나",
    wipe=False,     # Colab 노트북이 이 폴더에 산다면
),
```

돌리면 `results/10_*/README.md`, `docs/PIPELINE.md`의 단계 지도와 스크립트 역인덱스, `docs/DATA_INVENTORY.md`가 전부 갱신된다. 스크립트가 `scripts/`에 새로 생겼는데 역인덱스에서 "미분류"로 뜨면 `SCRIPT_RULES`에 접두어 한 줄만 추가한다.

---

## 올리지 않은 파일은 어떻게 확인하나

내용은 로컬에만 있다. 하지만 **무엇이 있는지는 저장소만 봐도 알 수 있다.**

[`docs/DATA_INVENTORY.md`](DATA_INVENTORY.md)에 `outputs/` 전체 2,336개 항목의 파일명·크기·행×열·컬럼명·수록 여부·제외 사유가 들어 있다. `repo_organize.py`를 돌릴 때마다 갱신된다.

그래서 다른 기기나 Colab에서도 이런 건 로컬 없이 확인된다.

- `stage2_windows_nl_v2.parquet`이 있는지, 몇 행인지, 컬럼이 뭔지
- 그게 왜 저장소에 없는지 (대용량 / 전사 포함)
- 무엇을 다시 돌리면 재생성되는지

내용 자체가 필요하면 두 가지 방법뿐이다.

1. 로컬에서 해당 스크립트를 다시 돌려 재생성 (외장하드 마운트 필요)
2. AIHub에서 원본을 받아 `scripts/build_index.py`부터 재현

---

## 로컬 유실 대비

`outputs/` 2.5GB는 어디에도 백업돼 있지 않다. 지금 유실되면 **재생성에 API 비용이 다시 든다** (라벨링·조건 실험 합계 수십 달러 규모, `PROJECT_STATE.md`의 비용 로그 참고).

우선순위를 매기면, 재생성 비용이 큰 순서는 이렇다.

| 파일 | 왜 |
|---|---|
| `gold_actual_batch1_final.parquet` | 19,847건 라벨링 결과. **이미 저장소에 있음** ✅ |
| `stage2d_gpt_predictions.parquet` | 315,160건 예측. **이미 저장소에 있음** ✅ |
| `stage2m_gpt_predictions_BD.parquet` | 118,185건. **이미 저장소에 있음** ✅ |
| `d04_dialog_index.parquet` | 162MB. API 비용은 없고 원본에서 재생성 가능 |
| `stage2_windows_nl_v2.parquet` | 38MB. 재생성 가능 |
| `*_requests.jsonl` 2,102개 | 재생성 가능, 보관 가치 낮음 |

API 비용이 든 산출물은 전부 저장소에 들어와 있다. 나머지는 원본만 있으면 계산으로 복원되므로, 추가 백업이 급한 것은 없다. 다만 외장하드 원본(AIHub)은 다시 받기 번거로우니 그쪽을 챙기는 편이 낫다.
