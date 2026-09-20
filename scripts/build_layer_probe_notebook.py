"""results/12_layer_probe/qwen25omni_layer_probe.ipynb 를 생성한다.

노트북 본문을 파이썬 문자열로 관리해 diff가 읽히게 하려는 목적. 노트북을
직접 고쳤다면 이 스크립트도 같이 고쳐야 한다(또는 이 스크립트를 버려도 된다).

코드 스타일은 results/09_audio_native_model_test/ 의 업로드용 노트북들
(qwen2_5_omni_test.ipynb, qwen2_audio_feasibility_gate.ipynb 등)을 따른다:
  - 코드 셀 첫 줄에 `# ===== 셀 N =====` 주석
  - print("="*72) 구분선
  - 판정은 정적 표가 아니라 실제로 계산해서 ✅/🔴/⚠️ 로 출력
  - GPU 확인을 맨 앞 셀에서 명시적으로 하고, 부족하면 SystemExit

실행: python scripts/build_layer_probe_notebook.py
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUT = BASE_DIR / "results" / "12_layer_probe" / "qwen25omni_layer_probe.ipynb"

CELLS = []


def _to_source(text: str) -> list:
    """nbformat 규약대로 각 줄에 개행을 남긴다.

    주의: source 를 재생 렌더러(Jupyter/GitHub/nbviewer)는 배열을
    ''.join() 으로 이어붙인다. 개행을 안 남기면 문단·헤더·리스트가
    전부 한 줄로 뭉개진다 — 이번에 실제로 터진 버그가 이거였다.
    """
    lines = text.strip("\n").split("\n")
    return [line + "\n" for line in lines]


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": _to_source(text)})


def code(text):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": _to_source(text)})


# ────────────────────────────────────────────────────────────── 0
md(r"""
# 1단계 — Qwen2.5-Omni-7B 층별 표현의 불만 분리도 (Koduru 재현)

Koduru et al. *"Heard but Not Heeded"* 의 층별 linear probe 분석을 우리 과제
(불만제기 이진 탐지)·우리 데이터(저음질 8kHz 한국어 콜센터)에 적용한다.

**이 노트북이 답하는 것은 하나다**: 어느 층 표현이 불만 vs 비불만을 가장 잘 가르는가.
후속 퓨전 실험("상위층 tap이 최종출력 tap보다 나은가")의 전제를 검증하는 예비
단계이며, 최종 성능을 내는 실험이 아니다.

## 산출물 4개

1. 정규화 층 깊이별 probe AUC 곡선 (그림 + 수치 테이블)
2. peak AUC/층 위치, 최종출력 AUC, 둘의 차이
3. 인코더/projector/디코더 경계 정의와 층 인덱싱 방식
4. 저음질 sanity check — 모델이 우리 오디오를 인코딩하긴 하는가

---

## ⚠️ 먼저 읽을 것 — 이 실험의 사전 위험 두 가지

### (1) 7B는 이 데이터에서 이미 한 번 실패했다

`results/09_audio_native_model_test/`의 기록:

| 모델 | 조건 | 결과 |
|---|---|---|
| Qwen2.5-Omni-**7B** | 5콜 스팟체크 | **0/5**. 배송 언급이 없는 콜에 "배송에 대한 확인을 요청합니다" 등 오디오와 어긋나는 요약 |
| Qwen3-Omni-**30B** | 250콜 오디오만 | accuracy **0.328** |
| Qwen3-Omni-30B | + 전사 병기 | macro-F1 0.482 (오디오만 0.450) |

즉 **"오디오만 주면 잘 못한다"가 이미 관측된 상태**다. 이건 이번 실험을 막을
이유는 아니다 — 생성 성능이 나쁜 것과 표현에 정보가 없는 것은 다른 문제이고,
Koduru의 논지가 정확히 "상위층엔 있는데 출력으로 못 간다"이기 때문이다.
오히려 그 격차를 직접 재는 게 이 실험이다.

다만 **전 층 AUC가 chance 근처로 깔릴 가능성**을 미리 염두에 둬야 하고,
그래서 아래 §4에 positive control을 넣었다.

### (2) positive control 없이는 음성 결과를 해석할 수 없다

전 층 AUC ≈ 0.5가 나왔을 때 두 해석이 갈린다.

- (가) 모델이 8kHz 한국어를 아예 표현 못 한다 → 층 비교 자체가 무의미
- (나) 모델은 잘 표현하는데 "불만"이 선형으로 없다 → 층 비교는 유효, 결론은 음성

이걸 가르려고 **같은 특징으로 성별을 예측하는 probe**를 같이 돌린다.
성별은 오디오 인코더가 정상이면 선형으로 쉽게 갈려야 하는 준언어 속성이다.

| 성별 AUC | 불만 AUC | 해석 |
|---|---|---|
| 높음 (>0.85) | 낮음 | (나) — 인코더는 정상. 불만이 선형으로 없다 |
| 낮음 (~0.5) | 낮음 | (가) — 인코더가 우리 오디오를 못 읽는다. **여기서 중단** |

---

## 데이터 세트 — 250콜로 확정

**결정됨(2026-09-20)**: 08단계가 이미 추출해 Drive에 올려 둔 **250콜**
(`model_pilot_sample.csv`, 불만제기 50건 오버샘플)로 즉시 착수한다.
별도 800콜 세트는 준비하지 않는다.

즉 `USE_LEGACY_250 = True` 그대로 두면 되고, 추가 오디오 추출은 없다.

### 다만 이 선택에는 대가가 하나 따른다

**n=250 / 불만 50건 / 특징 1280~3584차 → p ≫ n**이다. 그래서:

- AUC **절대값**은 낙관적이고 fold 분산이 크다. 성능 주장으로 쓰면 안 된다.
- AUC **상대** 비교(층 간)는 모든 층이 같은 n·p·fold를 쓰므로 공정하다.
  이번 단계가 답하려는 건 "어느 층이 제일 높은가"이므로 이걸로 충분하다.

결론이 애매하게 나오면(peak와 최종출력 차이가 fold SD 안에 묻히면) 그때
2,000콜로 키우면 된다: `python scripts/probe_audio_prep.py --calls unified2k`

### ⚠️ 먼저 할 일 — 성별 라벨 1개 파일

250콜 세트에는 성별 컬럼이 없어서, 아무것도 안 하면 **positive control이
통째로 꺼진다**. 위 (2)에서 봤듯 7B가 이미 실패한 데이터라 이 control이
이번 실행에서 가장 중요한 진단이다. 없으면 음성 결과를 해석할 수 없다.

`results/12_layer_probe/probe_labels.parquet` 가 저장소에 이미 수록돼 있다.
이 파일 하나를 Drive의 `MyDrive/audio_seg_2/` 에 올리면 셀 2가 알아서
찾아 붙인다. 없으면 경고만 내고 나머지는 그대로 돈다.
""")

# ────────────────────────────────────────────────────────────── 0
md("## 셀 0 — GPU 확인 (A100 기대)")
code(r"""
# ===== 셀 0 =====
import subprocess

try:
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                        "--format=csv,noheader"], capture_output=True, text=True)
    ok, out, err = (r.returncode == 0), r.stdout.strip(), r.stderr.strip()
except FileNotFoundError:
    ok, out, err = False, "", "nvidia-smi 없음 (GPU 미할당)"

if not ok:
    print("GPU가 없습니다:", err)
    print("런타임 → 런타임 유형 변경 → 하드웨어 가속기: GPU (A100) → 저장")
    raise SystemExit("GPU 미할당")

print("GPU:", out)
name, total, used = [x.strip() for x in out.split(",")]
GPU_NAME, GPU_MIB = name, int(total.split()[0])

# Qwen2.5-Omni-7B 는 talker·code2wav 포함 전체를 먼저 GPU에 올린 뒤
# disable_talker() 로 정리한다(셀 3). 순간 최대 사용량이 bf16 단순 7B
# 추정치(~16GB)보다 크므로 여유 있게 A100 40GB 기준으로 본다.
if GPU_MIB < 35000:
    print(f"\n⚠️ VRAM {total} — A100 40GB 미만이면 셀 3 모델 로드에서 OOM 날 수 있다.")
    print("   런타임 유형을 A100으로 바꾸는 것을 권한다.")
else:
    print(f"\n{name} / {total} — 진행 가능")
""")

# ────────────────────────────────────────────────────────────── 1
md("## 셀 1 — 환경")
code(r"""
# ===== 셀 1 =====
!pip install -q -U transformers accelerate
!pip install -q librosa soundfile scikit-learn
""")

# ────────────────────────────────────────────────────────────── 2
md(r"""
## 셀 2 — 데이터 로드

`USE_LEGACY_250 = True` 면 09단계에서 이미 Drive에 올려 둔 250콜 오디오를 그대로 쓴다.
`False` 면 `probe_audio_prep.py` 산출물(`probe_set_*/`)을 쓴다.
""")
code(r"""
# ===== 셀 2 =====
from google.colab import drive
drive.mount('/content/drive')

import os, pandas as pd, numpy as np

USE_LEGACY_250 = True   # False -> probe_audio_prep.py 산출물 사용

if USE_LEGACY_250:
    # 09단계가 올려둔 250콜 세트 (1,119발화)
    AUDIO_ROOT = '/content/drive/MyDrive/audio_seg'
    META_DIR   = '/content/drive/MyDrive/audio_seg_2'
    manifest = pd.read_parquet(os.path.join(META_DIR, 'audio_seg_manifest.parquet'))
    gold     = pd.read_parquet(os.path.join(META_DIR, 'gold_actual_batch1_final.parquet'))
    gcol = 'label' if 'label' in gold.columns else 'gold_actual'
    labels = (gold[['call_id', gcol]].rename(columns={gcol: 'gold_actual'})
              .merge(manifest[['call_id']].drop_duplicates(), on='call_id'))
    labels['is_complaint'] = (labels['gold_actual'] == '불만제기').astype(int)
    # 성별 라벨(positive control). results/12_layer_probe/probe_labels.parquet 를
    # audio_seg_2/ 에 올려두면 여기서 붙는다. 없으면 control만 꺼진다.
    _gpath = os.path.join(META_DIR, 'probe_labels.parquet')
    if os.path.exists(_gpath):
        _g = pd.read_parquet(_gpath)[['call_id', 'gender']]
        labels = labels.merge(_g, on='call_id', how='left')
        print(f"성별 라벨 결합: {labels['gender'].notna().sum()}/{len(labels)}콜")
    else:
        labels['gender'] = None
        print("!! 성별 라벨 없음 -> positive control 비활성화.\n"
              "   results/12_layer_probe/probe_labels.parquet 을\n"
              "   MyDrive/audio_seg_2/ 에 올리면 활성화된다.")
    # wav 경로: Drive에는 {call_id}/c{i}.wav 로 올라가 있다
    manifest['abs_path'] = manifest.apply(
        lambda r: os.path.join(AUDIO_ROOT, r.call_id, os.path.basename(str(r.wav_path)))
        if pd.notna(r.wav_path) else None, axis=1)
else:
    PROBE_ROOT = '/content/drive/MyDrive/probe_set'
    AUDIO_ROOT = os.path.join(PROBE_ROOT, 'audio')
    manifest = pd.read_parquet(os.path.join(PROBE_ROOT, 'probe_manifest.parquet'))
    labels   = pd.read_parquet(os.path.join(PROBE_ROOT, 'probe_labels.parquet'))
    manifest['abs_path'] = manifest['wav_path'].apply(
        lambda w: os.path.join(PROBE_ROOT, w) if pd.notna(w) else None)

manifest = manifest[manifest['abs_path'].notna()].copy()
manifest = manifest[manifest['abs_path'].apply(os.path.exists)]
manifest = manifest.sort_values(['call_id', 'utt_idx'])

# 라벨이 있는 콜로만 맞춘다
CALL_IDS = sorted(set(manifest.call_id) & set(labels.call_id))
manifest = manifest[manifest.call_id.isin(CALL_IDS)]
labels   = labels[labels.call_id.isin(CALL_IDS)].set_index('call_id').loc[CALL_IDS].reset_index()

print(f"콜 {len(CALL_IDS):,} / 발화 {len(manifest):,}")
print("콜당 발화 수:", manifest.groupby('call_id').size().value_counts().sort_index().to_dict())
print()
print("gold 분포:"); print(labels['gold_actual'].value_counts())
n_c = int(labels.is_complaint.sum())
print(f"\n이진: 불만 {n_c} ({n_c/len(labels)*100:.1f}%) / 비불만 {len(labels)-n_c}")
""")

# ────────────────────────────────────────────────────────────── 3
md(r"""
## 셀 3 — 모델 로드

probe에는 **Thinker만** 필요하다(talker/token2wav는 음성 합성용). `disable_talker()`로
떼면 A100 40GB에서 bf16 그대로 올라간다 — 09단계처럼 4bit 양자화할 이유가 없고,
양자화는 hidden state를 왜곡하므로 probe에는 **쓰면 안 된다**.
""")
code(r"""
# ===== 셀 3 =====
import os, torch, gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect(); torch.cuda.empty_cache()

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

MODEL_ID = "Qwen/Qwen2.5-Omni-7B"

def _vram():
    f, t = torch.cuda.mem_get_info(); return f / 1024**3, t / 1024**3

_f, _t = _vram(); print(f"로드 전 GPU 여유 {_f:.2f} / 전체 {_t:.2f} GiB")

load_error = None
try:
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
    )
    model.disable_talker()          # probe에 불필요. 메모리 절약
    model.eval()
except Exception as e:
    load_error = e
    raise

thinker = model.thinker
_f, _t = _vram(); print(f"✅ 모델 로드 완료 | GPU 여유 {_f:.2f} / 전체 {_t:.2f} GiB")
""")

# ────────────────────────────────────────────────────────────── 4
md(r"""
## 셀 4 — 구조 확인 + 인코더/디코더 경계 정의  ← **산출물 3**

여기서 층 인덱싱을 확정한다. 경계가 틀리면 "상위층 vs 최종출력" 비교 자체가
무의미해지므로, 하드코딩하지 않고 **런타임에 모듈을 직접 읽어** 정의한다.

`transformers`의 `Qwen2_5OmniAudioEncoder` 구현 기준 오디오 경로는 다음과 같다.

```
mel(128) ─ conv1 ─ conv2(stride2) ─ +pos
   └─ audio_tower.layers[0 .. L-1]        ← 오디오 인코더 본체 (d_model=1280)
        └─ ln_post ─ audio_tower.proj      ← projector (1280 → 3584, LM 공간)
             └─ thinker.model.layers[0 .. M-1]   ← 언어모델 디코더 (3584)
                  └─ lm_head → 토큰
```

**정규화 깊이**는 `[인코더 L층] + [projector 1] + [디코더 M층]` 을 한 축에 이어 붙이고
`depth = i / (L+1+M-1)` 로 잡는다. 즉 `depth=0` = 첫 인코더층 출력,
`depth=1` = 마지막 디코더층 출력(= lm_head 직전 = **최종 출력 표현**).
""")
code(r"""
# ===== 셀 4 =====
acfg = thinker.config.audio_config
tcfg = thinker.config.text_config

L_ENC = len(thinker.audio_tower.layers)
M_DEC = len(thinker.model.layers)

print("=" * 72)
print("오디오 인코더 (audio_tower)")
print("=" * 72)
print(f"  encoder_layers : {L_ENC}   (config.encoder_layers={acfg.encoder_layers})")
print(f"  d_model        : {acfg.d_model}")
print(f"  num_mel_bins   : {acfg.num_mel_bins}")
print(f"  n_window       : {acfg.n_window}")
print(f"  output_dim     : {acfg.output_dim}   <- projector 출력 = LM hidden")
print(f"  모듈           : conv1, conv2, layers[{L_ENC}], ln_post, proj")

print()
print("=" * 72)
print("언어모델 디코더 (thinker.model)")
print("=" * 72)
print(f"  num_hidden_layers : {M_DEC}")
print(f"  hidden_size       : {tcfg.hidden_size}")

print()
print("=" * 72)
print("오디오 입력 규격")
print("=" * 72)
SR_REQUIRED = processor.feature_extractor.sampling_rate
print(f"  feature_extractor.sampling_rate = {SR_REQUIRED}")
print(f"  우리 원음 = 8000 Hz  ->  {'리샘플 필요' if SR_REQUIRED != 8000 else '리샘플 불필요'}")

# audio 토큰 id (디코더에서 오디오 위치만 pooling 하는 데 필요)
AUDIO_TOKEN_ID = getattr(thinker.config, 'audio_token_id',
                  getattr(thinker.config, 'audio_token_index', None))
print(f"  audio_token_id = {AUDIO_TOKEN_ID}")

# ── tap 지점 정의: (이름, 모듈, 종류) ─────────────────────────────
TAPS = []
for i, m in enumerate(thinker.audio_tower.layers):
    TAPS.append((f"enc{i:02d}", m, "enc"))
TAPS.append(("proj", thinker.audio_tower.proj, "proj"))
for i, m in enumerate(thinker.model.layers):
    TAPS.append((f"dec{i:02d}", m, "dec"))

TAP_NAMES = [t[0] for t in TAPS]
N_TAP = len(TAPS)
DEPTH = np.array([i / (N_TAP - 1) for i in range(N_TAP)])
BOUND_ENC_END = (L_ENC - 1) / (N_TAP - 1)      # 마지막 인코더층
BOUND_PROJ    = L_ENC / (N_TAP - 1)            # projector

print()
print("=" * 72)
print("층 인덱싱")
print("=" * 72)
print(f"  tap 총 {N_TAP}개 = 인코더 {L_ENC} + projector 1 + 디코더 {M_DEC}")
print(f"  depth 0.000 = enc00 (첫 인코더층 출력)")
print(f"  depth {BOUND_ENC_END:.3f} = enc{L_ENC-1:02d} (마지막 인코더층)")
print(f"  depth {BOUND_PROJ:.3f} = proj (projector 출력, 오디오 인코더 끝)")
print(f"  depth 1.000 = dec{M_DEC-1:02d} (마지막 디코더층 = 최종 출력 표현)")
""")

# ────────────────────────────────────────────────────────────── 5
md(r"""
## 셀 5 — 오디오 로더 (8kHz → 16kHz)  ← **산출물 4의 일부**

`processor.feature_extractor.sampling_rate == 16000` 이다. 우리 원음은 8kHz이므로
**반드시 리샘플해야 한다.** 8kHz 파형을 그대로 넘기면서 sr=16000이라고 알리면
mel 필터뱅크가 2배로 어긋나 피치·시간축이 통째로 뒤틀린다 — 조용히 틀린 결과가 나온다.

리샘플이 정보를 **더해 주지는 않는다**. 원음이 4kHz에서 잘려 있다는 사실은 그대로이고
(나이퀴스트), 업샘플은 단지 모델 입력 규격을 맞추는 것뿐이다. 이 점은 결과 해석에서
"저음질이라 정보가 없다"와 "모델이 못 읽는다"를 구분할 때 중요하다.
""")
code(r"""
# ===== 셀 5 =====
import librosa, soundfile as sf

def load_utt(path):
    '''원음(8k) -> 모델 규격(16k) 리샘플. mono.'''
    y, sr = librosa.load(path, sr=SR_REQUIRED, mono=True)   # librosa가 리샘플까지 수행
    return y

# ── 리샘플 검증: 원음 sr과 로드 후 길이가 2배가 되는지 직접 확인
_p = manifest.abs_path.iloc[0]
_y_native, _sr_native = librosa.load(_p, sr=None, mono=True)
_y_16k = load_utt(_p)
print(f"원음      : sr={_sr_native}, samples={len(_y_native):,}, {len(_y_native)/_sr_native:.3f}s")
print(f"로드 후   : sr={SR_REQUIRED}, samples={len(_y_16k):,}, {len(_y_16k)/SR_REQUIRED:.3f}s")
print(f"길이(초) 일치: {abs(len(_y_native)/_sr_native - len(_y_16k)/SR_REQUIRED) < 1e-3}")

_srs = sorted(set(librosa.get_samplerate(p) for p in manifest.abs_path.sample(
    min(50, len(manifest)), random_state=42)))
print(f"\n표본 50발화의 원음 sr: {_srs}  (8000 단일이어야 정상)")

def call_audios(call_id):
    '''콜의 고객 발화 WAV들을 utt_idx 순서로 16kHz 파형 리스트로.'''
    rows = manifest[manifest.call_id == call_id].sort_values('utt_idx')
    return [load_utt(p) for p in rows.abs_path]
""")

# ────────────────────────────────────────────────────────────── 6
md(r"""
## 셀 6 — 저음질 sanity check ①: 모델이 우리 오디오를 알아듣는가  ← **산출물 4**

몇 개 콜을 모델에게 그대로 전사시켜 본다. 표현 probe와는 독립적인 확인이며,
여기서 완전히 엉뚱한 말이 나오면 층별 AUC를 해석하기 전에 멈춰야 한다.

> 출력에 고객 발화가 인용된다. 09단계에서와 같은 성격의 노출이며,
> 커밋 전 검토 대상이다.
""")
code(r"""
# ===== 셀 6 =====
SANITY_N = 5

SYS = "당신은 한국어 음성을 정확히 받아적는 전사기입니다."
ASR_PROMPT = ("아래 음성들은 한국어 콜센터 고객의 발화입니다. "
              "해석하거나 요약하지 말고, 들리는 말을 순서대로 그대로 받아적으세요.")

@torch.no_grad()
def transcribe(call_id, max_new_tokens=200):
    auds = call_audios(call_id)
    content = [{"type": "audio", "audio": a} for a in auds]
    content.append({"type": "text", "text": ASR_PROMPT})
    conv = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
            {"role": "user", "content": content}]
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=auds, sampling_rate=SR_REQUIRED,
                       return_tensors="pt", padding=True).to(model.device)
    out = thinker.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[0][inputs['input_ids'].shape[1]:]
    return processor.tokenizer.decode(gen, skip_special_tokens=True).strip()

for cid in CALL_IDS[:SANITY_N]:
    n_utt = (manifest.call_id == cid).sum()
    print(f"── {cid}  ({n_utt}발화, gold={labels.set_index('call_id').gold_actual[cid]})")
    print(transcribe(cid)[:400])
    print()
""")

# ────────────────────────────────────────────────────────────── 7
md(r"""
## 셀 7 — 층별 hidden 추출 (forward hook)

### pooling 규약 (바꾸면 결과가 달라지므로 고정)

- **인코더층 / projector**: 해당 콜의 **모든 오디오 프레임에 대한 mean**.
  Qwen2.5-Omni 오디오 인코더는 varlen 어텐션이라 hidden이 `(T_packed, d)` 로 **평탄하게
  패킹**돼 있다(`cu_seqlens`로 구간을 나눔). **콜당 forward를 1회만** 돌리면 패킹된
  행 전부가 그 콜의 유효 프레임이므로, `mean(dim=0)`이 곧 콜 단위 mean pooling이다.
  → 발화 길이로 가중된 평균이다(발화별 평균의 평균이 아님).
  **배치를 키우면 서로 다른 콜의 프레임이 같은 텐서에 섞여 조용히 오염된다. batch=1 고정.**
- **디코더층**: `(1, S, d)` 중 **오디오 토큰 위치만** 골라 mean.
  텍스트 프롬프트 토큰이 섞이면 층 비교가 프롬프트를 재는 꼴이 된다.

### hook을 쓰는 이유
`output_hidden_states=True` 도 이 버전에선 동작하지만(`_can_record_outputs`에
`Qwen2_5OmniAudioEncoderLayer` 등록됨), hook은 버전 변화에 덜 민감하고
projector처럼 중간 모듈 하나를 집는 데 더 직접적이다.
""")
code(r"""
# ===== 셀 7 =====
from collections import OrderedDict

_POOL = {}
_AUDIO_POS = None      # 디코더 pooling용 오디오 토큰 위치

def _flat_mean(h):
    '''(T,d) 든 (1,T,d) 든 마지막 축만 남기고 mean.'''
    return h.reshape(-1, h.shape[-1]).float().mean(dim=0)

def make_hook(name, kind):
    def hook(module, inp, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        h = h.detach()
        if kind in ("enc", "proj"):
            v = _flat_mean(h)
        else:                                   # dec
            if _AUDIO_POS is None or _AUDIO_POS.numel() == 0:
                return
            v = h[0][_AUDIO_POS].float().mean(dim=0)
        _POOL[name] = v.to(torch.float32).cpu().numpy()
    return hook

HANDLES = [m.register_forward_hook(make_hook(n, k)) for n, m, k in TAPS]
print(f"hook {len(HANDLES)}개 등록")

@torch.no_grad()
def extract_call(call_id):
    '''콜 1건 -> {tap_name: vector}. batch=1 고정.'''
    global _AUDIO_POS
    auds = call_audios(call_id)
    if not auds:
        return None
    content = [{"type": "audio", "audio": a} for a in auds]
    content.append({"type": "text", "text": "이 고객의 발화입니다."})
    conv = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(conv, add_generation_prompt=False, tokenize=False)
    inputs = processor(text=text, audio=auds, sampling_rate=SR_REQUIRED,
                       return_tensors="pt", padding=True).to(model.device)

    ids = inputs["input_ids"][0]
    _AUDIO_POS = (ids == AUDIO_TOKEN_ID).nonzero(as_tuple=True)[0]

    _POOL.clear()
    thinker(**inputs)                      # generate 아님 — forward 1회면 충분
    if len(_POOL) != N_TAP:
        raise RuntimeError(f"{call_id}: tap {len(_POOL)}/{N_TAP}만 채워짐")
    return dict(_POOL), int(_AUDIO_POS.numel())

# ── 1건으로 형태 확인
_f, _n_audio_tok = extract_call(CALL_IDS[0])
print(f"\n{CALL_IDS[0]}: 오디오 토큰 {_n_audio_tok}개")
for n in ["enc00", f"enc{L_ENC-1:02d}", "proj", "dec00", f"dec{M_DEC-1:02d}"]:
    print(f"  {n:8s} shape={_f[n].shape}  |v|={np.linalg.norm(_f[n]):.2f}")
""")

# ────────────────────────────────────────────────────────────── 8
md("## 셀 8 — 전 콜 추출")
code(r"""
# ===== 셀 8 =====
import time

FEATS = {n: [] for n in TAP_NAMES}
ok_calls, n_audio_tokens, failed = [], [], []

t0 = time.time()
for i, cid in enumerate(CALL_IDS):
    try:
        f, n_tok = extract_call(cid)
        for n in TAP_NAMES:
            FEATS[n].append(f[n])
        ok_calls.append(cid); n_audio_tokens.append(n_tok)
    except Exception as e:
        failed.append((cid, repr(e)[:120]))
    if (i + 1) % 25 == 0:
        el = time.time() - t0
        print(f"{i+1}/{len(CALL_IDS)}  {el:.0f}s  ({el/(i+1):.2f}s/call)  실패 {len(failed)}")

for n in TAP_NAMES:
    FEATS[n] = np.stack(FEATS[n]).astype(np.float32)

print(f"\n완료: {len(ok_calls):,}콜, 실패 {len(failed)}건, {time.time()-t0:.0f}s")
if failed:
    print("실패 예시:", failed[:5])
print(f"오디오 토큰 수/콜: mean={np.mean(n_audio_tokens):.0f}, "
      f"min={min(n_audio_tokens)}, max={max(n_audio_tokens)}")
print(f"enc00 {FEATS['enc00'].shape} / proj {FEATS['proj'].shape} / "
      f"dec00 {FEATS['dec00'].shape}")

y_df = labels.set_index('call_id').loc[ok_calls].reset_index()
y_complaint = y_df['is_complaint'].values
print(f"\n라벨: 불만 {y_complaint.sum()} / 비불만 {len(y_complaint)-y_complaint.sum()}")

for h in HANDLES:
    h.remove()
print("hook 해제")
""")

# ────────────────────────────────────────────────────────────── 9
md(r"""
## 셀 9 — 층별 linear probe

- **probe**: 로지스틱 회귀(L2)만. MLP 등 비선형 금지 — 정보 "존재" 여부는 선형 분리로 본다.
- **입력 단위 = 콜**. 한 콜이 한 행이므로 *같은 콜의 발화가 train/test에 동시에 들어가는
  누출이 구조적으로 불가능*하다(발화 단위로 풀었다면 GroupKFold가 필요했을 부분).
- **검증**: `StratifiedKFold(5, shuffle, seed=42)`. 지표는 **ROC-AUC** (불균형 때문에
  accuracy 금지). fold별 평균 ± 표준편차.
- **층마다 독립 학습**. 파라미터 공유 없음.
- **차원 통제**: 인코더는 1280차, projector·디코더는 3584차다. 차원이 다르면 경계를
  넘는 비교에 과적합 여력 차이가 섞인다. 그래서 fold 내부에서 PCA-128로 맞춘 변형을
  같이 돌려 경계 비교가 차원 때문이 아님을 확인한다.
""")
code(r"""
# ===== 셀 9 =====
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED, N_FOLD, N_PCA = 42, 5, 128

def build_pipe(X, y, use_pca):
    steps = [StandardScaler()]
    if use_pca:
        # PCA는 fold 내부에서 fit된다 -> 누출 없음. n_components는 train fold 크기 이하로
        k = min(N_PCA, X.shape[1], int(len(y) * (1 - 1 / N_FOLD)) - 1)
        steps.append(PCA(n_components=k, random_state=SEED))
    steps.append(LogisticRegression(max_iter=5000, C=1.0))
    return make_pipeline(*steps)

def probe_auc(X, y, use_pca=False):
    proto = build_pipe(X, y, use_pca)
    skf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)
    aucs = []
    for tr, te in skf.split(X, y):
        pipe = clone(proto)                    # fold마다 완전히 새 추정기
        pipe.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], pipe.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))

def run_all(y, tag, use_pca=False):
    rows = []
    for i, n in enumerate(TAP_NAMES):
        m, s = probe_auc(FEATS[n], y, use_pca=use_pca)
        rows.append({"tap": n, "idx": i, "depth": DEPTH[i],
                     "stage": "encoder" if n.startswith("enc")
                              else ("projector" if n == "proj" else "decoder"),
                     "dim": FEATS[n].shape[1], "auc": m, "std": s,
                     "target": tag, "pca": use_pca})
    return pd.DataFrame(rows)

res = [run_all(y_complaint, "complaint", False),
       run_all(y_complaint, "complaint", True)]
print("불만 probe 완료")
""")

# ────────────────────────────────────────────────────────────── 10
md(r"""
## 셀 10 — 🔴 저음질 sanity check ②: positive control  ← **산출물 4**

같은 특징으로 **성별**을 예측한다. 인코더가 우리 8kHz 한국어를 제대로 표현한다면
성별은 선형으로 쉽게 갈려야 한다(F0 차이가 남 136Hz / 여 225Hz로 크다).

이 셀은 **게이트다** — 여기서 실패하면 이후 층별 곡선을 해석할 근거가 없다.

- 성별 AUC 높음 + 불만 AUC 낮음 → 인코더는 정상. **불만이 선형으로 없는 것**
- 둘 다 낮음 → 인코더가 우리 오디오를 못 읽는 것. **층 비교 중단**

250콜 레거시 세트에는 성별 컬럼이 없다. `probe_labels.parquet`를 올리면 들어 있다.
없으면 이 셀은 건너뛰고, 대신 셀 6의 전사 결과로만 판단한다(근거가 약해진다).
""")
code(r"""
# ===== 셀 10 : positive control 게이트 =====
has_gender = ('gender' in y_df.columns) and y_df['gender'].notna().any()

C_GENDER = None   # 게이트 판정값. 셀 13에서 사용

if has_gender:
    g = y_df['gender'].astype(str)
    top2 = g.value_counts().index[:2]
    mask = g.isin(top2).values
    y_gender = (g[mask] == top2[0]).astype(int).values
    print(f"성별 control: {top2[0]}={y_gender.sum()} / {top2[1]}={len(y_gender)-y_gender.sum()}")

    rows = []
    for i, n in enumerate(TAP_NAMES):
        m, s = probe_auc(FEATS[n][mask], y_gender, use_pca=False)
        rows.append({"tap": n, "idx": i, "depth": DEPTH[i],
                     "stage": "encoder" if n.startswith("enc")
                              else ("projector" if n == "proj" else "decoder"),
                     "dim": FEATS[n].shape[1], "auc": m, "std": s,
                     "target": "gender(control)", "pca": False})
    res.append(pd.DataFrame(rows))
    gmax = max(r['auc'] for r in rows)
    C_GENDER = gmax > 0.85

    print(f"\n성별 peak AUC = {gmax:.3f}")
    print("✅ 통과 — 인코더 정상. 불만 AUC가 낮다면 '불만이 선형으로 없다'는 해석"
          if C_GENDER else
          "🔴 실패 — 성별조차 안 갈린다. 모델이 이 오디오를 표현 못 하는 쪽을 의심할 것")
else:
    print("⚠️ 성별 라벨 없음 (레거시 250 세트). positive control 생략 — "
          "probe_labels.parquet 을 올리면 활성화된다.")

RES = pd.concat(res, ignore_index=True)
""")

# ────────────────────────────────────────────────────────────── 11
md("## 셀 11 — 곡선 + 수치 테이블  ← **산출물 1·2**")
code(r"""
# ===== 셀 11 =====
# Colab 기본 matplotlib에는 한글 폰트가 없어 축·제목이 전부 두부(□)로 렌더된다.
!apt-get -qq install -y fonts-nanum > /dev/null 2>&1

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

_KFONT = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if os.path.exists(_KFONT):
    fm.fontManager.addfont(_KFONT)
    mpl.rcParams['font.family'] = fm.FontProperties(fname=_KFONT).get_name()
else:
    print('경고: 한글 폰트 설치 실패 - 라벨이 깨져 보일 수 있다')

mpl.rcParams.update({"font.size": 11, "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False})

# 검증된 3색 팔레트 (light) + 마커로 2차 인코딩
SERIES = [
    ("complaint", False, "#2a78d6", "o", "불만 (raw)"),
    ("complaint", True,  "#eb6834", "s", f"불만 (PCA-{N_PCA}, 차원 통제)"),
    ("gender(control)", False, "#1baf7a", "^", "성별 (positive control)"),
]

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.axhline(0.5, color="#8a8a86", lw=1, ls=":", zorder=1)
ax.text(1.004, 0.5, "chance", color="#52514e", fontsize=9, va="center")

ax.axvspan(0, BOUND_ENC_END, color="#2a78d6", alpha=0.045, zorder=0)
ax.axvline(BOUND_PROJ, color="#52514e", lw=1, ls="--", zorder=1)
ax.text(BOUND_PROJ, 1.008, " projector", color="#52514e", fontsize=9,
        ha="left", va="bottom", transform=ax.get_xaxis_transform())
ax.text(BOUND_ENC_END/2, 1.008, "오디오 인코더", color="#52514e", fontsize=9,
        ha="center", va="bottom", transform=ax.get_xaxis_transform())
ax.text((BOUND_PROJ+1)/2, 1.008, "언어모델 디코더", color="#52514e", fontsize=9,
        ha="center", va="bottom", transform=ax.get_xaxis_transform())

for tgt, pca, color, marker, lbl in SERIES:
    d = RES[(RES.target == tgt) & (RES.pca == pca)].sort_values("idx")
    if d.empty:
        continue
    ax.plot(d.depth, d.auc, color=color, lw=2, marker=marker, ms=4.5,
            mec="white", mew=0.7, label=lbl, zorder=3)
    ax.fill_between(d.depth, d.auc - d["std"], d.auc + d["std"],
                    color=color, alpha=0.13, lw=0, zorder=2)

ax.set_xlabel("정규화 층 깊이  (0 = 첫 인코더층, 1 = 최종 출력 표현)")
ax.set_ylabel("probe ROC-AUC  (5-fold 평균 ± SD)")
ax.set_title(f"Qwen2.5-Omni-7B 층별 불만 분리도 · n={len(ok_calls)}콜 "
             f"(불만 {int(y_complaint.sum())}건)", pad=26)
ax.set_xlim(-0.01, 1.01)
# 범례는 축 바깥 아래로. AUC가 chance 근처로 깔리면 축 안쪽 범례가 데이터를 가린다.
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
plt.tight_layout()
plt.savefig("layer_probe_curve.png", dpi=160, bbox_inches="tight")
plt.show()
""")

code(r"""
# ===== 셀 11b : 수치 테이블 + 판정값 =====
main = RES[(RES.target == "complaint") & (~RES.pca)].sort_values("idx")

peak = main.loc[main.auc.idxmax()]
final = main.iloc[-1]
enc = main[main.stage == "encoder"]
enc_peak = enc.loc[enc.auc.idxmax()]
proj_row = main[main.stage == "projector"].iloc[0]

print("=" * 72)
print("산출물 2 — peak / 최종출력")
print("=" * 72)
print(f"peak AUC        : {peak.auc:.4f} ± {peak['std']:.4f}   "
      f"@ {peak.tap} (depth {peak.depth:.3f}, {peak.stage})")
print(f"인코더 내 peak  : {enc_peak.auc:.4f} ± {enc_peak['std']:.4f}   "
      f"@ {enc_peak.tap} (depth {enc_peak.depth:.3f}, 인코더 {int(enc_peak.idx)+1}/{L_ENC}층)")
print(f"projector       : {proj_row.auc:.4f} ± {proj_row['std']:.4f}")
print(f"최종 출력층     : {final.auc:.4f} ± {final['std']:.4f}   @ {final.tap} (depth 1.000)")
print()
print(f"peak - 최종출력        : {peak.auc - final.auc:+.4f}")
print(f"인코더peak - 최종출력  : {enc_peak.auc - final.auc:+.4f}")
print("=" * 72)

pd.set_option("display.width", 140)
show = RES.assign(auc=lambda d: d.auc.round(4), std=lambda d: d["std"].round(4),
                  depth=lambda d: d.depth.round(3))
display(show[(show.target == "complaint") & (~show.pca)]
        [["idx", "tap", "stage", "depth", "dim", "auc", "std"]]
        .reset_index(drop=True))
""")

# ────────────────────────────────────────────────────────────── 12
md("## 셀 12 — 저장")
code(r"""
# ===== 셀 12 =====
import json, shutil
OUTDIR = '/content/drive/MyDrive/layer_probe_out'
os.makedirs(OUTDIR, exist_ok=True)

RES.to_parquet(f'{OUTDIR}/layer_probe_auc.parquet', index=False)
RES.to_csv(f'{OUTDIR}/layer_probe_auc.csv', index=False, encoding='utf-8-sig')
np.savez_compressed(f'{OUTDIR}/layer_probe_features.npz',
                    call_ids=np.array(ok_calls), y_complaint=y_complaint,
                    **{n: FEATS[n].astype(np.float16) for n in TAP_NAMES})
shutil.copy('layer_probe_curve.png', f'{OUTDIR}/layer_probe_curve.png')

meta = {
    "model": MODEL_ID, "dtype": "bfloat16", "n_calls": len(ok_calls),
    "n_complaint": int(y_complaint.sum()),
    "L_enc": L_ENC, "M_dec": M_DEC,
    "d_enc": int(acfg.d_model), "d_lm": int(tcfg.hidden_size),
    "sampling_rate_required": int(SR_REQUIRED), "native_sampling_rate": 8000,
    "pooling": "mean over all audio frames (enc/proj); mean over audio-token positions (dec)",
    "cv": f"StratifiedKFold({N_FOLD}, shuffle, seed={SEED}), call-level rows",
    "probe": "LogisticRegression(L2, C=1.0) on StandardScaler features",
}
with open(f'{OUTDIR}/layer_probe_meta.json', 'w') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("저장:", OUTDIR); print(json.dumps(meta, ensure_ascii=False, indent=2))
""")

# ────────────────────────────────────────────────────────────── 13
md(r"""
## 셀 13 — 최종 판정

셀 10(positive control), 셀 11(peak/최종출력)의 실측값으로 **직접 계산**한다.
정적 표를 눈으로 대조하지 않고, 아래 셀이 판정을 내린다. 다만 최종 결론은
사용자가 내리는 것이고 이건 근거 요약이다.
""")
code(r"""
# ===== 셀 13 : 최종 판정 (자립형) =====
# 앞 셀을 건너뛰어도 동작하도록 필요한 값을 globals()에서 직접 찾고,
# 없으면 "미실행"으로 표시한다.
def _g(name, default=None):
    return globals().get(name, default)

def _mark(v):
    return "✅ 통과" if v is True else ("🔴 실패" if v is False else "⚠️ 미실행")

print("=" * 72)
print("Qwen2.5-Omni-7B 층별 probe — 최종 판정")
print("=" * 72)

C1 = _g("C_GENDER")   # 셀 10: positive control
print(f"\n게이트 (셀 10 positive control)  {_mark(C1)}")
if "y_gender" in globals():
    _gmax = RES[RES.target == "gender(control)"].auc.max()
    print(f"   성별 peak AUC = {_gmax:.3f}  (기준 > 0.85)")

if C1 is False:
    print("\n" + "=" * 72)
    print("🔴 여기서 중단 — 모델이 8kHz 한국어를 표현하지 못한다.")
    print("   층 비교는 계속할 수 있지만 해석 근거가 없다.")
    print("=" * 72)
elif C1 is None:
    print("\n⚠️ positive control 미실행 — 판정 근거가 전사 스팟체크(셀 6)뿐이라 약하다.")

if "RES" in globals():
    main = RES[(RES.target == "complaint") & (~RES.pca)].sort_values("idx")
    peak = main.loc[main.auc.idxmax()]
    final = main.iloc[-1]
    enc = main[main.stage == "encoder"]
    enc_peak = enc.loc[enc.auc.idxmax()]
    gap = enc_peak.auc - final.auc

    rel = (enc_peak.idx + 1) / L_ENC
    # bool() 캐스팅 필수: pandas/numpy 비교 결과는 numpy.bool_ 이라
    # `is True`/`is False` identity 비교가 항상 False로 실패한다.
    C2 = bool(gap > enc_peak["std"])        # peak가 최종출력보다 fold SD 이상 높은가
    C3 = bool(rel >= 0.6)                   # peak가 인코더 상위 40% 안에 있는가

    print(f"\n인코더 내 peak    {enc_peak.auc:.4f} ± {enc_peak['std']:.4f}  "
          f"@ 인코더 {int(enc_peak.idx)+1}/{L_ENC}층 (상위 {(1-rel)*100:.0f}%)")
    print(f"최종 출력층       {final.auc:.4f} ± {final['std']:.4f}")
    print(f"peak - 최종출력   {gap:+.4f}")
    print()
    print(f"기준 1 peak가 최종출력보다 fold SD 이상 높은가   {_mark(C2)}")
    print(f"기준 2 peak가 인코더 상위 40% 안에 있는가        {_mark(C3)}")

    print()
    print("=" * 72)
    if C1 is False:
        print("🔴 판정 보류 — 게이트 실패로 위 수치를 해석할 근거가 없다.")
        print("   (셀 10에서 이미 중단을 권고했다. 아래는 참고용 원시 수치일 뿐이다)")
    elif C2 and C3:
        print("🟢 GO — 소실 전 층에 정보가 더 있다. 퓨전 실험 전제 확보")
    elif C2 and not C3:
        print("🟡 부분 GO — peak가 최종출력보다 높지만 인코더 하위층이다.")
        print("   Koduru 패턴(late encoder peak)과 다르다 — 저수준 음향을")
        print("   재고 있을 가능성. 퓨전 전에 원인 확인 필요.")
    else:
        print("🔴 NO-GO — peak와 최종출력 차이가 fold 분산 안에 묻힌다.")
        print("   상위층 tap 가설의 토대가 약하다. 설계 재고 필요.")
    print("=" * 72)

    raw = main
    pca = RES[(RES.target == "complaint") & (RES.pca)].sort_values("idx")
    if not pca.empty:
        shape_diff = np.abs(raw.auc.values - pca.auc.values).mean()
        print(f"\nraw vs PCA-{N_PCA} 평균 절대 차이: {shape_diff:.4f}  "
              f"({'차원 통제 후에도 패턴 유지' if shape_diff < 0.05 else '⚠️ 차원 차이가 결과에 영향 — PCA 쪽을 신뢰할 것'})")
else:
    print("\n⚠️ RES 없음 — 셀 9(probe)를 먼저 돌려야 한다.")
""")

# ────────────────────────────────────────────────────────────── 14
md(r"""
## 읽는 법 — 과신하면 안 되는 지점

셀 13이 판정을 계산해 주지만, 그 숫자 자체의 한계는 사람이 알아야 한다.

- **n=250이면 p ≫ n이다.** 불만 50건 / 특징 1280~3584차. AUC 절대값은 낙관적이고
  fold 분산이 크다. 층 간 *상대* 비교는 모든 층이 같은 n·p·fold를 쓰므로 공정하지만,
  "AUC 0.78이 나왔다"를 성능 주장으로 쓰면 안 된다. 2,000콜 세트로 재실행하면 안정된다.
- **pooling을 바꾸면 결과가 바뀐다.** 여기서는 발화 길이 가중 mean으로 고정했다.
- **8kHz→16kHz 업샘플은 정보를 더하지 않는다.** 모델 입력 규격을 맞춘 것뿐이다.
- 이 실험은 **선형** 분리만 본다. 비선형으로는 존재하는 정보를 놓칠 수 있다(Koduru도 동일).
""")

NB = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "A100"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(NB, f, ensure_ascii=False, indent=1)
print(f"저장: {OUT}  (셀 {len(CELLS)}개)")
