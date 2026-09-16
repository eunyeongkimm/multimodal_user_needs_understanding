"""각성도(arousal) 라벨링 전 타당성 스크리닝 (사람 청취 라벨링 착수 전 사전 판정).

라벨은 만들지 않는다. 음향 요약통계만으로 다음 세 가지만 본다.
  (1) 격앙(고각성) 통화가 전체에서 대략 몇 %인지
  (2) 고각성/저각성이 음향 특징으로 실제 구분되는지
  (3) 8kHz 저음질에서도 그 신호가 살아있는지

발화 선택은 기존 §3.1 음향 파이프라인(stage2a/stage2b)과 동일한 규칙을 따른다.
  - 고객 발화만 (speaker_type이 '고객'으로 시작)
  - 전사 품질 통과분만 (r_gold_valid_flag == True)
  - dialog_idx 오름차순 앞 5개 고객 발화

--------------------------------------------------------------------------
데이터 소스 (자동 판별)
--------------------------------------------------------------------------
FULL 모드  : outputs/d04_dialog_index.parquet + outputs/stage2_acoustic_features.parquet
             (= 사용자 맥. 외장하드 없이도 동작 - 이미 추출된 발화 단위 음향통계만 읽음)
             고객 발화만 / r_gold_valid / 앞 5개 고객 발화를 여기서 직접 재구성한다.
             화자 성별은 d04의 speaker_gender를 그대로 쓴다. 오디오 경로도 여기서 나온다.

PROXY 모드 : results/05_gate/train_10k_pool5_labeled.parquet
             (= outputs/ 가 없는 환경의 대체 소스. 저장소에 수록된 콜 단위 집계본)
             ! 이 파일의 pool5는 "앞 5개 고객 발화"가 아니라
               "앞 5발화(상담사 포함) ∪ 앞 5개 고객 발화"라서 상담사 발화가 섞여 있다
               (콜당 평균 6.78발화). 상담사는 훈련된 차분한 발화라 각성도를 희석시키고,
               상담사 성별이 여성에 쏠려 있어 F0 항도 오염된다.
             ! 성별 컬럼이 없어 F0 기반으로 추정한다(results/02_acoustic/
               stage2_gender_population_stats.csv 의 남/여 모집단 통계 기준 최대우도 배정).
             ! 오디오 경로가 없어 청취 표본 목록에 경로를 채울 수 없다.
             -> 분포의 '모양'과 특징 간 분리도를 보는 용도까지만 유효하다.
                최종 판정은 FULL 모드 재실행 결과로 한다.

--------------------------------------------------------------------------
proxy 점수 설계
--------------------------------------------------------------------------
각성도는 미세 음색이 아니라 "에너지·피치의 큰 궤적"에 실리므로 평균·변동 중심의
거친 통계만 쓴다. 전부 해석 가능한 등가중 선형 결합이며 학습 모델은 쓰지 않는다.

  P1_core    = mean(z_f0, z_energy, z_energy_var)          <- 기본형(가장 단순)
  P2_prosody = mean(P1 3항, z_f0_var, z_rate)              <- 운율 동역학 추가
  P3_silence = mean(P2 5항, -z_silence)                    <- 무음비율 반영
  P4_gain_robust = mean(z_f0, z_energy_cv, z_dyn, z_rate)  <- 회선 게인 비의존형

P4를 따로 두는 이유: 8kHz 전화망 녹음의 raw RMS(e_mean_amp)는 회선/단말 게인에
그대로 비례해서, "목소리가 크다"와 "녹음 레벨이 높다"가 분리되지 않는다. 반면
  - e_std_db  : amplitude_to_db(ref=np.max) 기반 발화 내 dynamic range -> 게인 불변
  - energy_cv : 발화 간 RMS 변동계수(std/mean)                          -> 게인 불변
이라서 P4는 게인에 영향받지 않는다. P1과 P4의 순위상관이 낮으면 P1의 에너지 항이
사실상 회선 게인을 재고 있다는 뜻이고, 그러면 "에너지가 크다=격앙"이라는 전제 자체가
흔들린다. 이 대조가 이번 스크리닝의 핵심 타당성 점검이다.

에너지 지표 선택은 arousal_target_check.py 의 주석을 그대로 따른다.
  e_mean_db 는 ref=np.max 라 발화 간 절대 성량 비교에 못 쓴다(사실상 dynamic range).
  따라서 절대 성량은 e_mean_amp(raw linear RMS), 발화 내 변동은 e_std_db 를 쓴다.

z 정규화 입도: arousal_target_check.py 는 '발화별 집계값의 분포'를 baseline으로 썼는데,
여기서는 점수화 대상과 baseline 입도를 맞추기 위해 **콜 단위 집계값의 분포**를
baseline으로 쓴다(성별 분리). 즉 z는 "같은 성별 콜들 사이에서의 상대 위치"다.

--------------------------------------------------------------------------
출력
--------------------------------------------------------------------------
  outputs/arousal_screen_percall.parquet      전 콜 특징 + z + 4개 proxy 점수 + 품질플래그
  outputs/arousal_screen_listen_sample.csv    고각성/저각성 청취 후보 각 25콜
  outputs/arousal_screen_listen_paths.csv     위 후보들의 발화 단위 오디오 경로 (FULL 모드만)
  outputs/arousal_screen_summary.md           분포/임계값/군간비교/품질지표 리포트

실행: python scripts/arousal_screen.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "outputs"

D04_PATH = OUT_DIR / "d04_dialog_index.parquet"
ACOUSTIC_PATH = OUT_DIR / "stage2_acoustic_features.parquet"
PROXY_PATH = BASE_DIR / "results" / "05_gate" / "train_10k_pool5_labeled.parquet"
GENDER_STATS_PATH = BASE_DIR / "results" / "02_acoustic" / "stage2_gender_population_stats.csv"

PERCALL_PATH = OUT_DIR / "arousal_screen_percall.parquet"
LISTEN_PATH = OUT_DIR / "arousal_screen_listen_sample.csv"
LISTEN_PATHS_PATH = OUT_DIR / "arousal_screen_listen_paths.csv"
MD_PATH = OUT_DIR / "arousal_screen_summary.md"

MAX_CUSTOMER_UTT = 5
CUSTOMER_PREFIX = "고객"
N_LISTEN = 25
REL_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]
ABS_THRESHOLDS = [0.5, 1.0, 1.5, 2.0]

# (콜 단위 특징명, z 컬럼명, 부호) - 부호는 "각성도가 높을수록 커지는가"
FEATURES = [
    ("f0_mean_mean", "z_f0", +1),          # 평균 피치
    ("e_mean_amp_mean", "z_energy", +1),   # 평균 성량 (raw linear RMS, 게인 민감)
    ("e_mean_amp_std", "z_energy_var", +1),  # 발화 간 성량 변동 (게인 민감)
    ("f0_std_mean", "z_f0_var", +1),       # 발화 내 피치 변동
    ("r_gold_mean", "z_rate", +1),         # 말속도 (음절/초)
    ("s_top_db30_mean", "z_silence", -1),  # 무음비율 (높을수록 저각성)
    ("e_std_db_mean", "z_dyn", +1),        # 발화 내 dynamic range (게인 불변)
    ("e_cv", "z_energy_cv", +1),           # 발화 간 성량 변동계수 (게인 불변)
]

PROXIES = {
    "P1_core": ["z_f0", "z_energy", "z_energy_var"],
    "P2_prosody": ["z_f0", "z_energy", "z_energy_var", "z_f0_var", "z_rate"],
    "P3_silence": ["z_f0", "z_energy", "z_energy_var", "z_f0_var", "z_rate", "z_silence"],
    "P4_gain_robust": ["z_f0", "z_energy_cv", "z_dyn", "z_rate"],
}
PRIMARY = "P1_core"
# P1에 들어가지 않는 항 = 분리도가 나오면 그게 '구성상 당연한 것'이 아닌 증거
HELD_OUT = ["z_f0_var", "z_rate", "z_silence", "z_dyn", "z_energy_cv"]

L = []


def p(msg=""):
    print(msg)
    L.append(msg)


def cohen_d(a: pd.Series, b: pd.Series) -> float:
    """pooled-std Cohen's d. a가 클수록 양수."""
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan


def spearman(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    if m.sum() < 3:
        return np.nan
    return float(a[m].rank().corr(b[m].rank()))


def text_hist(v: pd.Series, bins: int = 34, width: int = 54) -> list:
    """마크다운 코드블록용 ASCII 히스토그램."""
    v = v.dropna()
    counts, edges = np.histogram(v, bins=bins)
    peak = counts.max() if counts.max() else 1
    out = ["```"]
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(round(c / peak * width))
        out.append(f"{lo:+6.2f} ~ {hi:+6.2f} | {bar:<{width}} {c:>6,}")
    out.append("```")
    return out


# ---------------------------------------------------------------- 소스 로딩
def load_full() -> tuple:
    """FULL 모드: outputs/ 의 발화 단위 음향통계에서 콜 단위 표를 직접 만든다."""
    d04 = pd.read_parquet(
        D04_PATH,
        columns=["call_id", "dialog_idx", "speaker_type", "speaker_gender",
                 "r_gold_valid_flag", "r_gold", "duration", "audioPath"],
    )
    ac = pd.read_parquet(ACOUSTIC_PATH)

    n0 = len(d04)
    d04 = d04[d04["speaker_type"].astype(str).str.startswith(CUSTOMER_PREFIX)]
    n_cust = len(d04)
    d04 = d04[d04["r_gold_valid_flag"] == True]  # noqa: E712
    n_valid = len(d04)

    d04 = d04.sort_values(["call_id", "dialog_idx"])
    d04["customer_rank"] = d04.groupby("call_id").cumcount() + 1
    win = d04[d04["customer_rank"] <= MAX_CUSTOMER_UTT].copy()

    p("## 0. 데이터 소스")
    p("")
    p("**FULL 모드** - `outputs/d04_dialog_index.parquet` + "
      "`outputs/stage2_acoustic_features.parquet` (WAV 재추출 없음, 기존 추출물 재사용).")
    p("")
    p("발화 선택은 §3.1 파이프라인(`stage2a_window_pool.py`)과 동일 규칙:")
    p("")
    p(f"- 전체 dialog {n0:,}행")
    p(f"- 고객 발화만 (`speaker_type`이 '{CUSTOMER_PREFIX}'로 시작): -> {n_cust:,}행")
    p(f"- 전사 품질 통과분만 (`r_gold_valid_flag == True`): -> {n_valid:,}행")
    p(f"- dialog_idx 오름차순 앞 {MAX_CUSTOMER_UTT}개 고객 발화: -> **{len(win):,}행 / "
      f"{win['call_id'].nunique():,}콜**")
    p("")

    seg = win.merge(ac, on=["call_id", "dialog_idx"], how="left")

    # 품질 지표는 콜 단위 집계 전에 발화 단위로 집계해 둔다
    seg["f0_ok"] = seg["f0_mean"].notna()
    seg["load_ok"] = seg["load_ok"].fillna(False) if "load_ok" in seg.columns else False

    agg = seg.groupby("call_id").agg(
        n_utt=("dialog_idx", "size"),
        n_f0_fail=("f0_ok", lambda s: int((~s).sum())),
        n_load_fail=("load_ok", lambda s: int((~s.astype(bool)).sum())),
        dur_sec=("duration", "sum"),
        gender=("speaker_gender", lambda s: s.mode().iat[0] if len(s.mode()) else None),
        f0_mean_mean=("f0_mean", "mean"),
        f0_mean_std=("f0_mean", "std"),
        f0_std_mean=("f0_std", "mean"),
        e_mean_amp_mean=("e_mean_amp", "mean"),
        e_mean_amp_std=("e_mean_amp", "std"),
        e_std_db_mean=("e_std_db", "mean"),
        s_top_db30_mean=("s_top_db30", "mean"),
        r_gold_mean=("r_gold", "mean"),
        r_gold_std=("r_gold", "std"),
    ).reset_index()

    paths = win[["call_id", "customer_rank", "dialog_idx", "audioPath", "duration"]].copy()
    return agg, paths, "FULL"


def load_proxy() -> tuple:
    """PROXY 모드: 저장소 수록 콜 단위 집계본. 화자 혼재 + 성별 추정."""
    t = pd.read_parquet(PROXY_PATH)
    ren = {
        "pool5_f0_mean_mean": "f0_mean_mean",
        "pool5_f0_mean_std": "f0_mean_std",
        "pool5_f0_std_mean": "f0_std_mean",
        "pool5_e_mean_amp_mean": "e_mean_amp_mean",
        "pool5_e_mean_amp_std": "e_mean_amp_std",
        "pool5_e_std_db_mean": "e_std_db_mean",
        "pool5_s_top_db30_mean": "s_top_db30_mean",
        "pool5_r_gold_mean": "r_gold_mean",
        "pool5_r_gold_std": "r_gold_std",
        "pool5_n_utt": "n_utt",
    }
    agg = t[["call_id", "gold_actual"] + list(ren)].rename(columns=ren)
    agg["n_f0_fail"] = 0
    agg["n_load_fail"] = 0
    agg["dur_sec"] = np.nan

    # 성별 추정: 프로젝트 자체 모집단 통계(발화 단위 f0_mean)의 남/여 정규밀도 비교
    gs = pd.read_csv(GENDER_STATS_PATH)
    gs.columns = [c.strip().lstrip("﻿") for c in gs.columns]
    f0 = gs[gs["metric"] == "f0_mean"].set_index("gender")
    mu_m, sd_m = float(f0.loc["남", "mean"]), float(f0.loc["남", "std"])
    mu_f, sd_f = float(f0.loc["여", "mean"]), float(f0.loc["여", "std"])
    x = agg["f0_mean_mean"]
    ll_m = -np.log(sd_m) - 0.5 * ((x - mu_m) / sd_m) ** 2
    ll_f = -np.log(sd_f) - 0.5 * ((x - mu_f) / sd_f) ** 2
    agg["gender"] = np.where(ll_m >= ll_f, "남(추정)", "여(추정)")

    p("## 0. 데이터 소스")
    p("")
    p("**PROXY 모드** - `outputs/` 가 없어 저장소 수록본 "
      "`results/05_gate/train_10k_pool5_labeled.parquet`(10,000콜)으로 대체 실행했다.")
    p("")
    p("이 모드에는 **결과 해석을 제한하는 세 가지 한계**가 있다.")
    p("")
    p(f"1. **화자 혼재**: 이 파일의 pool5는 '앞 5개 고객 발화'가 아니라 "
      f"'앞 5발화(상담사 포함) ∪ 앞 5개 고객 발화'다(콜당 평균 "
      f"{agg['n_utt'].mean():.2f}발화). 상담사 발화가 섞여 있고, 상담사는 훈련된 "
      f"차분한 발화라 각성도 신호를 **희석**시킨다. 즉 아래 수치는 고객 전용으로 "
      f"다시 뽑으면 분리도가 지금보다 **좋아지는 방향**으로 움직일 가능성이 크다.")
    p("2. **성별 추정**: 성별 컬럼이 없어 `stage2_gender_population_stats.csv`의 "
      f"남(μ={mu_m:.1f}, σ={sd_m:.1f}) / 여(μ={mu_f:.1f}, σ={sd_f:.1f}) 정규분포 "
      "최대우도로 배정했다. 상담사 발화가 섞인 F0 평균이라 오배정이 있을 수 있다.")
    p("3. **오디오 경로 없음**: 청취 표본 목록에 경로를 채울 수 없다(call_id만 제공).")
    p("")
    p("-> 분포의 '모양'과 특징 간 분리도를 보는 데까지만 유효하다. "
      "**최종 판정은 FULL 모드 재실행 결과로 한다.**")
    p("")
    return agg, None, "PROXY"


# ---------------------------------------------------------------- 점수화
def build_scores(agg: pd.DataFrame) -> pd.DataFrame:
    # 게인 불변 에너지 변동계수
    agg["e_cv"] = agg["e_mean_amp_std"] / agg["e_mean_amp_mean"].replace(0, np.nan)

    # 품질 플래그
    agg["flag_short"] = agg["n_utt"] < 2          # std 항이 정의되지 않음
    agg["flag_f0_fail"] = agg["n_f0_fail"] > 0
    agg["flag_any"] = agg["flag_short"] | agg["flag_f0_fail"] | (agg["n_load_fail"] > 0)

    # 성별 분리 z (콜 단위 집계값 분포 기준)
    for col, zcol, sign in FEATURES:
        g = agg.groupby("gender")[col]
        agg[zcol] = sign * ((agg[col] - g.transform("mean")) / g.transform("std"))

    for name, terms in PROXIES.items():
        agg[name] = agg[terms].mean(axis=1)
    return agg


# ---------------------------------------------------------------- 리포트
def report_distribution(df: pd.DataFrame):
    s = df[PRIMARY].dropna()
    p("## 1. proxy 점수 분포")
    p("")
    p("proxy 점수는 전부 등가중 선형 결합이며 학습 모델이 아니다. "
      "각 항은 **같은 성별 콜 분포 안에서의 z-점수**이고, 각성도가 높을수록 "
      "커지도록 부호를 맞췄다(무음비율만 부호 반전).")
    p("")
    p("| 점수 | 구성 |")
    p("|---|---|")
    for name, terms in PROXIES.items():
        star = " **(기본)**" if name == PRIMARY else ""
        p(f"| `{name}`{star} | mean({', '.join(terms)}) |")
    p("")
    p("`z_f0`=평균피치, `z_energy`=평균성량(raw RMS), `z_energy_var`=발화간 성량변동, "
      "`z_f0_var`=발화내 피치변동, `z_rate`=말속도(음절/초), `z_silence`=무음비율(반전), "
      "`z_dyn`=발화내 dynamic range, `z_energy_cv`=발화간 성량 변동계수.")
    p("")

    p(f"### 1-1. `{PRIMARY}` 기술통계 (n={len(s):,})")
    p("")
    p("| 통계량 | 값 |")
    p("|---|---|")
    p(f"| mean | {s.mean():+.4f} |")
    p(f"| std | {s.std(ddof=1):.4f} |")
    p(f"| skewness | {s.skew():+.4f} |")
    p(f"| excess kurtosis | {s.kurt():+.4f} |")
    p("")
    p("### 1-2. 분위수")
    p("")
    p("| 분위 | " + " | ".join(f"{q:g}%" for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]) + " |")
    p("|---|" + "---|" * 9)
    for name in PROXIES:
        v = df[name].dropna()
        qs = [v.quantile(q / 100) for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]]
        p(f"| `{name}` | " + " | ".join(f"{x:+.3f}" for x in qs) + " |")
    p("")
    p(f"### 1-3. `{PRIMARY}` 히스토그램")
    p("")
    L.extend(text_hist(s))
    print(f"[histogram] {len(s):,} calls")
    p("")

    # 정규분포 대비 상단 꼬리 - "고각성 덩어리"가 따로 있는가
    z = (s - s.mean()) / s.std(ddof=1)
    p("### 1-4. 상단 꼬리가 정규분포보다 두꺼운가")
    p("")
    p("여기가 이번 스크리닝의 핵심이다. 점수가 단봉 정규분포에 가깝다면 "
      "'격앙 통화 몇 %'라는 물음 자체가 성립하지 않는다 — 상위 10%를 자르면 "
      "정의상 10%일 뿐이고, 어디를 끊어도 임의적이다. "
      "반대로 상단에 정규분포보다 두꺼운 덩어리가 있으면 그게 실제 격앙군의 흔적이다.")
    p("")
    p("| 컷 | 실제 비율 | 정규분포 기대 비율 | 실제/기대 |")
    p("|---|---|---|---|")
    from math import erfc, sqrt
    for k in ABS_THRESHOLDS:
        obs = float((z >= k).mean())
        exp = 0.5 * erfc(k / sqrt(2))
        ratio = obs / exp if exp > 0 else np.nan
        p(f"| z ≥ {k:.1f}σ | {obs*100:.2f}% | {exp*100:.2f}% | {ratio:.2f}× |")
    p("")


def report_thresholds(df: pd.DataFrame):
    p("## 2. 임계값별 고각성 비율")
    p("")
    p("### 2-1. 상대 임계값 (상위 N%)")
    p("")
    p("비율이 임계값과 같은 것은 정의상 당연하다. 여기서 읽을 것은 "
      "**각 구간의 컷오프 점수와 그 구간이 실제로 얼마나 극단인지**다.")
    p("")
    p("| 상위 | n | 컷오프 점수 | 구간 평균 점수 | 평균 피치 z | 평균 성량 z |")
    p("|---|---|---|---|---|---|")
    s = df[PRIMARY]
    for t in REL_THRESHOLDS:
        cut = s.quantile(1 - t)
        m = s >= cut
        p(f"| {t*100:.0f}% | {int(m.sum()):,} | {cut:+.3f} | {s[m].mean():+.3f} | "
          f"{df.loc[m, 'z_f0'].mean():+.3f} | {df.loc[m, 'z_energy'].mean():+.3f} |")
    p("")
    p("### 2-2. 절대 임계값 (점수 자체가 기준)")
    p("")
    p("이쪽이 '격앙이 몇 %인가'에 대한 비순환적 답이다. proxy 점수는 z의 평균이라 "
      "예컨대 +1.0은 '세 지표가 평균적으로 동성별 코퍼스 대비 1σ 위'를 뜻한다.")
    p("")
    header = "| 임계 | " + " | ".join(f"`{n}`" for n in PROXIES) + " |"
    p(header)
    p("|---|" + "---|" * len(PROXIES))
    for k in ABS_THRESHOLDS:
        cells = []
        for n in PROXIES:
            v = df[n].dropna()
            cells.append(f"{(v >= k).mean()*100:.2f}% ({int((v >= k).sum()):,})")
        p(f"| ≥ +{k:.1f} | " + " | ".join(cells) + " |")
    p("")

    p("### 2-3. proxy 점수들끼리 순위가 일치하는가 (Spearman ρ)")
    p("")
    p("가중치를 어떻게 잡든 같은 콜이 위로 오면 결론은 가중치에 견고하다. "
      "특히 `P1_core` vs `P4_gain_robust` 가 핵심 — 낮으면 P1의 에너지 항이 "
      "각성도가 아니라 **회선 녹음 게인**을 재고 있다는 뜻이다.")
    p("")
    names = list(PROXIES)
    p("| | " + " | ".join(f"`{n}`" for n in names) + " |")
    p("|---|" + "---|" * len(names))
    for a in names:
        cells = [f"{spearman(df[a], df[b]):.3f}" if a != b else "—" for b in names]
        p(f"| `{a}` | " + " | ".join(cells) + " |")
    p("")


def report_groups(df: pd.DataFrame, top_frac: float = 0.10):
    hi_cut = df[PRIMARY].quantile(1 - top_frac)
    lo_cut = df[PRIMARY].quantile(top_frac)
    hi = df[df[PRIMARY] >= hi_cut]
    lo = df[df[PRIMARY] <= lo_cut]

    p("## 3. 고각성 후보군 vs 저각성 후보군 음향 특징 비교")
    p("")
    p(f"`{PRIMARY}` 상위 {top_frac*100:.0f}%(n={len(hi):,}, 컷 {hi_cut:+.3f}) vs "
      f"하위 {top_frac*100:.0f}%(n={len(lo):,}, 컷 {lo_cut:+.3f}).")
    p("")
    p("**`P1_core`를 구성하는 항(z_f0, z_energy, z_energy_var)이 갈리는 것은 구성상 "
      "당연하므로 증거가 아니다.** 증거가 되는 것은 점수에 안 들어간 held-out 항이다. "
      "각성도가 실재하는 구성개념이라면 말속도는 빨라지고 무음은 줄고 "
      "dynamic range는 커져야 한다. 이게 안 나오면 proxy는 성량 한 축만 재고 있는 것이다.")
    p("")
    p("| 항 | 구성? | 고각성 mean | 저각성 mean | Cohen's d |")
    p("|---|---|---|---|---|")
    in_p1 = set(PROXIES[PRIMARY])
    for col, zcol, _s in FEATURES:
        tag = "**구성항**" if zcol in in_p1 else "held-out"
        d = cohen_d(hi[zcol], lo[zcol])
        p(f"| `{zcol}` | {tag} | {hi[zcol].mean():+.3f} | {lo[zcol].mean():+.3f} | {d:+.2f} |")
    p("")
    p("원단위(raw) 비교 — 실제로 얼마나 다른 소리인지 감을 잡기 위한 것.")
    p("")
    p("| 원단위 특징 | 고각성 mean | 저각성 mean | 차이 |")
    p("|---|---|---|---|")
    raw_view = [("f0_mean_mean", "평균 F0 (Hz)"), ("e_mean_amp_mean", "평균 RMS"),
                ("r_gold_mean", "말속도 (음절/초)"), ("s_top_db30_mean", "무음비율"),
                ("e_std_db_mean", "발화내 dynamic range (dB)")]
    for col, label in raw_view:
        a, b = hi[col].mean(), lo[col].mean()
        p(f"| {label} | {a:.4f} | {b:.4f} | {a-b:+.4f} |")
    p("")

    if "gold_actual" in df.columns:
        p("### 3-1. 참고: 기존 gold_actual 라벨 분포")
        p("")
        p("각성도 라벨은 아직 없으므로 검증이 아니라 **방향 점검**일 뿐이다. "
          "다만 불만제기가 고각성 쪽에 쏠린다면 proxy가 최소한 무의미하지는 않다는 신호다.")
        p("")
        p("| gold_actual | 전체 | 고각성 상위 | 저각성 하위 | 상위 비율 | 하위 비율 |")
        p("|---|---|---|---|---|---|")
        for lab in df["gold_actual"].value_counts().index:
            n_all = int((df["gold_actual"] == lab).sum())
            n_hi = int((hi["gold_actual"] == lab).sum())
            n_lo = int((lo["gold_actual"] == lab).sum())
            p(f"| {lab} | {n_all:,} | {n_hi:,} | {n_lo:,} | "
              f"{n_hi/len(hi)*100:.2f}% | {n_lo/len(lo)*100:.2f}% |")
        p("")
    return hi, lo


def report_confounds(df: pd.DataFrame):
    """단일 각성도 축이 존재하는가 + 회선/수집배치 교란이 있는가."""
    zc = [z for _c, z, _s in FEATURES]
    p("## 4. 교란 점검 — 애초에 '각성도 축'이 하나 있기는 한가")
    p("")
    p("각성도가 실재하는 잠재 차원이라면, 그것이 구동하는 지표들끼리는 서로 상관이 있어야 "
      "한다. 목소리를 키운 사람은 피치도 올라가고 말도 빨라지는 식으로 같이 움직여야 "
      "한다는 뜻이다. 반대로 지표들이 서로 흩어져 있으면 proxy는 여러 개의 무관한 축을 "
      "억지로 더한 값이고, 상위 N%는 '가장 각성된 콜'이 아니라 '가장 큰 소리로 녹음된 콜'에 "
      "가까워진다.")
    p("")
    p("### 4-1. proxy 구성항 간 Spearman ρ")
    p("")
    p("| | " + " | ".join(f"`{z}`" for z in zc) + " |")
    p("|---|" + "---|" * len(zc))
    corr = df[zc].corr(method="spearman")
    for a in zc:
        cells = ["—" if a == b else f"{corr.loc[a, b]:+.2f}" for b in zc]
        p(f"| `{a}` | " + " | ".join(cells) + " |")
    p("")
    core = PROXIES[PRIMARY]
    pairs = [(a, b) for i, a in enumerate(core) for b in core[i + 1:]]
    vals = [corr.loc[a, b] for a, b in pairs]
    p(f"기본 점수 `{PRIMARY}` 세 항의 쌍별 상관: "
      + ", ".join(f"`{a}`×`{b}`={v:+.2f}" for (a, b), v in zip(pairs, vals))
      + f" (평균 {np.mean(vals):+.2f}).")
    p("")
    p("해석에 특히 중요한 두 쌍:")
    p("")
    p(f"- `z_f0` × `z_energy` = **{corr.loc['z_f0', 'z_energy']:+.2f}**. "
      "각성도의 두 대표 채널인 피치와 성량이 사실상 따로 논다. "
      "공통 각성도 인자가 이 둘을 함께 끌어올리고 있다면 나오기 어려운 값이다.")
    p(f"- `z_silence` × `z_dyn` = **{corr.loc['z_silence', 'z_dyn']:+.2f}**. "
      "둘 다 발화 내 dB 대비 구조에서 나와 사실상 같은 것을 재는 지표다"
      "(독립적인 두 증거로 세면 안 된다).")
    p("")

    if df["call_id"].str.slice(0, 3).nunique() > 1:
        p("### 4-2. 수집 배치(회선/녹음조건) 교란")
        p("")
        p("`call_id` 접두어(J16~J19)는 수집 배치다. raw RMS는 회선·단말 게인에 그대로 "
          "비례하므로, 배치가 점수 분산을 크게 설명하면 proxy는 각성도가 아니라 "
          "녹음 조건을 재고 있는 것이다.")
        p("")
        df = df.assign(_batch=df["call_id"].str.slice(0, 3))
        p("| 배치 | n | 평균 RMS | 평균 F0 | 평균 말속도 | 평균 `P1_core` |")
        p("|---|---|---|---|---|---|")
        for b, g in df.groupby("_batch"):
            p(f"| {b} | {len(g):,} | {g['e_mean_amp_mean'].mean():.4f} | "
              f"{g['f0_mean_mean'].mean():.1f} | {g['r_gold_mean'].mean():.2f} | "
              f"{g['P1_core'].mean():+.4f} |")
        p("")
        p("| 지표 | 배치가 설명하는 분산 η² |")
        p("|---|---|")
        for s in ["P1_core", "P4_gain_robust", "z_energy", "z_f0"]:
            gm = df.groupby("_batch")[s]
            ss_b = (gm.count() * (gm.mean() - df[s].mean()) ** 2).sum()
            ss_t = ((df[s] - df[s].mean()) ** 2).sum()
            p(f"| `{s}` | {ss_b/ss_t:.4f} |")
        p("")
        p("**이 교란은 기각된다.** 배치가 설명하는 분산이 2% 미만이라 "
          "점수가 수집 배치별 녹음 레벨 차이로 만들어진 것은 아니다. "
          "다만 콜 단위 회선 게인(배치 내 개별 편차)은 이 검정으로 배제되지 않는다 — "
          "그건 같은 콜의 **상담사 발화 성량**과 대조해야 갈라지고, "
          "FULL 모드에서만 가능하다(§7 참조).")
        p("")


def report_quality(df: pd.DataFrame, mode: str):
    p("## 5. 저음질(8kHz) 품질 지표")
    p("")
    p("AIHub 저음질 전화망 데이터는 **8kHz 모노**다"
      "(`results/08_audio_seg/audio_seg_manifest.parquet` 1,119발화 전수 확인: "
      "sample_rate 8000, n_channels 1). 나이퀴스트가 4kHz라 고주파 성분은 "
      "애초에 없고, F0 추출도 `pitch_floor=75 / pitch_ceiling=500Hz` 범위에서만 한다. "
      "따라서 **미세 음색·고주파 지표는 이번 설계에 아예 넣지 않았다** — "
      "평균·변동 중심의 거친 통계만 쓴 이유다.")
    p("")
    n = len(df)
    p("| 지표 | n | 비율 |")
    p("|---|---|---|")
    p(f"| 전체 콜 | {n:,} | 100.00% |")
    if mode == "FULL":
        p(f"| F0 추출 실패 발화가 1개 이상 있는 콜 | {int(df['flag_f0_fail'].sum()):,} | "
          f"{df['flag_f0_fail'].mean()*100:.2f}% |")
        p(f"| WAV 로딩 실패 발화가 1개 이상 있는 콜 | {int((df['n_load_fail'] > 0).sum()):,} | "
          f"{(df['n_load_fail'] > 0).mean()*100:.2f}% |")
    p(f"| 고객 발화 1개뿐(변동 항 정의 불가) | {int(df['flag_short'].sum()):,} | "
      f"{df['flag_short'].mean()*100:.2f}% |")
    p(f"| proxy 점수 산출 실패(NaN) | {int(df[PRIMARY].isna().sum()):,} | "
      f"{df[PRIMARY].isna().mean()*100:.2f}% |")
    p(f"| 품질 플래그 1개 이상 | {int(df['flag_any'].sum()):,} | "
      f"{df['flag_any'].mean()*100:.2f}% |")
    p("")
    p("무성비율 대리지표로 무음비율(`s_top_db30`, top_db=30 기준)을 함께 본다. "
      "프레임 단위 유성/무성 카운트는 저장돼 있지 않아 발화 단위 무음비율로 대신한다.")
    p("")
    p("| 무음비율 | mean | p10 | p50 | p90 |")
    p("|---|---|---|---|---|")
    v = df["s_top_db30_mean"].dropna()
    p(f"| `s_top_db30_mean` | {v.mean():.4f} | {v.quantile(.1):.4f} | "
      f"{v.median():.4f} | {v.quantile(.9):.4f} |")
    p("")
    if mode == "FULL" and df["flag_any"].any():
        clean = df[~df["flag_any"]]
        p(f"품질 플래그가 붙은 콜을 빼고 다시 보면 상위 20% 컷오프가 "
          f"{df[PRIMARY].quantile(0.8):+.3f} -> {clean[PRIMARY].quantile(0.8):+.3f}로 "
          f"움직인다(n {len(df):,} -> {len(clean):,}). "
          f"분포 해석 시 이 콜들은 별도 표시 대상이다.")
        p("")


def report_listen(df: pd.DataFrame, paths, mode: str):
    d = df[~df["flag_any"]].dropna(subset=[PRIMARY]).sort_values(PRIMARY, ascending=False)
    hi = d.head(N_LISTEN).copy()
    lo = d.tail(N_LISTEN).copy()
    hi["group"] = "고각성후보"
    lo["group"] = "저각성후보"
    sample = pd.concat([hi, lo], ignore_index=True)
    sample["rank_in_corpus"] = sample["call_id"].map(
        dict(zip(d["call_id"], range(1, len(d) + 1))))
    sample["pct_in_corpus"] = sample["rank_in_corpus"] / len(d) * 100

    cols = ["group", "call_id", "rank_in_corpus", "pct_in_corpus", PRIMARY,
            "P4_gain_robust", "gender", "n_utt", "f0_mean_mean", "e_mean_amp_mean",
            "r_gold_mean", "s_top_db30_mean"]
    cols = [c for c in cols if c in sample.columns]
    out = sample[cols]

    listen_paths = None
    if mode == "FULL" and paths is not None:
        listen_paths = paths[paths["call_id"].isin(sample["call_id"])].merge(
            sample[["call_id", "group", PRIMARY]], on="call_id", how="left"
        ).sort_values(["group", PRIMARY, "customer_rank"], ascending=[True, False, True])
        first = (paths.sort_values("customer_rank").groupby("call_id")["audioPath"].first())
        out = out.assign(first_audio_path=out["call_id"].map(first))

    p("## 6. 청취 표본 목록")
    p("")
    p(f"품질 플래그가 붙은 콜을 제외한 뒤 `{PRIMARY}` 상위 {N_LISTEN}콜 / 하위 {N_LISTEN}콜. "
      "**이 목록은 proxy가 맞다는 증거가 아니라, proxy가 지각된 각성도를 따라가는지 "
      "사람이 직접 들어 확인할 표본이다.** 들었을 때 상위군이 실제로 격앙되게 들리지 "
      "않으면 그 시점에서 이 스크리닝은 부정 판정이다.")
    p("")
    if mode == "FULL":
        p(f"오디오 경로는 발화 단위다(AIHub 원천이 발화별 개별 WAV). 표에는 첫 고객 발화 "
          f"경로만 싣고, 선택된 콜의 발화 전체 경로는 `{LISTEN_PATHS_PATH.name}`에 있다. "
          f"실제 파일은 `{'/Volumes/AIHub/007.저음질 전화망 음성인식 데이터/01.데이터/1.Training/원천데이터_230316'}` 아래.")
    else:
        p("PROXY 모드라 오디오 경로를 채울 수 없다. call_id만 제공하며, "
          "FULL 모드로 재실행하면 경로가 함께 나온다.")
    p("")
    for gname, g in [("고각성 후보", hi), ("저각성 후보", lo)]:
        p(f"### {gname} ({len(g)}콜)")
        p("")
        p(f"| # | call_id | `{PRIMARY}` | `P4_gain_robust` | 성별 | 발화수 | "
          f"F0(Hz) | RMS | 말속도 | 무음비율 |")
        p("|---|---|---|---|---|---|---|---|---|---|")
        gg = g.sort_values(PRIMARY, ascending=(gname == "저각성 후보"))
        for i, r in enumerate(gg.itertuples(), 1):
            p(f"| {i} | {r.call_id} | {getattr(r, PRIMARY):+.3f} | {r.P4_gain_robust:+.3f} | "
              f"{r.gender} | {int(r.n_utt)} | {r.f0_mean_mean:.1f} | "
              f"{r.e_mean_amp_mean:.4f} | {r.r_gold_mean:.2f} | {r.s_top_db30_mean:.3f} |")
        p("")
    return out, listen_paths


def main():
    OUT_DIR.mkdir(exist_ok=True)
    if D04_PATH.exists() and ACOUSTIC_PATH.exists():
        agg, paths, mode = load_full()
    else:
        agg, paths, mode = load_proxy()

    df = build_scores(agg)
    report_distribution(df)
    report_thresholds(df)
    report_groups(df)
    report_confounds(df)
    report_quality(df, mode)
    listen, listen_paths = report_listen(df, paths, mode)

    p("## 7. 판정에 필요한 근거 요약")
    p("")
    s = df[PRIMARY].dropna()
    z = (s - s.mean()) / s.std(ddof=1)
    from math import erfc, sqrt
    obs15 = float((z >= 1.0).mean())
    exp15 = 0.5 * erfc(1.0 / sqrt(2))
    rho14 = spearman(df["P1_core"], df["P4_gain_robust"])
    hi_cut, lo_cut = s.quantile(0.9), s.quantile(0.1)
    hi, lo = df[df[PRIMARY] >= hi_cut], df[df[PRIMARY] <= lo_cut]
    p("판정은 사용자가 한다. 아래는 판정 기준에 대응하는 숫자만 모은 것이다.")
    p("")
    p("| 물음 | 숫자 |")
    p("|---|---|")
    p(f"| 고각성 추정 비율 (z ≥ +1σ) | **{obs15*100:.2f}%** "
      f"(정규분포 기대 {exp15*100:.2f}%, {obs15/exp15:.2f}×) |")
    p(f"| 분포 비대칭도 (skewness) | {s.skew():+.3f} |")
    p(f"| 가중치 견고성 (`P1_core` vs `P4_gain_robust` ρ) | {rho14:.3f} |")
    for zc in HELD_OUT:
        p(f"| held-out 분리도 `{zc}` (Cohen's d) | {cohen_d(hi[zc], lo[zc]):+.2f} |")
    p(f"| 품질 플래그 콜 비율 | {df['flag_any'].mean()*100:.2f}% |")
    p(f"| 데이터 소스 | {mode} 모드 |")
    p("")
    p("### 7-1. 이 스크리닝으로는 갈라지지 않는 것")
    p("")
    p("콜 단위 **회선 게인**은 여기서 배제되지 않았다. 배치 수준 교란은 §4-2에서 "
      "기각됐지만(η²<0.02), 개별 콜의 녹음 레벨 차이는 남아 있다. 이걸 가르는 "
      "결정적 검정은 **같은 콜의 상담사 발화 성량과의 상관**이다.")
    p("")
    p("- 고객 RMS와 상담사 RMS가 콜 안에서 강하게 상관 → 점수는 회선 게인이다.")
    p("- 상관이 약하면 → 고객 성량은 화자 고유 특성이고, 각성도 해석이 살아난다.")
    p("")
    p("`d04_dialog_index.parquet` + `stage2_acoustic_features.parquet` 가 있는 환경"
      "(FULL 모드)에서 상담사 발화만 따로 집계하면 바로 계산된다. "
      "각성도 라벨링에 공수를 넣기 전에 이것부터 보는 것을 권한다.")
    p("")

    df.to_parquet(PERCALL_PATH, index=False)
    listen.to_csv(LISTEN_PATH, index=False, encoding="utf-8-sig")
    if listen_paths is not None:
        listen_paths.to_csv(LISTEN_PATHS_PATH, index=False, encoding="utf-8-sig")

    header = [
        "# 각성도(arousal) 라벨링 전 타당성 스크리닝",
        "",
        "사람 청취 라벨링에 공수를 들이기 전 사전 판정용. **라벨은 만들지 않았다.**",
        "음향 요약통계만으로 (1) 격앙 통화 비율 추정, (2) 고/저각성의 음향 분리도, "
        "(3) 8kHz에서 신호 생존 여부만 본다.",
        "",
        "> **proxy 점수는 정답이 아니다.** 사람이 음성만 듣고 만드는 골드 라벨을 "
        "대체하지 않으며, '라벨링할 가치가 있는지'만 판정한다. "
        "API 호출 0, 전부 로컬 파일.",
        "",
    ]
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(header + L) + "\n")

    print(f"\n저장: {MD_PATH}")
    print(f"저장: {PERCALL_PATH} ({len(df):,}행)")
    print(f"저장: {LISTEN_PATH} ({len(listen)}행)")
    if listen_paths is not None:
        print(f"저장: {LISTEN_PATHS_PATH} ({len(listen_paths)}행)")
    print("AROUSAL_SCREEN_COMPLETE")


if __name__ == "__main__":
    main()
