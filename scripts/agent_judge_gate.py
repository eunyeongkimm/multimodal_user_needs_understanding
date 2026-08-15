"""아웃풋 ① 게이트(층별 AUC) + ③ 점수 분포.

mismatch flag = stage2g 정의: 조건 A / N=1 의 predicted_label != gold_actual(label).
판정 기준은 gold!=불만제기 층(불만 교란 분리)의 AUC.

  gold!=불만 층 AUC >= 0.6  -> 통과 (층화 확대 진행)
  ~0.5                      -> 상담사 워크스트림 종료

gold_actual / mismatch 는 여기(평가)에서만 쓰이고 점수 산출 프롬프트에는 미사용.

출력:
  outputs/agent_judge_gate_summary.md
  outputs/agent_judge_percall.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, pointbiserialr, spearmanr
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_PATH = BASE_DIR / "outputs" / "model_pilot_sample.csv"
SCORES_PATH = BASE_DIR / "outputs" / "agent_judge_scores.parquet"
PRED_PATH = BASE_DIR / "outputs" / "stage2d_gpt_predictions.parquet"
GOLD_PATH = BASE_DIR / "outputs" / "gold_actual_batch1_final.parquet"
HUMAN_PATH = BASE_DIR / "outputs" / "agent_judge_human_eval_slots.csv"
MD_PATH = BASE_DIR / "outputs" / "agent_judge_gate_summary.md"
PERCALL_PATH = BASE_DIR / "outputs" / "agent_judge_percall.parquet"

COMPLAINT = "불만제기"
SURFACE_CONDITION, SURFACE_N = "A", 1
GATE_THRESHOLD = 0.60
SCORE_COLS = [("친절도", "친절도"), ("적절대응", "적절대응"), ("합산", "합산")]

L = []


def p(msg=""):
    print(msg)
    L.append(msg)


def main():
    sample = pd.read_csv(SAMPLE_PATH)[["call_id", "gold_actual"]]
    scores = pd.read_parquet(SCORES_PATH)[["call_id", "친절도", "적절대응", "judge_error"]]

    pred = pd.read_parquet(PRED_PATH)
    sur = pred[(pred["condition"] == SURFACE_CONDITION) & (pred["n"] == SURFACE_N)]
    sur = sur.dropna(subset=["predicted_label"])[["call_id", "predicted_label"]]
    gold = pd.read_parquet(GOLD_PATH)[["call_id", "label"]].rename(
        columns={"label": "gold_label"})
    sur = sur.merge(gold, on="call_id", how="left")
    sur["gold_mismatch"] = (sur["predicted_label"] != sur["gold_label"]).astype(int)

    df = sample.merge(scores, on="call_id", how="left").merge(
        sur[["call_id", "gold_mismatch", "gold_label"]], on="call_id", how="left")
    df["is_불만"] = (df["gold_label"] == COMPLAINT).astype(int)
    df["합산"] = df["친절도"] + df["적절대응"]

    n_fail = int(df["친절도"].isna().sum())
    n_no_mm = int(df["gold_mismatch"].isna().sum())

    p("# 상담사 텍스트 대응 점수 게이트 (250건)")
    p("")
    p("상담사 점수는 **텍스트 전용** GPT-as-judge(`gpt-5.4`, `reasoning.effort=low`, "
      "Responses API, full-call 전사). acoustic 미사용.")
    p("")
    p("**누수 차단 확인 로그**")
    p("")
    p("- 프롬프트는 `d04_dialog_index.parquet` 전사 텍스트만으로 조립되며 "
      "`gold_actual`/`mismatch` 컬럼은 어떤 경로로도 읽지 않는다(구조적 보장).")
    p("- 전사 밖(지시문·평가항목)에 7개 카테고리명 등장: **0건** (자동 검증, 있으면 중단).")
    p("- 전사 본문에 카테고리명이 우발적으로 등장한 콜: **1건**(그중 gold와 같은 단어 1건). "
      "화자가 통화 중 실제로 발화한 원본 텍스트이며 정답 주입이 아니라 유지.")
    p("- 프롬프트에 \"상담사가 실제 의도를 맞췄는가\"로 채점하지 말라는 금지 문구 포함.")
    p(f"- `gold_actual`/`mismatch`는 아래 AUC·상관 계산에만 사용.")
    p("")
    p(f"**파싱/채점 실패**: {n_fail}건 (재시도 발생 0건). "
      f"mismatch flag 결측: {n_no_mm}건.")
    p("")
    p("**mismatch 정의**: `stage2g_mismatch_subset_analysis.py` 기준 — "
      f"조건 {SURFACE_CONDITION} / N={SURFACE_N}(표면: 고객 첫 1발화, text-only)의 "
      "`predicted_label != gold_actual`.")
    p("")

    valid = df.dropna(subset=["친절도", "gold_mismatch"]).copy()
    layers = [
        ("전체", valid, "참고"),
        ("gold≠불만★", valid[valid["is_불만"] == 0], "판정 기준"),
        ("gold=불만", valid[valid["is_불만"] == 1], "참고"),
    ]

    p("## 아웃풋 ① 게이트 — 층별 AUC")
    p("")
    p("| 층 | n | mismatch | AUC(친절) | AUC(적절) | AUC(합산) |")
    p("|---|---|---|---|---|---|")
    aucs = {}
    for name, sub, _note in layers:
        y = sub["gold_mismatch"].astype(int)
        row = [f"| {name} | {len(sub)} | {int(y.sum())} "]
        if y.nunique() < 2:
            row.append("| - | - | - |")
            aucs[name] = {}
        else:
            vals = {}
            for label, col in SCORE_COLS:
                a = roc_auc_score(y, sub[col])
                vals[label] = a
                row.append(f"| {a:.3f} ")
            row.append("|")
            aucs[name] = vals
        p("".join(row))
    p("")
    p("AUC는 \"점수가 높을수록 mismatch\" 방향으로 계산했다. 0.5 미만이면 "
      "\"점수가 낮을수록 mismatch\"라는 뜻이며, 방향과 무관한 판별력은 "
      "|AUC-0.5|로 읽으면 된다.")
    p("")

    p("### point-biserial 상관 (점수 vs mismatch)")
    p("")
    p("| 층 | n | r(친절) | p | r(적절) | p | r(합산) | p |")
    p("|---|---|---|---|---|---|---|---|")
    for name, sub, _note in layers:
        y = sub["gold_mismatch"].astype(int)
        cells = [f"| {name} | {len(sub)} "]
        if y.nunique() < 2:
            cells.append("| - | - | - | - | - | - |")
        else:
            for _label, col in SCORE_COLS:
                r, pv = pointbiserialr(y, sub[col])
                cells.append(f"| {r:+.3f} | {pv:.4f} ")
            cells.append("|")
        p("".join(cells))
    p("")

    gate_layer = "gold≠불만★"
    best = max(aucs[gate_layer].values()) if aucs[gate_layer] else float("nan")
    best_name = (max(aucs[gate_layer], key=aucs[gate_layer].get)
                 if aucs[gate_layer] else "-")
    passed = best >= GATE_THRESHOLD
    p("### 판정")
    p("")
    p(f"- 판정 기준 층: **{gate_layer}** (n={len(layers[1][1])}, "
      f"mismatch={int(layers[1][1]['gold_mismatch'].sum())})")
    p(f"- 세 점수 중 최고 AUC: **{best:.3f}** ({best_name})")
    p(f"- 기준: AUC ≥ {GATE_THRESHOLD:.2f} 통과 / ~0.5 종료")
    p("")
    p(f"→ **{'통과 — 층화 확대 진행' if passed else '미통과 — 상담사 워크스트림 종료'}**")
    if not passed:
        p("")
        p("gold≠불만 층에서 상담사 텍스트 대응 점수는 mismatch를 판별하지 못한다. "
          "층화 비교·표본 확대·acoustic 재추출은 진행하지 않는다.")
    p("")

    p("## 아웃풋 ③ 점수 분포")
    p("")
    p("| 점수 | 1 | 2 | 3 | 4 | 5 | mean | median | std |")
    p("|---|---|---|---|---|---|---|---|---|")
    for col in ["친절도", "적절대응"]:
        v = df[col].dropna()
        counts = " | ".join(str(int((v == k).sum())) for k in range(1, 6))
        p(f"| {col} | {counts} | {v.mean():.3f} | {v.median():.1f} | {v.std(ddof=1):.3f} |")
    p("")
    v = df["합산"].dropna()
    p(f"합산(2~10): mean={v.mean():.3f}, median={v.median():.1f}, "
      f"std={v.std(ddof=1):.3f}, range={int(v.min())}~{int(v.max())}")
    p("")

    ok = df.dropna(subset=["친절도", "적절대응"])
    pr, ppv = pearsonr(ok["친절도"], ok["적절대응"])
    sr, spv = spearmanr(ok["친절도"], ok["적절대응"])
    p("### 두 점수 간 상관")
    p("")
    p(f"- Pearson r = {pr:.3f} (p={ppv:.2e}), Spearman rho = {sr:.3f} (p={spv:.2e}), n={len(ok)}")
    p(f"- 두 점수가 완전히 같은 콜: {int((ok['친절도'] == ok['적절대응']).sum())}/{len(ok)}건")
    p("")
    p("### 교차표 (행=친절도, 열=적절대응)")
    p("")
    ct = pd.crosstab(ok["친절도"].astype(int), ok["적절대응"].astype(int))
    ct = ct.reindex(index=range(1, 6), columns=range(1, 6), fill_value=0)
    p("| 친절\\적절 | 1 | 2 | 3 | 4 | 5 |")
    p("|---|---|---|---|---|---|")
    for k in range(1, 6):
        p(f"| **{k}** | " + " | ".join(str(int(x)) for x in ct.loc[k]) + " |")
    p("")

    p("### 사람 검증 Kappa 슬롯")
    p("")
    human = pd.read_csv(HUMAN_PATH)
    filled = human["human_친절도"].notna() & (human["human_친절도"].astype(str).str.strip() != "")
    p(f"- `{HUMAN_PATH.name}` 40건 무작위 추출(seed=42). "
      f"수동 채점 입력된 행: **{int(filled.sum())}건**.")
    if filled.sum() == 0:
        p("- 아직 비어 있어 Kappa 미계산. 채점 입력 후 이 스크립트를 다시 돌리면 산출된다.")
    else:
        from sklearn.metrics import cohen_kappa_score
        h = human[filled]
        for c in ["친절도", "적절대응"]:
            k = cohen_kappa_score(h[f"gpt_{c}"].astype(int), h[f"human_{c}"].astype(int),
                                  weights="quadratic")
            p(f"- {c}: quadratic-weighted Kappa = {k:.3f} (n={len(h)})")
    p("")

    out = df[["call_id", "gold_mismatch", "gold_label", "친절도", "적절대응",
              "합산", "is_불만"]]
    out.to_parquet(PERCALL_PATH, index=False)
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n저장: {MD_PATH}")
    print(f"저장: {PERCALL_PATH} ({len(out)}행)")
    print(f"GATE_{'PASSED' if passed else 'FAILED'}")


if __name__ == "__main__":
    main()
