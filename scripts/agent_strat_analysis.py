"""상담사 적절대응 점수를 층화축으로, 불만·mismatch·acoustic 효과와의 연관 탐색.

250건 텍스트 층화만. 재추출·표본확대·acoustic 재산출 없음.
아웃풋 ① 그룹 x 불만/mismatch 비율, ② 그룹 x acoustic 효과(A vs B).

★해석 가드: 연관만 보고하며 인과 주장을 하지 않는다.
  "불만이라 상담사가 힘들었나 vs 상담사가 못해 불만이 됐나"는 이 데이터로 판별 불가.

출력:
  outputs/agent_strat_summary.md
  outputs/agent_strat_percall.parquet
"""

from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.metrics import f1_score, recall_score

BASE_DIR = Path(__file__).resolve().parent.parent
SCORES_PATH = BASE_DIR / "outputs" / "agent_judge_percall.parquet"
AB_PATH = BASE_DIR / "outputs" / "modality_ab_percall.parquet"
MD_PATH = BASE_DIR / "outputs" / "agent_strat_summary.md"
PERCALL_PATH = BASE_DIR / "outputs" / "agent_strat_percall.parquet"

CATEGORIES = ["환불요청", "주문취소", "불만제기", "배송확인", "교환반품", "구매진행", "서비스이용"]
COMPLAINT = "불만제기"
AXIS = "적절대응"
GOOD, BAD = "잘 대응", "못 대응"
GROUP_ORDER = [GOOD, BAD]
MIN_SUPPORT_WARN = 30

L = []


def p(msg=""):
    print(msg)
    L.append(msg)


def prop_test(a_pos, a_n, b_pos, b_n):
    """2x2 비율 차이 검정. 기대빈도 5 미만 셀이 있으면 Fisher."""
    table = [[a_pos, a_n - a_pos], [b_pos, b_n - b_pos]]
    chi2, pv, _dof, exp = chi2_contingency(table, correction=True)
    if (exp < 5).any():
        _odds, pv_f = fisher_exact(table)
        return "Fisher exact", pv_f
    return "카이제곱(Yates 보정)", pv


def main():
    sc = pd.read_parquet(SCORES_PATH)
    ab = pd.read_parquet(AB_PATH)[["call_id", "pred_A", "pred_B"]]
    df = sc.merge(ab, on="call_id", how="inner")
    assert len(df) == len(sc) == 250, f"병합 후 행 수 이상: {len(df)}"

    v = df[AXIS]
    df["그룹"] = (v >= 4).map({True: GOOD, False: BAD})
    n_good = int((df["그룹"] == GOOD).sum())
    n_bad = int((df["그룹"] == BAD).sum())

    p("# 상담사 적절대응 층화 — 불만·mismatch·acoustic 효과 연관 탐색 (250건)")
    p("")
    p("게이트(적절대응 vs mismatch, gold≠불만 층 AUC 0.467)는 이미 미통과. "
      "여기서는 연결처를 불만·acoustic으로 넓혀 **연관만** 확인한다.")
    p("")
    p("**상담사 점수 출처 재확인**: `agent_judge_run.py`가 `d04_dialog_index.parquet` "
      "전사 텍스트만으로 조립한 프롬프트로 산출. `gold_actual`/`mismatch` 컬럼 미접근, "
      "전사 밖 카테고리명 등장 0건(자동 검증). 화자 식별은 `speaker_type` 접두만 사용.")
    p("")
    p("**A/B 예측 출처**: `modality_ab_percall.parquet` — 조건 A(customer-only, "
      "text-only) / B(customer-only, text+acoustic), 둘 다 gpt-5.4 / "
      "`reasoning.effort=low`, 동일 250 call_id. 기존 예측 재사용(신규 호출 없음).")
    p("")

    p("## 층화 축과 분할 기준")
    p("")
    p(f"층화축 = **{AXIS}**. 친절도는 250건 중 238건(95%)이 3~4에 눌려 있어 "
      "(5점 2건, 1점 0건, std 0.587) 축에서 제외하고 보조 표기만 한다.")
    p("")
    p(f"{AXIS} 분포: " + ", ".join(f"{k}점 {int((v == k).sum())}건" for k in range(1, 6))
      + f" (중앙값 {v.median():.0f})")
    p("")
    p("| 기준 | 잘 대응 | 못 대응 | 비율 | 채택 |")
    p("|---|---|---|---|---|")
    n_g2 = int((v > v.median()).sum())
    p(f"| (기준1) 4~5=잘 / 1~3=못 | {n_good} | {n_bad} | "
      f"{n_good/len(df)*100:.1f} : {n_bad/len(df)*100:.1f} | **채택** |")
    p(f"| (기준2) >중앙값({v.median():.0f})=잘 | {n_g2} | {len(df)-n_g2} | "
      f"{n_g2/len(df)*100:.1f} : {(len(df)-n_g2)/len(df)*100:.1f} | 기각(쏠림) |")
    p(f"| (참고) ≥중앙값=잘 | {int((v >= v.median()).sum())} | "
      f"{int((v < v.median()).sum())} | — | 기준1과 동일 |")
    p("")
    p("정수 5점 척도이고 중앙값이 4라 중앙값 분할이 독립 기준이 되지 못한다"
      "(`≥`는 기준1과 같고 `>`는 5점만 남아 16:84로 쏠린다). 기준1을 쓴다.")
    p("")
    p(f"보조: 그룹별 친절도 평균 — {GOOD} "
      f"{df.loc[df['그룹'] == GOOD, '친절도'].mean():.3f} / {BAD} "
      f"{df.loc[df['그룹'] == BAD, '친절도'].mean():.3f}")
    p("")

    # ---------- 아웃풋 ① ----------
    p("## 아웃풋 ① 그룹 × 불만 / mismatch 비율")
    p("")
    p("| 그룹(적절대응) | n | 불만 건수 | 불만비율 | mismatch 건수 | mismatch비율 |")
    p("|---|---|---|---|---|---|")
    stats = {}
    for g in GROUP_ORDER:
        sub = df[df["그룹"] == g]
        n_c = int(sub["is_불만"].sum())
        n_m = int(sub["gold_mismatch"].sum())
        stats[g] = {"n": len(sub), "c": n_c, "m": n_m}
        p(f"| {g} | {len(sub)} | {n_c} | {n_c/len(sub)*100:.1f}% | "
          f"{n_m} | {n_m/len(sub)*100:.1f}% |")
    p("")
    for label, key in [("불만비율", "c"), ("mismatch비율", "m")]:
        test, pv = prop_test(stats[GOOD][key], stats[GOOD]["n"],
                             stats[BAD][key], stats[BAD]["n"])
        sig = "유의(p<0.05)" if pv < 0.05 else "유의하지 않음"
        p(f"- {label} 그룹 간 차이: {test}, p = {pv:.4f} — {sig}")
    p("")
    p("> **해석 가드**: 위는 연관일 뿐 인과가 아니다. \"고객이 불만 상태라 상담사가 "
      "제대로 대응하기 어려웠다\"와 \"상담사가 제대로 대응하지 못해 고객이 불만이 "
      "됐다\"는 이 데이터로 구분할 수 없다. 방향을 주장하지 않는다.")
    p("")

    # ---------- 아웃풋 ② ----------
    p("## 아웃풋 ② 그룹 × acoustic 효과 (고객 의도 A vs B)")
    p("")
    p("| 그룹 | n | 불만 support | A(text) macroF1 | B(text+ac) macroF1 | B−A(macroF1) | "
      "A 불만recall | B 불만recall | B−A(불만recall) |")
    p("|---|---|---|---|---|---|---|---|---|")
    rows = {}
    for g in GROUP_ORDER:
        sub = df[df["그룹"] == g]
        y = sub["gold_label"]
        supp = int((y == COMPLAINT).sum())
        vals = {}
        for cond, col in [("A", "pred_A"), ("B", "pred_B")]:
            vals[f"{cond}_f1"] = f1_score(y, sub[col], labels=CATEGORIES,
                                          average="macro", zero_division=0)
            vals[f"{cond}_rec"] = recall_score((y == COMPLAINT).astype(int),
                                               (sub[col] == COMPLAINT).astype(int),
                                               zero_division=0)
        rows[g] = {"n": len(sub), "supp": supp, **vals}
        p(f"| {g} | {len(sub)} | {supp} | {vals['A_f1']:.3f} | {vals['B_f1']:.3f} | "
          f"{vals['B_f1']-vals['A_f1']:+.3f} | {vals['A_rec']:.3f} | {vals['B_rec']:.3f} | "
          f"{vals['B_rec']-vals['A_rec']:+.3f} |")
    p("")

    p("### ⚠️ 불만 support 경고")
    p("")
    for g in GROUP_ORDER:
        r = rows[g]
        n_hit_a = int(round(r["A_rec"] * r["supp"]))
        n_hit_b = int(round(r["B_rec"] * r["supp"]))
        warn = " **← 30건 미만, 노이즈 지배**" if r["supp"] < MIN_SUPPORT_WARN else ""
        p(f"- {g}: 불만 support **{r['supp']}건**{warn}. "
          f"불만 recall은 A {n_hit_a}건 / B {n_hit_b}건 맞힌 것에 해당 — "
          f"1건 차이가 recall {1/r['supp']:.3f}를 움직인다.")
    p("")
    p(f"전체 250건의 불만 50건이 두 그룹으로 갈리면서 각 그룹 support가 "
      f"{rows[GOOD]['supp']} / {rows[BAD]['supp']}건이 됐다. "
      "아래 방향 판독은 **참고용**이며 유의성 주장은 하지 않는다.")
    p("")

    p("### 상호작용 방향 (참고용)")
    p("")
    d_f1 = {g: rows[g]["B_f1"] - rows[g]["A_f1"] for g in GROUP_ORDER}
    d_rec = {g: rows[g]["B_rec"] - rows[g]["A_rec"] for g in GROUP_ORDER}
    for label, d in [("macro-F1", d_f1), ("불만 recall", d_rec)]:
        same = (d[GOOD] >= 0) == (d[BAD] >= 0)
        p(f"- B−A({label}): {GOOD} {d[GOOD]:+.3f} / {BAD} {d[BAD]:+.3f} — "
          f"두 그룹 부호 {'같음' if same else '다름'}, "
          f"차이 {abs(d[GOOD]-d[BAD]):.3f}")
    p("")
    p("> 상담사품질 × 모달리티 상호작용의 **방향만** 읽는다. 각 그룹 불만 support가 "
      "얇아 유의성 검정은 하지 않았다.")
    p("")

    out = df[["call_id", AXIS, "친절도", "그룹", "gold_label", "gold_mismatch",
              "is_불만", "pred_A", "pred_B"]]
    out.to_parquet(PERCALL_PATH, index=False)
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\n저장: {MD_PATH}")
    print(f"저장: {PERCALL_PATH} ({len(out)}행)")
    print("AGENT_STRAT_COMPLETE")


if __name__ == "__main__":
    main()
