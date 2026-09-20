"""어휘 애매성 그룹별 음성 효과 파일럿 — 기존 예측 재집계 (추론 재실행 없음).

원래 질문: "어휘로 불만 여부를 판단하기 어려운 콜에서는, 음성을 추가하면
텍스트만보다 더 나은가?" 층별 probe에서 불만이 어휘(디코더)에 실린 것으로
나와, 음성 기여가 작았던 게 "어휘가 이미 답을 알아서"일 수 있다는 반박을
검증한다. 가설: Δ_ambiguous > Δ_clear (Δ = Text+Audio − Text-only).

핵심 설계 — 새 추론 없음. n=1 게이트에서 이미 돌린 Text-only(modality=text)
/ Text+Audio(modality=both) 800콜 예측(outputs/qwen3_30b_nsweep_800.parquet,
n=1·prompt=gate 서브셋)을 재사용한다. 800콜이 expand800_call_ids.csv와
완전 일치함을 확인했다(circularity 유지: eval 콜 자신의 gold는 애매성
판정에 쓰지 않는다 — 애매성은 paper_lex_ambig_pilot.neighbor_search 가
reference 이웃만으로 계산한 S_c/S_n/A 그대로 재사용).

★ 인사말 제거: 공백제외 12자 미만 콜은 별도 short 트랙으로 빼고 메인
  분석(ambiguous/clear)에서 제외한다 — 이전 파일럿(A 랭킹)에서 인사말이
  상위를 오염시켰던 것과 같은 이유.
★ 항의표현 필터: ambiguous 후보 중 첫 발화에 이미 명백한 불만 어휘가 있는
  콜은 "순수 중립"이 아니므로 플래그만 하고(자동 제거는 안 함), 원문을
  전부 출력해 사용자가 육안으로 최종 판단하게 한다. 키워드는 감정/문제
  어휘로만 구성했고 환불/배송/주문 같은 순수 토픽어는 넣지 않았다 —
  "환불 여부 ≠ 불만 여부"가 이 연구의 정의이기 때문이다.

이번 단계는 파일럿이다. threshold(τ,δ) 확정, 유의성 검정, human
validation, eval 확장은 다음 단계. 여기서는 점추정 + 방향만 본다.

실행: python scripts/paper_ambig_speech_effect_pilot.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_lex_ambig_pilot as LEX  # noqa: E402  (build_texts/get_device/embed/neighbor_search)

BASE = LEX.BASE
OUT = LEX.OUT
NSWEEP_PATH = BASE / "outputs" / "qwen3_30b_nsweep_800.parquet"

COMPLAINT = "불만제기"
SHORT_LEN = 12              # 공백제외 글자수 하한 — 이 미만은 short 트랙
N_CANDIDATES = [75, 100, 150]
N_PRIMARY = 150              # 두 그룹 다 불만 >=20건을 만족하는 값으로 선택(분포 보고 확정)
MIN_COMPLAINT_WARN = 20

# 항의표현 키워드(1차 자동 필터). 감정·문제 어휘만 — 환불/배송/주문 등
# 순수 토픽어는 의도적으로 뺐다(그건 "불만 여부"가 아니라 "용건"이므로).
COMPLAINT_KEYWORDS = [
    "안 왔", "안왔", "안 되", "안돼", "안됩니다", "잘못", "왜 이렇게", "왜 자꾸",
    "화나", "화가", "짜증", "답답", "불편", "황당", "어이없", "여러 번",
    "몇 번이나", "몇번이나", "말이 다르", "이상하", "문제가", "항의", "실망",
    "어렵", "힘들", "불만",
]


# ──────────────────────────────────────────────────────────── 태깅 ────
def tag_groups():
    """eval 800콜 각각에 S_c/S_n/A(reference 이웃 기반) + 길이 + 그룹."""
    eval_df, ref_df = LEX.build_texts()
    eval_df["len_nospace"] = eval_df.text.str.replace(r"\s+", "", regex=True).str.len()

    device = LEX.get_device()
    eval_emb = LEX.embed("kure", LEX.MODELS["kure"], eval_df, "eval", device)
    ref_emb = LEX.embed("kure", LEX.MODELS["kure"], ref_df, "ref", device)
    res = LEX.neighbor_search(eval_df, eval_emb, ref_df, ref_emb)
    res = res.merge(eval_df[["call_id", "len_nospace", "label", "is_complaint"]], on="call_id")

    res["is_short"] = res.len_nospace < SHORT_LEN
    pat = "|".join(re.escape(k) for k in COMPLAINT_KEYWORDS)
    res["has_complaint_expr"] = res.eval_text.str.contains(pat, regex=True)

    rest = res[~res.is_short].copy()
    # 상위 N개의 컷오프 = 상위 N개 중 최솟값(A >= 컷오프로 상위 N개 포함).
    # 하위 N개의 컷오프 = 하위 N개 중 최댓값(A <= 컷오프로 하위 N개 포함) —
    # 여기를 .min()으로 잘못 쓰면 컷오프가 전체 최솟값이 되어 하위 그룹이
    # 사실상 1건(동률만)으로 무너진다. 실제로 그렇게 터져서 잡은 버그.
    hi_cut = rest.A.nlargest(N_PRIMARY).min()
    lo_cut = rest.A.nsmallest(N_PRIMARY).max()
    res["group"] = "excluded"
    res.loc[res.is_short, "group"] = "short"
    res.loc[(~res.is_short) & (res.A >= hi_cut), "group"] = "ambiguous"
    res.loc[(~res.is_short) & (res.A <= lo_cut), "group"] = "clear"
    return res.rename(columns={"label": "gold"})


def n_sweep_report(res):
    """N 후보별 그룹 구성 — 최종 N 선택 근거."""
    rest = res[~res.is_short]
    rows = []
    for N in N_CANDIDATES:
        hi = rest.nlargest(N, "A")
        lo = rest.nsmallest(N, "A")
        rows.append({"N": N,
                    "ambiguous_n": len(hi), "ambiguous_complaint": int(hi.is_complaint.sum()),
                    "ambiguous_rate": hi.is_complaint.mean(),
                    "clear_n": len(lo), "clear_complaint": int(lo.is_complaint.sum()),
                    "clear_rate": lo.is_complaint.mean()})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────── 예측 결합 ────
def load_predictions():
    """n=1·gate 서브셋의 text/both 예측. 800/800 paired 검증 포함."""
    d = pd.read_parquet(NSWEEP_PATH)
    sub = d[(d.n == 1) & (d.prompt == "gate") & (d.modality.isin(["text", "both"]))]
    assert sub.is_failure.sum() == 0, "예측 실패 건 존재 — 재확인 필요"
    wide = sub.pivot(index="call_id", columns="modality", values="pred").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"text": "text_pred", "both": "audio_pred"})
    gold_map = sub.drop_duplicates("call_id").set_index("call_id")["gold"]
    assert wide.call_id.map(gold_map).notna().all()
    return wide, gold_map


def metrics_for(pred, gold):
    """이진(불만 vs 비불만) F1/recall/precision/FPR."""
    yp = (pred == COMPLAINT).astype(int)
    yg = (gold == COMPLAINT).astype(int)
    non_idx = yg == 0
    fpr = float(yp[non_idx].mean()) if non_idx.sum() else np.nan
    return {
        "n": len(gold), "n_complaint": int(yg.sum()),
        "F1": f1_score(yg, yp, zero_division=0),
        "recall": recall_score(yg, yp, zero_division=0),
        "precision": precision_score(yg, yp, zero_division=0),
        "FPR": fpr,
    }


def delta_table(res, wide, gold_map):
    """그룹(ambiguous/clear/short) × 모달리티(text/audio=both) 지표 + Δ."""
    rows = []
    for group in ["ambiguous", "clear", "short"]:
        ids = res.loc[res.group == group, "call_id"]
        sub = wide[wide.call_id.isin(ids)]
        gold = sub.call_id.map(gold_map)
        m_text = metrics_for(sub.text_pred, gold)
        m_audio = metrics_for(sub.audio_pred, gold)
        for modality, m in [("text_only", m_text), ("text_audio", m_audio)]:
            rows.append({"group": group, "modality": modality, **m})
    tab = pd.DataFrame(rows)

    delta_rows = []
    for group in ["ambiguous", "clear", "short"]:
        t = tab[(tab.group == group) & (tab.modality == "text_only")].iloc[0]
        a = tab[(tab.group == group) & (tab.modality == "text_audio")].iloc[0]
        delta_rows.append({"group": group, "n": t.n, "n_complaint": t.n_complaint,
                          "dF1": a.F1 - t.F1, "dRecall": a.recall - t.recall,
                          "dPrecision": a.precision - t.precision, "dFPR": a.FPR - t.FPR})
    delta = pd.DataFrame(delta_rows)
    return tab, delta


def main():
    print("=" * 72)
    print("1. 그룹 태깅 (reference 이웃 기반 S_c/S_n/A, eval 자신의 gold 미사용)")
    print("=" * 72)
    res = tag_groups()
    res.to_parquet(OUT / "ambig_speech_800_tagged.parquet", index=False)

    print(f"\nshort(길이<{SHORT_LEN}) 트랙: {(res.group=='short').sum()}건")
    sweep = n_sweep_report(res)
    sweep.to_csv(OUT / "table_ambig_speech_N_sweep.csv", index=False, encoding="utf-8-sig")
    print(f"\nN 후보별 그룹 구성 (short 제외 {(~res.is_short).sum()}콜 중):")
    print(sweep.round(3).to_string(index=False))
    print(f"\n선택된 N={N_PRIMARY} (두 그룹 다 불만 >={MIN_COMPLAINT_WARN}건 만족하는 최소값)")

    print("\n" + "=" * 72)
    print("2. 그룹 구성 진단")
    print("=" * 72)
    diag = (res[res.group.isin(["ambiguous", "clear", "short"])]
            .groupby("group").agg(n=("call_id", "size"), n_complaint=("is_complaint", "sum"))
            .reset_index())
    diag["complaint_rate"] = diag.n_complaint / diag.n
    diag.to_csv(OUT / "table_ambig_speech_group_diag.csv", index=False, encoding="utf-8-sig")
    print(diag.round(4).to_string(index=False))
    for _, r in diag.iterrows():
        if r.group != "short" and r.n_complaint < MIN_COMPLAINT_WARN:
            print(f"  ⚠️ {r.group} 그룹 불만 {int(r.n_complaint)}건 < {MIN_COMPLAINT_WARN} — Δ 재현율 불안정")
    amb_rate = diag.loc[diag.group == "ambiguous", "complaint_rate"].iloc[0]
    clr_rate = diag.loc[diag.group == "clear", "complaint_rate"].iloc[0]
    print(f"\n불만 비율 ambiguous {amb_rate:.1%} vs clear {clr_rate:.1%} "
          f"(차이 {abs(amb_rate-clr_rate):.1%}p) — 크면 Δ 비교가 교란될 수 있음")

    print("\n" + "=" * 72)
    print("3. Δ 재집계 (n=1·gate, Text-only vs Text+Audio, 새 추론 없음)")
    print("=" * 72)
    wide, gold_map = load_predictions()
    tab, delta = delta_table(res, wide, gold_map)
    tab.round(4).to_csv(OUT / "table_ambig_speech_metrics.csv", index=False, encoding="utf-8-sig")
    delta.round(4).to_csv(OUT / "table_ambig_speech_delta.csv", index=False, encoding="utf-8-sig")
    print("\n지표 (그룹 × 모달리티):")
    print(tab.round(4).to_string(index=False))
    print("\nΔ = Text+Audio − Text-only:")
    print(delta.round(4).to_string(index=False))

    d_amb = delta.loc[delta.group == "ambiguous", "dF1"].iloc[0]
    d_clr = delta.loc[delta.group == "clear", "dF1"].iloc[0]
    print(f"\nΔF1_ambiguous − ΔF1_clear = {d_amb - d_clr:+.4f} "
          f"({'가설 지지 방향' if d_amb > d_clr else '가설 반대 방향'} — 점추정, 유의성 검정 안 함)")
    d_amb_r = delta.loc[delta.group == "ambiguous", "dRecall"].iloc[0]
    d_clr_r = delta.loc[delta.group == "clear", "dRecall"].iloc[0]
    print(f"ΔRecall_ambiguous − ΔRecall_clear = {d_amb_r - d_clr_r:+.4f}")

    print("\n" + "=" * 72)
    print("4. ambiguous 그룹 — 항의표현 플래그 (자동 1차, 육안 확정용)")
    print("=" * 72)
    amb = res[res.group == "ambiguous"].sort_values("A", ascending=False)
    flagged = amb[amb.has_complaint_expr]
    flagged[["call_id", "eval_text", "A", "S_c", "S_n", "gold", "is_complaint"]].to_csv(
        OUT / "ambig_speech_complaint_expr_flagged.csv", index=False, encoding="utf-8-sig")
    print(f"\nambiguous {len(amb)}건 중 항의표현 플래그 {len(flagged)}건 "
          f"({len(flagged)/len(amb)*100:.1f}%) — 제거하면 {len(amb)-len(flagged)}건 남음")
    print("플래그 리스트(전문):")
    pd.set_option("display.max_colwidth", 60)
    print(flagged[["call_id", "eval_text", "gold"]].to_string(index=False))

    print("\n" + "=" * 72)
    print("산출물")
    print("=" * 72)
    for f in ["ambig_speech_800_tagged.parquet", "table_ambig_speech_N_sweep.csv",
             "table_ambig_speech_group_diag.csv", "table_ambig_speech_metrics.csv",
             "table_ambig_speech_delta.csv", "ambig_speech_complaint_expr_flagged.csv"]:
        print(f"  outputs/paper/{f}")
    print("\n다음: ambig_speech_complaint_expr_flagged.csv 를 육안 확인해 순수중립만")
    print("      남기고, 그 부분집합으로 §3 Δ를 다시 계산해 보는 것을 권함(선택).")
    print("      τ·δ 확정, 유의성 검정, human validation은 다음 단계(범위 밖).")
    print("\nPAPER_AMBIG_SPEECH_EFFECT_PILOT_COMPLETE")


if __name__ == "__main__":
    main()
