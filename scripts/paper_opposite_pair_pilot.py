"""opposite-label pair 파일럿 — reference 내 "결말이 갈리는" 발화 탐색 (2트랙).

앞 파일럿(paper_lex_ambig_pilot.py, A=S_min−ΔS 랭킹)이 무정보 인사말과
"환불" 토픽에 쏠렸다. 이번엔 정의를 더 엄격하게 좁히되 두 종류로 나눈다:

  트랙 B (context_dependent): 어휘는 충분한데 결말이 갈리는 쌍
    reference의 complaint 발화 하나하나에 대해, non 발화 중 가장 가까운
    (cosine 최댓값) 1개를 짝짓는다. 양쪽 다 길이 하한 이상이어야
    "어휘가 충분하다"는 트랙 정의가 성립하므로, comp·non 풀 둘 다 같은
    길이 하한으로 거른다.

  트랙 A (underspecified): 어휘 정보가 거의 없는데도 결말이 갈리는 발화
    길이 하한 미만 풀에 같은 매칭 로직을 적용(재사용) + 대표 무정보
    발화(빈도 상위, exact-text 그룹)별 gold 분포를 별도로 본다.

circularity: reference(약 18.7k, eval 800콜 제외)만 쓴다. eval은 건드리지
않는다 — 그래서 build_texts()가 돌려주는 eval_df 는 이 스크립트에서
로드만 하고 참조하지 않는다(assert 검증용으로만 쓰임).

이번 단계는 "정의가 데이터에서 성립하는가, 규모가 얼마인가"만 본다.
τ·δ 확정, eval 매칭, 음성 실험은 다음 단계.

임베딩: KURE-v1(primary). 이전 파일럿이 만든 reference 임베딩 캐시
(outputs/paper/lex_ambig_emb_kure_ref.npy)를 그대로 재사용한다 —
build_texts()가 결정적이라 ref_df 순서가 동일하면 캐시가 그대로 맞는다
(스크립트가 로드 시 call_id 순서를 직접 검증한다). KoSimCSE는 각 트랙
상위 몇 건의 교차확인용으로만(둘 다 이미 인코딩돼 있어 재계산 없음).

실행: python scripts/paper_opposite_pair_pilot.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_lex_ambig_pilot as LEX  # noqa: E402  (build_texts/get_device/embed 재사용)

BASE = LEX.BASE
OUT = LEX.OUT

LEN_CANDIDATES = [8, 10, 12, 15]
PRIMARY_LEN = 12          # 트랙 B/A를 가르는 기본 하한(task 예시값)
TOP_B = 40                 # 트랙 B 상위 원문 테이블 건수
TOP_A_PAIR = 15            # 트랙 A pair 상위 건수(KoSimCSE 교차확인 대상)
TOP_A_TYPE = 15            # 트랙 A 대표 무정보 발화 유형 개수
SIM_BINS = [0.95, 0.90, 0.85]
N_CROSSCHECK = 15          # 각 트랙에서 KoSimCSE로 재확인할 상위 건수


# ──────────────────────────────────────────────────────────── 매칭 ────
def find_opposite_pairs(ref_df, ref_emb, min_len, max_len=None):
    """comp/non 각각 길이 조건을 만족하는 풀에서, comp 발화마다 non 중
    cosine 최댓값 1개를 짝짓는다. min_len<=len(<max_len) 구간.
    """
    len_mask = ref_df.len_nospace >= min_len
    if max_len is not None:
        len_mask &= ref_df.len_nospace < max_len
    comp_idx = np.where(len_mask & (ref_df.is_complaint == 1))[0]
    non_idx = np.where(len_mask & (ref_df.is_complaint == 0))[0]
    if len(comp_idx) == 0 or len(non_idx) == 0:
        return pd.DataFrame(columns=["comp_call_id", "comp_text", "non_call_id",
                                     "non_text", "non_gold", "sim", "exact_dup"])

    sim = ref_emb[comp_idx] @ ref_emb[non_idx].T     # (n_comp, n_non)
    best_j = sim.argmax(axis=1)
    best_sim = sim[np.arange(len(comp_idx)), best_j]

    rows = []
    for k in range(len(comp_idx)):
        ci, ni = comp_idx[k], non_idx[best_j[k]]
        ct, nt = ref_df.text.iat[ci], ref_df.text.iat[ni]
        rows.append({
            "comp_call_id": ref_df.call_id.iat[ci], "comp_text": ct,
            "non_call_id": ref_df.call_id.iat[ni], "non_text": nt,
            "non_gold": ref_df.label.iat[ni],
            "sim": float(best_sim[k]), "exact_dup": ct == nt,
        })
    return pd.DataFrame(rows)


def length_sweep(ref_df, ref_emb, lens):
    """길이 하한별 쌍 수·sim 구간별 개수·완전동일 쌍 수 변화."""
    rows = []
    for L in lens:
        pairs = find_opposite_pairs(ref_df, ref_emb, min_len=L)
        row = {"min_len": L, "n_pairs": len(pairs),
              "n_exact_dup": int(pairs.exact_dup.sum()) if len(pairs) else 0}
        for s in SIM_BINS:
            row[f"n_sim_ge_{int(s*100)}"] = int((pairs.sim >= s).sum()) if len(pairs) else 0
        rows.append(row)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────── 트랙 A ────
def underspecified_gold_dist(ref_df, max_len, top_n):
    """길이 하한 미만 풀 — 대표 발화(exact-text)별 gold 분포 + 전체 규모."""
    short = ref_df[ref_df.len_nospace < max_len]
    g = (short.groupby("text")
         .agg(n_calls=("call_id", "size"), n_complaint=("is_complaint", "sum"))
         .reset_index())
    g["n_non"] = g.n_calls - g.n_complaint
    g["both_sides"] = (g.n_complaint > 0) & (g.n_non > 0)   # gold가 실제로 양쪽에 갈리는가
    g = g.sort_values("n_calls", ascending=False).head(top_n).reset_index(drop=True)

    scale = {"min_len_threshold": max_len,
            "n_calls_underspecified": len(short),
            "pct_of_reference": len(short) / len(ref_df) * 100,
            "n_complaint_underspecified": int(short.is_complaint.sum()),
            "n_unique_texts": short.text.nunique()}
    return g, scale


# ──────────────────────────────────────────────────────────── KoSimCSE ────
def kosimcse_crosscheck(pairs_top, model_key="kosimcse"):
    """트랙 pair 상위 건의 comp/non call_id를 KoSimCSE 임베딩에서 찾아
    같은 쌍의 cosine을 재계산. 캐시가 있으면 재인코딩 없음.
    """
    ids = np.load(OUT / f"lex_ambig_ids_{model_key}_ref.npy", allow_pickle=True)
    emb = np.load(OUT / f"lex_ambig_emb_{model_key}_ref.npy")
    pos = {cid: i for i, cid in enumerate(ids)}

    sims = []
    for r in pairs_top.itertuples():
        ci, ni = pos.get(r.comp_call_id), pos.get(r.non_call_id)
        if ci is None or ni is None:
            sims.append(np.nan)
            continue
        sims.append(float(emb[ci] @ emb[ni]))
    out = pairs_top.copy()
    out[f"sim_{model_key}"] = sims
    out["sim_delta"] = out[f"sim_{model_key}"] - out["sim"]
    return out


def main():
    print("=" * 72)
    print("1. 데이터 로드 (reference만 — eval 800콜은 참조하지 않는다)")
    print("=" * 72)
    eval_df, ref_df = LEX.build_texts()
    del eval_df   # circularity: 이 스크립트에서 쓰지 않는다(로드만, 참조 금지)
    ref_df["len_nospace"] = ref_df.text.str.replace(r"\s+", "", regex=True).str.len()

    device = LEX.get_device()
    ref_emb = LEX.embed("kure", LEX.MODELS["kure"], ref_df, "ref", device)
    print(f"reference 임베딩 {ref_emb.shape} (캐시 재사용됨 — 재인코딩 없음)")

    print("\n" + "=" * 72)
    print("2. 트랙 B — context_dependent (opposite-label pair)")
    print("=" * 72)
    sweep = length_sweep(ref_df, ref_emb, LEN_CANDIDATES)
    sweep.to_csv(OUT / "table_opp_trackB_length_sweep.csv", index=False, encoding="utf-8-sig")
    print(f"\n길이 하한별 쌍 수 / sim 구간별 개수 (기준: complaint {ref_df.is_complaint.sum()}개 중):")
    print(sweep.to_string(index=False))

    pairs_b = find_opposite_pairs(ref_df, ref_emb, min_len=PRIMARY_LEN)
    print(f"\n기본 하한 L={PRIMARY_LEN}: 쌍 {len(pairs_b)}개, "
          f"완전동일 {int(pairs_b.exact_dup.sum())}개")

    top_b = pairs_b.sort_values("sim", ascending=False).head(TOP_B).reset_index(drop=True)
    for c in ["truly_opposite", "topic", "comp_gold_ok", "non_gold_ok"]:
        top_b[c] = ""
    top_b.to_csv(OUT / "pilot_opp_pairB_top40.csv", index=False, encoding="utf-8-sig")
    print(f"저장: pilot_opp_pairB_top40.csv ({len(top_b)}행)")

    print("\n" + "=" * 72)
    print("3. 트랙 A — underspecified (어휘 0인데 결말 갈림)")
    print("=" * 72)
    gold_dist, scale = underspecified_gold_dist(ref_df, PRIMARY_LEN, TOP_A_TYPE)
    gold_dist.to_csv(OUT / "table_opp_trackA_gold_dist.csv", index=False, encoding="utf-8-sig")
    print(f"\n트랙 A 규모(길이<{PRIMARY_LEN}): {scale['n_calls_underspecified']:,}콜 "
          f"({scale['pct_of_reference']:.1f}% of reference), "
          f"그중 complaint {scale['n_complaint_underspecified']}건, "
          f"고유 텍스트 {scale['n_unique_texts']:,}종")
    print(f"\n대표 무정보 발화 top-{TOP_A_TYPE} (both_sides=True면 같은 발화가 "
          f"complaint·non 양쪽 콜에 다 나타남):")
    print(gold_dist.to_string(index=False))
    pd.DataFrame([scale]).to_csv(OUT / "table_opp_trackA_scale.csv", index=False,
                                 encoding="utf-8-sig")

    pairs_a = find_opposite_pairs(ref_df, ref_emb, min_len=0, max_len=PRIMARY_LEN)
    top_a = pairs_a.sort_values("sim", ascending=False).head(TOP_A_PAIR).reset_index(drop=True)
    for c in ["truly_opposite", "topic", "comp_gold_ok", "non_gold_ok"]:
        top_a[c] = ""
    top_a.to_csv(OUT / "pilot_opp_pairA_top15.csv", index=False, encoding="utf-8-sig")
    print(f"\n트랙 A pair(길이<{PRIMARY_LEN} 풀 안에서 동일 매칭): {len(pairs_a)}쌍, "
          f"상위 {len(top_a)}건 저장 (완전동일 {int(pairs_a.exact_dup.sum())}/{len(pairs_a)})")

    print("\n" + "=" * 72)
    print(f"4. KoSimCSE 교차확인 (각 트랙 상위 {N_CROSSCHECK}건)")
    print("=" * 72)
    cc_b = kosimcse_crosscheck(top_b.head(N_CROSSCHECK))
    cc_b.to_csv(OUT / "pilot_opp_pairB_top40_kosimcse_check.csv", index=False,
               encoding="utf-8-sig")
    print(f"\n트랙 B 상위 {N_CROSSCHECK}건: KURE·KoSimCSE sim 평균 "
          f"{cc_b.sim.mean():.4f} vs {cc_b.sim_kosimcse.mean():.4f} "
          f"(Δ평균 {cc_b.sim_delta.mean():+.4f})")

    cc_a = kosimcse_crosscheck(top_a.head(N_CROSSCHECK))
    cc_a.to_csv(OUT / "pilot_opp_pairA_top15_kosimcse_check.csv", index=False,
               encoding="utf-8-sig")
    print(f"트랙 A 상위 {min(N_CROSSCHECK, len(top_a))}건: KURE·KoSimCSE sim 평균 "
          f"{cc_a.sim.mean():.4f} vs {cc_a.sim_kosimcse.mean():.4f} "
          f"(Δ평균 {cc_a.sim_delta.mean():+.4f})")

    print("\n" + "=" * 72)
    print("산출물")
    print("=" * 72)
    for f in ["table_opp_trackB_length_sweep.csv", "pilot_opp_pairB_top40.csv",
             "table_opp_trackA_gold_dist.csv", "table_opp_trackA_scale.csv",
             "pilot_opp_pairA_top15.csv",
             "pilot_opp_pairB_top40_kosimcse_check.csv",
             "pilot_opp_pairA_top15_kosimcse_check.csv"]:
        print(f"  outputs/paper/{f}")
    print("\n다음: pilot_opp_pairB_top40.csv 의 truly_opposite/topic/comp_gold_ok/")
    print("      non_gold_ok 를 채워 τ·δ 확정 근거로 쓸 것. eval 콜 매칭·음성")
    print("      실험은 다음 단계(이 스크립트 범위 밖).")
    print("\nPAPER_OPPOSITE_PAIR_PILOT_COMPLETE")


if __name__ == "__main__":
    main()
