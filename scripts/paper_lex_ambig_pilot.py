"""어휘 애매성(lexical ambiguity) subset 파일럿 — 임베딩 이웃 탐색 + 50건 수동검토.

배경: 층별 probe에서 불만 정보가 인코더(음향)보다 디코더(어휘)에서 더 크게
나타났다 — 이 과제는 전역적으로 어휘 의존이 크다는 뜻이다. 이 파일럿은
"어휘로도 판단이 애매한 지역"을 데이터에서 찾아, 그 지역에서 음성이 추가정보를
주는지 이후 검증하기 위한 선행 단계다. 이번 목표는 subset 확정이 아니라
"이 정의가 우리 데이터에서 먹히는지" 감을 잡는 것.

circularity 차단 (절대 지킬 것):
  - eval 800콜 "자신의 gold"는 ambiguity 판정에 절대 쓰지 않는다.
    (그래서 review CSV에 eval 콜의 gold를 아예 넣지 않았다 — 사람이 라벨링할 때
     eval gold를 보고 앵커링되는 것 자체를 구조적으로 차단)
  - reference = eval 800콜을 제외한 나머지(gold ∩ n=1 유효, 약 18.7k).
  - 각 eval 콜의 초기발화 텍스트만으로 reference에서 이웃을 찾는다.

데이터: 고객 발화만 / r_gold_valid_flag==True / dialog_idx 최소=첫 발화(n=1).
§3.1 파이프라인과 동일 규칙(paper_a7_acoustic_only.py 등과 동일 소스).

임베딩 2종:
  - primary: KURE-v1 (nlpai-lab/KURE-v1, 1024d, 한국어 retrieval)
  - 비교군: KoSimCSE-roberta-multitask (BM-K/KoSimCSE-roberta-multitask, 768d)
  둘 다 동일 파이프라인으로 돌려 상위 이웃 품질을 수동으로 비교한다.

sim(c,n) 정의(명시): 뽑힌 complaint top-3 이웃과 non-complaint top-3 이웃의
  교차 3×3=9쌍 cosine 평균. top-1끼리보다 노이즈에 덜 민감해서 택함.

A = S_min − ΔS 는 탐색적 순위용일 뿐이다. 어휘 애매성의 operational 정의
(S_min ≥ τ AND ΔS ≤ δ)는 이번 파일럿 분포를 보기 전에 τ·δ를 고정하지 않는다
— 이번엔 계산만 하고 확정 안 함.

실행: python scripts/paper_lex_ambig_pilot.py
재실행 시 임베딩은 캐시(outputs/paper/lex_ambig_emb_*.npy)를 재사용한다.
GPU 불필요(Mac MPS 사용, ~19,548건 인코딩 ~4분/모델). API 호출 0.
"""

import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

D04_PATH = BASE / "outputs" / "d04_dialog_index.parquet"
GOLD_PATH = BASE / "outputs" / "gold_actual_batch1_final.parquet"
EVAL_IDS_PATH = BASE / "outputs" / "expand800_call_ids.csv"

COMPLAINT = "불만제기"
CUSTOMER_PREFIX = "고객"
SEED = 42
TOP_K = 3
N_REVIEW = 50
N_OVERLAP = 30

MODELS = {
    "kure": "nlpai-lab/KURE-v1",
    "kosimcse": "BM-K/KoSimCSE-roberta-multitask",
}
PRIMARY = "kure"
BATCH_SIZE = 64


# ──────────────────────────────────────────────────────────── 데이터 ────
def build_texts():
    """eval(800) / reference(~18.7k) 텍스트+라벨 데이터프레임.

    §3.1과 동일: 고객 발화만, r_gold_valid_flag==True, dialog_idx 오름차순
    첫 번째(n=1)만. gold는 gold_actual_batch1_final(최종본).
    """
    d04 = pd.read_parquet(D04_PATH, columns=[
        "call_id", "dialog_idx", "speaker_type", "r_gold_valid_flag", "text_clean"])
    sub = d04[d04.speaker_type.astype(str).str.startswith(CUSTOMER_PREFIX)
              & (d04.r_gold_valid_flag == True)].sort_values(["call_id", "dialog_idx"])  # noqa: E712
    sub["rank"] = sub.groupby("call_id").cumcount() + 1
    n1 = sub[sub["rank"] == 1][["call_id", "text_clean"]].rename(columns={"text_clean": "text"})

    n_empty = int((n1.text.str.strip() == "").sum())
    n1 = n1[n1.text.str.strip() != ""].copy()

    gold = pd.read_parquet(GOLD_PATH)[["call_id", "label"]]
    merged = n1.merge(gold, on="call_id", how="inner")
    merged["is_complaint"] = (merged.label == COMPLAINT).astype(int)

    eval_ids = set(pd.read_csv(EVAL_IDS_PATH)["call_id"])
    eval_df = merged[merged.call_id.isin(eval_ids)].reset_index(drop=True)
    ref_df = merged[~merged.call_id.isin(eval_ids)].reset_index(drop=True)

    assert len(eval_ids - set(eval_df.call_id)) == 0, \
        f"eval 800콜 중 n=1/gold 못 찾은 콜 있음: {len(eval_ids - set(eval_df.call_id))}건"

    print(f"빈 텍스트(n=1이 공백) 제외: {n_empty}건")
    print(f"eval: {len(eval_df):,}콜 (불만 {eval_df.is_complaint.sum()}, "
          f"{eval_df.is_complaint.mean()*100:.1f}%)")
    print(f"reference: {len(ref_df):,}콜 (불만 {ref_df.is_complaint.sum()}, "
          f"{ref_df.is_complaint.mean()*100:.2f}%)")
    return eval_df, ref_df


# ──────────────────────────────────────────────────────────── 임베딩 ────
def get_device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def embed(model_key, model_name, df, tag, device):
    """df(call_id, text) -> (N,d) 임베딩. 디스크 캐시, call_id 순서 일치 검증."""
    cache_emb = OUT / f"lex_ambig_emb_{model_key}_{tag}.npy"
    cache_ids = OUT / f"lex_ambig_ids_{model_key}_{tag}.npy"
    ids = df.call_id.tolist()
    if cache_emb.exists() and cache_ids.exists():
        cached_ids = np.load(cache_ids, allow_pickle=True).tolist()
        if cached_ids == ids:
            emb = np.load(cache_emb)
            print(f"  [캐시 재사용] {model_key}/{tag} {emb.shape}")
            return emb
        print(f"  [캐시 불일치 — 재계산] {model_key}/{tag}")

    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    m = SentenceTransformer(model_name, device=device)
    emb = m.encode(df.text.tolist(), batch_size=BATCH_SIZE,
                   normalize_embeddings=True, show_progress_bar=True)
    del m
    print(f"  {model_key}/{tag}: {len(df):,}건 {time.time()-t0:.0f}s, shape={emb.shape}")
    np.save(cache_emb, emb)
    np.save(cache_ids, np.array(ids, dtype=object))
    return emb


# ──────────────────────────────────────────────────────────── 이웃 탐색 ────
def neighbor_search(eval_df, eval_emb, ref_df, ref_emb):
    """eval 콜마다 reference에서 class별(complaint/non) top-3 이웃 + 지표.

    sim(c,n) = comp top-3 × non top-3 교차 9쌍 cosine 평균(명시된 정의).
    """
    is_c = ref_df.is_complaint.values.astype(bool)
    comp_idx_pool = np.where(is_c)[0]
    non_idx_pool = np.where(~is_c)[0]
    comp_emb, non_emb = ref_emb[comp_idx_pool], ref_emb[non_idx_pool]

    sim_comp_all = eval_emb @ comp_emb.T   # (n_eval, n_comp_ref)
    sim_non_all = eval_emb @ non_emb.T     # (n_eval, n_non_ref)

    rows = []
    for i, r in enumerate(eval_df.itertuples()):
        c_top = np.argsort(-sim_comp_all[i])[:TOP_K]
        n_top = np.argsort(-sim_non_all[i])[:TOP_K]
        c_cos = sim_comp_all[i, c_top]
        n_cos = sim_non_all[i, n_top]
        c_glob = comp_idx_pool[c_top]     # ref_df 전역 인덱스
        n_glob = non_idx_pool[n_top]

        S_c, S_n = float(c_cos.mean()), float(n_cos.mean())
        S_min, dS = min(S_c, S_n), abs(S_c - S_n)

        # sim(c,n): top-3 × top-3 교차 9쌍 평균
        cross = comp_emb[c_top] @ non_emb[n_top].T   # (3,3)
        sim_cn = float(cross.mean())

        rec = {"call_id": r.call_id, "eval_text": r.text,
               "S_c": S_c, "S_n": S_n, "S_min": S_min, "deltaS": dS,
               "sim_cn": sim_cn, "A": S_min - dS}
        for k in range(TOP_K):
            j = c_glob[k]
            rec[f"comp_neighbor_{k+1}_call_id"] = ref_df.call_id.iat[j]
            rec[f"comp_neighbor_{k+1}_text"] = ref_df.text.iat[j]
            rec[f"comp_neighbor_{k+1}_cos"] = float(c_cos[k])
        for k in range(TOP_K):
            j = n_glob[k]
            rec[f"non_neighbor_{k+1}_call_id"] = ref_df.call_id.iat[j]
            rec[f"non_neighbor_{k+1}_text"] = ref_df.text.iat[j]
            rec[f"non_neighbor_{k+1}_gold"] = ref_df.label.iat[j]
            rec[f"non_neighbor_{k+1}_cos"] = float(n_cos[k])
        rows.append(rec)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────── 산출물 ────
def _format_review_columns(df):
    """공통 표시 컬럼 선정 + 반올림. 행 선택/정렬은 호출부 책임."""
    cols = ["call_id", "eval_text"]
    for k in range(1, TOP_K + 1):
        cols += [f"comp_neighbor_{k}_text", f"comp_neighbor_{k}_cos"]
    for k in range(1, TOP_K + 1):
        cols += [f"non_neighbor_{k}_text", f"non_neighbor_{k}_gold", f"non_neighbor_{k}_cos"]
    cols += ["S_c", "S_n", "S_min", "deltaS", "sim_cn", "A"]
    out = df[cols].round({"S_c": 4, "S_n": 4, "S_min": 4, "deltaS": 4, "sim_cn": 4, "A": 4,
                          **{f"comp_neighbor_{k}_cos": 4 for k in range(1, TOP_K + 1)},
                          **{f"non_neighbor_{k}_cos": 4 for k in range(1, TOP_K + 1)}})
    return out.reset_index(drop=True)


def build_review_table(res_primary):
    """A 상위 50건 — 이웃 원문 + 지표 + 사람이 채울 빈 컬럼 3개.

    eval 콜 자신의 gold는 넣지 않는다(circularity 차단 — 라벨링 시 앵커링 방지).
    """
    top = res_primary.sort_values("A", ascending=False).head(N_REVIEW)
    out = _format_review_columns(top)
    # 사용자가 채울 빈 컬럼
    out["type_label"] = ""          # underspecified / context_dependent / artifact
    out["neighbor_gold_ok"] = ""    # 딸려온 이웃 gold가 초기발화만 봐도 명백히 틀리진 않았는가
    out["note"] = ""
    return out


def build_compare_table(res_other, call_id_order):
    """다른 임베딩의 결과를 primary 표와 **같은 콜 순서**로 재배열.

    KoSimCSE 자체 A로 재정렬하면 KURE 50건과 나란히 못 본다 — 반드시
    call_id_order(=primary 표의 행 순서)를 그대로 따라야 육안 비교가 된다.
    """
    aligned = (res_other.set_index("call_id").loc[call_id_order].reset_index())
    return _format_review_columns(aligned)


def distribution_summary(all_res):
    """모델별 S_min·ΔS·sim(c,n) 분위수."""
    rows = []
    for model_key, res in all_res.items():
        for metric in ["S_min", "deltaS", "sim_cn"]:
            v = res[metric]
            rows.append({"model": model_key, "metric": metric, "n": len(v),
                        "mean": v.mean(), "std": v.std(),
                        "p10": v.quantile(.1), "p25": v.quantile(.25),
                        "p50": v.quantile(.5), "p75": v.quantile(.75),
                        "p90": v.quantile(.9)})
    return pd.DataFrame(rows)


def embedding_overlap(all_res):
    """모델별 A-상위 30건 call_id 겹침(Jaccard)."""
    tops = {k: set(res.sort_values("A", ascending=False).head(N_OVERLAP).call_id)
           for k, res in all_res.items()}
    keys = list(tops)
    rows = []
    for a, b in combinations(keys, 2):
        inter = tops[a] & tops[b]
        union = tops[a] | tops[b]
        rows.append({"모델A": a, "모델B": b, "상위N": N_OVERLAP,
                    "교집합": len(inter), "Jaccard": len(inter) / len(union),
                    "겹치는_call_id": sorted(inter)})
    return pd.DataFrame(rows)


def gold_noise_flag(ref_df):
    """reference 내 완전 동일 텍스트인데 is_complaint(0/1)가 갈리는 쌍.

    이 파일럿이 보는 경계는 complaint vs non-complaint 하나뿐이므로, label
    7종 중 아무거나 다르면(예: 환불요청 vs 배송확인, 둘 다 non) 잡지 않고
    is_complaint 가 갈릴 때만 잡는다 — 그게 이 파일럿과 무관한 노이즈를
    솎아내는 기준이다. 자동 계산만 한다 — 어느 쪽이 옳은지 판정은 사용자 몫.
    """
    dup = ref_df[ref_df.duplicated("text", keep=False)].copy()
    if dup.empty:
        return pd.DataFrame(columns=["text", "n_calls", "labels", "call_ids", "mixed_complaint"])
    g = dup.groupby("text").agg(
        n_calls=("call_id", "size"),
        labels=("label", lambda s: sorted(s.unique().tolist())),
        call_ids=("call_id", lambda s: sorted(s.tolist())),
        n_complaint_classes=("is_complaint", "nunique"),
    ).reset_index()
    g["mixed_complaint"] = g.n_complaint_classes > 1
    return (g[g.mixed_complaint].drop(columns=["n_complaint_classes"])
            .sort_values("n_calls", ascending=False).reset_index(drop=True))


def analyze_labels():
    """review CSV에 사람이 type_label 을 채워 넣은 뒤 재실행하면 그룹별
    sim(c,n) 분포를 비교해 준다(가설: artifact는 낮고 context_dependent는 높다).
    """
    path = OUT / "pilot_neighbors_top50.csv"
    if not path.exists():
        print("리뷰 CSV 없음 — 먼저 본 스크립트를 기본 모드로 실행하세요.")
        return
    df = pd.read_csv(path)
    labeled = df[df.type_label.fillna("").str.strip() != ""]
    if labeled.empty:
        print("type_label 이 아직 비어 있습니다. 50건을 라벨링한 뒤 다시 실행하세요.")
        return
    print(f"라벨링된 {len(labeled)}/{len(df)}건")
    print(labeled.groupby("type_label")["sim_cn"].describe()[["count", "mean", "std", "min", "max"]]
          .round(4).to_string())


def main():
    import sys
    if "--analyze" in sys.argv:
        analyze_labels()
        return

    print("=" * 72)
    print("1. 데이터 로드 (eval 800 / reference ~18.7k, n=1 첫 고객발화)")
    print("=" * 72)
    eval_df, ref_df = build_texts()

    gflag = gold_noise_flag(ref_df)
    gflag.to_csv(OUT / "table_lex_ambig_gold_flag.csv", index=False, encoding="utf-8-sig")
    print(f"\nreference gold 이상 의심(동일 텍스트, gold class 갈림): {len(gflag)}건 "
          f"(파일: table_lex_ambig_gold_flag.csv)")

    device = get_device()
    print(f"\n디바이스: {device}")

    all_res = {}
    for model_key, model_name in MODELS.items():
        print("\n" + "=" * 72)
        print(f"2. 임베딩 + 이웃 탐색 — {model_key} ({model_name})")
        print("=" * 72)
        eval_emb = embed(model_key, model_name, eval_df, "eval", device)
        ref_emb = embed(model_key, model_name, ref_df, "ref", device)
        res = neighbor_search(eval_df, eval_emb, ref_df, ref_emb)
        res.to_parquet(OUT / f"lex_ambig_neighbors_{model_key}_full.parquet", index=False)
        all_res[model_key] = res
        print(f"  S_min mean={res.S_min.mean():.4f} / ΔS mean={res.deltaS.mean():.4f} "
              f"/ sim_cn mean={res.sim_cn.mean():.4f}")

    print("\n" + "=" * 72)
    print(f"3. 50건 수동검토 테이블 (primary={PRIMARY}, A=S_min−ΔS 상위)")
    print("=" * 72)
    review = build_review_table(all_res[PRIMARY])
    review.to_csv(OUT / "pilot_neighbors_top50.csv", index=False, encoding="utf-8-sig")
    print(f"저장: {OUT / 'pilot_neighbors_top50.csv'} ({len(review)}행)")

    # 같은 50콜·같은 행 순서로 비교군 임베딩 결과(라벨링은 primary 표 하나로만
    # — 중복 노동 방지. 순서를 primary와 맞춰야 나란히 육안 비교가 된다).
    other_keys = [k for k in MODELS if k != PRIMARY]
    if other_keys:
        ok = other_keys[0]
        cmp_review = build_compare_table(all_res[ok], review.call_id.tolist())
        cmp_path = OUT / f"pilot_neighbors_top50_{ok}_compare.csv"
        cmp_review.to_csv(cmp_path, index=False, encoding="utf-8-sig")
        print(f"저장: {cmp_path} (같은 50콜·같은 행 순서, {ok} 결과 — 이웃 품질 육안 비교용)")

    print("\n" + "=" * 72)
    print("4. 분포 요약 (S_min · ΔS · sim(c,n))")
    print("=" * 72)
    dist = distribution_summary(all_res)
    dist.to_csv(OUT / "table_lex_ambig_dist.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 160)
    print(dist.round(4).to_string(index=False))

    print("\n" + "=" * 72)
    print(f"5. 임베딩 2종 상위 {N_OVERLAP}건 겹침")
    print("=" * 72)
    ov = embedding_overlap(all_res)
    ov.drop(columns=["겹치는_call_id"]).to_csv(
        OUT / "table_lex_ambig_embedding_overlap.csv", index=False, encoding="utf-8-sig")
    print(ov.drop(columns=["겹치는_call_id"]).to_string(index=False))

    print("\n" + "=" * 72)
    print("산출물")
    print("=" * 72)
    for f in ["pilot_neighbors_top50.csv",
             f"pilot_neighbors_top50_{other_keys[0]}_compare.csv" if other_keys else None,
             "table_lex_ambig_dist.csv", "table_lex_ambig_embedding_overlap.csv",
             "table_lex_ambig_gold_flag.csv"]:
        if f:
            print(f"  outputs/paper/{f}")
    print("\n다음: pilot_neighbors_top50.csv 의 type_label/neighbor_gold_ok 를 채운 뒤")
    print("      `python scripts/paper_lex_ambig_pilot.py --analyze` 로 유형별 sim(c,n) 비교.")
    print("\nPAPER_LEX_AMBIG_PILOT_COMPLETE")


if __name__ == "__main__":
    main()
