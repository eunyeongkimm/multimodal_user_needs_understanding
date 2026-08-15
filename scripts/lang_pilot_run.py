"""reasoning 언어 효과 250건 paired 파일럿. 3조건 전부 새로 실행(기존 B 재사용 안 함).

  1. ko      : 입력 한국어 전사 + 한국어 description, developer 지시 "한국어로 추론"
  2. en_rsn  : 입력 한국어 전사 + 한국어 description, developer 지시 "영어로 추론"
  3. en_full : 입력 영어 전사(MT 캐시) + 영어 description(템플릿 재생성), "영어로 추론"

공통 고정: 같은 250 call_id, 조건 B(customer-only + text + acoustic),
gpt-5.4 / reasoning.effort=low, version="base" 평문 출력, Responses API.
바뀌는 건 (입력 언어 + 추론 지시 언어)뿐이며 지시문·카테고리 정의·질문부는
3조건 모두 한국어로 동일하다(출력 라벨이 한국어 카테고리명이라 필수).

음성 description은 번역하지 않고 nl_en_template 으로 같은 quintile level에서
영어 문장을 직접 생성한다.

사용:
  python3 scripts/lang_pilot_run.py --verify
  python3 scripts/lang_pilot_run.py --variant ko
"""

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nl_en_template import build_nl_description_en, verify_parity
from relabel_v2_and_kappa import normalize_gpt_label
from stage2d_prompt import CATEGORIES, make_prompt
import stage2c_v2_normalize_nl_gender as ko_tmpl

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_PATH = BASE_DIR / "outputs" / "model_pilot_sample.csv"
WINDOWS_PATH = BASE_DIR / "outputs" / "stage2_windows_nl_v2.parquet"
CACHE_PATH = BASE_DIR / "outputs" / "lang_pilot_translation_cache.parquet"
OUT_TMPL = str(BASE_DIR / "outputs" / "lang_pilot_pred_{variant}.parquet")

WINDOW_N = 5
WINDOW_VERSION = "customer_only"
MODEL = "gpt-5.4"
EFFORT = "low"
CONCURRENCY = 6
MAX_RETRIES = 4

RSN_KO = "모든 추론 과정을 한국어로 진행하세요."
RSN_EN = "Conduct all of your reasoning in English."

# variant -> (입력 언어, developer 지시)
VARIANTS = {
    "ko": ("ko", RSN_KO),
    "en_rsn": ("ko", RSN_EN),
    "en_full": ("en", RSN_EN),
}


def build_prompts(variant: str) -> pd.DataFrame:
    input_lang, _ = VARIANTS[variant]
    sample = pd.read_csv(SAMPLE_PATH)
    win = pd.read_parquet(WINDOWS_PATH)
    win = win[(win["window_n"] == WINDOW_N) & (win["window_version"] == WINDOW_VERSION)
              & win["call_id"].isin(sample["call_id"])]

    if input_lang == "en":
        cache = pd.read_parquet(CACHE_PATH)[["call_id", "dialog_idx", "en_text"]]
        win = win.merge(cache, on=["call_id", "dialog_idx"], how="left")
        n_missing_mt = int(win["en_text"].isna().sum())
        if n_missing_mt:
            raise SystemExit(f"번역 캐시 결측 {n_missing_mt}건 - 중단")
        win = win.assign(text_used=win["en_text"],
                         desc_used=win.apply(build_nl_description_en, axis=1))
    else:
        win = win.assign(text_used=win["text_clean"], desc_used=win["nl_description"])

    rows, n_missing = [], 0
    for cid in sample["call_id"]:
        g = win[win["call_id"] == cid].sort_values("dialog_idx")
        if g.empty:
            n_missing += 1
            continue
        utt = [{"speaker_group": r.speaker_group, "text_clean": r.text_used,
                "nl_description": r.desc_used} for r in g.itertuples()]
        rows.append({"call_id": cid, "prompt": make_prompt(utt, with_acoustic=True)})
    if n_missing:
        print(f"경고: 윈도우 없는 콜 {n_missing}건 (제외)")
    return sample.merge(pd.DataFrame(rows), on="call_id", how="inner")


def parse_response(raw):
    if raw is None or not str(raw).strip():
        return None, "빈응답"
    cat = normalize_gpt_label(str(raw).strip())
    if cat not in CATEGORIES:
        return None, f"형식오류({cat[:60]!r})"
    return cat, None


def call_one(client, instruction, call_id, prompt):
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.responses.create(
                model=MODEL, instructions=instruction, input=prompt,
                reasoning={"effort": EFFORT},
            )
            label, perr = parse_response(resp.output_text)
            return {"call_id": call_id, "predicted_label": label,
                    "parse_error": perr, "n_attempts": attempt + 1}
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 16))
    return {"call_id": call_id, "predicted_label": None,
            "parse_error": f"API실패({type(last_err).__name__}: {last_err})",
            "n_attempts": MAX_RETRIES}


def do_verify():
    """250 call_id 동일성 + 조건 B 구조 동일성 + KO/EN 템플릿 정합성 확인."""
    problems = verify_parity(ko_tmpl)
    print(f"KO/EN 템플릿 정합성: {'OK (불일치 0)' if not problems else problems}")

    dfs = {v: build_prompts(v) for v in VARIANTS}
    base_ids = list(dfs["ko"]["call_id"])
    for v, d in dfs.items():
        assert list(d["call_id"]) == base_ids, f"{v}: call_id 순서/집합 불일치"
    print(f"250 call_id 동일: OK ({len(base_ids)}건, 3조건 순서까지 일치)")

    # 조건 B 구조: ko/en_rsn 프롬프트는 완전 동일해야 하고(입력 언어 같음),
    # en_full 은 발화/음성 줄만 달라야 한다.
    same = (dfs["ko"]["prompt"] == dfs["en_rsn"]["prompt"]).all()
    print(f"ko vs en_rsn 프롬프트 바이트 동일: {'OK' if same else '불일치!'} "
          f"(입력이 같고 developer 지시만 다름)")

    import difflib
    p_ko, p_en = dfs["ko"]["prompt"].iat[0], dfs["en_full"]["prompt"].iat[0]
    changed = [l for l in difflib.unified_diff(p_ko.split("\n"), p_en.split("\n"),
                                                lineterm="", n=0)
               if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    struct = [l for l in changed
              if not (l[1:].startswith("발화") or l[1:].strip().startswith("[음성 특징:"))]
    print(f"ko vs en_full 변경 줄 {len(changed)}개 중 발화/음성 이외 변경: {len(struct)}개 "
          f"{'(구조 동일 OK)' if not struct else struct}")

    win = pd.read_parquet(WINDOWS_PATH)
    win = win[(win["window_n"] == WINDOW_N) & (win["window_version"] == WINDOW_VERSION)]
    print("\n=== 음성 description KO/EN 스팟체크 5건 (같은 level -> 같은 의미) ===")
    lv = [f"{m}_level" for m in ko_tmpl.METRICS]
    for i, (_, r) in enumerate(win.head(5).iterrows(), 1):
        print(f"\n[{i}] levels: {dict(zip(ko_tmpl.METRICS, [r[c] for c in lv]))}")
        print(f"  KO: {r['nl_description']}")
        print(f"  EN: {build_nl_description_en(r)}")
    print("\nLANG_PILOT_VERIFY_OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS))
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if args.verify:
        do_verify()
        return
    if not args.variant:
        ap.error("--variant 또는 --verify 필요")

    variant = args.variant
    input_lang, instruction = VARIANTS[variant]
    df = build_prompts(variant)
    print(f"프롬프트 생성: {len(df)}건 (variant={variant}, 입력={input_lang}, "
          f"developer 지시={instruction!r})")

    load_dotenv(BASE_DIR / ".env")
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key or api_key == "REPLACE_ME":
        print("ERROR: OPENAI_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    print(f"\n=== {variant} | {MODEL} (effort={EFFORT}) {len(df)}건, "
          f"concurrency={CONCURRENCY} ===")
    done, lock, t0 = [0], threading.Lock(), time.time()

    def work(row):
        r = call_one(client, instruction, row.call_id, row.prompt)
        with lock:
            done[0] += 1
            if done[0] % 50 == 0 or done[0] == len(df):
                print(f"  {done[0]}/{len(df)} ({time.time()-t0:.0f}s)", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        results = list(ex.map(work, df.itertuples()))

    res = pd.DataFrame(results)
    out = df[["call_id", "gold_actual"]].merge(res, on="call_id", how="left")
    n_fail = out["predicted_label"].isna().sum()
    n_retried = (out["n_attempts"] > 1).sum()
    print(f"\n완료: {len(out)}건 / 파싱·API 실패 {n_fail}건 / 재시도 발생 {n_retried}건 "
          f"/ 소요 {time.time()-t0:.0f}s")
    if n_fail:
        for r in out[out["predicted_label"].isna()].itertuples():
            print(f"  {r.call_id}: {r.parse_error}")

    path = OUT_TMPL.format(variant=variant)
    out.to_parquet(path, index=False)
    print(f"저장: {path}")
    print(f"LANG_PILOT_RUN_COMPLETE::{variant}")


if __name__ == "__main__":
    main()
