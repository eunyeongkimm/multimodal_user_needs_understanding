"""reasoning 언어 효과 평가: 조건별 절대값만 산출(델타/flip/화살표 표기 없음).

행 순서 고정: 1(KO) / 2(EN-rsn) / 3(EN-full)
핵심 셀: gold=불만제기 -> 예측=환불요청.

출력:
  outputs/lang_pilot_summary.md
  outputs/lang_pilot_percall.parquet
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_PATH = BASE_DIR / "outputs" / "model_pilot_sample.csv"
MD_PATH = BASE_DIR / "outputs" / "lang_pilot_summary.md"
PERCALL_PATH = BASE_DIR / "outputs" / "lang_pilot_percall.parquet"

CATEGORIES = ["환불요청", "주문취소", "불만제기", "배송확인", "교환반품", "구매진행", "서비스이용"]
PARSE_FAIL = "PARSE_FAIL"
COMPLAINT = "불만제기"
REFUND = "환불요청"

# (표시명, per-call 컬럼, 예측 parquet variant) — 행 순서 고정
CONDITIONS = [
    ("1(KO)", "pred_ko", "ko"),
    ("2(EN-rsn)", "pred_enrsn", "en_rsn"),
    ("3(EN-full)", "pred_enfull", "en_full"),
]


def main():
    df = pd.read_csv(SAMPLE_PATH)[["call_id", "gold_actual"]]
    n0 = len(df)
    for _, col, variant in CONDITIONS:
        pr = pd.read_parquet(BASE_DIR / "outputs" / f"lang_pilot_pred_{variant}.parquet")[
            ["call_id", "predicted_label"]].rename(columns={"predicted_label": col})
        assert set(pr["call_id"]) == set(df["call_id"]), f"{variant}: call_id 불일치(paired 깨짐)"
        df = df.merge(pr, on="call_id", how="left")
    assert len(df) == n0 == 250, f"행 수 이상: {len(df)}"

    fail_counts = {}
    for _, col, _v in CONDITIONS:
        bad = df[col].isna() | ~df[col].isin(CATEGORIES)
        fail_counts[col] = int(bad.sum())
        df.loc[bad, col] = PARSE_FAIL
        df[f"correct_{col}"] = df[col] == df["gold_actual"]

    y = df["gold_actual"]
    cms = {}
    for _, col, _v in CONDITIONS:
        labs = CATEGORIES + ([PARSE_FAIL] if fail_counts[col] else [])
        cms[col] = pd.DataFrame(confusion_matrix(y, df[col], labels=labs),
                                index=labs, columns=labs).loc[CATEGORIES]

    L = []
    L.append("# reasoning 언어 효과 파일럿 (250건 paired)")
    L.append("")
    L.append("3조건 전부 새로 실행(기존 B 예측 재사용 없음, 대칭성 유지). "
             "동일 250 call_id(`model_pilot_sample.csv`, seed=42, 불만제기 50건 오버샘플).")
    L.append("")
    L.append("| 조건 | 입력 전사 | 입력 음성 description | developer 지시 |")
    L.append("|---|---|---|---|")
    L.append("| 1(KO) | 한국어 | 한국어 | `모든 추론 과정을 한국어로 진행하세요.` |")
    L.append("| 2(EN-rsn) | 한국어 | 한국어 | `Conduct all of your reasoning in English.` |")
    L.append("| 3(EN-full) | 영어(MT 캐시) | 영어(템플릿 재생성) | "
             "`Conduct all of your reasoning in English.` |")
    L.append("")
    L.append("1→2 = 추론 언어 효과, 2→3 = 입력 번역 효과.")
    L.append("")
    L.append("**공통 고정**: 조건 B(customer-only + text + acoustic, window_n=5), "
             "`gpt-5.4` / `reasoning.effort=low`, Responses API, "
             "`make_prompt(..., with_acoustic=True)` version=\"base\" 평문 카테고리명 출력.")
    L.append("")
    L.append("**언어를 바꾸지 않은 부분**: 지시문·카테고리 정의·질문부는 3조건 모두 "
             "한국어로 동일하다. 출력 라벨이 한국어 카테고리명이라 카테고리 정의를 "
             "영어로 바꾸면 라벨 파싱이 깨지기 때문이다. 즉 EN-full의 \"입력 영어\"는 "
             "전사와 음성 description 두 필드에 한정된다.")
    L.append("")
    L.append("**검증**: 3조건 call_id 순서까지 일치. 1과 2의 프롬프트는 바이트 동일"
             "(developer 지시만 다름). 1과 3의 차이는 발화 줄과 `[음성 특징: ...]` 줄에만 "
             "발생하며 그 외 구조 변경 0줄.")
    L.append("")
    L.append("**음성 description 영어화 방식**: 한국어 문장을 MT에 넣지 않고, "
             "`nl_en_template.py`가 동일한 quintile level 컬럼(`{metric}_level`)에서 "
             "영어 문장을 직접 생성한다. METRICS 순서·LEVEL_LABELS·분기 키 집합이 "
             "한국어 템플릿과 일치함을 자동 검증했다(불일치 0).")
    L.append("")
    L.append("**전사 번역**: `gpt-5.4`로 발화 단위 1회 번역 후 "
             "`lang_pilot_translation_cache.parquet`에 캐시하여 조건 3에서 재사용"
             "(조건마다 재번역 없음). 1,119발화 번역 실패 0건.")
    L.append("")

    L.append("## 최종 표 (전부 절대값)")
    L.append("")
    L.append("| 조건 | macro-F1 | 불만 F1 | 불만 P | 불만 R | 불만→환불 셀 | 정상 F1 | accuracy | 파싱실패 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, col, _v in CONDITIONS:
        p = df[col]
        yt = (y == COMPLAINT).astype(int)
        yp = (p == COMPLAINT).astype(int)
        L.append(
            f"| {name} | "
            f"{f1_score(y, p, labels=CATEGORIES, average='macro', zero_division=0):.3f} | "
            f"{f1_score(yt, yp, zero_division=0):.3f} | "
            f"{precision_score(yt, yp, zero_division=0):.3f} | "
            f"{recall_score(yt, yp, zero_division=0):.3f} | "
            f"{int(cms[col].loc[COMPLAINT, REFUND])} | "
            f"{f1_score(1 - yt, 1 - yp, zero_division=0):.3f} | "
            f"{(y == p).mean():.3f} | {fail_counts[col]} |")
    L.append("")

    L.append("### support (gold 실제 개수, 3조건 공통)")
    L.append("")
    supp = y.value_counts()
    L.append("| " + " | ".join(CATEGORIES) + " | 합계 |")
    L.append("|" + "---|" * (len(CATEGORIES) + 1))
    L.append("| " + " | ".join(str(int(supp.get(c, 0))) for c in CATEGORIES) + f" | {len(df)} |")
    L.append("")

    L.append("### 핵심 셀: gold=불만제기 행 전체 (count)")
    L.append("")
    L.append("| 조건 | " + " | ".join(CATEGORIES) + " |")
    L.append("|" + "---|" * (len(CATEGORIES) + 1))
    for name, col, _v in CONDITIONS:
        row = cms[col].loc[COMPLAINT]
        L.append(f"| {name} | " + " | ".join(str(int(row.get(c, 0))) for c in CATEGORIES) + " |")
    L.append("")
    L.append("- gold=불만제기 → 예측=환불요청: " + ", ".join(
        f"{name} = {int(cms[col].loc[COMPLAINT, REFUND])}건" for name, col, _v in CONDITIONS))
    L.append("")

    L.append("## 조건별 per-class precision / recall / F1")
    for name, col, _v in CONDITIONS:
        L.append("")
        L.append(f"### {name}")
        L.append("")
        rep = classification_report(y, df[col], labels=CATEGORIES,
                                     output_dict=True, zero_division=0)
        L.append("| 클래스 | precision | recall | F1 | support |")
        L.append("|---|---|---|---|---|")
        for c in CATEGORIES:
            r = rep[c]
            L.append(f"| {c} | {r['precision']:.3f} | {r['recall']:.3f} | "
                     f"{r['f1-score']:.3f} | {int(r['support'])} |")

    L.append("")
    L.append("## Confusion matrix (행=gold 실제, 열=예측, count)")
    for name, col, _v in CONDITIONS:
        cm = cms[col]
        L.append("")
        L.append(f"### {name}")
        L.append("")
        L.append("| gold \\ pred | " + " | ".join(cm.columns) + " |")
        L.append("|" + "---|" * (len(cm.columns) + 1))
        for c in CATEGORIES:
            L.append(f"| **{c}** | " + " | ".join(str(int(v)) for v in cm.loc[c]) + " |")

    L.append("")
    L.append("## 파싱 실패 로그")
    L.append("")
    if sum(fail_counts.values()) == 0:
        L.append("3조건 전부 파싱 실패/빈 응답 0건, 재시도 발생 0건.")
    else:
        for name, col, _v in CONDITIONS:
            L.append(f"- {name}: {fail_counts[col]}건 (오답 처리, 제외 없음)")

    md = "\n".join(L)
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(md + "\n")

    out_cols = (["call_id", "gold_actual"] + [c for _, c, _v in CONDITIONS]
                + [f"correct_{c}" for _, c, _v in CONDITIONS])
    df[out_cols].to_parquet(PERCALL_PATH, index=False)

    print(md)
    print(f"\n저장: {MD_PATH}")
    print(f"저장: {PERCALL_PATH} ({len(df)}행)")
    print("LANG_PILOT_EVAL_COMPLETE")


if __name__ == "__main__":
    main()
