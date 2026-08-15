"""모달리티 A/B 비교 평가: 같은 250건 paired, model=gpt-5.4, effort=low 고정.

  A = customer-only + text-only        (make_prompt(..., with_acoustic=False))
  B = customer-only + text + acoustic  (make_prompt(..., with_acoustic=True))

바뀌는 건 발화별 "  [음성 특징: ...]" 줄의 유무뿐이며 그 외 프롬프트 텍스트는 동일.
조건별 절대값만 산출한다(델타/flip/화살표 표기 없음).

핵심 지표: confusion matrix의 gold=불만제기 -> 예측=환불요청 셀,
          gold=불만제기 -> 예측=배송확인 셀.

출력:
  outputs/modality_ab_summary.md
  outputs/modality_ab_percall.parquet
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_PATH = BASE_DIR / "outputs" / "model_pilot_sample.csv"
MD_PATH = BASE_DIR / "outputs" / "modality_ab_summary.md"
PERCALL_PATH = BASE_DIR / "outputs" / "modality_ab_percall.parquet"

CATEGORIES = ["환불요청", "주문취소", "불만제기", "배송확인", "교환반품", "구매진행", "서비스이용"]
PARSE_FAIL = "PARSE_FAIL"
COMPLAINT = "불만제기"
FOCUS_CELLS = ["환불요청", "배송확인"]

# (표시명, per-call 컬럼, 예측 parquet 태그) — 행 순서 고정
CONDITIONS = [
    ("A(5.4-low)", "pred_A", "A_54low"),
    ("B(5.4-low)", "pred_B", "54low"),
]


def main():
    df = pd.read_csv(SAMPLE_PATH)[["call_id", "gold_actual"]]
    n0 = len(df)
    for _, col, tag in CONDITIONS:
        path = BASE_DIR / "outputs" / f"model_pilot_pred_{tag}.parquet"
        pr = pd.read_parquet(path)[["call_id", "predicted_label"]].rename(
            columns={"predicted_label": col})
        assert set(pr["call_id"]) == set(df["call_id"]), f"{tag}: call_id 집합 불일치(paired 깨짐)"
        df = df.merge(pr, on="call_id", how="left")
    assert len(df) == n0 == 250, f"행 수 이상: {len(df)}"

    fail_counts = {}
    for _, col, _t in CONDITIONS:
        bad = df[col].isna() | ~df[col].isin(CATEGORIES)
        fail_counts[col] = int(bad.sum())
        df.loc[bad, col] = PARSE_FAIL
        df[f"correct_{col}"] = df[col] == df["gold_actual"]

    y = df["gold_actual"]
    cms = {}
    for _, col, _t in CONDITIONS:
        labs = CATEGORIES + ([PARSE_FAIL] if fail_counts[col] else [])
        cms[col] = pd.DataFrame(
            confusion_matrix(y, df[col], labels=labs), index=labs, columns=labs).loc[CATEGORIES]

    lines = []
    lines.append("# 모달리티 A/B 비교 (250건 paired, model=gpt-5.4, reasoning.effort=low 고정)")
    lines.append("")
    lines.append("두 조건 모두 **동일 250 call_id**(`model_pilot_sample.csv`, seed=42, "
                 "불만제기 50건 오버샘플). 바뀌는 건 모달리티 하나뿐이다.")
    lines.append("")
    lines.append("- **A** = customer-only + text-only — `make_prompt(rows, with_acoustic=False)`")
    lines.append("- **B** = customer-only + text + acoustic — `make_prompt(rows, with_acoustic=True)`")
    lines.append("")
    lines.append("A는 B에서 발화별 `  [음성 특징: ...]` 줄만 제거된 형태이고, 지시문·카테고리 "
                 "정의·발화 텍스트·화자 표기·질문부·평문 출력 포맷은 전부 동일하다. "
                 "voice_guide 블록은 `version=\"base\"`라 A/B 모두 비어 있어 변수가 아니다. "
                 "250건 전부 원본 배치 JSONL(A=stage2d, B=stage2m)과 바이트 동일 검증됨.")
    lines.append("")
    lines.append("화자는 `speaker_group`(type 필드)만 사용하며 gold_actual은 프롬프트에 미포함. "
                 "두 run 모두 Responses API, 동기 호출, concurrency=6, temperature 미지정.")
    lines.append("")

    lines.append("## 최종 표 (전부 절대값)")
    lines.append("")
    lines.append("| 조건 | macro-F1 | 불만 F1 | 불만 P | 불만 R | 불만→환불 셀 | 불만→배송 셀 | accuracy | 파싱실패 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, col, _t in CONDITIONS:
        p = df[col]
        macro = f1_score(y, p, labels=CATEGORIES, average="macro", zero_division=0)
        yt = (y == COMPLAINT).astype(int)
        yp = (p == COMPLAINT).astype(int)
        cell_refund = int(cms[col].loc[COMPLAINT, "환불요청"])
        cell_deliv = int(cms[col].loc[COMPLAINT, "배송확인"])
        lines.append(
            f"| {name} | {macro:.3f} | {f1_score(yt, yp, zero_division=0):.3f} | "
            f"{precision_score(yt, yp, zero_division=0):.3f} | "
            f"{recall_score(yt, yp, zero_division=0):.3f} | "
            f"{cell_refund} | {cell_deliv} | {(y == p).mean():.3f} | {fail_counts[col]} |")
    lines.append("")
    lines.append(f"불만제기 support = {int((y == COMPLAINT).sum())}건 (두 조건 공통)")
    lines.append("")

    lines.append("## 핵심 셀: gold=불만제기 행 전체 (count)")
    lines.append("")
    lines.append("| 조건 | " + " | ".join(CATEGORIES) + " |")
    lines.append("|" + "---|" * (len(CATEGORIES) + 1))
    for name, col, _t in CONDITIONS:
        row = cms[col].loc[COMPLAINT]
        lines.append(f"| {name} | " + " | ".join(str(int(row.get(c, 0))) for c in CATEGORIES) + " |")
    lines.append("")
    for cell in FOCUS_CELLS:
        vals = ", ".join(f"{name} = {int(cms[col].loc[COMPLAINT, cell])}건"
                         for name, col, _t in CONDITIONS)
        lines.append(f"- gold=불만제기 → 예측={cell}: {vals}")
    lines.append("")

    lines.append("### support (gold 실제 개수, 두 조건 공통)")
    lines.append("")
    supp = y.value_counts()
    lines.append("| " + " | ".join(CATEGORIES) + " | 합계 |")
    lines.append("|" + "---|" * (len(CATEGORIES) + 1))
    lines.append("| " + " | ".join(str(int(supp.get(c, 0))) for c in CATEGORIES) + f" | {len(df)} |")
    lines.append("")

    lines.append("## 조건별 per-class precision / recall / F1")
    for name, col, _t in CONDITIONS:
        lines.append("")
        lines.append(f"### {name}")
        lines.append("")
        rep = classification_report(y, df[col], labels=CATEGORIES,
                                     output_dict=True, zero_division=0)
        lines.append("| 클래스 | precision | recall | F1 | support |")
        lines.append("|---|---|---|---|---|")
        for c in CATEGORIES:
            r = rep[c]
            lines.append(f"| {c} | {r['precision']:.3f} | {r['recall']:.3f} | "
                         f"{r['f1-score']:.3f} | {int(r['support'])} |")

    lines.append("")
    lines.append("## Confusion matrix (행=gold 실제, 열=예측, count)")
    for name, col, _t in CONDITIONS:
        cm = cms[col]
        lines.append("")
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| gold \\ pred | " + " | ".join(cm.columns) + " |")
        lines.append("|" + "---|" * (len(cm.columns) + 1))
        for c in CATEGORIES:
            lines.append(f"| **{c}** | " + " | ".join(str(int(v)) for v in cm.loc[c]) + " |")

    lines.append("")
    lines.append("## 파싱 실패 로그")
    lines.append("")
    if sum(fail_counts.values()) == 0:
        lines.append("두 조건 모두 파싱 실패/빈 응답 0건. A run은 재시도 발생 0건.")
    else:
        for name, col, _t in CONDITIONS:
            lines.append(f"- {name}: {fail_counts[col]}건 (오답 처리, 제외 없음)")

    md = "\n".join(lines)
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(md + "\n")

    out_cols = (["call_id", "gold_actual"] + [c for _, c, _t in CONDITIONS]
                + [f"correct_{c}" for _, c, _t in CONDITIONS])
    df[out_cols].to_parquet(PERCALL_PATH, index=False)

    print(md)
    print(f"\n저장: {MD_PATH}")
    print(f"저장: {PERCALL_PATH} ({len(df)}행)")
    print("MODALITY_AB_EVAL_COMPLETE")


if __name__ == "__main__":
    main()
