"""reasoning effort·모델 비용 축 파일럿 평가: 조건별 절대 성능만 산출
(델타/flip/화살표 등 변화량 표기 없음).

조건 5개(행 순서 고정): 4.1현재(재사용) / 5.4-low / 5.4-med / luna-low / luna-med
전부 동일 250건 paired 샘플, 조건 B(customer-only + text + acoustic).

출력:
  outputs/model_pilot_summary.md      요약 표 + support + per-class + confusion matrix 5개
  outputs/model_pilot_percall.parquet call_id, gold, pred_*, correct_*
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_PATH = BASE_DIR / "outputs" / "model_pilot_sample.csv"
MD_PATH = BASE_DIR / "outputs" / "model_pilot_summary.md"
PERCALL_PATH = BASE_DIR / "outputs" / "model_pilot_percall.parquet"

CATEGORIES = ["환불요청", "주문취소", "불만제기", "배송확인", "교환반품", "구매진행", "서비스이용"]
PARSE_FAIL = "PARSE_FAIL"

# (표시명, per-call 컬럼명, 예측 parquet 태그) — 행 순서 고정
CONDITIONS = [
    ("4.1현재", "pred_41cur", None),          # 기존 재사용(chat completions, temperature=0)
    ("5.4-low", "pred_54low", "54low"),
    ("5.4-med", "pred_54med", "54med"),
    ("luna-low", "pred_lunalow", "lunalow"),
    ("luna-med", "pred_lunamed", "lunamed"),
]


def main():
    df = pd.read_csv(SAMPLE_PATH)[["call_id", "gold_actual", "pred_41mini_cur"]].rename(
        columns={"pred_41mini_cur": "pred_41cur"})
    n0 = len(df)
    for _, col, tag in CONDITIONS:
        if tag is None:
            continue
        path = BASE_DIR / "outputs" / f"model_pilot_pred_{tag}.parquet"
        pr = pd.read_parquet(path)[["call_id", "predicted_label"]].rename(
            columns={"predicted_label": col})
        assert set(pr["call_id"]) == set(df["call_id"]), f"{tag}: call_id 집합 불일치(paired 깨짐)"
        df = df.merge(pr, on="call_id", how="left")
    assert len(df) == n0, "병합 후 행 수 변동"

    fail_counts = {}
    for _, col, _t in CONDITIONS:
        # 카테고리 외 응답/결측 = 파싱 실패, 오답으로 처리(제외 금지)
        bad = df[col].isna() | ~df[col].isin(CATEGORIES)
        fail_counts[col] = int(bad.sum())
        df.loc[bad, col] = PARSE_FAIL

    for _, col, _t in CONDITIONS:
        df[f"correct_{col}"] = df[col] == df["gold_actual"]

    y = df["gold_actual"]
    lines = []
    lines.append("# reasoning effort · 모델 비용 축 파일럿 "
                 "(250건 paired, 조건 B: customer-only + text + acoustic)")
    lines.append("")
    lines.append("모집단: acoustically-valid subset(B/N=5/customer_only, 정보충분) 16,777건 "
                 "중 층화 250건, seed=42. 불만제기는 비례추출 7건 → 50건으로 오버샘플. "
                 "5조건 전부 **동일 250 call_id**(paired).")
    lines.append("")
    lines.append("프롬프트/입력/출력형식은 5조건 전부 동일: `stage2d_prompt.make_prompt("
                 "rows, with_acoustic=True)` (version=\"base\"), 출력은 평문 카테고리명 "
                 "(JSON schema 미사용). 250건 전부 stage2m 제출 JSONL과 바이트 동일 검증됨. "
                 "화자는 `speaker_group`(type 필드)만 사용하며 gold_actual은 프롬프트에 미포함.")
    lines.append("")
    lines.append("**API 경로(숨은 변수)**: `4.1현재`만 Chat Completions "
                 "(`/v1/chat/completions`, Batch API, `temperature=0`, reasoning 없음)로 "
                 "실행된 기존 예측 재사용. 나머지 4조건은 Responses API + "
                 "`reasoning.effort`, temperature 미지정(동기 호출, concurrency=6).")
    lines.append("")

    lines.append("## 요약 (전부 절대값)")
    lines.append("")
    lines.append("| 조건 | macro-F1 | 불만 F1 | 불만 P | 불만 R | 정상 F1 | accuracy | 파싱실패 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, col, _t in CONDITIONS:
        p = df[col]
        macro = f1_score(y, p, labels=CATEGORIES, average="macro", zero_division=0)
        yt = (y == "불만제기").astype(int)
        yp = (p == "불만제기").astype(int)
        c_f1 = f1_score(yt, yp, zero_division=0)
        c_p = precision_score(yt, yp, zero_division=0)
        c_r = recall_score(yt, yp, zero_division=0)
        # "정상" = 불만제기가 아닌 6개 클래스를 하나로 묶은 이진 관점
        n_f1 = f1_score(1 - yt, 1 - yp, zero_division=0)
        acc = (y == p).mean()
        lines.append(f"| {name} | {macro:.3f} | {c_f1:.3f} | {c_p:.3f} | {c_r:.3f} | "
                     f"{n_f1:.3f} | {acc:.3f} | {fail_counts[col]} |")
    lines.append("")

    lines.append("### support (gold 실제 개수, 5조건 공통)")
    lines.append("")
    supp = y.value_counts()
    lines.append("| " + " | ".join(CATEGORIES) + " | 합계 |")
    lines.append("|" + "---|" * (len(CATEGORIES) + 1))
    lines.append("| " + " | ".join(str(int(supp.get(c, 0))) for c in CATEGORIES)
                 + f" | {len(df)} |")
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
        labs = CATEGORIES + ([PARSE_FAIL] if fail_counts[col] else [])
        cm = confusion_matrix(y, df[col], labels=labs)
        cm_df = pd.DataFrame(cm, index=CATEGORIES + ([PARSE_FAIL] if fail_counts[col] else []),
                              columns=labs).loc[CATEGORIES]
        lines.append("")
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| gold \\ pred | " + " | ".join(labs) + " |")
        lines.append("|" + "---|" * (len(labs) + 1))
        for c in CATEGORIES:
            lines.append(f"| **{c}** | " + " | ".join(str(int(v)) for v in cm_df.loc[c]) + " |")

    lines.append("")
    lines.append("## 파싱 실패 로그")
    lines.append("")
    if sum(fail_counts.values()) == 0:
        lines.append("5조건 전부 파싱 실패/빈 응답 0건. 신규 3 run은 재시도 발생 0건.")
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
    print("MODEL_PILOT_EVAL_COMPLETE")


if __name__ == "__main__":
    main()
