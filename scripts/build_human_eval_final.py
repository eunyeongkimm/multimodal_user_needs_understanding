"""human_eval_sample.csv + pilot200_gpt_labels.csv + inbound_outbound_gpt.csv를 합쳐
사람이 한 파일에서 니즈 라벨링과 인바운드/아웃바운드 판별을 같이 할 수 있는
outputs/human_eval_final.csv를 만든다.

컬럼: call_id, call_text, gpt55_label, gpt41mini_label, call_type_gpt, reason_gpt,
      human_label(비어있음), human_call_type(비어있음)

call_text는 원래 "화자: 발화" 줄바꿈(\n) 연결 텍스트인데, 엑셀/구글시트에서 한 콜이
한 행에 깔끔히 들어가도록 줄바꿈을 " / "로 바꿔서 한 줄로 펼친다.
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
HUMAN_EVAL_PATH = BASE_DIR / "outputs" / "human_eval_sample.csv"
GPT_LABELS_PATH = BASE_DIR / "outputs" / "pilot200_gpt_labels.csv"
INBOUND_OUTBOUND_PATH = BASE_DIR / "outputs" / "inbound_outbound_gpt.csv"
FINAL_PATH = BASE_DIR / "outputs" / "human_eval_final.csv"


def main():
    human_df = pd.read_csv(HUMAN_EVAL_PATH)
    gpt_labels_df = pd.read_csv(GPT_LABELS_PATH)
    io_df = pd.read_csv(INBOUND_OUTBOUND_PATH)

    df = human_df.merge(gpt_labels_df, on="call_id", how="left")
    df = df.merge(io_df, on="call_id", how="left")

    # 기존 human_label 값(있다면) 보존, 없으면 빈 문자열
    df["human_label"] = df["human_label"].fillna("")
    df["human_call_type"] = ""

    # 엑셀/구글시트에서 한 콜 = 한 행으로 깔끔하게 보이도록 줄바꿈을 구분자로 치환
    df["call_text"] = df["call_text"].str.replace("\n", " / ", regex=False)

    df = df[
        [
            "call_id",
            "call_text",
            "gpt55_label",
            "gpt41mini_label",
            "call_type_gpt",
            "reason_gpt",
            "human_label",
            "human_call_type",
        ]
    ]

    missing = df[df["gpt55_label"].isna() | df["call_type_gpt"].isna()]
    if len(missing):
        print(f"경고: {len(missing)}개 call_id가 gpt55_label 또는 call_type_gpt와 매칭되지 않았습니다.")
        print(missing["call_id"].tolist())

    df.to_csv(FINAL_PATH, index=False, encoding="utf-8-sig")
    print(f"저장: {FINAL_PATH} ({len(df)}행)")


if __name__ == "__main__":
    main()
