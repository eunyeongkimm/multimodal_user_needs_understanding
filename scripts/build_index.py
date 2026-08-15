import json
import os
import re
import sys
import time
import pandas as pd

# 2026-07-12: 라벨링데이터/D04 압축 해제 완료 확인 (J16~J19 개수가 원천데이터 S-폴더 수와 정확히
# 일치, dir 기반 재빌드 결과가 zip 기반 결과와 100% 동일함을 diff로 검증함).
# 이후부터는 LBL_DIR을 정본 소스로 사용. zip 경로는 미추출 상황을 위한 폴백으로만 남겨둠.
LBL_DIR = "/Volumes/AIHub/007.저음질 전화망 음성인식 데이터/01.데이터/1.Training/라벨링데이터_230316/D04"
LBL_ZIP = "/Volumes/AIHub/007.저음질 전화망 음성인식 데이터/01.데이터/1.Training/라벨링데이터_230316/TL_D04.zip"  # fallback

PROJECT_DIR = "/Users/eunyeongkim/Desktop/skku-ads/1.논문"
OUT_PARQUET = os.path.join(PROJECT_DIR, "outputs", "d04_dialog_index.parquet")


def parse_json_record(call_id, j_code, s_code, data):
    ds = data["dataSet"]
    type_info = ds.get("typeInfo", {})
    category = type_info.get("category")
    subcategory = type_info.get("subcategory")
    speakers = {s["id"]: s for s in type_info.get("speakers", [])}
    dialogs = ds.get("dialogs", [])
    records = []
    for dialog_idx, dlg in enumerate(dialogs):
        sp_id = dlg.get("speaker")
        sp = speakers.get(sp_id, {})
        records.append({
            "call_id": call_id,
            "j_code": j_code,
            "s_code": s_code,
            "category": category,
            "subcategory": subcategory,
            "speaker_id": sp_id,
            "speaker_type": sp.get("type"),
            "speaker_gender": sp.get("gender"),
            "speaker_age": sp.get("age"),
            "telephone_network": sp.get("telephone_network"),
            "dialog_idx": dialog_idx,
            "audioPath": dlg.get("audioPath"),
            "duration": dlg.get("duration"),
            "text": dlg.get("text"),
        })
    return records


def iter_json_from_dir(lbl_dir):
    for j_code in sorted(os.listdir(lbl_dir)):
        j_path = os.path.join(lbl_dir, j_code)
        if not os.path.isdir(j_path):
            continue
        for s_code in sorted(os.listdir(j_path)):
            s_path = os.path.join(j_path, s_code)
            json_path = os.path.join(s_path, f"{s_code}.json")
            if os.path.isfile(json_path):
                with open(json_path, encoding="utf-8") as f:
                    yield j_code, s_code, json.load(f)


def iter_json_from_zip(lbl_zip):
    import zipfile
    json_re = re.compile(r"^D04/(J\d+)/(S\d+)/\2\.json$")
    with zipfile.ZipFile(lbl_zip) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".json")):
            m = json_re.match(name)
            if not m:
                continue
            with z.open(name) as f:
                yield m.group(1), m.group(2), json.load(f)


if __name__ == "__main__":
    rows = []
    n_ok = 0
    n_err = 0
    t0 = time.time()

    source = iter_json_from_dir(LBL_DIR) if os.listdir(LBL_DIR) else iter_json_from_zip(LBL_ZIP)

    for i, (j_code, s_code, data) in enumerate(source, 1):
        call_id = f"{j_code}_{s_code}"
        try:
            rows.extend(parse_json_record(call_id, j_code, s_code, data))
            n_ok += 1
        except Exception as e:
            n_err += 1
            print(f"[ERROR] {call_id}: {e}", file=sys.stderr)

        if i % 1000 == 0:
            elapsed = time.time() - t0
            print(f"[{i}] json 처리 완료 (ok={n_ok}, err={n_err}, {elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"[{n_ok + n_err}] json 처리 완료 (ok={n_ok}, err={n_err}, {elapsed:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\nDONE. rows={len(df)}, calls_ok={n_ok}, calls_err={n_err}")
    print(f"parquet saved to: {OUT_PARQUET}")
