import os
import re
import shutil
import pandas as pd

PROJECT_DIR = "/Users/eunyeongkim/Desktop/skku-ads/1.논문"
PARQUET = os.path.join(PROJECT_DIR, "outputs", "d04_dialog_index.parquet")
BACKUP = os.path.join(PROJECT_DIR, "outputs", "d04_dialog_index_backup_before_clean.parquet")

if not os.path.exists(BACKUP):
    shutil.copy(PARQUET, BACKUP)
    print(f"백업 저장: {BACKUP}")
else:
    print(f"백업 이미 존재: {BACKUP} (재사용)")

df = pd.read_parquet(PARQUET)
# 혹시 이전 실행에서 컬럼이 이미 추가돼있으면 원본 백업에서 다시 시작
if "text_clean" in df.columns:
    df = pd.read_parquet(BACKUP)

texts = df["text"].fillna("").astype(str)

# ---------------------------------------------------------------------------
# 조사 결과 요약 (investigate_alpha_runs.py):
#   - [A-Za-z]+/ 형태로 나타나는 alpha-run은 총 53종.
#   - 길이 1(단일 글자)이면서, preceded-by-context 검사 결과 전부(또는 거의 전부)
#     공백/문장시작/한글 뒤에서만 등장 -> 진짜 disfluency 태그로 확인:
#       o(503556), n(179553), b(27041), l(3556), O(73), I(32), p(27), N(2), i(2), f(3), h(1)
#     ('T'는 유일하게 항상 "(" 바로 뒤에서만 등장 -> "(T/티)" 처럼 영문자 발음
#      교정쌍이지 태그가 아님 -> 화이트리스트에서 제외)
#   - 길이 2 이상 alpha-run(ID, cash, site, page, pass, PC, ARS, EBS, Gmail, Wifi,
#     COM, TABLET, click, tab, phone, mail, plus, tablet, button, homepage, start,
#     premium, mileage, one, book, basic, center, run, set, no, on, in, it, pc,
#     raping, bl, oi, ll, oo, ln, xxn 등 41종, 89회) -> 전부 영어 단어/약어 발음
#     교정쌍의 원문 쪽이며 태그가 아님 -> (원문)/(교정) 로직으로 넘김.
#
#   버그 발견 및 수정: 처음에 "\(*...\)*" 형태로 괄호 결손을 관용적으로 허용하는
#   교정쌍 정규식을 썼더니, 괄호가 전혀 없는 오탈자 슬래시(예: "0/", 잘못 붙은
#   "/")가 시작점이 되어 최대 120자 떨어진 무관한 문장의 ")"까지 통째로 삼켜버리는
#   버그가 있었음 (예: "...0/ (한 권만)/(1권만)" 앞의 멀쩡한 문장이 통째로 지워짐).
#   -> RAW 바로 앞에 실제 "(" 가 있어야만 매치가 시작되도록 강제해서 해결.
# ---------------------------------------------------------------------------

TAG_WHITELIST = {"o", "n", "b", "l", "i", "p", "f", "h"}
TAG_RE = re.compile(r"(?<![A-Za-z])([A-Za-z])/")
DOUBLE_PAREN_RE = re.compile(r"\(\(.*?\)\)+")
# 정상: (원문)/(교정)  |  파손(괄호 한쪽 누락): (원문/(교정)  또는  (원문)/교정)
# RAW 앞에 반드시 실제 "(" 존재 -> 무관한 위치에서 시작해 멀리 있는 ")"와
# 잘못 이어붙는 것을 방지.
CORRECTION_PAIR_RE = re.compile(r"\(([^()/]*)\)?/\(?([^()]*)\)")
SINGLE_PAREN_RE = re.compile(r"\([^()]*\)")
STRAY_BRACKET_RE = re.compile(r"[()/]")
WS_RE = re.compile(r"\s+")
HANGUL_SYLLABLE_RE = re.compile(r"[가-힣]")


def remove_whitelisted_tags(t):
    def repl(m):
        letter = m.group(1).lower()
        return " " if letter in TAG_WHITELIST else m.group(0)
    return TAG_RE.sub(repl, t)


MAIN_TAGS = {"o", "n", "b", "l"}


def count_tags(t):
    counts = {f"disfluency_{c}_count": 0 for c in MAIN_TAGS}
    total = 0
    for m in TAG_RE.finditer(t):
        letter = m.group(1).lower()
        if letter in TAG_WHITELIST:
            total += 1
            if letter in MAIN_TAGS:
                counts[f"disfluency_{letter}_count"] += 1
    counts["disfluency_tag_count"] = total
    return counts


def count_correction_pairs(t):
    return len(CORRECTION_PAIR_RE.findall(t))


def has_double_paren(t):
    return bool(DOUBLE_PAREN_RE.search(t))


def clean_text(t):
    # 1. (( ... )) 불명확 발화 블록 통째로 제거 (내부에 중첩된 태그도 함께 제거됨)
    t = DOUBLE_PAREN_RE.sub(" ", t)
    # 2. 화이트리스트 disfluency 태그 제거 (교정쌍 안에 중첩된 태그도 먼저 제거해서
    #    이후 교정쌍 매칭이 깨끗해지도록 함)
    t = remove_whitelisted_tags(t)
    # 3. (원문)/(교정) 쌍 -> 교정만 남김 (괄호 한쪽 누락된 파손 케이스도 관용적으로 처리)
    t = CORRECTION_PAIR_RE.sub(lambda m: m.group(2), t)
    # 4. 기타 단일 괄호 ( ... ) 통째로 제거 (태그류로 간주)
    t = SINGLE_PAREN_RE.sub(" ", t)
    # 5. 안전망: 남은 미짝 괄호/슬래시 잔재 제거 (손상된 라벨링 잔여물, 오탈자 슬래시 등)
    t = STRAY_BRACKET_RE.sub(" ", t)
    # 6. 공백 정리
    t = WS_RE.sub(" ", t).strip()
    return t


if __name__ == "__main__":
    print("태그/교정쌍 카운트 계산 중 (원본 text 기준)...")
    tag_counts = [count_tags(t) for t in texts]
    tag_counts_df = pd.DataFrame(tag_counts)

    correction_pair_count = texts.apply(count_correction_pairs)
    double_paren_present = texts.apply(has_double_paren)

    print("text_clean 생성 중...")
    text_clean = texts.apply(clean_text)

    unintelligible_flag = double_paren_present & (text_clean.str.len() == 0)
    text_clean_too_short_flag = text_clean.str.len() <= 2

    syllable_count = text_clean.apply(lambda s: len(HANGUL_SYLLABLE_RE.findall(s)))

    duration = df["duration"]
    r_gold = syllable_count / duration.replace(0, pd.NA)

    df["text_clean"] = text_clean
    df = pd.concat([df, tag_counts_df], axis=1)
    df["correction_pair_count"] = correction_pair_count
    df["unintelligible_flag"] = unintelligible_flag
    df["text_clean_too_short_flag"] = text_clean_too_short_flag
    df["syllable_count"] = syllable_count
    df["r_gold"] = r_gold

    df.to_parquet(PARQUET, index=False)
    print(f"\n업데이트된 parquet 저장: {PARQUET}")
    print(f"컬럼: {list(df.columns)}")

    # -----------------------------------------------------------------------
    # 검증
    # -----------------------------------------------------------------------
    print("\n=== 검증 ===")
    n_total = len(df)
    n_empty_clean = (df["text_clean"].str.len() == 0).sum()
    print(f"text_clean 완전히 빈 문자열: {n_empty_clean} / {n_total} ({n_empty_clean/n_total*100:.3f}%)")
    print(f"  - unintelligible_flag=True (원인: (( ))): {df.loc[df['text_clean'].str.len()==0, 'unintelligible_flag'].sum()}")
    print(f"  - unintelligible_flag=False (다른 원인): {(~df.loc[df['text_clean'].str.len()==0, 'unintelligible_flag']).sum()}")

    n_short = df["text_clean_too_short_flag"].sum()
    print(f"text_clean 길이 <= 2: {n_short} ({n_short/n_total*100:.3f}%)")

    n_unintelligible = df["unintelligible_flag"].sum()
    print(f"unintelligible_flag=True: {n_unintelligible} ({n_unintelligible/n_total*100:.3f}%)")

    n_stray = df["text_clean"].str.contains(r"[()/]", regex=True).sum()
    print(f"text_clean에 남아있는 (),/ 잔여 문자 (0이어야 정상): {n_stray}")
    if n_stray > 0:
        print("  예시:")
        print(df.loc[df['text_clean'].str.contains(r'[()/]', regex=True), 'text_clean'].head(5).to_string())

    print(f"\ndisfluency_tag_count 합계 (o/n/b/l 종류별, 화이트리스트 소수태그 포함 total):")
    for c in ["o", "n", "b", "l"]:
        print(f"  {c}/ : {df[f'disfluency_{c}_count'].sum()}")
    print(f"  total(i/p/f/h 등 소수 화이트리스트 태그 포함): {df['disfluency_tag_count'].sum()}")

    print(f"\ncorrection_pair_count 합계: {df['correction_pair_count'].sum()}")
    print(f"r_gold 통계: mean={df['r_gold'].mean():.3f}, median={df['r_gold'].median():.3f}, "
          f"min={df['r_gold'].min():.3f}, max={df['r_gold'].max():.3f}, NaN 개수={df['r_gold'].isna().sum()}")

    # 텍스트 손상(과도 매칭) 여부 최종 안전 점검: text_clean 길이가 원본보다
    # 비정상적으로 많이 줄었는지(이상치) 체크 -> 정상 범위면 이상 없음
    len_ratio = text_clean.str.len() / texts.str.len().replace(0, pd.NA)
    print(f"\ntext_clean/원본 길이 비율: mean={len_ratio.mean():.3f}, median={len_ratio.median():.3f}, min={len_ratio.min():.3f}")
