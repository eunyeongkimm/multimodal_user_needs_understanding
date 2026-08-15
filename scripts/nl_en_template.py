"""음성 description의 영어 템플릿.

한국어 description을 번역한 것이 아니라, 동일한 quintile level 값에서 영어 문장을
직접 생성한다. stage2c_v2_normalize_nl_gender.py 의 METRIC_DESC / build_nl_description
와 **같은 METRICS 순서·같은 LEVEL_LABELS·같은 분기**를 쓰며, 입력으로 받는
`{metric}_level` 컬럼도 동일한 것을 쓴다(= 같은 숫자 -> 같은 분기가 구조적으로 보장).

한국어 원본(참조):
    LEVEL_LABELS = ["매우낮음", "낮음", "보통", "높음", "매우높음"]
    METRICS = ["f0_mean", "f0_std", "e_mean_db", "e_std_db", "r_gold", "s_top_db30"]
"""

LEVEL_LABELS = ["매우낮음", "낮음", "보통", "높음", "매우높음"]
METRICS = ["f0_mean", "f0_std", "e_mean_db", "e_std_db", "r_gold", "s_top_db30"]

# (영어 prefix, {level: 영어 phrase}) — 한국어 METRIC_DESC와 1:1 대응
METRIC_DESC_EN = {
    "f0_mean": ("vocal pitch is", {
        "매우낮음": "very low", "낮음": "somewhat low", "보통": "average",
        "높음": "somewhat high", "매우높음": "very high"}),
    "f0_std": ("pitch variation is", {
        "매우낮음": "very small", "낮음": "somewhat small", "보통": "average",
        "높음": "somewhat large", "매우높음": "very large"}),
    "e_mean_db": ("loudness (energy) is", {
        "매우낮음": "very low", "낮음": "somewhat low", "보통": "average",
        "높음": "somewhat high", "매우높음": "very high"}),
    "e_std_db": ("loudness variation is", {
        "매우낮음": "very small", "낮음": "somewhat small", "보통": "average",
        "높음": "somewhat large", "매우높음": "very large"}),
    "r_gold": ("speaking rate is", {
        "매우낮음": "very slow", "낮음": "somewhat slow", "보통": "average",
        "높음": "somewhat fast", "매우높음": "very fast"}),
    "s_top_db30": ("silence ratio is", {
        "매우낮음": "very low", "낮음": "somewhat low", "보통": "average",
        "높음": "somewhat high", "매우높음": "very high"}),
}

DEFAULT_LEVEL = "보통"


def build_nl_description_en(row) -> str:
    """한국어 build_nl_description과 동일한 루프 구조. 구분자만 ', '로 같다."""
    parts = []
    for m in METRICS:
        label = row[f"{m}_level"]
        prefix, phrase_map = METRIC_DESC_EN[m]
        phrase = phrase_map.get(label, phrase_map[DEFAULT_LEVEL])
        parts.append(f"{prefix} {phrase}")
    return ", ".join(parts)


def verify_parity(ko_desc_module) -> list:
    """한국어 템플릿과 metric 순서/level 키 집합이 동일한지 검증. 불일치 목록 반환."""
    problems = []
    if list(ko_desc_module.METRICS) != list(METRICS):
        problems.append(f"METRICS 순서 불일치: {ko_desc_module.METRICS} vs {METRICS}")
    if list(ko_desc_module.LEVEL_LABELS) != list(LEVEL_LABELS):
        problems.append("LEVEL_LABELS 불일치")
    for m in METRICS:
        ko_keys = set(ko_desc_module.METRIC_DESC[m][1])
        en_keys = set(METRIC_DESC_EN[m][1])
        if ko_keys != en_keys:
            problems.append(f"{m}: level 키 집합 불일치 {ko_keys ^ en_keys}")
    return problems
