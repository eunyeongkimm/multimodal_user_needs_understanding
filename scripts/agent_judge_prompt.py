"""상담사 대응 품질 GPT-as-judge 프롬프트 (텍스트 전용 게이트용).

acoustic 미사용 — 이번 게이트는 상담사의 텍스트 대응만 평가한다.

★누수 차단 원칙 (이 파일에서 반드시 지켜야 함)
  - gold_actual / 정답 카테고리 / 카테고리 taxonomy / mismatch 어떤 것도
    프롬프트에 넣지 않는다.
  - "상담사가 고객의 실제 의도를 맞췄는가"로 채점하지 않는다.
  - 채점 근거는 전사 텍스트에 드러난 상담사의 말로 한정한다.

출력: 태그 포맷 <친절도>N</친절도><적절대응>N</적절대응> (N = 1~5 정수)
"""

import re

JUDGE_INSTRUCTION = (
    "당신은 고객센터 상담 품질 평가자입니다. 상담사의 응대 태도와 방식만 평가하며, "
    "고객의 문의가 어떤 유형인지 분류하거나 추측하지 않습니다."
)

PROMPT_TEMPLATE = """[지시문]
아래는 전자상거래(온라인 교육 플랫폼) 고객센터 통화의 전체 전사입니다.
이 통화에서 **상담사의 응대 품질**만 평가하세요.
판단 근거는 전사에 실제로 적힌 상담사의 말로만 삼으세요.

[평가 항목]
1. 친절도 (1~5): 공손한 말투, 고객 감정에 대한 공감 표현, 배려하는 표현.
   - 5: 일관되게 공손하고 공감·배려 표현이 뚜렷함
   - 3: 사무적이지만 무례하지 않음
   - 1: 무뚝뚝하거나 퉁명스럽고 공감·배려 표현이 없음

2. 적절대응 (1~5): 고객 말을 재확인·요약하는지, 필요한 정보를 되묻는지,
   구체적 해결을 시도하는지, 상황에 맞는 응대를 하는지.
   - 5: 재확인·되묻기가 충실하고 구체적 해결을 끝까지 시도함
   - 3: 기본적인 안내는 하나 재확인이나 해결 시도가 부족함
   - 1: 고객 말을 흘려듣거나 해결 시도 없이 대화가 끊김

[중요 - 평가 대상이 아닌 것]
- 고객의 문의가 어떤 범주인지 분류하거나 추측하지 마세요.
- 상담사가 고객의 "진짜 의도"를 알아맞혔는지 여부로 점수를 매기지 마세요.
- 문제가 최종적으로 해결되었는지(결과)가 아니라, 상담사의 **응대 방식**만 보세요.
- 고객의 태도는 평가 대상이 아닙니다.

[통화 전체 전사]
{transcript}

[질문]
위 두 항목을 각각 1~5 정수로 채점하세요.

출력 형식(반드시 준수, 다른 설명 금지):
<친절도>정수</친절도>
<적절대응>정수</적절대응>
"""

TRANSCRIPT_MARK = "[통화 전체 전사]"

KIND_RE = re.compile(r"<친절도>\s*([0-9]+)\s*</친절도>")
APPR_RE = re.compile(r"<적절대응>\s*([0-9]+)\s*</적절대응>")


def build_prompt(transcript: str) -> str:
    return PROMPT_TEMPLATE.format(transcript=transcript)


def parse_judge(raw: str):
    """(친절도, 적절대응, error) 반환. 1~5 범위 밖이면 실패 처리."""
    if raw is None or not str(raw).strip():
        return None, None, "빈응답"
    mk, ma = KIND_RE.search(str(raw)), APPR_RE.search(str(raw))
    if not mk or not ma:
        missing = []
        if not mk:
            missing.append("친절도")
        if not ma:
            missing.append("적절대응")
        return None, None, f"형식오류(태그누락:{','.join(missing)})"
    k, a = int(mk.group(1)), int(ma.group(1))
    if not (1 <= k <= 5) or not (1 <= a <= 5):
        return None, None, f"범위오류(친절도={k}, 적절대응={a})"
    return k, a, None
