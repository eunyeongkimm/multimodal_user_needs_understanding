"""v3 프롬프트: v2 프롬프트에 '불만제기 우선판단 규칙'을 추가한 버전.

배경: 샘플 점검 결과, "환불요청"/"배송확인"으로 분류된 콜 중 일부가
실제로는 항의성 콜(반복 통화, 약속 불이행, 처리 지연에 대한 명시적 불만 등)인데도
용건(환불/배송)만 보고 분류되어 "불만제기"가 흡수되는 경향이 확인됨.
v3는 이런 항의 신호를 다른 용건보다 우선 판단하도록 규칙을 추가한다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relabel_v2_and_kappa import CATEGORIES, PROMPT_TEMPLATE_V2  # noqa: F401,E402

PRIORITY_RULE_BLOCK = """[판단 원칙 - 우선순위 추가]
- 통화에서 아래와 같은 항의성 신호가 하나라도 뚜렷하게 나타나면, 다른 용건(환불/배송 등)과
  무관하게 최우선으로 "불만제기"로 분류할 것:
  - 이전 통화/약속에 대한 불이행을 지적 ("지난번에 해준다고 했는데", "몇 번을 전화했는지 아세요")
  - 처리 지연에 대한 명시적 불만 ("한 달이 넘어가는데 왜", "몇 시간이나 지난 거 아니에요")
  - 상담사/회사의 응대 방식에 대한 직접적 비판 ("일방적으로 처리하시니까", "매번 이렇게 안 오고")
  - 반복적 압박·강조 표현으로 감정이 실린 경우 ("당연히 되야 되는 거 아니에요?", "이해가 안 가서")
- 용건 자체(환불, 배송 등)가 명확해 보여도 항의 톤이 함께 있다면 불만제기를 우선한다.
- 단순 지연 문의라도, 반복 접촉이 있었지만 톤이 담담하고 사실 확인 위주라면 원래 용건
  (환불요청/배송확인)을 유지한다 (반복 접촉 자체만으로 불만제기로 분류하지 말 것 -
  어조/표현이 핵심 판단 기준)

"""

# v2 템플릿의 "[판단 원칙 - 공통]" 섹션 바로 앞에 우선순위 규칙 블록을 삽입한다.
_MARKER = "[판단 원칙 - 공통]"
assert _MARKER in PROMPT_TEMPLATE_V2, "v2 프롬프트 구조가 변경된 것 같습니다. 마커를 확인하세요."
PROMPT_TEMPLATE_V3 = PROMPT_TEMPLATE_V2.replace(_MARKER, PRIORITY_RULE_BLOCK + _MARKER)


def make_prompt(call_text: str) -> str:
    return PROMPT_TEMPLATE_V3.format(call_text=call_text)


if __name__ == "__main__":
    print(PROMPT_TEMPLATE_V3.format(call_text="[예시 통화 텍스트]"))
