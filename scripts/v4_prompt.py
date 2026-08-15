"""v4 프롬프트: 2단계 판단(항의 여부 게이트 -> 카테고리 분류) + JSON 출력.
v3(우선순위 규칙을 카테고리 정의 사이에 끼워넣는 방식)이 human_eval에서
환불요청/주문취소 경계를 흔들어 Kappa를 떨어뜨린 문제를 피하기 위해,
"항의 여부"를 완전히 분리된 1단계 게이트로 만들고, 게이트를 통과하지 못한
콜만 기존 6개 카테고리로 분류하도록 구조 자체를 바꾼 버전.
"""

import json
import re

CATEGORIES_V4 = ["환불요청", "주문취소", "배송확인", "교환반품", "구매진행", "서비스이용"]

PROMPT_TEMPLATE_V4 = """[지시문]
당신은 전자상거래(온라인 교육 플랫폼) 고객센터 상담 통화를 분석하는
전문가입니다. 아래 통화를 읽고 두 단계로 판단하세요.

[1단계 - 항의 여부 판단]
이 통화에서 고객이 아래와 같은 항의성 신호를 뚜렷하게 보이는지 판단하세요:
- 이전 통화/약속 불이행 지적 ("지난번에 해준다고 했는데", "몇 번을 전화했는지 아세요")
- 처리 지연에 대한 명시적 불만 ("한 달이 넘어가는데 왜")
- 응대 방식에 대한 직접적 비판 ("일방적으로 처리하시니까")
- 반복 압박·감정적 강조 표현 ("당연히 되야 되는 거 아니에요?")
※ 단순 반복 접촉(재문의) 자체만으로는 해당하지 않음 - 어조가 담담하고
  사실 확인 위주면 항의 아님

[2단계 - 카테고리 분류]
1단계에서 항의성이 뚜렷하면 -> "불만제기"로 최종 분류하고 2단계는 건너뛸 것.
항의성이 없으면 -> 아래 6개 카테고리 중 하나로 분류:

1. 환불요청: 결제한 금액을 돌려받는 것이 최종 목적인 경우
   - "취소", "반품", "반송"이라는 단어가 나와도, 최종 목적이
     금전 반환이면 환불요청
   - 예: "재결제하고 전체 카드취소 해주세요" (재결제는 절차일 뿐)
2. 주문취소: 배송/수강 전, 환불 절차 없이 순수 주문 취소
3. 배송확인: 배송 상태·도착 문의
4. 교환반품: 불량·오배송 -> 물건↔물건 교체 (금전 반환 아님)
5. 구매진행: 결제 완료를 위한 도움 요청
6. 서비스이용: 로그인·기기·앱·시스템 이용 문제

[통화 전체 내용]
{call_text}

[출력 형식 - JSON만 출력]
{{"has_complaint": true 또는 false, "label": "카테고리명"}}
(has_complaint가 true면 label은 반드시 "불만제기")
"""


def make_prompt(call_text: str) -> str:
    return PROMPT_TEMPLATE_V4.format(call_text=call_text)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_v4_response(raw: str):
    """모델 응답에서 {"has_complaint":..., "label":...}를 추출한다.
    파싱 실패 시 (None, None) 반환. has_complaint가 true인데 label이
    다른 값이면 스펙대로 강제로 "불만제기"로 보정한다."""
    if raw is None:
        return None, None
    m = _JSON_RE.search(raw)
    if not m:
        return None, None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None, None

    has_complaint = obj.get("has_complaint")
    label = obj.get("label")
    if not isinstance(has_complaint, bool):
        return None, None
    if has_complaint:
        label = "불만제기"
    if label not in (CATEGORIES_V4 + ["불만제기"]):
        return has_complaint, None
    return has_complaint, label


if __name__ == "__main__":
    print(PROMPT_TEMPLATE_V4.format(call_text="[예시 통화 텍스트]"))
    print(parse_v4_response('설명: 네 알겠습니다\n{"has_complaint": true, "label": "환불요청"}'))
    print(parse_v4_response('{"has_complaint": false, "label": "배송확인"}'))
