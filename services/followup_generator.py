from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json
import logging
llm = ChatOpenAI(model="gpt-4o")
logger = logging.getLogger(__name__)

def safe_parse_followups(raw: str):
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], str):
            nested = json.loads(parsed[0])
            if isinstance(nested, list):
                return nested
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    fallback = [line.strip("•-–— []\",") for line in raw.splitlines() if line.strip()]
    return fallback

def safe_parse_followups(raw: str):
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], str):
            nested = json.loads(parsed[0])
            if isinstance(nested, list):
                return nested
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    fallback = [line.strip("•-–— []\",") for line in raw.splitlines() if line.strip()]
    return fallback

def generate_followups(state: dict):
    try:
        prompt = f"""
사용자 질문: "{state['message']}"

이 질문과 관련해 사용자가 더 탐색할 수 있는 주제 키워드를 3가지 제안해줘.
- 반드시 ~~추천 해줘 라는 키워드로 끝날것
- 반드시 JSON 배열로만 출력할 것
- 설명, 줄바꿈, 따옴표, 특수기호 절대 사용하지 마

"""

        result = llm.invoke([HumanMessage(content=prompt)])
        raw = result.content.strip()

        followups = safe_parse_followups(raw)

        logger.info("📌 파생 질문 생성 완료: %s", followups)
        return {**state, "followups": followups}

    except Exception as e:
        logger.exception("❌ [generate_followups] 오류 발생: %s", e)
        return {**state, "followups": []}