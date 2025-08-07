import openai
import os
import json
import re
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def clean_response(text: str, max_lines=3) -> str:
    # 문자열 내부의 \n 제거
    text = text.replace("\n", "")

    # 특수기호 제거 및 정제
    lines = [re.sub(r"[^\w\s가-힣,.]", "", line.strip()) for line in text.splitlines()]
    lines = [line for line in lines if line]

    return " ".join(lines[:max_lines])

def generate_response(state: dict):

    
    try:
        filtered_results = state["filtered_results"][:3]

        if not filtered_results:
            response = "죄송해요, 조건에 맞는 메뉴를 찾지 못했어요."
        else:

            completion = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "당신은 음식 추천 챗봇입니다. "
                            "사용자에게 각 메뉴를 한 문장으로 간결하게 소개해 주세요. "
                            "특수기호, 이모지, 번호, 별표 등은 절대 사용하지 말고, 간단한 문장만 3개 써주세요."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"다음 3가지 음식 메뉴를 간단히 소개해줘:\n{json.dumps(filtered_results, ensure_ascii=False)}"
                    }
                ],
                temperature=0.6
            )

            response = completion.choices[0].message.content
            response = clean_response(response)

        logger.info("💬 GPT 응답 생성 완료")
        return {**state, "gpt_response": response}

    except Exception as e:
        logger.exception("❌ [generate_response] 오류 발생: %s", e)
        raise
