import os
import json
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import openai

# ✅ 최신 LangChain 구조
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain.retrievers.multi_query import MultiQueryRetriever
from qdrant_client import QdrantClient

from geopy.distance import geodesic
from langchain_core.messages import HumanMessage

# ====================
# 🛠️ Logging 설정
# ====================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ====================
# 🔐 환경 변수 로딩
# ====================
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ====================
# 🚀 Flask 앱 시작
# ====================
app = Flask(__name__)

# ====================
# 🧠 Qdrant + LangChain 설정
# ====================
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

qdrant_store = Qdrant(
    client=qdrant_client,
    collection_name="menus",
    embeddings=embedding_model,
    content_payload_key="page_content",
)

llm = ChatOpenAI(model="gpt-4o")

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=qdrant_store.as_retriever(search_kwargs={"k": 5}),
    llm=llm
)

# ====================
# 🔁 파생 질문 생성 함수
# ====================
def generate_followup_queries_with_llm(message, llm):
    prompt = f"""
사용자 질문: "{message}"

위 질문에 대해 사용자가 관심 가질 만한 관련 추가 질문을 3가지 자연스럽게 만들어줘.
각 질문은 한 줄로, 리스트 형태 없이 문자열 배열로 생성해줘.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()

    # JSON 형식일 경우 처리
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [q.strip() for q in parsed]
    except json.JSONDecodeError:
        pass

    # 아닐 경우 개행 기준 처리
    return [line.strip("•-— ") for line in content.split("\n") if line.strip()]

# ====================
# 🎯 추천 API
# ====================
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    message = data.get("message")
    lat = data.get("latitude")
    lon = data.get("longitude")
    restrictions = data.get("restrictions", [])
    previousResults = data.get("previousResults", [])

    logger.info(f"📥 사용자 메시지: {message}")
    logger.info(f"📍 위치: ({lat}, {lon})")
    logger.info(f"⛔ 제한 음식: {restrictions}")
    logger.info(f"🔁 이전 결과: {previousResults}")

    def is_within_radius(user_lat, user_lon, target_lat, target_lon, radius_km=10):
        try:
            return geodesic((user_lat, user_lon), (target_lat, target_lon)).km <= radius_km
        except Exception as e:
            logger.warning(f"❗ 거리 계산 실패: {e}")
            return False

    try:
        docs = multi_query_retriever.get_relevant_documents(message)
        logger.info(f"🔎 문서 검색 결과 수: {len(docs)}")

        ENABLE_FILTER = False

        filtered_results = []
        for doc in docs:
            meta = getattr(doc, "metadata", {})
            if not meta:
                continue

            if ENABLE_FILTER:
                loc = meta.get("location")
                if loc:
                    if not is_within_radius(lat, lon, loc.get("lat"), loc.get("lon"), 3):
                        continue

            menu_name = meta.get("menu", "").lower()
            if any(res.lower() in menu_name for res in restrictions):
                continue

            filtered_results.append(meta)

        logger.info(f"✅ 필터링 결과 수: {len(filtered_results)}")

        results = [{
            "menuId": p.get("menuId"),
            "menu": p.get("menu"),
            "category": p.get("category"),
            "price": p.get("price"),
            "restaurant": p.get("restaurant"),
            "restaurantId": p.get("restaurantId"),
            "address": p.get("address"),
            "hasAR": p.get("hasAR"),
            "hasCoupon": p.get("hasCoupon"),
            "location": p.get("location"),
            "description": p.get("description"),
            "page_content": p.get("page_content"),
        } for p in filtered_results]

        logger.info("🍽️ 최종 추천 결과:\n%s", json.dumps(results, ensure_ascii=False, indent=2))

        gpt_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "음식 추천 결과를 사용자에게 자연스럽게 설명해줘."},
                {"role": "user", "content": f"다음 추천 리스트를 자연스럽게 한 문장으로 소개해줘:\n{json.dumps(results, ensure_ascii=False)}"}
            ],
            temperature=0.7
        )
        natural_response = gpt_response.choices[0].message.content

        followups = generate_followup_queries_with_llm(message, llm)

        return jsonify({
            "recommendation": results,
            "response": natural_response,
            "followUpSuggestions": followups
        })

    except Exception as e:
        logger.exception("❌ 추천 처리 중 예외 발생")
        return jsonify({"error": f"추천 실패: {str(e)}"}), 500

# ====================
# 🧪 앱 실행
# ====================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
