import os
import requests
import openai
from dotenv import load_dotenv
from datetime import datetime
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# 1. .env 불러오기
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# 2. Qdrant 연결
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# 3. 기존 컬렉션 삭제 후 새로 만들기
def recreate_collection():
    if client.collection_exists("restaurants"):
        client.delete_collection("restaurants")
        print("🗑️ 기존 'restaurants' 컬렉션 삭제 완료")
    client.create_collection(
        collection_name="menus",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    print("✅ 새 컬렉션 'restaurants' 생성 완료")

# 4. Spring Boot에서 메뉴 리스트 가져오기
def fetch_restaurants():
    try:
        response = requests.get("http://52.79.235.168:8080/api/restaurants/vectorize")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ 메뉴 데이터 불러오기 실패: {e}")
        return []

# 5. 주소 문자열 변환
def format_address(address):
    parts = [
        address.get("province"),
        address.get("city"),
        address.get("roadName"),
        address.get("buildingNumber"),
        address.get("detail")
    ]
    return " ".join(filter(None, parts))

# 6. 전체 실행
def run_langchain_insert():
    recreate_collection()
    restaurants = fetch_restaurants()
    if not restaurants:
        print("⚠️ 가져온 메뉴 데이터가 없습니다. 종료합니다.")
        return

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    qdrant_store = Qdrant(
        client=client,
        collection_name="restaurants",
        embeddings=embedding_model,
        content_payload_key="page_content"
    )

    documents = []
    ids = []

    for restaurant in restaurants:
        try:
            addr = restaurant["address"]
            address_str = format_address(addr)
            lat = addr.get("latitude")
            lon = addr.get("longitude")

            description = (
                f"{restaurant['category']} {restaurant['restaurant']}의 메뉴 '{restaurant['menu']}'은 "
                f"{restaurant['price']}원입니다. "
                f"{'AR을 통해 미리 확인할 수 있는' if restaurant['hasAR'] else '일반'} 메뉴이며, "
                f"식당의 주소는 {address_str}입니다. "
            )

            doc = Document(
                page_content=description,
                metadata={
                    "menuId": restaurant["id"],
                    "menu": restaurant["menu"],
                    "restaurant": restaurant["restaurant"],
                    "restaurantId": restaurant["restaurantId"],
                    "category": restaurant["category"],
                    "price": restaurant["price"],
                    "hasAR": restaurant["hasAR"],
                    "hasCoupon": restaurant["hasCoupon"],
                    "address": address_str,
                    "location": {"lat": lat, "lon": lon},
                    "tags": [merestaurantnu["category"], "전체"],
                    "description": description,
                    "createdAt": datetime.utcnow().isoformat(),
                    "embeddingVersion": "openai/text-embedding-3-small"
                }
            )
            documents.append(doc)
            ids.append(restaurant["id"])  # 고유 ID로 삽입
        except Exception as e:
            print(f"❌ 메뉴 변환 실패 (menuId: {restaurant.get('id')}): {e}")
            continue

    try:
        qdrant_store.add_documents(documents, ids=ids)
        print(f"✅ LangChain 기반으로 {len(documents)}건 저장 완료")
    except Exception as e:
        print(f"❌ Qdrant 저장 실패: {e}")

# 7. 실행 시작
if __name__ == "__main__":
    run_langchain_insert()
