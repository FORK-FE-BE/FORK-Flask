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
    if client.collection_exists("menus"):
        client.delete_collection("menus")
        print("🗑️ 기존 'menus' 컬렉션 삭제 완료")
    client.create_collection(
        collection_name="menus",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    print("✅ 새 컬렉션 'menus' 생성 완료")

# 4. Spring Boot에서 메뉴 리스트 가져오기
def fetch_menus():
    try:
        response = requests.get("http://52.79.235.168:8080/api/menus/vectorize")
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
    menus = fetch_menus()
    if not menus:
        print("⚠️ 가져온 메뉴 데이터가 없습니다. 종료합니다.")
        return

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    qdrant_store = Qdrant(
        client=client,
        collection_name="menus",
        embeddings=embedding_model,
        content_payload_key="page_content"
    )

    documents = []
    ids = []

    for menu in menus:
        try:
            addr = menu["address"]
            address_str = format_address(addr)
            lat = addr.get("latitude")
            lon = addr.get("longitude")

            description = (
                f"{menu['category']} {menu['restaurant']}의 메뉴 '{menu['menu']}'은 "
                f"{menu['price']}원입니다. "
                f"{'AR을 통해 미리 확인할 수 있는' if menu['hasAR'] else '일반'} 메뉴이며, "
                f"식당의 주소는 {address_str}입니다. "
            )

            doc = Document(
                page_content=description,
                metadata={
                    "menuId": menu["id"],
                    "menu": menu["menu"],
                    "restaurant": menu["restaurant"],
                    "restaurantId": menu["restaurantId"],
                    "category": menu["category"],
                    "price": menu["price"],
                    "hasAR": menu["hasAR"],
                    "hasCoupon": menu["hasCoupon"],
                    "address": address_str,
                    "location": {"lat": lat, "lon": lon},
                    "tags": [menu["category"], "전체"],
                    "description": description,
                    "createdAt": datetime.utcnow().isoformat(),
                    "embeddingVersion": "openai/text-embedding-3-small"
                }
            )
            documents.append(doc)
            ids.append(menu["id"])  # 고유 ID로 삽입
        except Exception as e:
            print(f"❌ 메뉴 변환 실패 (menuId: {menu.get('id')}): {e}")
            continue

    try:
        qdrant_store.add_documents(documents, ids=ids)
        print(f"✅ LangChain 기반으로 {len(documents)}건 저장 완료")
    except Exception as e:
        print(f"❌ Qdrant 저장 실패: {e}")

# 7. 실행 시작
if __name__ == "__main__":
    run_langchain_insert()
