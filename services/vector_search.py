import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain.retrievers.multi_query import MultiQueryRetriever
from qdrant_client import QdrantClient
import logging
load_dotenv()
logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

qdrant_store = Qdrant(
    client=QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    ),
    collection_name="menus",
    embeddings=embedding_model,
    content_payload_key="page_content"
)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=qdrant_store.as_retriever(search_kwargs={"k": 1}),
    llm=llm
)

def search_vectors(state: dict):
    try:
        message = state["message"]
        docs = multi_query_retriever.get_relevant_documents(message)
        metas = [doc.metadata for doc in docs if doc.metadata]

        logger.info("🔎 검색된 문서 수: %d", len(metas))
        return {**state, "search_results": metas}
    except Exception as e:
        logger.exception("❌ [search_vectors] 오류 발생: %s", e)
        raise