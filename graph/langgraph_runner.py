import logging
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END

from services.vector_search import search_vectors
from services.filter import apply_filter
from services.gpt_response import generate_response
from services.followup_generator import generate_followups

# 🔹 Logger 설정
logger = logging.getLogger(__name__)

# ✅ 상태 타입 정의
class RecommendationState(TypedDict, total=False):
    message: str
    latitude: float
    longitude: float
    restrictions: List[str]
    previousResults: List[Dict[str, Any]]
    search_results: List[Any]
    filtered_results: List[Any]
    gpt_response: str
    followups: List[str]

# ✅ LangGraph 실행 함수
def run_recommendation_graph(input_state: dict):
    try:
        logger.info("🧠 [LangGraph 시작] 입력 상태:")
        logger.info(input_state)

        builder = StateGraph(RecommendationState)

        # ✅ 각 단계 로그 포함 래핑 함수 정의
        def logged_node(name, func):
            def wrapper(state):
                logger.info(f"🔹 [{name}] 시작")
                result = func(state)
                logger.info(f"✅ [{name}] 완료 - 상태 키: {list(result.keys())}")
                return result
            return wrapper

        # ✅ 노드 등록 (로그 포함)
        builder.add_node("search", logged_node("search", search_vectors))
        builder.add_node("filter", logged_node("filter", apply_filter))
        builder.add_node("respond", logged_node("respond", generate_response))
        builder.add_node("followup", logged_node("followup", generate_followups))

        # ✅ 그래프 흐름 연결
        builder.set_entry_point("search")
        builder.add_edge("search", "filter")
        builder.add_edge("filter", "respond")
        builder.add_edge("respond", "followup")
        builder.add_edge("followup", END)

        graph = builder.compile()

        logger.info("🚀 [LangGraph 컴파일 완료] 실행 중...")
        result = graph.invoke(input_state)
        logger.info("🎉 [LangGraph 완료] 최종 상태:")
        logger.info(result)

        return result

    except Exception as e:
        logger.exception("❌ LangGraph 실행 중 오류 발생")
        raise
