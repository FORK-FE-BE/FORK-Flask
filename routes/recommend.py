from flask import Blueprint, request, jsonify
from graph.langgraph_runner import run_recommendation_graph
from messages.message_store import save_message, link_followups

recommend_bp = Blueprint("recommend", __name__)

@recommend_bp.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    message = data["message"]

    # 사용자 메시지 저장
    user_msg_id = save_message(role="user", content=message, parent_id=data.get("parentId"))

    # LangGraph 실행
    final_state = run_recommendation_graph(data)

    # AI 응답 저장
    assistant_msg_id = save_message(role="assistant", content=final_state["gpt_response"], parent_id=user_msg_id)

    # 파생 질문 저장
    follow_ids = link_followups(final_state["followups"], assistant_msg_id)

    return jsonify({
        "response": final_state["gpt_response"],
        "recommendation": final_state["filtered_results"],
        "followUpSuggestions": final_state["followups"],
        "messageId": assistant_msg_id
    })
