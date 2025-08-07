import uuid

# 간단한 인메모리 메시지 저장소
MESSAGES = {}
FOLLOWUPS = {}

def save_message(role: str, content: str, parent_id=None):
    msg_id = str(uuid.uuid4())
    MESSAGES[msg_id] = {
        "id": msg_id,
        "role": role,
        "content": content,
        "parentId": parent_id
    }
    return msg_id

def link_followups(questions: list, parent_id: str):
    ids = []
    for q in questions:
        follow_id = str(uuid.uuid4())
        FOLLOWUPS[follow_id] = {
            "id": follow_id,
            "content": q,
            "parentId": parent_id
        }
        ids.append(follow_id)
    return ids
