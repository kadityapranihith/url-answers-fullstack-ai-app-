from app.firebase_config import db

def create_chat(user_id, chat_id, urls):

    chat_ref = db.collection("users")\
                 .document(user_id)\
                 .collection("chats")\
                 .document(chat_id)

    chat_ref.set({
        "urls": urls,
        "messages": [],
        "title": None
    })


def save_message(user_id, chat_id, role, content):

    chat_ref = db.collection("users")\
                 .document(user_id)\
                 .collection("chats")\
                 .document(chat_id)

    chat_doc = chat_ref.get().to_dict() or {}

    messages = chat_doc.get("messages", [])

    messages.append({
        "role": role,
        "content": content
    })

    update_data = {"messages": messages}

    # If this is the first user question, use it as the chat title
    if role == "user" and not chat_doc.get("title"):
        update_data["title"] = content

    chat_ref.update(update_data)

def get_user_chats(user_id):

    chats_ref = db.collection("users")\
                  .document(user_id)\
                  .collection("chats")

    chats = []

    for chat in chats_ref.stream():
        chat_data = chat.to_dict() or {}
        chats.append({
            "chat_id": chat.id,
            "urls": chat_data.get("urls", []),
            "title": chat_data.get("title")
        })

    return chats
def get_chat_history(user_id, chat_id):

    chat_ref = db.collection("users")\
                 .document(user_id)\
                 .collection("chats")\
                 .document(chat_id)

    chat_doc = chat_ref.get().to_dict()

    return chat_doc