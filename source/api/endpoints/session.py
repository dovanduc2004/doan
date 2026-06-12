from fastapi import APIRouter, HTTPException
from source.schema.message_input import MessageInput
from source.data.db.utils_db import DB_Utils
from source.data.db.db_connection import DBConnection
from source.core.config import Settings

# Khởi tạo các đối tượng
router = APIRouter()
setting = Settings()
db = DBConnection(setting)
db_utils = DB_Utils(db)

@router.post("/start-session")
def start_session():
    try:
        session_id = db_utils.Create_Session()
        return {"session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting session: {str(e)}")

@router.post("/save-message")
def save_message(data: MessageInput):
    try:
        message_id = db_utils.Insert_Message(data.session_id, data.sender, data.message)
        if data.sender == 'bot' and hasattr(data, 'references'):
            db_utils.Insert_References(message_id, data.references)
        return {"status": "success", "message": "Message saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving message: {str(e)}")

@router.get("/get-sessions")
def get_sessions():
    try:
        sessions = db_utils.Get_Session()
        return {
            "sessions": [
                {"id": s[0], "create_at": s[1], "first_message": s[2]} for s in sessions
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching sessions: {str(e)}")

@router.get("/get-chat-history/{session_id}")
def get_chat_history(session_id: int):
    try:
        messages = db_utils.Get_History(session_id)

        def _fmt(value):
            if hasattr(value, "strftime"):
                return value.strftime('%Y-%m-%d %H:%M:%S')
            return str(value)

        return {
            "chat_history": [
                {
                    "id": m[0],
                    "sender": m[1],
                    "message": m[2],
                    "send_at": _fmt(m[3])
                } for m in messages
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chat history: {str(e)}")

@router.get("/get-message-references/{message_id}")
def get_message_references(message_id: int):
    try:
        references = db_utils.Get_References(message_id)
        return {"references": references}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching message references: {str(e)}")

@router.delete("/delete-session/{session_id}")
def delete_session(session_id: int):
    try:
        db_utils.Delete_Session(session_id)
        return {"status": "success", "message": "Session deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")
