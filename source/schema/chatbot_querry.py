from pydantic import BaseModel
from typing import Optional


class ChatbotQuery(BaseModel):
    query: str
    use_web_fallback: bool = False
    session_id: Optional[int] = None
