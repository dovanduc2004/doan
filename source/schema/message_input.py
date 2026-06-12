from pydantic import BaseModel, validator
from typing import List, Optional, Union

class MessageInput(BaseModel):
    session_id: int
    sender: str
    message: str
    references: Optional[Union[List[str], str]] = None

    @validator('references')
    def validate_references(cls, v):
        if v == "":
            return []
        return v