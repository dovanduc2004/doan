# SCHEMA - Request/Response Models

## Các file liên quan:
- `source/schema/chatbot_querry.py`
- `source/schema/message_input.py`

---

## 1. ChatbotQuery (`source/schema/chatbot_querry.py`)

### Chức năng
Pydantic model cho request body của endpoint chat agentic.

### Định nghĩa
```python
class ChatbotQuery(BaseModel):
    query: str
    use_web_fallback: bool = False
    session_id: Optional[int] = None
```

### Các trường
| Trường | Type | Mặc định | Mô tả |
|---|---|---|---|
| `query` | str | (bắt buộc) | Câu hỏi người dùng |
| `use_web_fallback` | bool | `false` | Bật chế độ Web Search (planner tự quyết khi nào tra web) |
| `session_id` | int? | `null` | Session để lấy/lưu bộ nhớ hội thoại |

### Sử dụng
- Endpoint: `POST /api/chat/chatbot-agentic`

### Request body ví dụ
```json
{
    "query": "Bao nhiêu tuổi thì được kết hôn?",
    "use_web_fallback": false,
    "session_id": 42
}
```

> **Thay đổi**: trước đây model chỉ có `query` (hai endpoint riêng cho RAG / web). Nay gộp về một endpoint, toggle web nằm ở `use_web_fallback`, và `session_id` được truyền để dùng bộ nhớ hội thoại.

---

## 2. MessageInput (`source/schema/message_input.py`)

### Chức năng
Pydantic model cho request body khi lưu tin nhắn thủ công.

### Định nghĩa
```python
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
```

### Các trường
| Trường | Type | Required | Mô tả |
|---|---|---|---|
| `session_id` | int | ✅ | ID session chat |
| `sender` | str | ✅ | "user" hoặc "bot" |
| `message` | str | ✅ | Nội dung tin nhắn |
| `references` | Optional[List[str] \| str] | ❌ | Tài liệu tham khảo |

### Validator
- Nếu `references == ""` → chuyển thành `[]` (xử lý frontend gửi empty string).

### Sử dụng
- Endpoint: `POST /api/session/save-message`

### Request body ví dụ
```json
{
    "session_id": 1,
    "sender": "bot",
    "message": "Theo quy định pháp luật...",
    "references": ["Tài liệu tham khảo: 1\nLoại văn bản: Luật\n..."]
}
```
