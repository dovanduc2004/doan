# API LAYER - Tầng API

## Các file liên quan:
- `source/api/router.py`
- `source/api/endpoints/chat.py`
- `source/api/endpoints/session.py`

---

## 1. Router (`source/api/router.py`)

### Chức năng
Router tổng hợp, gom các sub-router và gắn prefix.

```python
api_router = APIRouter()
api_router.include_router(session.router, prefix="/session", tags=["session"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
```

### Cấu trúc URL cuối cùng
```
/api/session/start-session
/api/session/save-message
/api/session/get-sessions
/api/session/get-chat-history/{session_id}
/api/session/get-message-references/{message_id}
/api/session/delete-session/{session_id}
/api/chat/chatbot-agentic                 ← endpoint chính (SSE)
/api/chat/memory/facts                    ← GET liệt kê facts
/api/chat/memory/facts/{fact_id}          ← DELETE 1 fact
```

---

## 2. Chat Endpoints (`source/api/endpoints/chat.py`)

### Chức năng
Khởi tạo toàn bộ AI stack ở module level và expose endpoint agentic (SSE) cùng các endpoint quản lý long-term memory.

### Khởi tạo Components (Module-level)

Khi module được import lần đầu (lúc server khởi động), tất cả model và utility được khởi tạo theo thứ tự phụ thuộc:

```
Settings (config)
 ├── Gemini (cấu hình LLM: Bedrock model id / Gemini)
 ├── Bert_Extract (BERT QA model)
 ├── Sentences_Transformer_Embedding (embedding model)
 ├── Qdrant_Vector (kết nối Qdrant)
 ├── RerankModelFinetune (Cross-Encoder fine-tuned)
 ├── Rerank_Utils
 ├── Extract_Information
 ├── Gemini_Generate (primitive sinh nội dung qua LLM)
 ├── Qdrant_Utils
 │
 ├── TavilySearchTool (web search backend)
 ├── Utils_Search_Tools (_synthesize web answer)
 │
 ├── DBConnection + DB_Utils (SQLite)
 ├── Memory (short-term summary + long-term facts)
 │
 ├── PlannerAgent
 ├── GraderAgent (threshold = GRADER_THRESHOLD)
 ├── HallucinationChecker
 └── Orchestrator (ráp tất cả lại, max_iterations = MAX_ITERATIONS)
```

**Lưu ý quan trọng**: Model được load một lần vào memory khi khởi động → startup lâu (2-5 phút), memory cao (~4-8 GB), nhưng inference nhanh vì không load lại mỗi request.

---

### Endpoint chính: `POST /api/chat/chatbot-agentic`

**Mô tả**: Endpoint **duy nhất** cho chat. Thay thế hai endpoint cũ (`/chatbot-with-gemini` và `/chatbot-with-search-web`). Việc bật/tắt web search nay nằm ở field `use_web_fallback`; bản thân orchestrator tự quyết định khi nào tra Qdrant vs. Tavily dựa trên intent của planner.

**Input**: `ChatbotQuery` (JSON body)
```json
{
    "query": "Bao nhiêu tuổi thì được kết hôn?",
    "use_web_fallback": false,
    "session_id": 42
}
```

| Field | Type | Mặc định | Ý nghĩa |
|---|---|---|---|
| `query` | str | (bắt buộc) | Câu hỏi người dùng |
| `use_web_fallback` | bool | `false` | Bật chế độ Web Search |
| `session_id` | int? | `null` | Session để lấy/lưu bộ nhớ hội thoại |

**Output**: `StreamingResponse` với `media_type="text/event-stream"`.

Mỗi frame có dạng SSE: `data: {<json>}\n\n`. Các loại event (`type`):

| type | agent | Ý nghĩa |
|---|---|---|
| `plan` | planner | Quyết định intent + tool + reasoning |
| `template` / `clarify` / `reject` | planner | Nhánh trả lời mẫu / hỏi lại / từ chối |
| `retrieve_start` / `retrieve_done` | retriever | Bắt đầu / xong truy xuất Qdrant (kèm số doc) |
| `extract` | extractor | Tóm tắt số snippet BERT trích được |
| `grade` | grader | Verdict của grader (score, decision, missing) |
| `iterate` | orchestrator | Lặp lại với query mới / chuyển web |
| `web_start` / `web_done` | web_researcher | Gọi Tavily / nhận link |
| `generate_start` / `generate_done` | generator | Bắt đầu / xong tổng hợp |
| `check` | fact_checker | Verdict hallucination |
| `regenerate` | generator | Sẽ viết lại với hướng dẫn cụ thể |
| `final` | — | Câu trả lời cuối + references (terminal) |
| `error` | * | Lỗi không phục hồi (terminal) |
| `done` | — | Trace tổng thời gian |

**Logic xử lý**:
1. Đọc `query`, `session_id`, `use_web_fallback`.
2. `event_stream()` gọi `orchestrator.run(...)` và **yield từng frame** xuống client.
3. Đồng thời "peek" mỗi frame: khi gặp `type=="final"` thì capture `answer` + `references`.
4. Khi stream kết thúc, đăng ký `BackgroundTasks` để lưu lượt + cập nhật bộ nhớ (người dùng không phải chờ).

```python
def _persist_and_update_memory(session_id, user_msg, bot_msg, references=None):
    # 1) Insert user + bot message vào SQLite (+ references nếu có)
    # 2) memory.refresh_summary_if_needed(session_id)
    # 3) memory.extract_and_save_facts(user_msg)
```

Header trả về có `Cache-Control: no-cache` và `X-Accel-Buffering: no` (tắt buffering nếu chạy sau nginx).

---

### Endpoint quản lý Long-term Memory

#### `GET /api/chat/memory/facts`
- Trả về danh sách facts đã ghi nhớ về người dùng.
- Output: `{"facts": [{"id":..., "text":..., "created_at":...}, ...]}`

#### `DELETE /api/chat/memory/facts/{fact_id}`
- Xóa một fact theo id.
- Output: `{"status": "ok"}`

---

## 3. Session Endpoints (`source/api/endpoints/session.py`)

### Chức năng
Quản lý phiên chat và lịch sử tin nhắn. Lưu trữ trong **SQLite** (`chat_history.db`).

### Khởi tạo Components
```python
setting = Settings()
db = DBConnection(setting)
db_utils = DB_Utils(db)
```

### Các Endpoints

#### 3.1 `POST /api/session/start-session`
- Tạo session mới → trả `{"session_id": <id>}`.

#### 3.2 `POST /api/session/save-message`
- Input: `MessageInput` (`session_id`, `sender`, `message`, `references?`).
- Insert message; nếu `sender == 'bot'` và có `references` → insert vào `Chat_References`.
- Output: `{"status": "success", "message": "Message saved successfully"}`.

> Lưu ý: ở luồng agentic, việc lưu message thường do **BackgroundTasks** trong `chatbot-agentic` đảm nhiệm. Endpoint này vẫn dùng được cho frontend lưu thủ công.

#### 3.3 `GET /api/session/get-sessions`
- Lấy các session có ít nhất 1 message từ user, kèm `first_message`, mới nhất trước.
- Output: `{"sessions": [{"id", "create_at", "first_message"}, ...]}`.

#### 3.4 `GET /api/session/get-chat-history/{session_id}`
- Lấy toàn bộ message theo session, sắp theo thời gian tăng dần.
- `send_at` được format `'%Y-%m-%d %H:%M:%S'`.
- Output: `{"chat_history": [{"id","sender","message","send_at"}, ...]}`.

#### 3.5 `GET /api/session/get-message-references/{message_id}`
- Lấy references của một message.
- Output: `{"references": [...]}`.

#### 3.6 `DELETE /api/session/delete-session/{session_id}`
- Xóa session (cascade xóa messages + references nhờ `ON DELETE CASCADE`).
- Output: `{"status": "success", "message": "Session deleted successfully"}`.

### Error Handling
Tất cả endpoint session wrap trong try-except, trả HTTP 500 kèm detail khi lỗi.
