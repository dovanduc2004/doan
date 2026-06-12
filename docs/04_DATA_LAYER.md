# DATA LAYER - Tầng Dữ liệu

## Các file liên quan:
- `source/data/db/db_connection.py`
- `source/data/db/utils_db.py`
- `source/data/db/memory.py`
- `source/data/vectordb/qdrant.py`

> **Thay đổi quan trọng**: hệ thống đã chuyển từ **Microsoft SQL Server** sang **SQLite** (file đơn `chat_history.db`). Thư mục `source/data/file_db/Law_ChatBot_DB.sql` chỉ còn là di sản (legacy) và không được dùng. Schema hiện tại được tạo tự động bằng Python khi khởi động.

---

## 1. Database Connection (`source/data/db/db_connection.py`)

### Chức năng
Quản lý kết nối SQLite và **tự tạo schema** lần đầu.

### Class: `DBConnection`

#### Constructor
```python
def __init__(self, setting: Settings):
    self.setting = setting
    self.db_path = setting.DB_PATH
    os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    self._ensure_schema()
```
- Tạo thư mục chứa DB nếu chưa có.
- Gọi `_ensure_schema()` để tạo bảng (idempotent).

#### Method: `Get_DB_Connection()`
```python
conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
conn.execute("PRAGMA foreign_keys = ON;")
return conn
```
- Mỗi lần gọi tạo connection mới (không pooling).
- Bật `foreign_keys` để cascade delete hoạt động.

#### Method: `_ensure_schema()`
Tạo 4 bảng nếu chưa tồn tại (xem mục Schema). Có **migration best-effort**: thử `ALTER TABLE Chat_Sessions ADD COLUMN summary` cho DB cũ chưa có cột `summary` (bỏ qua nếu đã có).

### Schema (SQLite)

```sql
CREATE TABLE IF NOT EXISTS Chat_Sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    create_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    summary TEXT DEFAULT NULL          -- rolling summary của session
);

CREATE TABLE IF NOT EXISTS Chat_Messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    sender TEXT NOT NULL,              -- "user" | "bot"
    message TEXT NOT NULL,
    send_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES Chat_Sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS Chat_References (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL,
    reference_content TEXT NOT NULL,
    FOREIGN KEY (message_id) REFERENCES Chat_Messages(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS User_Facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_text TEXT NOT NULL UNIQUE,    -- long-term fact (toàn cục)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Quan hệ
```
Chat_Sessions (1) ── (N) Chat_Messages (1) ── (N) Chat_References
User_Facts: bảng toàn cục, độc lập với session (single-user demo)
```

---

## 2. Database Utilities (`source/data/db/utils_db.py`)

### Class: `DB_Utils`
CRUD cho session / message / reference. Tất cả dùng parameterized query (chống SQL injection) và pattern: open → execute → (fetch) → commit → close.

| Method | Mô tả |
|---|---|
| `Create_Session()` | Insert `Chat_Sessions` → trả `lastrowid` (session_id) |
| `Insert_Message(session_id, sender, message)` | Insert message → trả `message_id` |
| `Insert_References(message_id, references)` | Loop insert từng reference (return sớm nếu rỗng) |
| `Get_References(message_id)` | Trả `List[str]` reference_content |
| `Get_Session()` | Các session có ≥1 message user, kèm `first_message`, mới nhất trước |
| `Get_History(session_id)` | Tuple `(id, sender, message, send_at)` theo thời gian tăng dần |
| `Delete_Session(session_id)` | Xóa session (cascade messages + references) |

---

## 3. Memory Subsystem (`source/data/db/memory.py`) ⭐ MỚI

### Chức năng
Cung cấp **bộ nhớ hội thoại** hai tầng, được Orchestrator dùng để xây "ngữ cảnh" inject vào mỗi lượt.

### Hằng số
- `SHORT_TERM_KEEP = 5`: số message gần nhất giữ nguyên văn.

### Short-term memory (theo từng session)
- Giữ nguyên văn **5 message gần nhất**.
- Các message cũ hơn được **nén thành rolling summary** lưu ở `Chat_Sessions.summary`.

| Method | Mô tả |
|---|---|
| `get_session_summary(session_id)` / `set_session_summary(...)` | Đọc/ghi summary |
| `get_recent_turns(session_id, limit=5)` | 5 lượt gần nhất (chronological) |
| `count_messages(session_id)` | Đếm message |
| `get_messages_older_than_last_n(session_id, n)` | Lấy phần cũ hơn n message cuối |
| `build_context(session_id)` | **Ráp khối ngữ cảnh** `[facts][summary][recent turns]` |
| `refresh_summary_if_needed(session_id)` | Nếu có message cũ hơn 5 lượt → gọi LLM tạo/nối summary |

`build_context` tạo chuỗi tiếng Việt gồm:
```
## Thông tin về người dùng (ghi nhớ qua nhiều phiên): ...   (long-term facts)
## Tóm tắt các phần đã trao đổi trước đó trong phiên: ...    (session summary)
## Các lượt gần nhất trong phiên: ...                        (recent turns)
```

### Long-term memory (facts toàn cục)
- Facts được LLM trích từ tin nhắn người dùng (tên, nghề nghiệp, vấn đề pháp lý đang theo dõi, sở thích trình bày...), lưu trong `User_Facts` (UNIQUE để tránh trùng).

| Method | Mô tả |
|---|---|
| `get_facts(limit=20)` | List fact mới nhất |
| `add_fact(fact)` | `INSERT OR IGNORE` |
| `delete_fact(fact_id)` | Xóa fact |
| `list_facts()` | List dict `{id, text, created_at}` (cho API) |
| `extract_and_save_facts(user_message)` | Gọi LLM trích fact dài hạn từ tin nhắn, bỏ qua thông tin nhất thời, lưu fact mới |

### Khi nào chạy
- `build_context` được gọi đầu mỗi lượt trong `Orchestrator.run`.
- `refresh_summary_if_needed` và `extract_and_save_facts` chạy trong **BackgroundTasks** sau khi stream xong (xem doc 02).
- Nếu `generator is None`, các thao tác cần LLM (summary/fact) được bỏ qua an toàn.

---

## 4. Qdrant Vector Database (`source/data/vectordb/qdrant.py`)

### Class: `Qdrant_Vector`

#### Constructor
```python
def __init__(self, setting: Settings, embedding: Sentences_Transformer_Embedding):
    self.setting = setting
    self.embedding = embedding
```

#### Method: `Open_Qdrant()`
```python
open_collection = Qdrant.from_existing_collection(
    embedding=self.embedding.embeddings_bkai,
    url=self.setting.URL_QDRANT_LOCAL,
    collection_name=self.setting.EXIST_COLLECTION_NAME,
    metadata_payload_key=self.setting.metadata_payload_key,
    check_compatibility=False,
)
```
- Kết nối collection đã tồn tại (không tạo mới).
- `check_compatibility=False`: tắt cảnh báo version giữa client/server Qdrant (server 1.17.x vs client 1.14.x) để giảm spam log.
- Trả về LangChain Qdrant object hỗ trợ `similarity_search_with_score()`.

### Qdrant Data Structure
Mỗi document gồm vector embedding + payload metadata:
```json
{
  "metadata": {
    "LoaiVanBan": "Luật",
    "NoiBanHanh": "Quốc hội",
    "SoHieu": "52/2014/QH13",
    "LinhVucNganh": "Hôn nhân và Gia đình",
    "NgayBanHanh": "19/06/2014",
    "ChuDe": "...",
    "Chapter": "Chương III",
    "Section": "Mục 1",
    "MiniSection": "...",
    "Article": "Điều 8",
    "LoaiVanBanFilter": "luật",
    "NoibanHanhFilter": "quốc hội",
    "NgayBanHanhFilter": ["19/06/2014", "2014"]
  },
  "page_content": "Nội dung văn bản pháp luật (đã tokenize ViTokenizer)..."
}
```
Các field `*Filter` (lowercase) dùng cho metadata filtering do planner trích ra.

### Vai trò
- Knowledge base chính của chatbot, deploy như Docker container riêng (xem doc 14).
- Hỗ trợ similarity search + metadata filtering.
