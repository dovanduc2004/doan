# TỔNG QUAN KIẾN TRÚC HỆ THỐNG - LAW CHATBOT (AGENTIC RAG)

## 1. Giới thiệu

**Law ChatBot** là một hệ thống chatbot tư vấn pháp luật Việt Nam, xây dựng trên kiến trúc **Agentic RAG (Retrieval-Augmented Generation)**. Thay vì một pipeline RAG tuyến tính cố định, hệ thống dùng nhiều **agent** phối hợp với nhau (Planner, Grader, Hallucination Checker) dưới sự điều phối của một **Orchestrator** có vòng lặp suy luận (reflective loop). Toàn bộ quá trình suy luận được **stream về frontend qua Server-Sent Events (SSE)**, cho phép người dùng nhìn thấy từng bước agent đang làm gì (thay vì một thanh tiến trình giả).

Hệ thống trả lời dựa trên:
- **Cơ sở tri thức pháp luật nội bộ** lưu trong Qdrant (văn bản luật đã chunk + embed).
- **Web Search (Tavily)** cho câu hỏi thời gian thực / ngoài phạm vi luật, khi người dùng bật chế độ Web Search.

## 2. Đặc điểm kiến trúc cốt lõi

| Khía cạnh | Mô tả |
|---|---|
| Mô hình điều phối | Agentic, level 3b (ReAct / Self-Reflection): `Planner → (Retrieve → Grade → retry)* → Generate → Check → Final` |
| LLM backbone | **AWS Bedrock — Claude Sonnet 4.5** (mặc định). Google Gemini là backend legacy/fallback (`LLM_PROVIDER=gemini`) |
| Giao tiếp client | **SSE streaming** — mỗi bước agent là một event JSON |
| Bộ nhớ hội thoại | Short-term (rolling summary theo session) + Long-term (facts toàn cục) |
| Vector DB | Qdrant |
| Relational DB | **SQLite** (file `chat_history.db`) |
| Web search | **Tavily** (`/search` trả về URL + nội dung + answer tóm tắt trong 1 lần gọi) |
| Embedding | HuggingFace Sentence Transformers (`bhuy71/embedding_finetuned`) |
| Rerank | Cross-Encoder fine-tuned local (`bhuy71/rerank_model`) + Reciprocal Rank Fusion |
| Extract QA | BERT fine-tuned (`bhuy71/bert_extract_full_fine-tuned`) |
| Tokenizer tiếng Việt | PyVi (`ViTokenizer`) |

## 3. Kiến trúc tổng thể

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT (Frontend)                            │
│         index.html + static/script.js (đọc luồng SSE)                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ POST /api/chat/chatbot-agentic
                               │ (Server-Sent Events stream)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI Application (main.py)                    │
│   CORS · StaticFiles · Jinja2 · API Router (/api)                    │
│            ┌────────────────┴────────────────┐                       │
│            ▼                                  ▼                       │
│      /api/chat/*                        /api/session/*               │
│   (chatbot-agentic SSE,                (session + history +          │
│    memory facts CRUD)                   references CRUD)             │
└──────┬───────────────────────────────────────┬──────────────────────┘
       │                                        │
       ▼                                        ▼
┌─────────────────────────────┐      ┌──────────────────────────────┐
│      ORCHESTRATOR            │      │      DATA LAYER (SQLite)     │
│  (agentic control loop)      │      │  Chat_Sessions / Messages /  │
│                              │      │  References / User_Facts     │
│  ┌────────────────────────┐ │      └──────────────────────────────┘
│  │ PlannerAgent           │ │
│  │  intent + tool routing │ │
│  ├────────────────────────┤ │
│  │ Retrieve (iterative)   │ │────────► Qdrant Vector DB
│  │  multi-query + filter  │ │
│  │  + RRF + Rerank        │ │
│  ├────────────────────────┤ │
│  │ BERT Extract           │ │
│  ├────────────────────────┤ │
│  │ GraderAgent            │ │  ◄── quyết định retry / web fallback
│  ├────────────────────────┤ │
│  │ Generate (Claude)      │ │────────► AWS Bedrock (Claude Sonnet 4.5)
│  ├────────────────────────┤ │
│  │ HallucinationChecker   │ │  ◄── quyết định regenerate
│  └────────────────────────┘ │
│           │                  │
│           ▼ (tool=web)       │
│   TavilySearchTool ──────────┼────────► Tavily Web Search API
└─────────────────────────────┘
```

## 4. Luồng xử lý chính (Agentic Loop)

Một lượt hỏi đi qua các giai đoạn sau (mỗi giai đoạn phát một hoặc nhiều event SSE):

### Giai đoạn 1 — PLAN (PlannerAgent)
Một lệnh gọi LLM **duy nhất** quyết định toàn bộ cách xử lý lượt này:
- `intent`: `chitchat` | `legal` | `realtime` | `out_of_scope`
- `tool`: `template` | `rag` | `web` | `reject` | `ask_user`
- `rewritten_query`: câu hỏi đã giải tham chiếu hội thoại, tự chứa
- `search_query`: query keyword cho Tavily (chỉ khi `tool=web`)
- `entities`: metadata filter cho Qdrant (chỉ khi `tool=rag` và có văn bản cụ thể)

Routing được "kẹp" lại bằng bảng quy tắc (xem doc 10) để tránh model chọn tool sai so với trạng thái nút Web Search.

### Giai đoạn 2 — BRANCH theo tool
- **template** → trả lời chào hỏi/giới thiệu (cosine-sim trên `information.json` + LLM).
- **ask_user** → hỏi lại người dùng để lấy thông tin còn thiếu.
- **reject** → từ chối lịch sự (khi Web Search tắt và câu hỏi ngoài phạm vi).
- **web** → Tavily search → tổng hợp → hallucination check.
- **rag** → pipeline RAG nội bộ có vòng lặp (mục dưới).

### Giai đoạn 3 — RAG branch (vòng lặp truy xuất + chấm điểm)
Lặp tối đa `MAX_ITERATIONS` (mặc định 2):
1. **Retrieve & Rerank**: sinh 3 biến thể query → tokenize ViTokenizer → search Qdrant (kèm metadata filter) → **RRF** gộp kết quả → **Cross-Encoder rerank** lấy top 5.
2. **BERT Extract**: trích đoạn trả lời tốt nhất từ mỗi document.
3. **Grade** (GraderAgent): chấm `relevance_score` + `decision` (`sufficient`/`partial`/`insufficient`).
   - Thoát sớm khi `score ≥ GRADER_THRESHOLD` (mặc định 0.6) hoặc `decision=sufficient`.
   - Ngược lại: viết lại query (`suggested_query`) và lặp lại.
   - Luôn giữ **iteration tốt nhất theo score** (không phải iteration cuối).
4. **Web fallback**: chỉ khi best score `< 0.4` **và** Web Search bật → chuyển sang nhánh web.

### Giai đoạn 4 — GENERATE (Claude)
Tổng hợp câu trả lời từ **full document + BERT highlight** (hybrid input — xem doc 10). LLM trả về JSON `{"answer": ..., "key": [...]}`. `key` là danh sách chỉ số document được trích dẫn → dựng khối tài liệu tham khảo.

### Giai đoạn 5 — CHECK (HallucinationChecker)
Kiểm tra câu trả lời có được hỗ trợ bởi các document được trích dẫn không. Chỉ **regenerate** khi bằng chứng hallucination mạnh (`confidence ≥ 0.75` **và** `≥ 2` claim không có trong nguồn) — lúc đó feedback các claim thiếu cho generator để viết lại chính xác hơn.

### Giai đoạn 6 — FINAL
Phát event `final` chứa `answer` + `references`. Frontend hiển thị câu trả lời (typing effect) và panel tài liệu tham khảo. Sau khi stream xong, `BackgroundTasks` lưu lượt vào SQLite và cập nhật bộ nhớ (summary + facts).

## 5. Cấu trúc thư mục (đã cập nhật)

```
├── main.py                          # Entry point - FastAPI app + asset versioning
├── requirements.txt                 # Python dependencies
├── docker-compose.yml               # Chỉ chạy Qdrant container
├── run_mac.sh                       # Script chạy local trên macOS (venv + qdrant + uvicorn)
├── SETUP_MAC.md                     # Hướng dẫn cài đặt macOS
├── upload_to_hf.py                  # Tiện ích upload model lên HuggingFace
├── .env / .env.example              # Biến môi trường
│
├── source/
│   ├── agent/                       # ⭐ TẦNG AGENTIC (mới)
│   │   ├── orchestrator.py          # Vòng lặp điều phối + phát SSE
│   │   ├── planner.py               # PlannerAgent (routing 1 LLM call)
│   │   ├── grader.py                # GraderAgent (CRAG-style evaluator)
│   │   ├── hallucination_checker.py # HallucinationChecker (fact-check)
│   │   └── schemas.py               # Pydantic: Plan / GraderResult / CheckResult
│   │
│   ├── api/
│   │   ├── router.py                # Router tổng hợp
│   │   └── endpoints/
│   │       ├── chat.py              # Khởi tạo toàn bộ stack + endpoint SSE + memory CRUD
│   │       └── session.py           # Quản lý session / history / references
│   │
│   ├── core/
│   │   ├── config.py                # Settings (env vars + tuning agentic)
│   │   ├── gemini_prompting.yaml    # Prompt templates (query_generator, response, ...)
│   │   └── information.json         # 17 đoạn giới thiệu chatbot
│   │
│   ├── data/
│   │   ├── db/
│   │   │   ├── db_connection.py     # Kết nối SQLite + tự tạo schema
│   │   │   ├── utils_db.py          # CRUD session/message/reference
│   │   │   └── memory.py            # ⭐ Bộ nhớ short-term + long-term (mới)
│   │   ├── file_db/                 # Schema MSSQL cũ (legacy, không còn dùng)
│   │   └── vectordb/
│   │       └── qdrant.py            # Kết nối Qdrant
│   │
│   ├── model/
│   │   ├── embedding_model.py       # Sentence Transformer embedding
│   │   ├── extract_model.py         # BERT QA model
│   │   ├── generate_model.py        # Cấu hình LLM (Bedrock / Gemini)
│   │   ├── rerank_model_finetune.py # Cross-Encoder fine-tuned
│   │   └── reset_apikey.py          # Round-robin API key (Gemini legacy)
│   │
│   ├── extract/utils_extract.py     # Logic BERT extraction
│   ├── search/utils_search.py       # Qdrant search + build metadata filter
│   ├── rerank/utils_rerank.py       # RRF + Cross-Encoder rerank
│   ├── generate/generate.py         # Gemini_Generate (các primitive sinh nội dung)
│   ├── function/utils_shared.py     # Tiện ích dùng chung (load prompt, parse JSON, ...)
│   ├── schema/                      # ChatbotQuery / MessageInput
│   └── tool/
│       ├── tavily_search.py         # TavilySearchTool (web search backend)
│       └── utils_search.py          # Utils_Search_Tools (_synthesize web answer)
│
├── static/                          # script.js (SSE client) + style.css
├── templates/index.html
├── dataset/                         # Dữ liệu + qdrant_data
├── rerank_model/                    # Checkpoint Cross-Encoder fine-tuned
└── source_research/                 # Notebook nghiên cứu (không deploy)
```

> **Lưu ý**: Các module cũ như `source/function/utils_result.py` (class `RAG`), `source/model/rerank_model.py` (Cohere), `source/tool/google_search.py` đã được thay thế trong kiến trúc agentic. Logic điều phối nay nằm ở `source/agent/orchestrator.py`.

## 6. Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Backend Framework | FastAPI + Uvicorn |
| Streaming | Server-Sent Events (StreamingResponse) |
| LLM (mặc định) | AWS Bedrock — Claude Sonnet 4.5 (`langchain-aws`) |
| LLM (fallback) | Google Gemini (`langchain-google-genai`) |
| Embedding | HuggingFace Sentence Transformers (BKAI/fine-tuned) |
| Vector Database | Qdrant (`langchain-qdrant`) |
| Relational Database | SQLite |
| Rerank | Cross-Encoder fine-tuned (`sentence-transformers`) + RRF |
| Extract QA | BERT fine-tuned for QA (`transformers`) |
| NLP Tokenizer | PyVi |
| Web Search | Tavily (`tavily-python`) |
| Orchestration | LangChain (LCEL) |
| Containerization | Docker Compose (chỉ Qdrant) |

## 7. API Endpoints

| Method | Endpoint | Mô tả |
|---|---|---|
| GET | `/` | Trang chủ (HTML) |
| POST | `/api/chat/chatbot-agentic` | **Endpoint chính** — chạy agentic loop, trả về SSE stream |
| GET | `/api/chat/memory/facts` | Liệt kê long-term facts |
| DELETE | `/api/chat/memory/facts/{fact_id}` | Xóa một fact |
| POST | `/api/session/start-session` | Tạo session mới |
| POST | `/api/session/save-message` | Lưu tin nhắn |
| GET | `/api/session/get-sessions` | Lấy danh sách sessions |
| GET | `/api/session/get-chat-history/{session_id}` | Lấy lịch sử chat |
| GET | `/api/session/get-message-references/{message_id}` | Lấy tài liệu tham khảo |
| DELETE | `/api/session/delete-session/{session_id}` | Xóa session |

## 8. Tham số tinh chỉnh agentic (env)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `LLM_PROVIDER` | `bedrock` | `bedrock` hoặc `gemini` |
| `MAX_ITERATIONS` | `2` | Số vòng lặp retrieve/grade tối đa |
| `GRADER_THRESHOLD` | `0.6` | Ngưỡng score để chấp nhận retrieval |
| `HALLUCINATION_CHECK_ENABLED` | `true` | Bật/tắt kiểm tra hallucination |

(Web fallback threshold `0.4` được hardcode trong orchestrator.)

## 9. Tài liệu chi tiết theo thành phần

| Doc | Nội dung |
|---|---|
| 01 | Main entry point (`main.py`) |
| 02 | API layer (router + chat + session) |
| 03 | Core config & prompts |
| 04 | Data layer (SQLite + memory) |
| 05 | Model layer (embedding/BERT/LLM/rerank) |
| 06 | Search component (Qdrant) |
| 07 | Rerank component (RRF + Cross-Encoder) |
| 08 | Extract component (BERT QA) |
| 09 | Generate component (LLM primitives) |
| 10 | **Agentic Orchestrator & Agents** (Planner/Grader/Checker) |
| 11 | Shared utilities |
| 12 | Web search tool (Tavily) |
| 13 | Schema (request models) |
| 14 | Infrastructure (Docker/run scripts) |
| 15 | End-to-end flow |
