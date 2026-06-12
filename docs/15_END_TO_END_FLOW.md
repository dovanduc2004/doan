# END-TO-END FLOW - Luồng xử lý từ đầu đến cuối (Agentic + SSE)

> Toàn bộ luồng chat đi qua **một endpoint duy nhất** `POST /api/chat/chatbot-agentic`, trả về **SSE stream**. Frontend (`static/script.js`) đọc từng event và render panel "Agent đang suy luận", rồi hiển thị câu trả lời cuối khi nhận event `final`.

---

## 1. Luồng hoàn chỉnh: Câu hỏi pháp luật (intent = legal, tool = rag)

### Ví dụ: "Bao nhiêu tuổi thì được đăng ký kết hôn?"

```
┌──────────────────────────────────────────────────────────────────────┐
│ FRONTEND (script.js)                                                  │
│ POST /api/chat/chatbot-agentic                                        │
│ { query, use_web_fallback:false, session_id:42 }                      │
│ → đọc body.getReader() từng SSE frame, parse JSON, render panel       │
└──────────────────────────────┬───────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ memory.build_context(42)  → [facts][summary][recent turns]            │
└──────────────────────────────┬───────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ PLAN  (LLM call #1 — PlannerAgent)            event: plan             │
│ intent=legal, tool=rag, rewritten_query="Điều kiện độ tuổi đăng ký    │
│ kết hôn theo pháp luật VN", entities=null                             │
└──────────────────────────────┬───────────────────────────────────────┘
                               ▼ (vòng lặp ≤ MAX_ITERATIONS=2)
┌──────────────────────────────────────────────────────────────────────┐
│ RETRIEVE (vòng 1)                  event: retrieve_start / retrieve_done│
│  generate_query (LLM call) → [original + 3 biến thể]                   │
│  build_metadata_filter(null) → no filter                              │
│  ViTokenizer mỗi biến thể → search_documents (top 25, cùng filter)    │
│  RRF gộp → top 50 → CrossEncoder rerank → top 5                       │
└──────────────────────────────┬───────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ EXTRACT (BERT QA, local)                       event: extract         │
│ {0:"Nam từ đủ 20 tuổi, nữ từ đủ 18 tuổi...", 1:"...", ...}            │
└──────────────────────────────┬───────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ GRADE (LLM call — GraderAgent)                 event: grade           │
│ relevance_score=0.9, decision="sufficient"                            │
│ → score ≥ 0.6 → THOÁT vòng lặp (giữ làm best)                         │
└──────────────────────────────┬───────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ GENERATE (LLM call — Claude)        event: generate_start/generate_done│
│ input = full doc + BERT highlight (per doc)                           │
│ output JSON: {"answer":"Theo pháp luật VN, nam từ đủ 20 tuổi...",     │
│               "key":[0,1]}                                            │
│ → dựng references từ key (metadata + nội dung doc)                    │
└──────────────────────────────┬───────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ CHECK (LLM call — HallucinationChecker)        event: check           │
│ scope = các doc được trích dẫn (used_indices)                         │
│ is_grounded=true, confidence=0.4 → KHÔNG regenerate                   │
└──────────────────────────────┬───────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ FINAL                                          event: final           │
│ { type:"final", answer:"...", references:[...] }                      │
│ event: done (total_ms)                                                │
└──────────────────────────────┬───────────────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ BackgroundTasks (sau khi stream xong)                                 │
│  - Insert user + bot message (+ references) vào SQLite                │
│  - memory.refresh_summary_if_needed(42)                               │
│  - memory.extract_and_save_facts(user_msg)                            │
└──────────────────────────────────────────────────────────────────────┘
```

### Khi grader chưa đủ (decision != sufficient và score < 0.6)
```
GRADE vòng 1 → partial, suggested_query="..."     event: iterate
   → query = suggested_query, lặp lại RETRIEVE/EXTRACT/GRADE (vòng 2)
   → giữ iteration có score CAO NHẤT
Sau 2 vòng:
   best_score < 0.4 và Web ON → chuyển sang nhánh WEB (fallback)
   best_score ∈ [0.4, 0.6)    → vẫn GENERATE bằng RAG
```

---

## 2. Tổng kết LLM call cho 1 request (nhánh RAG, 1 vòng lặp)

| Bước | Agent / Service | Số call | Ghi chú |
|---|---|---|---|
| Plan | LLM (Planner) | 1 | routing + entities |
| Generate query | LLM | 1 / vòng | multi-query |
| Search | Qdrant | ≤4 / vòng | mỗi biến thể |
| RRF | local | 0 | |
| Rerank | local (CrossEncoder) | 0 | |
| BERT Extract | local (BERT) | 0 | |
| Grade | LLM (Grader) | 1 / vòng | |
| Generate response | LLM (Claude) | 1 (+1 nếu regenerate) | |
| Hallucination check | LLM (Checker) | 1 | |
| **Tổng (1 vòng)** | | **~4 LLM + ≤4 Qdrant** | nhân thêm cho mỗi vòng lặp |

---

## 3. Luồng nhánh khác (theo plan.tool)

### chitchat (tool = template)
```
plan → _ensure_intro_loaded (lazy embed information.json)
     → search_from_json (top 5 đoạn) → generate_information (LLM)
     → final (references rỗng)
```

### realtime/out_of_scope, Web OFF (tool = reject)
```
plan → invalid_query (LLM) → final (từ chối lịch sự, gợi ý bật Web Search)
```

### realtime thiếu thông tin (tool = ask_user)
```
plan → clarify (phát clarifying_question) → final (chính câu hỏi đó)
```

### realtime/out_of_scope, Web ON (tool = web)
```
plan → web_start → Tavily.search → web_done (links)
     → get_last_answer + extract_texts_from_links → excerpts
     → generate_start → _synthesize (LLM) → generate_done
     → HallucinationChecker (regenerate có điều kiện)
     → final (references = links)
```

---

## 4. Luồng Web Search Fallback (từ nhánh RAG)

```
Nhánh RAG kết thúc với:
  best_score < 0.4 (CSDL thiếu)  HOẶC  generator trả key=[] (không trích được)
        │
        ├── Web Search BẬT  → orchestrator tự chuyển sang _run_web
        │                      (tool=web, search_query = rewritten_query)
        │
        └── Web Search TẮT  → final: thông báo "thông tin không có trong CSDL,
                                vui lòng bật Web Search"
```

---

## 5. Luồng Session & Memory

```
1. User mở app
   → POST /api/session/start-session  ← {session_id}

2. User gửi câu hỏi
   → POST /api/chat/chatbot-agentic {query, use_web_fallback, session_id}
       ├── memory.build_context(session_id)  (facts + summary + recent turns)
       ├── (SSE stream các bước agent)
       └── event final {answer, references}
   → BackgroundTasks: lưu message + references, refresh summary, extract facts

3. User xem lại sessions
   → GET /api/session/get-sessions

4. User xem lịch sử
   → GET /api/session/get-chat-history/{id}

5. User xem references của 1 message
   → GET /api/session/get-message-references/{message_id}

6. Quản lý long-term facts
   → GET /api/chat/memory/facts
   → DELETE /api/chat/memory/facts/{fact_id}
```

### Bộ nhớ hai tầng
- **Short-term** (theo session): 5 lượt gần nhất nguyên văn + rolling summary cho phần cũ hơn.
- **Long-term** (toàn cục): facts về người dùng (tên, vấn đề pháp lý đang theo dõi...), inject vào context mỗi lượt.

---

## 6. Hợp đồng sự kiện SSE (frontend `script.js`)

Frontend đọc stream, mỗi frame `data: {...}\n\n`:
- `type=="final"` → lưu `answer` + `references`, KHÔNG render vào panel.
- `type=="done"` → bỏ qua.
- `type=="error"` → lưu thông báo lỗi.
- còn lại → `panel.addStep(ev)` render thành một bước trong panel suy luận.

Mỗi event gắn `agent` (planner/retriever/extractor/grader/web_researcher/generator/fact_checker/orchestrator) → frontend gán icon + màu riêng (`AGENT_META`). Sau khi stream xong và có `finalAnswer`, frontend chạy hiệu ứng "typing" hiển thị câu trả lời và panel tài liệu tham khảo.

---

## 7. Performance Characteristics (ước tính)

| Metric | Ước tính |
|---|---|
| Startup time | 2-5 phút (load embedding + BERT + CrossEncoder) |
| Memory usage | ~4-8 GB (model cục bộ) |
| Latency (chitchat / reject) | ~1-3s (1 LLM call) |
| Latency (RAG, 1 vòng) | ~8-20s (~4 LLM + ≤4 Qdrant + inference cục bộ) |
| Latency (RAG, 2 vòng) | cao hơn (nhân đôi retrieve/grade) |
| Latency (web) | ~10-25s (Tavily + tổng hợp + check) |
| LLM calls / request (RAG) | ~4 mỗi vòng lặp + 1 plan + 1 check |

> Vì dùng SSE, người dùng thấy tiến trình ngay lập tức (panel suy luận) thay vì chờ toàn bộ pipeline xong — cảm nhận độ trễ tốt hơn so với phiên bản blocking cũ.
