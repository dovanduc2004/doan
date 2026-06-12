# AGENTIC ORCHESTRATOR & AGENTS - Điều phối suy luận

## Các file liên quan:
- `source/agent/orchestrator.py` — vòng lặp điều phối + phát SSE
- `source/agent/planner.py` — PlannerAgent (routing)
- `source/agent/grader.py` — GraderAgent (CRAG-style evaluator)
- `source/agent/hallucination_checker.py` — HallucinationChecker
- `source/agent/schemas.py` — Pydantic models (Plan / GraderResult / CheckResult)

> **Thay đổi lớn**: class `RAG` cũ (`source/function/utils_result.py`, pipeline tuyến tính classify→search→...) đã được thay bằng kiến trúc **agentic**: nhiều agent chuyên biệt do `Orchestrator` điều phối trong một vòng lặp phản chiếu (reflective loop), và stream tiến trình qua SSE.

---

## 1. Schemas (`source/agent/schemas.py`)

```python
Intent = Literal["chitchat", "legal", "realtime", "out_of_scope"]
Tool   = Literal["template", "rag", "web", "reject", "ask_user"]

class Plan(BaseModel):
    intent: Intent
    tool: Tool
    rewritten_query: str               # câu hỏi tự chứa, đã giải tham chiếu
    search_query: Optional[str]        # query keyword cho Tavily (chỉ khi tool=web)
    entities: Optional[Dict[str,str]]  # metadata filter Qdrant (chỉ khi tool=rag)
    needs_clarification: bool = False
    clarifying_question: Optional[str]
    reason: str                        # giải thích ngắn (hiển thị cho người dùng)

GraderDecision = Literal["sufficient", "partial", "insufficient"]

class GraderResult(BaseModel):
    relevance_score: float             # 0.0 - 1.0
    decision: GraderDecision
    missing_aspects: List[str]         # gap cần bù khi retry
    suggested_query: Optional[str]     # query viết lại cho vòng sau
    reason: str

class CheckResult(BaseModel):
    is_grounded: bool
    confidence: float                  # 0.0 - 1.0
    unsupported_claims: List[str]
    reason: str
```

---

## 2. PlannerAgent (`source/agent/planner.py`)

### Vai trò
**Một lệnh gọi LLM duy nhất** quyết định toàn bộ cách xử lý lượt: intent, tool, query đã chuẩn hóa, search query, entities filter, có cần hỏi lại không. Planner KHÔNG trả lời câu hỏi — chỉ định tuyến.

### 4 INTENT
- `chitchat`: chào hỏi, hỏi về bot, cảm ơn.
- `legal`: câu hỏi pháp luật VN (trả lời được từ CSDL).
- `realtime`: phụ thuộc thời gian thực (thời tiết, tin tức, giá cả...).
- `out_of_scope`: ngoài luật VN và không phải realtime.

### 5 TOOL
- `template`: trả lời mẫu (chitchat).
- `rag`: pipeline RAG nội bộ (legal).
- `web`: Tavily + tổng hợp (realtime/out_of_scope khi Web Search bật).
- `reject`: từ chối lịch sự (realtime/out_of_scope khi Web Search tắt).
- `ask_user`: hỏi lại (realtime thiếu thông tin critical).

### Bảng định tuyến (tool ↔ intent × web_search_enabled)
| Intent | Web OFF | Web ON |
|---|---|---|
| chitchat | template | template |
| legal | rag | rag |
| realtime | reject | web (hoặc ask_user nếu thiếu info) |
| out_of_scope | reject | web |

### entities (chỉ khi tool=rag và có văn bản cụ thể)
- `LoaiVanBanFilter`: Luật / Nghị định / Quyết định / Thông tư / Văn bản hợp nhất
- `NoibanHanhFilter`: Bộ Công An / Bộ GTVT / ... / Quốc hội / Chính phủ / Thủ tướng Chính phủ
- `NgayBanHanhFilter`: `dd-mm-yyyy` hoặc `yyyy`

### `plan(user_message, memory_context, web_search_enabled) -> Plan`
1. Build prompt (system inline + human) với `today`, memory context, toggle web.
2. `_make_model(temperature=0.0, max_tokens=800)` → invoke → parse JSON (`_extract_json`).
3. **Làm sạch entities**: bỏ key có value `null`/rỗng; nếu rỗng → `None`.
4. Nếu parse lỗi → `_fallback_plan`: mặc định `intent=legal, tool=rag` (giữ hành vi cũ).
5. `_enforce_routing`: kẹp tool theo bảng định tuyến (model thỉnh thoảng chọn sai so với toggle).

```python
@staticmethod
def _enforce_routing(plan, web_search_enabled):
    if plan.intent == "chitchat":   plan.tool = "template"
    elif plan.intent == "legal":    plan.tool = "rag"
    elif plan.intent in ("realtime","out_of_scope"):
        if not web_search_enabled:  plan.tool = "reject"; ...
        else:                       plan.tool = plan.tool if plan.tool in ("web","ask_user") else "web"
    if plan.entities is not None and not plan.entities: plan.entities = None
    return plan
```

---

## 3. GraderAgent (`source/agent/grader.py`)

### Vai trò
Lấy cảm hứng từ **CRAG** (Yan et al. 2024): một retrieval evaluator nhẹ, chấm xem các **snippet BERT** có đủ trả lời câu hỏi không, để orchestrator quyết định: generate / viết lại query & retry / chuyển web.

> Chấm điểm sau bước rerank + BERT extract (chấm trên snippet generator sẽ thấy, không chấm trên top-50 thô).

### `grade(query, snippets: Dict[int,str]) -> GraderResult`
- Nếu `snippets` rỗng → `insufficient`, score 0.0.
- Build prompt liệt kê snippet → `_make_model(temperature=0.0, max_tokens=500)` → parse JSON.
- Parse lỗi → mặc định **an toàn** `sufficient` score 0.6 (tránh kẹt vòng lặp).
- Chuẩn hóa: nếu `decision=sufficient` nhưng `score < threshold` → hạ về `partial`.

`threshold` = `GRADER_THRESHOLD` (mặc định 0.6).

---

## 4. HallucinationChecker (`source/agent/hallucination_checker.py`)

### Vai trò
Kiểm tra câu trả lời có **được hỗ trợ bởi nguồn** không (một LLM call). Áp dụng cho cả nhánh RAG (nguồn = doc được trích dẫn) và web (nguồn = Tavily excerpt + answer).

### Nguyên tắc thiết kế
- **Ưu tiên tránh false positive**: paraphrase/tóm tắt KHÔNG phải hallucination.
- Phân loại mỗi claim: `exact_match` / `semantic_match` / `unsupported`. Chỉ `unsupported` mới vào `unsupported_claims`.
- Không đánh giá routing/chủ đề (vd answer thời tiết không phải hallucination dù bot vốn là chatbot luật).
- Không chắc → `is_grounded=true`, confidence thấp (0.3-0.5).

### `check(answer, sources: List[str]) -> CheckResult`
- `answer` rỗng → grounded (bỏ qua).
- `sources` rỗng → not grounded, confidence 0.5.
- Trim mỗi source 1500 ký tự → prompt → `_make_model(temperature=0.0, max_tokens=600)` → parse JSON.
- Parse lỗi → **fail-open**: grounded, confidence 0.3 (không chặn câu trả lời).

---

## 5. Orchestrator (`source/agent/orchestrator.py`)

### Vai trò
"Bộ não" điều phối: ráp Planner/Grader/Checker với stack retrieval+rerank+BERT+generate, chạy vòng lặp phản chiếu và **phát SSE** từng bước.

Cấp độ: **3b (ReAct / Self-Reflection)**:
```
Planner → (Retrieve → Grade → maybe retry)* → Generate → Check → Final
```

### Helper SSE
```python
def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
```

### Entry point: `run(user_input, session_id, web_search_enabled)` (generator)
```
1) memory_context = memory.build_context(session_id)        # nếu có session
2) plan = planner.plan(...)            → yield {type:"plan", ...}
3) Branch theo plan.tool:
     template → _run_template
     ask_user → _run_clarify
     reject   → _run_reject
     web      → _run_web
     rag      → _run_rag
4) yield {type:"done", total_ms}
```

### Nhánh `_run_template` (chitchat)
- Lazy-load `information.json` + embeddings (`_ensure_intro_loaded`).
- `search_from_json` lấy top đoạn → `generate_information` → `final` (references rỗng).

### Nhánh `_run_clarify` / `_run_reject`
- `clarify`: phát câu hỏi lại (`clarifying_question`) rồi `final` chính câu hỏi đó.
- `reject`: `invalid_query` → `final`.

### Nhánh `_run_web`
```
1) q = search_query | rewritten_query
2) web_start → Tavily search(q) → web_done (links)
3) Nếu không có link → final (xin lỗi)
4) Lấy tavily_answer + extract_texts_from_links → excerpts
5) generate_start → Utils_Search_Tools._synthesize(...) → generate_done
6) _maybe_regenerate_web: hallucination check trên excerpts/answer
     - chỉ regenerate khi !grounded & confidence ≥ 0.75 & ≥2 unsupported claim
7) final (references = links)
```

### Nhánh `_run_rag` ⭐ (vòng lặp truy xuất + chấm điểm)

Chiến lược lặp (tối đa `max_iterations`, mặc định 2):
```python
best_result, best_score = None, -1.0
accept_threshold = grader.threshold        # 0.6
WEB_FALLBACK_THRESHOLD = 0.4

for iteration in 1..max_iterations:
    retrieve_start → _retrieve_and_rerank(query, entities) → retrieve_done
    snippets = extract_utils.predict(doc_contents, query)  → extract
    grade = grader.grade(query, snippets)                  → grade
    if grade.relevance_score > best_score:                 # giữ BEST theo score
        best_result, best_score = (docs, snippets, grade), grade.relevance_score
    if grade.decision == "sufficient" or score >= accept_threshold:  # thoát sớm
        break
    if iteration == max_iterations: break
    query = grade.suggested_query or query                 # iterate (viết lại query)

rerank_docs, snippets, grade = best_result
```

Ba quyết định thiết kế (ghi chú trong code là "Bug fix"):
1. **Thoát sớm theo score**: score ≥ threshold là đủ, bất kể grader gắn nhãn `partial` (grader hay quá khắt khe, liệt kê gap nhỏ).
2. **Giữ iteration tốt nhất theo score** (không phải iteration cuối — vòng retry có thể tệ hơn).
3. **Web fallback chỉ khi best score thật sự thấp** (`< 0.4`) và Web Search bật. Khoảng `[0.4, 0.6)` vẫn generate bằng RAG.

Sau vòng lặp:
```
Nếu best_score < 0.4 và web bật → _run_web (fallback) → return
generate_start → answer, references, used_indices = _generate_rag_answer(...) → generate_done
Nếu references rỗng:
    web bật → _run_web; ngược lại → final (gợi ý bật Web Search)
Hallucination check (scope theo used_indices):
    chỉ regenerate khi !grounded & confidence ≥ 0.75 & ≥2 unsupported claim
    (feedback các claim thiếu cho generator để viết lại)
final (answer, references)
```

### Helper `_retrieve_and_rerank(query, entities)`
```
queries = generator.generate_query(query)              # original + 3 biến thể
metadata_filter = qdrant_utils.build_metadata_filter(entities)   # 1 filter dùng chung
for q in queries: ViTokenizer → qdrant_utils.search_documents(q, filter)
rrf = rerank_utils.reciprocal_rank_fusion(query_results)
return rerank_utils.rerank_documents_finetune(query, rrf)        # top 5
```

### Helper `_build_generator_input(rerank_docs, snippets)` ⭐
Gộp **full document + BERT highlight** thành input cho generator:
```python
result[idx] = full_doc
if hint and hint != "Không có câu trả lời":
    result[idx] = f"{full_doc}\n[BERT highlight (...): {hint}]"
```
Lý do: BERT chỉ trả 1 span/doc; câu hỏi nhiều phần (vd nhiều ngưỡng mức phạt) sẽ thiếu nếu chỉ dùng span. Claude Sonnet 4.5 context 200k token dư sức nhận full doc; BERT highlight chỉ làm gợi ý focus.

### Helper `_generate_rag_answer(query, snippets, rerank_docs, memory_context, feedback_claims=None)`
- Nếu `feedback_claims` (đường regeneration): chèn block hướng dẫn yêu cầu giữ số liệu/mức phạt/điều khoản cụ thể, trả lời đầy đủ mọi ngưỡng, không bịa.
- Gọi `generate_response` → `clean_code_fence_safe` → `parse_raw_json`.
- Parse lỗi → degrade: strip scaffolding JSON, trả text thô, references rỗng.
- Dựng `references` từ `key` (chỉ số doc hợp lệ) gồm metadata (LoaiVanBan, NoiBanHanh, SoHieu, ...) + nội dung doc.
- Trả `(answer, references, selected_keys)`.

### Helper `_format_doc_label(infor)`
Tạo nhãn ngắn cho chip UI: `"<LoaiVanBan> <SoHieu> — <Article|ChuDe>"`, fallback về preview để không bao giờ rỗng.

---

## 6. Sơ đồ tổng hợp nhánh RAG

```
                 user_input + session_id + web_enabled
                              │
                    memory.build_context
                              │
                        PlannerAgent  ──► plan (intent, tool, query, entities)
                              │ tool=rag
        ┌─────────────────────┴──────────────────────────────┐
        ▼                                                     │
  ┌─ vòng lặp (≤ max_iterations) ───────────────────────┐     │
  │  _retrieve_and_rerank → top 5                         │     │
  │  BERT extract → snippets                              │     │
  │  GraderAgent → score/decision                         │     │
  │  giữ BEST theo score                                  │     │
  │  score ≥ 0.6 hoặc sufficient → break                  │     │
  │  else: query = suggested_query, lặp lại               │     │
  └───────────────────────────────────────────────────────┘     │
        │ best_score < 0.4 & web ON → _run_web ─────────────────┘
        ▼
  Generate (full doc + BERT highlight) → {answer, key}
        │ references rỗng → web hoặc gợi ý bật web
        ▼
  HallucinationChecker (scope theo doc trích dẫn)
        │ !grounded & conf≥0.75 & ≥2 claim → regenerate (kèm feedback)
        ▼
  final: {answer, references}
```
