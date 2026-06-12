# WEB SEARCH TOOL - Tìm kiếm trên Web (Tavily)

## Các file liên quan:
- `source/tool/tavily_search.py`
- `source/tool/utils_search.py`

> **Thay đổi**: hệ thống chuyển từ **Google Custom Search** (đóng đăng ký khách mới năm 2025) sang **Tavily**. Pipeline web không còn là method blocking `Search_Docs_From_Tools`; logic retrieve được inline trong `Orchestrator._run_web` để phát SSE từng bước, còn `Utils_Search_Tools` chỉ giữ lại `_synthesize`.

---

## 1. Tổng quan
Web Search là nhánh dành cho câu hỏi **realtime / out_of_scope** khi người dùng bật Web Search. Planner định tuyến `tool=web`; orchestrator gọi Tavily, trích nội dung, tổng hợp bằng LLM, rồi kiểm tra hallucination.

---

## 2. TavilySearchTool (`source/tool/tavily_search.py`)

### Chức năng
Web search + content extraction qua Tavily. Một lần gọi `/search` trả về **đồng thời**: URL, nội dung trang (raw), và một **answer tóm tắt do LLM của Tavily tạo**.

### Class: `TavilySearchTool`
```python
def __init__(self, setting: Settings):
    self.api_key = setting.TAVILY_API_KEY
    self.client = TavilyClient(api_key=self.api_key) if self.api_key else None
    self._cache = {}
    self._last_answer = ""
```

#### `search(query, num_results=5) -> List[str]`
```python
response = self.client.search(
    query=query, max_results=num_results,
    include_raw_content=True, include_answer="advanced", search_depth="advanced",
)
self._last_answer = response.get("answer") or ""
results = response.get("results", [])
self._cache = {r["url"]: r for r in results}
return [r["url"] for r in results]
```
- Cache full response để các bước sau đọc lại (không gọi mạng thêm).
- Raise `RuntimeError` nếu thiếu `TAVILY_API_KEY`.

#### `get_last_answer() -> str`
- Trả answer tóm tắt của Tavily từ lần `search` gần nhất.

#### `extract_text_from_url(url) -> str`
- Đọc từ cache: `raw_content` hoặc `content`. Trả `""` nếu không có.

#### `extract_texts_from_links(links) -> List[str]`
- Batch extract text cho danh sách URL.

---

## 3. Search Utils (`source/tool/utils_search.py`)

### Class: `Utils_Search_Tools`
```python
def __init__(self, setting, gemini_func, extract_func, search_backend):
    self.setting = setting
    self.gemini_func = gemini_func        # Gemini_Generate
    self.extract_func = extract_func      # Extract_Information (giữ để tương thích)
    self.search_backend = search_backend  # TavilySearchTool
```

#### `_synthesize(query, tavily_answer, excerpts, lst_links) -> str`
Gọi LLM tổng hợp câu trả lời tiếng Việt từ **hai nguồn**:
1. **Tóm tắt Tavily** — nguồn đáng tin cho dữ liệu thời gian thực (được phép trích số liệu từ đây).
2. **Trích đoạn từ trang web** — bổ sung chi tiết, ghi (Nguồn N).

Quy tắc trong prompt: ưu tiên Tavily cho số liệu/sự kiện realtime; chỉ trả "không đủ dữ liệu" khi cả hai nguồn đều không đề cập; không bịa; nêu cả hai nếu mâu thuẫn. Prompt chèn ngày hiện tại (`today`).

```python
model = self.gemini_func._make_model(temperature=0.2, max_tokens=4096)
answer = (model | StrOutputParser()).invoke(prompt).strip()
# fallback: nếu LLM lỗi → trả tavily_answer hoặc thông báo không đủ thông tin
```

---

## 4. Luồng Web Search (trong `Orchestrator._run_web`)

```
plan.tool == "web"  (hoặc fallback từ nhánh RAG)
        │
        ▼
 q = search_query | rewritten_query
        │
   web_start ──► Tavily.search(q) ──► web_done (links + n_links)
        │
   (không có link) → final: "không tìm thấy kết quả"
        │
   get_last_answer() → tavily_answer
   extract_texts_from_links(links) → docs → excerpts (mỗi nguồn ≤4000 ký tự)
        │
   generate_start ──► _synthesize(q, tavily_answer, excerpts, links) ──► generate_done
        │
   _maybe_regenerate_web:
     HallucinationChecker.check(answer, sources)   (sources = excerpts + tavily_answer)
     chỉ regenerate khi !grounded & confidence ≥ 0.75 & ≥2 unsupported claim
        │
   final: {answer, references = links}
```

---

## 5. Khi nào Web Search được kích hoạt

1. **Planner định tuyến trực tiếp** `tool=web` (intent realtime/out_of_scope + Web Search bật).
2. **Fallback từ nhánh RAG** khi:
   - best relevance score `< 0.4` (CSDL thiếu nhiều thông tin), HOẶC
   - generator không chọn được văn bản trích dẫn (`references` rỗng),
   - **và** Web Search đang bật.

Nếu Web Search **tắt** mà câu hỏi ngoài CSDL → orchestrator trả thông báo gợi ý người dùng bật Web Search.

---

## 6. So sánh RAG vs Web Search

| Tiêu chí | RAG (Qdrant) | Web (Tavily) |
|---|---|---|
| Nguồn | CSDL pháp luật đã kiểm duyệt | Internet (realtime) |
| Độ chính xác pháp lý | Cao | Trung bình |
| References | Metadata chi tiết (điều, khoản, số hiệu) | URLs |
| Khi dùng | intent `legal` | intent `realtime`/`out_of_scope` hoặc fallback |
| Phụ thuộc | Qdrant local | Tavily API + Internet |
