# SHARED UTILITIES - Các hàm tiện ích dùng chung

## File: `source/function/utils_shared.py`

---

## 1. Chức năng tổng quan
Các utility dùng xuyên suốt hệ thống: load prompt, load data giới thiệu, cosine search, lọc query, parse JSON từ output LLM.

---

## 2. Các hàm chi tiết

### `load_prompt_from_yaml(settings, section)` — Load prompt template
```python
def load_prompt_from_yaml(settings, section):
    current_dir = os.path.dirname(__file__)
    yaml_path = os.path.abspath(os.path.join(current_dir, '..', 'core', settings.YAML_PATH))
    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
    messages = yaml_data['prompts'][section]['messages']
    return ChatPromptTemplate.from_messages([(m['role'], m['content']) for m in messages])
```
- Trả về `ChatPromptTemplate` cho section yêu cầu.
- Sections hiện hành: `query_generator`, `response`, `invalid_query`, `information_query`.
- Dùng bởi `Gemini_Generate` (tất cả method sinh nội dung).

---

### `load_information_from_json(settings, model_embedding)` — Load thông tin chatbot
```python
def load_information_from_json(settings, model_embedding):
    json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core', settings.PATH_INFOR))
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    corpus = [item['content'] for item in data]
    corpus_embedding = model_embedding.embeddings_bkai.embed_documents(corpus)
    return corpus, corpus_embedding
```
- Trả `(corpus, corpus_embedding)` cho 17 đoạn trong `information.json`.
- Được gọi **lazy** một lần trong `Orchestrator._ensure_intro_loaded` (nhánh chitchat), cache trong orchestrator.

---

### `search_from_json(corpus_embedding, corpus, query, model_embedding)` — Cosine search
```python
def search_from_json(corpus_embedding, corpus, query, model_embedding):
    query_embedding = model_embedding.embeddings_bkai.embed_query(query)
    cos_scores = util.cos_sim(query_embedding, corpus_embedding)[0]
    top_results = cos_scores.argsort(descending=True)[:5]
    return "\n".join(corpus[idx] for idx in top_results)
```
- Top 5 đoạn tương tự nhất → context cho prompt `information_query`.
- Dùng trong `Orchestrator._run_template` (chitchat).

---

### `clean_generated_queries(queries)` — Lọc query sinh ra
```python
def clean_generated_queries(queries):
    cleaned = []
    for q in queries:
        if '```' in q: continue          # bỏ code block
        if len(q.split()) < 5: continue  # bỏ query quá ngắn
        cleaned.append(q)
    return cleaned
```
- Dùng bởi `Gemini_Generate.generate_query`.

---

### `extract_json_dict(text)` — Trích JSON dict (an toàn)
```python
def extract_json_dict(text):
    if text is None: return {}
    # Case 1: fenced ```json {...} ```
    # Case 2: bare {...} bất kỳ đâu
    # Case 3: không có JSON → return {}
```
- **Khác bản cũ**: nay trả `{}` thay vì raise khi không tìm thấy / parse lỗi (an toàn hơn). Là tiện ích chung; luồng entity nay do planner xử lý.

---

### `clean_code_fence_safe(text)` — Xóa code fence
```python
def clean_code_fence_safe(text):
    lines = text.strip().splitlines()
    if lines and lines[0].strip().startswith("```"): lines = lines[1:]
    if lines and lines[-1].strip() == "```": lines = lines[:-1]
    return "\n".join(lines).strip()
```
- Bỏ ``` markers trước khi parse JSON. Dùng trong `Orchestrator._generate_rag_answer`.

---

### `parse_raw_json(raw_text)` — Parse JSON `{answer, key}` (chịu lỗi)
Parse output generator có dạng `{"answer": ..., "key": [...]}`. Xử lý 3 trường hợp:

1. **Fast path** — `json.loads` trực tiếp (Claude thường trả JSON chuẩn). Tự thêm `answer`/`key` mặc định nếu thiếu.
2. **Slow path** — `answer` chứa dấu `"`/newline thô chưa escape: dùng regex bắt trường `answer`, escape lại nội dung, ghép thành JSON hợp lệ rồi parse.
3. **Empty answer** — cho phép `answer` rỗng (`.*?`).

```python
text = raw_text.replace('“','"').replace('”','"').strip()
try:
    obj = json.loads(text)
    if isinstance(obj, dict):
        obj.setdefault("answer",""); obj.setdefault("key",[]); return obj
except json.JSONDecodeError: pass
# regex-repair trường "answer" ...
```
- Lỗi không sửa được → raise `ValueError` (orchestrator bắt và degrade về text thô).

---

## 3. Tổng kết vai trò

| Hàm | Được gọi bởi | Mục đích |
|---|---|---|
| `load_prompt_from_yaml` | `Gemini_Generate` | Load prompt template |
| `load_information_from_json` | `Orchestrator._ensure_intro_loaded` | Pre-load thông tin chatbot |
| `search_from_json` | `Orchestrator._run_template` | Cosine search chitchat |
| `clean_generated_queries` | `Gemini_Generate.generate_query` | Lọc query |
| `extract_json_dict` | tiện ích chung (an toàn) | Parse JSON dict |
| `clean_code_fence_safe` | `Orchestrator._generate_rag_answer` | Xóa code fence |
| `parse_raw_json` | `Orchestrator._generate_rag_answer` | Parse JSON câu trả lời |
