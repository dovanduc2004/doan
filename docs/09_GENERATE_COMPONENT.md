# GENERATE COMPONENT - Các primitive sinh nội dung qua LLM

## File: `source/generate/generate.py`

> **Thay đổi**: sau refactor agentic, class `Gemini_Generate` chỉ còn giữ các **primitive sinh nội dung**. Việc routing / phân loại câu hỏi / trích entity đã chuyển sang `PlannerAgent` (doc 10). Các method cũ `classify_query`, `extract_entities` **không còn**. LLM backend mặc định là **AWS Bedrock (Claude)**, Gemini chỉ là fallback. Tên class giữ nguyên `Gemini_Generate` để tránh phải sửa import khắp nơi.

---

## 1. Chức năng tổng quan
`Gemini_Generate` là lớp gọi LLM thống nhất. Nó cung cấp:
- `generate_query`: sinh truy vấn mở rộng (multi-query).
- `generate_response`: prompt tổng hợp câu trả lời pháp luật.
- `generate_information`: prompt giới thiệu chatbot (chitchat).
- `invalid_query`: prompt từ chối lịch sự.
- `_make_model`: factory tạo client LLM (Bedrock hoặc Gemini).

Ngoài ra, các agent (Planner/Grader/Checker) và bộ tổng hợp web đều gọi `_make_model` để lấy client LLM.

---

## 2. Class: `Gemini_Generate`

### Constructor
```python
def __init__(self, gemini_model: Gemini, settings: Settings):
    self.gemini_model = gemini_model     # cấu hình model (Bedrock id / Gemini)
    self.yaml_path = settings            # dùng load prompt từ YAML
    self.settings = settings
    self.provider = settings.LLM_PROVIDER
```

---

### Method: `_make_model(temperature=0.0, max_tokens=4096, top_p=1.0)` ⭐

Factory tạo client LLM theo provider:

```python
def _make_model(self, temperature=0.0, max_tokens=4096, top_p=1.0):
    if self.provider == "bedrock":
        # Claude trên Bedrock không nhận temperature + top_p cùng lúc
        model_kwargs = {"temperature": temperature}
        return ChatBedrock(
            model_id=self.gemini_model.bedrock_model_id,
            region_name=self.gemini_model.aws_region,
            model_kwargs=model_kwargs,
            max_tokens=max_tokens,
        )
    # Legacy Gemini path
    ... # monkey-patch BaseCache/Callbacks rồi tạo ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        google_api_key=self.gemini_model.key_manager.get_next_key(),
        model=self.gemini_model.model_gemini,
        temperature=temperature, max_tokens=max_tokens, top_p=top_p,
    )
```

#### Chi tiết
- **Bedrock (mặc định)**: `ChatBedrock` (langchain-aws). Chỉ truyền `temperature` trong `model_kwargs` (Claude từ chối nếu có cả `top_p`).
- **Gemini (fallback)**: monkey-patch `BaseCache`/`Callbacks` để fix tương thích Pydantic của `langchain-google-genai`, rồi tạo `ChatGoogleGenerativeAI` với key rotation.

---

### Method: `generate_query(original_query, memory_context="")` — Multi-query
```python
def generate_query(self, original_query, memory_context=""):
    prompt = load_prompt_from_yaml(self.yaml_path, 'query_generator')
    model = self._make_model(temperature=0)
    effective_query = (f"Ngữ cảnh hội thoại trước đó:\n{memory_context}\n\nCâu hỏi hiện tại: {original_query}"
                       if memory_context else original_query)
    prompt = prompt.format_messages(original_query=effective_query)
    result = (model | StrOutputParser()).invoke(prompt)
    generated_queries = clean_generated_queries(result.strip().split('\n'))
    return [original_query] + generated_queries
```
- Sinh 3 biến thể + giữ query gốc → trả `[original, v1, v2, v3]`.
- `clean_generated_queries` lọc bỏ dòng có ``` hoặc < 5 từ.
- Gọi bởi `Orchestrator._retrieve_and_rerank`.

---

### Method: `generate_response(query, docs, memory_context="")` — Sinh câu trả lời
```python
def generate_response(self, query, docs, memory_context=""):
    docs_str = "\n".join(f"{k}: {v}" for k, v in docs.items())
    if memory_context:
        docs_str = f"[NGỮ CẢNH HỘI THOẠI TRƯỚC ĐÓ]\n{memory_context}\n\n[VĂN BẢN LUẬT LIÊN QUAN]\n{docs_str}"
    prompt_template = load_prompt_from_yaml(self.yaml_path, "response")
    model = self._make_model(temperature=0.2, max_tokens=8192, top_p=0.6)
    prompt = prompt_template.format_messages(original_query=query, context=docs_str)
    return (model | StrOutputParser()).invoke(prompt).strip()
```
- `docs`: dict `{index: hybrid_text}` (full doc + BERT highlight, do `Orchestrator._build_generator_input` dựng).
- Inject memory context nếu có.
- Cấu hình: `temperature=0.2`, `max_tokens=8192`, `top_p=0.6` (top_p chỉ có hiệu lực ở Gemini path).
- **Output**: chuỗi JSON `{"answer": "...", "key": [...]}` (sẽ được parse ở orchestrator).

---

### Method: `invalid_query(query, memory_context="")` — Từ chối lịch sự
- Load prompt `invalid_query`, `temperature=0`.
- Dùng trong nhánh `reject` của orchestrator (Web Search tắt, câu hỏi ngoài phạm vi).

---

### Method: `generate_information(query, context, memory_context="")` — Giới thiệu chatbot
- Load prompt `information_query`, `temperature=0`.
- `context`: top đoạn từ `information.json` (cosine similarity).
- Dùng trong nhánh `template` (chitchat).

---

## 3. LangChain Pattern (LCEL)
Mọi method theo pattern:
```python
prompt = load_prompt_from_yaml(settings, section)
model  = self._make_model(temperature=..., max_tokens=..., top_p=...)
prompt = prompt.format_messages(**kwargs)
result = (model | StrOutputParser()).invoke(prompt)
```

---

## 4. Ai gọi gì (per request, nhánh RAG điển hình)

| LLM call | Nơi gọi | Số lần |
|---|---|---|
| Planner (routing + entities) | `PlannerAgent.plan` | 1 |
| `generate_query` (multi-query) | `Orchestrator._retrieve_and_rerank` | 1 mỗi vòng lặp |
| `generate_response` | `Orchestrator._generate_rag_answer` | 1 (+1 nếu regenerate) |
| Grader | `GraderAgent.grade` | 1 mỗi vòng lặp |
| Hallucination check | `HallucinationChecker.check` | 1 |

(Chi tiết end-to-end ở doc 15.)
