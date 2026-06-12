# SEARCH COMPONENT - Tìm kiếm Vector

## File: `source/search/utils_search.py`

> **Thay đổi**: method `search_With_Similarity_Queries` (sinh 3 query + gọi LLM trích entity cho TỪNG biến thể, 4 lần) đã bị **loại bỏ** sau khi refactor agentic. Nay Orchestrator tự sinh biến thể query và áp **một** metadata filter duy nhất do PlannerAgent trích ra. `Qdrant_Utils` chỉ còn 2 method nhỏ gọn.

---

## 1. Chức năng tổng quan
Component Search là lớp mỏng (thin wrapper) trên Qdrant: mở collection, chạy similarity search, và dựng metadata filter từ entities của planner.

## 2. Class: `Qdrant_Utils`

### Constructor
```python
def __init__(self, qdrant: Qdrant_Vector, gemini_util=None):
    self.qdrant = qdrant
    self.gemini = gemini_util   # optional, không còn dùng nội bộ
```
- `gemini_util` giữ lại như tham số optional để không phá vỡ caller cũ, nhưng không dùng nữa.

---

### Method: `search_documents(query, top_k=25, filter=None)`
```python
def search_documents(self, query, top_k=25, filter: Optional[Filter] = None):
    connection = self.qdrant.Open_Qdrant()
    return connection.similarity_search_with_score(
        query=query, k=top_k, timeout=300, filter=filter,
    )
```
- Mở connection → similarity search.
- `k=25` mỗi query, `timeout=300`s, `filter` là Qdrant `Filter` (optional).
- LangChain tự embed `query` (đã được tokenize ViTokenizer trước khi truyền vào).
- **Return**: list `(Document, score)`.

---

### Method: `build_metadata_filter(entity_dict)`
```python
def build_metadata_filter(self, entity_dict: dict) -> Optional[Filter]:
    conditions = []
    for key, value in (entity_dict or {}).items():
        if not value:
            continue
        if key == "NgayBanHanhFilter":
            conditions.append(FieldCondition(key="metadata."+key, match=MatchAny(any=[value])))
        else:
            conditions.append(FieldCondition(key="metadata."+key, match=MatchValue(value=str(value).lower())))
    if not conditions:
        return None
    return Filter(should=conditions)
```

#### Chi tiết
- Duyệt entity dict (do planner trích): `LoaiVanBanFilter`, `NoibanHanhFilter`, `NgayBanHanhFilter`.
- **NgayBanHanhFilter**: dùng `MatchAny` (ngày có thể match nhiều format).
- Field khác: `MatchValue` với `.lower()` (case-insensitive).
- `should` = OR (match bất kỳ điều kiện nào). Trả `None` nếu không có condition.

#### Ví dụ
Input:
```python
{"LoaiVanBanFilter": "Thông tư", "NoibanHanhFilter": "Bộ Y tế", "NgayBanHanhFilter": "2015"}
```
Output:
```
Filter(should=[
  FieldCondition(key="metadata.LoaiVanBanFilter",  match=MatchValue(value="thông tư")),
  FieldCondition(key="metadata.NoibanHanhFilter",  match=MatchValue(value="bộ y tế")),
  FieldCondition(key="metadata.NgayBanHanhFilter", match=MatchAny(any=["2015"])),
])
```

---

## 3. Search được gọi ở đâu

Trong `Orchestrator._retrieve_and_rerank` (xem doc 10):

```
rewritten_query (từ Planner) + entities (từ Planner)
        │
        ▼
generate_query()  → [original + 3 biến thể]   (1 LLM call, KHÔNG trích entity từng biến thể)
        │
        ▼  build_metadata_filter(entities)  → 1 filter dùng chung cho mọi biến thể
   ┌────┴──────────────┬──────────────┬──────────────┐
   ▼                   ▼              ▼              ▼
 ViTokenizer        ViTokenizer    ViTokenizer    ViTokenizer
   │                   │              │              │
 search_documents   search_...     search_...     search_...   (mỗi cái top 25, cùng filter)
   └───────────────────┴──────────────┴──────────────┘
                       │
                       ▼
            query_results (≤4 list × 25 doc)  →  RRF (doc 07)
```

### Khác biệt so với phiên bản cũ
- **Trước**: mỗi biến thể query gọi `extract_entities` riêng → 4 LLM call chỉ để trích entity, mỗi biến thể có filter khác nhau.
- **Nay**: planner trích entity **một lần**, filter dùng chung cho tất cả biến thể → tiết kiệm LLM call và nhất quán hơn.
