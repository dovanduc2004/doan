# RERANK COMPONENT - Xếp hạng lại tài liệu

## File: `source/rerank/utils_rerank.py`

> **Thay đổi**: bỏ hoàn toàn nhánh Cohere API. `Rerank_Utils` nay chỉ nhận `RerankModelFinetune` và luôn dùng Cross-Encoder fine-tuned local.

---

## 1. Chức năng tổng quan
Component Rerank đảm nhiệm 2 việc:
1. **Reciprocal Rank Fusion (RRF)**: gộp kết quả nhiều query, loại trùng, xếp hạng → top 50.
2. **Rerank bằng Cross-Encoder fine-tuned**: top 50 → top 5.

## 2. Class: `Rerank_Utils`

### Constructor
```python
def __init__(self, model_finetune: RerankModelFinetune):
    self.model_finetune = model_finetune.model   # CrossEncoder local
```

---

### Method: `reciprocal_rank_fusion(documents_nested, k=60)`

#### Thuật toán RRF
```
RRF_score(doc) = Σ 1 / (k + rank + 1)   với k = 60
```
- `rank`: vị trí doc trong kết quả mỗi query (0-indexed).
- Tổng qua tất cả query mà doc xuất hiện → doc xuất hiện ở nhiều query được boost.

#### Logic
```python
def reciprocal_rank_fusion(self, documents_nested, k=60):
    document_scores = {}
    for query_idx, result_list in enumerate(documents_nested):
        for rank, (doc, _) in enumerate(result_list):
            doc_key = doc.page_content        # dùng content làm key → dedup
            rrf_score = 1 / (k + rank + 1)
            if doc_key not in document_scores:
                document_scores[doc_key] = {"score": 0, "doc_metadata": doc.metadata, "count": 0}
            document_scores[doc_key]["score"] += rrf_score
            document_scores[doc_key]["count"] += 1
    reranked_docs = sorted(document_scores.items(), key=lambda x: x[1]["score"], reverse=True)
    return reranked_docs[:50]
```
- **Dedup**: dùng `page_content` làm key.
- **Output**: list `(content_string, {"score", "doc_metadata", "count"})`, top 50.
- Bọc try-except: lỗi → trả `[]`.

---

### Method: `rerank_documents_finetune(query, documents)` — Cross-Encoder local ⭐

#### Chức năng
Rerank top ≤50 doc → top 5. **Đây là reranker duy nhất dùng trong production.**

```python
def rerank_documents_finetune(self, query, documents) -> List[Tuple[str, dict]]:
    if len(documents) > 50:
        raise ValueError("Số lượng documents không được vượt quá 50.")
    doc_contents = [doc for doc, _ in documents]
    doc_metadata = [info['doc_metadata'] for _, info in documents]
    # clean text: bỏ "_" của ViTokenizer + sửa spacing trước dấu câu
    doc_contents = [d.replace("_"," ").replace(' .','.')... for d in doc_contents]
    pairs = [(query, doc) for doc in doc_contents]
    scores = self.model_finetune.predict(pairs)
    scores = [float(s) for s in scores]
    ranked = sorted(zip(scores, range(len(doc_contents))), key=lambda x: x[0], reverse=True)
    return [(doc_contents[idx], {'score': score, 'doc_metadata': doc_metadata[idx]})
            for score, idx in ranked[:5]]
```

#### Chi tiết
- **Validation**: tối đa 50 doc (Cross-Encoder chậm hơn bi-encoder).
- Tạo pair `(query, doc)` → `predict` ra score → sort giảm dần → top 5.
- **Return format**: `[(content, {'score': float, 'doc_metadata': dict})]`.
- Lỗi → trả `[]`.

---

## 3. Luồng Rerank trong Pipeline

```
≤4 query × 25 doc = ~100 doc (có trùng)
            │
            ▼
    ┌───────────────┐
    │ RRF Fusion    │  dedup theo page_content + xếp hạng
    │ (k=60)        │
    └───────┬───────┘
            │
            ▼
        Top 50 doc
            │
            ▼
    ┌───────────────┐
    │ CrossEncoder  │  fine-tuned rerank
    │ (top 5)       │
    └───────┬───────┘
            │
            ▼
   Top 5 (content, {score, metadata})  →  BERT Extract (doc 08)
```
