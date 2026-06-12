# EXTRACT COMPONENT - Trích xuất thông tin bằng BERT

## File: `source/extract/utils_extract.py`

---

## 1. Chức năng tổng quan

Component Extract sử dụng model BERT fine-tuned cho task Question Answering để trích xuất đoạn trả lời chính xác nhất từ mỗi document.

> **Vai trò trong kiến trúc agentic**: kết quả BERT extract được dùng cho **hai mục đích**:
> 1. Làm đầu vào cho **GraderAgent** chấm điểm xem retrieval có đủ trả lời không (doc 10).
> 2. Làm **"highlight"** gợi ý cho generator. Lưu ý: generator nhận **toàn bộ nội dung điều luật** (full doc) cộng với BERT highlight, KHÔNG chỉ nhận riêng span BERT — vì BERT chỉ trả 1 span/doc nên với câu hỏi nhiều phần (vd "mức phạt nồng độ cồn" có 3-4 ngưỡng) span đơn lẻ dễ thiếu. Xem `Orchestrator._build_generator_input` (doc 10).

## 2. Class: `Extract_Information`

### Constructor
```python
def __init__(self, bert: Bert_Extract):
    self.bert = bert
```
- Nhận `Bert_Extract` object đã load sẵn model và tokenizer

---

### Method: `predict(contexts, question)`

#### Chức năng
Nhận danh sách contexts và một question, trả về đoạn trả lời tốt nhất từ mỗi context.

#### Thuật toán chi tiết

```
Input:
  - contexts: ["Điều 8 Luật HNGĐ 2014 quy định...", "Theo Nghị định 126..."]
  - question: "Bao nhiêu tuổi thì được kết hôn?"

For each context:
  1. Tokenize (question, context) bằng ViTokenizer
  2. Encode bằng BERT tokenizer (max_length=512, stride=150)
  3. Forward pass qua BERT → start_logits, end_logits
  4. Tìm top N_BEST start positions và end positions
  5. Kết hợp start-end pairs → tính score
  6. Lấy answer có score cao nhất

Output:
  {0: "Nam từ đủ 20 tuổi trở lên, nữ từ đủ 18 tuổi trở lên",
   1: "Không có câu trả lời"}
```

#### Code flow chi tiết

```python
def predict(self, contexts, question):
    lst_Answer_Final = {}
    question = ViTokenizer.tokenize(question)  # Tokenize tiếng Việt
    
    for idx, context in enumerate(contexts):
        context = ViTokenizer.tokenize(context)
        
        # 1. Tokenize input
        inputs = self.bert.tokenizer(
            question, context,
            max_length=self.bert.max_length,      # 512
            truncation="only_second",              # Chỉ cắt context, giữ nguyên question
            stride=self.bert.stride,               # 150 tokens overlap
            return_offsets_mapping=True,            # Mapping token → character positions
            padding="max_length",
            return_tensors="pt"
        )
        
        # 2. BERT inference
        with torch.no_grad():
            outputs = self.bert.model_extract(
                **{k: v for k, v in inputs.items() 
                   if k in ['input_ids', 'attention_mask']}
            )
        
        # 3. Lấy logits
        start_logits = outputs.start_logits.squeeze().cpu().numpy()
        end_logits = outputs.end_logits.squeeze().cpu().numpy()
        offsets = inputs["offset_mapping"][0].cpu().numpy()
        
        # 4. Tìm best answer
        answers = []
        start_indexes = np.argsort(start_logits)[-self.bert.nbert:][::-1].tolist()
        end_indexes = np.argsort(end_logits)[-self.bert.nbert:][::-1].tolist()
        
        for start_index in start_indexes:
            for end_index in end_indexes:
                # Validate: end >= start, length <= max_answer_length
                if end_index < start_index:
                    continue
                if end_index - start_index + 1 > self.bert.max_answer_length:
                    continue
                    
                # Extract answer text using offset mapping
                if offsets[start_index][0] is not None and offsets[end_index][1] is not None:
                    answer_text = context[offsets[start_index][0]:offsets[end_index][1]].strip()
                    if answer_text:
                        answers.append({
                            "text": answer_text,
                            "score": start_logits[start_index] + end_logits[end_index]
                        })
        
        # 5. Chọn best answer
        if answers:
            answers.sort(key=lambda x: x["score"], reverse=True)
            best_answer = answers[0]['text']
            # Clean text: bỏ underscore từ ViTokenizer, sửa spacing
            lst_Answer_Final[idx] = best_answer.replace("_", " ")...
        else:
            lst_Answer_Final[idx] = "Không có câu trả lời"
    
    return lst_Answer_Final
```

---

## 3. Giải thích các tham số quan trọng

### `max_length = 512`
- Giới hạn tối đa tokens cho input (question + context)
- BERT có giới hạn 512 tokens
- Nếu context dài hơn → sử dụng stride

### `stride = 150`
- Khi context dài hơn max_length, chia thành chunks overlap
- Overlap 150 tokens giữa các chunks
- Đảm bảo không mất thông tin ở biên

### `truncation = "only_second"`
- Chỉ cắt context (second segment), giữ nguyên question (first segment)
- Đảm bảo question luôn đầy đủ

### `N_BEST = 380`
- Xem xét top 380 start positions và top 380 end positions
- Tổng combinations: 380 × 380 = 144,400 pairs (nhưng nhiều bị filter)
- Giá trị cao để không bỏ sót answer tốt

### `max_answer_length = 2000`
- Giới hạn độ dài answer (tính bằng token positions)
- Tránh extract quá dài (gần như toàn bộ context)

### `offset_mapping`
- Mapping từ token index → character positions trong text gốc
- Dùng để extract answer text chính xác từ context

---

## 4. Scoring

```
answer_score = start_logits[start_index] + end_logits[end_index]
```
- `start_logits`: Xác suất mỗi token là điểm bắt đầu answer
- `end_logits`: Xác suất mỗi token là điểm kết thúc answer
- Score = tổng 2 logits → answer có cả start và end confidence cao nhất

---

## 5. Text Cleaning

```python
best_answer.replace("_", " ")      # ViTokenizer: "kết_hôn" → "kết hôn"
    .replace(' .', '.')             # Sửa spacing trước dấu chấm
    .replace(' ,', ',')             # Sửa spacing trước dấu phẩy
    .replace(' !', '!')
    .replace(' ?', '?')
    .replace(' :', ':')
    .replace(' ;', ';')
```
- ViTokenizer nối từ ghép bằng `_` → cần thay lại thành space
- Sửa các lỗi spacing do tokenization

---

## 6. Vai trò trong Pipeline

```
Top 5 documents (sau Rerank)
    │
    ▼
┌─────────────────────────────┐
│ BERT Extract                │
│                             │
│ Doc 1: "Điều 8 Luật HNGĐ   │ → "Nam từ đủ 20 tuổi, nữ từ đủ 18 tuổi"
│ 2014 quy định nam từ đủ    │
│ 20 tuổi trở lên, nữ từ     │
│ đủ 18 tuổi trở lên mới     │
│ được kết hôn. Ngoài ra..."  │
│                             │
│ Doc 2: "Nghị định 126..."   │ → "Không có câu trả lời"
│                             │
│ Doc 3: ...                  │ → "..."
└─────────────────────────────┘
    │
    ▼
{0: "Nam từ đủ 20 tuổi...", 1: "Không có câu trả lời", 2: "..."}
    │
    ├──► GraderAgent: chấm điểm relevance của các snippet (doc 10)
    │
    └──► Generator: dùng làm highlight kèm full doc (doc 09, 10)
```

### Lợi ích
1. **Tín hiệu cho Grader**: span trích được là cái grader chấm điểm để quyết định retry/web fallback.
2. **Highlight cho Generator**: hướng "focus attention" vào đoạn quan trọng nhất (kết hợp full doc để không bỏ sót).
3. **Extractive**: span lấy từ text gốc, hỗ trợ giảm hallucination.
