# MODEL LAYER - Tầng AI Models

## Các file liên quan:
- `source/model/embedding_model.py`
- `source/model/extract_model.py`
- `source/model/generate_model.py`
- `source/model/rerank_model_finetune.py`
- `source/model/reset_apikey.py`

> **Thay đổi**: `source/model/rerank_model.py` (wrapper Cohere) đã bị loại bỏ — hệ thống chỉ dùng Cross-Encoder fine-tuned local. `generate_model.py` nay cấu hình cho **AWS Bedrock (Claude)** mặc định, Gemini chỉ là fallback.

---

## 1. Embedding Model (`source/model/embedding_model.py`)

### Class: `Sentences_Transformer_Embedding`
```python
def __init__(self, setting: Settings):
    self.embeddings_bkai = HuggingFaceEmbeddings(
        model_name=setting.MODEL_EMBEDDING,
        model_kwargs=setting.DEVICE,                  # {"device": "mps"/"cuda"/"cpu"}
        encode_kwargs={"normalize_embeddings": True}
    )
```
- **model_name**: từ env `MODEL_EMBEDDING` (vd `bhuy71/embedding_finetuned`).
- **model_kwargs**: device tự phát hiện (MPS/CUDA/CPU).
- **normalize_embeddings=True**: chuẩn hóa vector → cosine similarity tốt hơn.
- Wrapper `langchain_huggingface.HuggingFaceEmbeddings`.

### Cách dùng
- `embed_documents(texts)` / `embed_query(text)`.
- Truyền vào `Qdrant.from_existing_collection()` cho similarity search.
- Dùng cho cosine similarity trên `information.json` (nhánh template).

---

## 2. Extract Model - BERT QA (`source/model/extract_model.py`)

### Class: `Bert_Extract`
```python
def __init__(self, config: Settings):
    self.max_length = config.MAX_LENGTH                  # 512
    self.max_answer_length = config.MAX_ANSWER_LENGTH    # 2000
    self.stride = config.STRIDE                          # 150
    self.model_extract = AutoModelForQuestionAnswering.from_pretrained(config.MODEL_EXTRACT)
    self.tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER)
    self.nbert = config.N_BEST                           # 380
```
- Model BERT QA (predict start/end positions) + tokenizer tương ứng.
- Code load LoRA (PEFT) bị comment lại → hiện dùng full fine-tuned model.

### Vai trò
- Nhận `(question, context)` → trả đoạn text trong context trả lời question.
- Giảm noise từ document dài, làm "highlight" cho generator.

---

## 3. Generate Model (`source/model/generate_model.py`)

### Class: `Gemini`
> Tên class giữ là `Gemini` để tương thích import, nhưng backend hoạt động mặc định là **AWS Bedrock**.

```python
def __init__(self, setting: Settings):
    self.setting = setting
    self.provider = setting.LLM_PROVIDER          # "bedrock" | "gemini"
    self.model_gemini = setting.MODEL_GEMINI       # tên model Gemini (fallback)
    self.bedrock_model_id = setting.BEDROCK_MODEL_ID
    self.aws_region = setting.AWS_REGION
    if setting.APIS_GEMINI_LIST:
        self.key_manager = APIKeyManager(setting.APIS_GEMINI_LIST)
    else:
        self.key_manager = None
```

### Chi tiết
- Chỉ **giữ cấu hình** (model id, region, key manager); không khởi tạo client ở đây.
- Việc tạo client LLM thực sự nằm ở `Gemini_Generate._make_model()` (doc 09), chọn `ChatBedrock` hay `ChatGoogleGenerativeAI` theo `provider`.
- `key_manager` chỉ tạo khi có danh sách key Gemini (dùng cho fallback).

---

## 4. Rerank Model Fine-tuned (`source/model/rerank_model_finetune.py`)

### Class: `RerankModelFinetune`
```python
def __init__(self, setting: Settings):
    self.model = CrossEncoder(
        setting.RERANK,
        trust_remote_code=True,
        device='cpu' if torch.cuda.is_available() else 'cpu'   # luôn 'cpu'
    )
```
- Cross-Encoder từ `sentence-transformers`, nhận pair `(query, doc)` → relevance score.
- **Lưu ý**: biểu thức device luôn cho ra `'cpu'` (cả hai nhánh đều `'cpu'`). Cross-Encoder rerank top ≤50 doc nên chạy CPU vẫn chấp nhận được.
- Đây là reranker duy nhất trong production (dùng bởi `Rerank_Utils.rerank_documents_finetune`).

---

## 5. API Key Manager (`source/model/reset_apikey.py`)

### Class: `APIKeyManager`
Round-robin nhiều API key để tránh rate limit — **chỉ liên quan tới Gemini fallback** (Bedrock dùng xác thực AWS, không cần quản lý key ở đây).

```python
def __init__(self, keys):
    self.keys = {key: 0 for key in keys}   # key → usage count
    self.index = 0
    self.key_list = list(self.keys.keys())

def get_next_key(self):
    if len(set(self.keys.values())) == 1:
        self.index = 0
    key = self.key_list[self.index]
    self.keys[key] += 1
    self.index = (self.index + 1) % len(self.key_list)
    return key
```

#### Round-robin
```
Keys: [A, B, C]
Call 1 → A   Call 2 → B   Call 3 → C   Call 4 → A (reset) ...
```
- `get_key_usage()` trả về `{key: count}` để monitor.

> Khi `LLM_PROVIDER=bedrock` (mặc định), key manager hầu như không được dùng tới.
