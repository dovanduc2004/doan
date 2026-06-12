# CORE - Configuration & Prompts

## Các file liên quan:
- `source/core/config.py`
- `source/core/gemini_prompting.yaml`
- `source/core/information.json`

---

## 1. Config (`source/core/config.py`)

### Chức năng
Class `Settings` tập trung quản lý cấu hình, load từ environment variables (`.env`) và định nghĩa hằng số. Đọc trực tiếp bằng `os.getenv()`.

### Hàm phụ trợ: `_default_device()`
Tự chọn device cho PyTorch theo phần cứng:
```python
def _default_device() -> str:
    # MPS (Apple Silicon) → "mps"
    # CUDA               → "cuda"
    # còn lại            → "cpu"
```
Có thể override bằng env `TORCH_DEVICE`.

### Chi tiết các cấu hình

#### Qdrant Vector Database
| Biến | Mô tả |
|---|---|
| `URL_QDRANT_LOCAL` | URL Qdrant (vd `http://localhost:6333/`) |
| `EXIST_COLLECTION_NAME` | Tên collection văn bản luật (vd `Domain_Law`) |
| `metadata_payload_key` | Key metadata trong payload Qdrant (hardcode `"metadata"`) |

#### LLM Provider
| Biến | Mặc định | Mô tả |
|---|---|---|
| `LLM_PROVIDER` | `bedrock` | `bedrock` (Claude) hoặc `gemini` |

#### AWS Bedrock (LLM mặc định)
| Biến | Mặc định | Mô tả |
|---|---|---|
| `AWS_REGION` | `us-east-1` | Region Bedrock |
| `BEDROCK_MODEL_ID` | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | Model id Claude Sonnet 4.5 |

> Xác thực AWS theo cơ chế chuẩn của boto3 (env `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, profile, hoặc IAM role).

#### Google Gemini (legacy / fallback)
| Biến | Mô tả |
|---|---|
| `APIS_GEMINI_LIST` | Danh sách API key Gemini (phân tách `,`) — chỉ dùng khi `LLM_PROVIDER=gemini` |
| `MODEL_GEMIMI` | Tên model Gemini (lưu ý chính tả env: `MODEL_GEMIMI`) |

#### Rerank
| Biến | Mô tả |
|---|---|
| `RERANK` (`MODEL_RERANK`) | Path/name Cross-Encoder fine-tuned (vd `bhuy71/rerank_model`) |
| `MODEL_RERANKER` | Tên model Cohere (legacy, không bắt buộc) |
| `API_RERANKER` | Danh sách API key Cohere (legacy) |

#### Embedding
| Biến | Mô tả |
|---|---|
| `MODEL_EMBEDDING` | Model HuggingFace embedding (vd `bhuy71/embedding_finetuned`) |

#### BERT Extract Model
| Biến | Giá trị | Mô tả |
|---|---|---|
| `MODEL_EXTRACT` (`GENERATE_MODEL_EXTRACT`) | — | Model BERT QA |
| `TOKENIZER` (`GENERATE_MODEL_TOKENIZER`) | — | Tokenizer cho BERT |
| `MAX_LENGTH` | 512 | Độ dài tối đa input |
| `STRIDE` | 150 | Overlap khi chia chunk |
| `N_BEST` | 380 | Số start/end positions xem xét |
| `MAX_ANSWER_LENGTH` | 2000 | Giới hạn độ dài answer |

#### Database (SQLite)
| Biến | Mặc định | Mô tả |
|---|---|---|
| `DB_PATH` | `../data/chat_history.db` | Đường dẫn file SQLite |

#### Paths
| Biến | Mô tả |
|---|---|
| `YAML_PATH` (`PATH_PROMPT`) | Tên file YAML prompt (`gemini_prompting.yaml`) |
| `PATH_INFOR` | Tên file JSON giới thiệu (`information.json`) |

#### Web Search
| Biến | Mô tả |
|---|---|
| `TAVILY_API_KEY` | API key Tavily |

#### Hardware
| Biến | Mô tả |
|---|---|
| `DEVICE` | `{"device": <TORCH_DEVICE hoặc auto>}` |

#### Tuning Agentic RAG
| Biến | Mặc định | Mô tả |
|---|---|---|
| `MAX_ITERATIONS` | 2 | Số vòng retrieve/grade tối đa |
| `GRADER_THRESHOLD` | 0.6 | Ngưỡng score chấp nhận retrieval |
| `HALLUCINATION_CHECK_ENABLED` | true | Bật/tắt kiểm tra hallucination |

### Đặc điểm thiết kế
- Dùng `os.getenv()` trực tiếp (không validate bằng Pydantic BaseSettings).
- Tự phát hiện device (MPS/CUDA/CPU).
- Hằng số model (MAX_LENGTH, STRIDE...) hardcode trong class.

---

## 2. Prompting (`source/core/gemini_prompting.yaml`)

### Chức năng
Chứa các prompt template dạng LCEL `ChatPromptTemplate`. Tên file vẫn là `gemini_prompting.yaml` (lý do lịch sử) nhưng được dùng cho cả Claude (Bedrock) lẫn Gemini.

### Cấu trúc
```yaml
prompts:
  <tên_prompt>:
    messages:
      - role: system
        content: "..."
      - role: human
        content: "..."
```

### Các Prompt Templates (hiện hành)

> Lưu ý: các prompt routing/classification/entity-extraction kiểu cũ (`classify_query`, `query_extract_entities`) **không còn dùng** — việc định tuyến và trích entity đã chuyển vào `PlannerAgent` (prompt nằm inline trong `source/agent/planner.py`, không nằm trong YAML này).

#### 2.1 `query_generator` — Sinh truy vấn mở rộng
- **Mục đích**: từ 1 câu hỏi gốc, sinh thêm **3** truy vấn tương tự để multi-query search.
- **Input**: `{original_query}`.
- **Output**: 3 câu hỏi (mỗi câu 1 dòng).
- **Logic**: CASE 1 (user không biết luật → diễn đạt dễ hiểu hơn); CASE 2 (user biết luật, có số hiệu/điều khoản → diễn đạt chính xác/pháp lý hơn). Giữ semantic similarity ≥ 85%.
- Dùng bởi `Gemini_Generate.generate_query` → gọi trong `Orchestrator._retrieve_and_rerank`.

#### 2.2 `response` — Sinh câu trả lời pháp luật
- **Mục đích**: tổng hợp context (full doc + BERT highlight) thành câu trả lời.
- **Input**: `{original_query}`, `{context}`.
- **Output**: JSON
  ```json
  {"answer": "Câu trả lời tiếng Việt", "key": [0, 2, 4]}
  ```
  `key` là danh sách chỉ số document được trích dẫn.
- **Quy tắc**: chỉ dựa trên context, không bịa; nếu thiếu context → câu xin lỗi mẫu; xây "key list" trước khi viết; output phải `json.loads()` được.
- Dùng bởi `Gemini_Generate.generate_response`.

#### 2.3 `invalid_query` — Từ chối lịch sự
- **Mục đích**: trả lời khi câu hỏi ngoài phạm vi (nhánh `reject`, Web Search tắt).
- **Input**: `{query}`. Output: câu từ chối lịch sự, gợi ý bật Web Search.
- Dùng bởi `Gemini_Generate.invalid_query`.

#### 2.4 `information_query` — Giới thiệu chatbot
- **Mục đích**: trả lời chào hỏi / câu hỏi về bot (nhánh `template`).
- **Input**: `{query}`, `{context}` (top đoạn từ `information.json`). Output: giới thiệu ngắn gọn, lịch sự.
- Dùng bởi `Gemini_Generate.generate_information`.

> Prompt cho **Planner**, **Grader**, **Hallucination Checker** và **tổng hợp web** (`_synthesize`) được viết inline trong code Python tương ứng (`source/agent/*.py`, `source/tool/utils_search.py`), không nằm trong YAML.

---

## 3. Information JSON (`source/core/information.json`)

### Chức năng
Chứa **17 đoạn** văn bản giới thiệu chatbot. Dùng cho nhánh `template` (chitchat).

### Cách sử dụng
1. Khi nhánh template chạy lần đầu, 17 đoạn được embed thành vectors (lazy-load, cache trong Orchestrator).
2. Query người dùng được embed → tìm **top 5** đoạn tương tự nhất (cosine similarity, `search_from_json`).
3. Top 5 đoạn làm context cho prompt `information_query` để sinh câu trả lời.

### Nội dung chính
- Trợ lý pháp luật Việt Nam, do "Bùi Huy" phát triển, dựa trên mô hình AI tiên tiến (kiến trúc Agentic RAG).
- Hỗ trợ các lĩnh vực: hôn nhân & gia đình, giao thông, giáo dục, lao động, hình sự...
- Trích dẫn điều luật cụ thể, nguồn từ Quốc hội / Bộ Tư pháp.
- Không phải luật sư, chỉ mang tính tham khảo.
