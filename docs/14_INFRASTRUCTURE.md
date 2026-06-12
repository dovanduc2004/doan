# INFRASTRUCTURE - Hạ tầng triển khai

## Các file liên quan:
- `docker-compose.yml`
- `run_mac.sh`
- `SETUP_MAC.md`
- `requirements.txt`
- `.env` / `.env.example`
- `.dockerignore`
- `upload_to_hf.py`

> **Thay đổi**: kiến trúc hạ tầng đơn giản hơn nhiều. Không còn MSSQL container, không còn `Dockerfile`/`init_and_run.sh` cho app. App **chạy trực tiếp bằng uvicorn** (local venv), chỉ **Qdrant** chạy trong Docker. SQLite thay cho MSSQL nên không cần init DB script. LLM dùng AWS Bedrock (cloud) nên không load LLM cục bộ.

---

## 1. Docker Compose (`docker-compose.yml`)

Chỉ orchestrate **một service**: Qdrant.

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant_law
    restart: always
    ports:
      - "6333:6333"      # REST API
      - "6334:6334"      # gRPC API
    volumes:
      - ./dataset/qdrant_data:/qdrant/storage
    networks:
      - mynetwork

networks:
  mynetwork:
    driver: bridge
```
- **Volume**: persist dữ liệu vector tại `./dataset/qdrant_data`.
- App KHÔNG nằm trong compose — chạy ở host và kết nối Qdrant qua `URL_QDRANT_LOCAL` (vd `http://localhost:6333/`).

---

## 2. Script chạy local (`run_mac.sh`)

```bash
#!/bin/bash
set -e
cd "$(dirname "$0")"
PORT="${PORT:-8080}"

# 1. Tạo venv nếu chưa có (python3.11 ưu tiên)
if [ ! -d ".venv" ]; then
  python3.11 -m venv .venv 2>/dev/null || python3 -m venv .venv
fi
source .venv/bin/activate

# 2. Cài dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Khởi động Qdrant container
docker compose up -d qdrant

# 4. Chạy FastAPI
exec uvicorn main:app --host 0.0.0.0 --port "${PORT}" --reload
```

### Luồng
```
run_mac.sh
   │
   ├── tạo / kích hoạt .venv
   ├── pip install -r requirements.txt
   ├── docker compose up -d qdrant     (port 6333/6334)
   └── uvicorn main:app --port 8080 --reload
         │
         └── import main.py → import chat.py
               ├── load embedding / BERT QA / Cross-Encoder
               ├── lazy info.json embeddings (khi cần)
               ├── khởi tạo Orchestrator + agents
               └── server sẵn sàng trên :8080
```
- `--reload`: hot reload khi sửa code (development).
- Mặc định PORT 8080 (override bằng env `PORT`).

---

## 3. Cài đặt macOS (`SETUP_MAC.md`)
Hướng dẫn cài đặt cho macOS / Apple Silicon. Điểm chính:
- Python 3.11, tạo venv.
- `TORCH_DEVICE="mps"` để tận dụng GPU Apple Silicon (config tự phát hiện nếu không set).
- Cấu hình AWS credentials cho Bedrock + `TAVILY_API_KEY`.
- Chạy `./run_mac.sh`.

---

## 4. Biến môi trường (`.env.example`)

Nhóm chính:
```dotenv
# LLM provider
LLM_PROVIDER="bedrock"            # hoặc "gemini"

# AWS Bedrock (LLM mặc định)
AWS_REGION="us-east-1"
BEDROCK_MODEL_ID="us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# Gemini (fallback)
APIS_GEMINI_LIST=KEY1,KEY2
MODEL_GEMIMI="models/gemini-2.0-flash"

# Qdrant
URL_QDRANT_LOCAL="http://localhost:6333/"
EXIST_COLLECTION_NAME="Domain_Law"

# HuggingFace models (tự tải lần đầu)
GENERATE_MODEL_EXTRACT="bhuy71/bert_extract_full_fine-tuned"
GENERATE_MODEL_TOKENIZER="bhuy71/bert_extract_full_fine-tuned"
MODEL_RERANK="bhuy71/rerank_model"
MODEL_EMBEDDING="bhuy71/embedding_finetuned"

# Cohere (optional, legacy)
MODEL_RERANKER="rerank-v3.5"
API_RERANKER=KEY1,KEY2

# Prompt / info
PATH_PROMPT="gemini_prompting.yaml"
PATH_INFOR="information.json"

# Device & DB
TORCH_DEVICE="mps"                # "mps" | "cuda" | "cpu"
DB_PATH="./source/data/chat_history.db"

# Tavily web search
TAVILY_API_KEY="..."
```

> AWS credentials (`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` hoặc profile/IAM role) cấu hình theo cơ chế chuẩn boto3, không nằm trong `.env.example`.

---

## 5. `.dockerignore`
```
source_research/
```
- Bỏ qua thư mục notebook nghiên cứu khi build image.

---

## 6. `upload_to_hf.py`
Tiện ích upload model (embedding / rerank / extract) lên HuggingFace Hub. Không thuộc runtime của chatbot.

---

## 7. Kiến trúc triển khai

```
┌──────────────────── Host (macOS / Linux) ────────────────────┐
│                                                              │
│   uvicorn main:app  (:8080)                                  │
│   ├── FastAPI + Orchestrator + agents                        │
│   ├── Embedding / BERT QA / Cross-Encoder (local, CPU/MPS)   │
│   └── SQLite (chat_history.db)                                │
│            │                          │                       │
│            │ HTTP                     │ boto3 / HTTPS         │
│            ▼                          ▼                       │
│   ┌─────────────────┐        ┌──────────────────┐            │
│   │ Qdrant (Docker) │        │  AWS Bedrock      │ (cloud)    │
│   │  :6333 / :6334  │        │  Claude Sonnet 4.5│            │
│   └─────────────────┘        └──────────────────┘            │
│                                       │                       │
│                              ┌──────────────────┐            │
│                              │  Tavily API      │ (cloud)     │
│                              └──────────────────┘            │
└──────────────────────────────────────────────────────────────┘
```

---

## 8. Startup Sequence

```
1. docker compose up -d qdrant         (Qdrant sẵn sàng nhanh)
2. uvicorn main:app
     ├── load .env
     ├── import chat.py → load model cục bộ + khởi tạo agent (2-5 phút)
     ├── DBConnection tự tạo schema SQLite (idempotent)
     └── server ready :8080
```
- Thời gian khởi động chủ yếu do load model embedding/BERT/Cross-Encoder.
- LLM (Claude) gọi qua API nên không tốn thời gian load.
