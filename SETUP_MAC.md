# Setup Guide — macOS (Apple Silicon M1/M2/M3/M4)

This document walks you through every environment variable you must update in `.env` before deploying the Agentic RAG system on macOS, where to obtain each credential, and how to verify everything works.

---

## 1. Prerequisites

Install once on your Mac:

| Tool | Install command | Why |
|---|---|---|
| Homebrew | `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"` | Package manager |
| Python 3.11 | `brew install python@3.11` | Runtime for FastAPI + ML stack |
| Docker Desktop | Download from https://www.docker.com/products/docker-desktop/ | Runs Qdrant vector DB |
| Git (optional) | `brew install git` | If cloning the repo |

Open Docker Desktop at least once and make sure the whale icon is in the menu bar before running the project.

---

## 2. Environment Variables — Full Reference

Open the `.env` file at the project root:

```bash
cd "/Users/admin/đồ án tốt nghiệp/Retrieval_Information_System_With_Agentic_RAG-main"
open -e .env
```

All variables below live in that file. Replace the values marked **MUST CHANGE** with your own credentials.

### 2.1 LLM Provider — AWS Bedrock (default) or Google Gemini

```
LLM_PROVIDER="bedrock"

# --- AWS Bedrock path ---
AWS_REGION="us-east-1"
BEDROCK_MODEL_ID="us.anthropic.claude-sonnet-4-5-20250929-v1:0"

# --- Gemini path (optional fallback) ---
APIS_GEMINI_LIST=key1,key2,key3,key4
MODEL_GEMIMI="models/gemini-2.0-flash"
```

**What it does:** The LLM powers the query classifier, query rewriter, entity extractor, and final answer generator. Every user question hits the LLM 4–6 times.

---

#### Option A — AWS Bedrock (recommended, already wired up)

**Why Bedrock:** Gemini free tier on `gemini-2.5-flash` is 5 RPM per project — not enough for a single chat turn. Bedrock is pay-per-token (~$0.003/input 1K tokens on Sonnet 4.5) with no per-minute cap. 100 demo chats ≈ $1–2.

**Prerequisites:**
1. **AWS account** with billing enabled.
2. **AWS CLI installed and configured:**
   ```bash
   brew install awscli
   aws configure
   # enter Access Key ID, Secret Key, region (us-east-1), output (json)
   ```
   Verify with `aws sts get-caller-identity` — should print your account ID.

3. **Request Bedrock model access** (one-time, takes ~1 minute):
   - Go to https://console.aws.amazon.com/bedrock/ → make sure you're in **us-east-1**
   - Left sidebar → **Model access** → **Manage model access**
   - Tick **Claude Sonnet 4.5** under Anthropic → **Request access** → Submit
   - Wait for "Access granted" (usually instant for Anthropic models)

4. **Verify Bedrock works from CLI:**
   ```bash
   aws bedrock list-foundation-models --region us-east-1 \
     --query "modelSummaries[?contains(modelId, 'sonnet-4-5')].modelId" \
     --output text
   ```
   Should print `anthropic.claude-sonnet-4-5-20250929-v1:0`.

**Model choice (`BEDROCK_MODEL_ID`):**
- `us.anthropic.claude-sonnet-4-5-20250929-v1:0` — recommended, balanced quality/cost
- `us.anthropic.claude-haiku-4-5-20251001-v1:0` — cheapest, faster
- `us.anthropic.claude-opus-4-7` — highest quality, most expensive

The `us.` prefix means cross-region inference profile (routes across US regions automatically for better availability).

**IAM permissions:** Your AWS user needs `bedrock:InvokeModel` on the model ARN. The default admin policy or `AmazonBedrockFullAccess` managed policy works.

---

#### Option B — Google Gemini (legacy fallback)

Set `LLM_PROVIDER="gemini"` in `.env` and fill in `APIS_GEMINI_LIST`.

**How to get Gemini keys:**
1. Go to https://aistudio.google.com/app/apikey
2. Sign in with a Google account
3. Click **Create API key** → choose a new project
4. Copy the key (starts with `AIza...`)
5. Paste comma-separated into `.env`:

   ```
   APIS_GEMINI_LIST=AIzaSyA...,AIzaSyB...,AIzaSyC...,AIzaSyD...
   ```

**Model choice (`MODEL_GEMIMI`):**
- `models/gemini-2.0-flash` — **recommended for free tier**, 15 RPM
- `models/gemini-2.5-flash` — only 5 RPM on free tier, too low for this system

Quotas are per Google Cloud project, not per key. Creating multiple keys in the same project doesn't raise the limit — create separate Google accounts if you need more throughput.

---

### 2.2 Cohere Rerank — document re-ranking (MUST CHANGE)

```
MODEL_RERANKER="rerank-v3.5"
API_RERANKER=key1,key2
```

**What it does:** After Qdrant retrieves candidate documents, Cohere's reranker re-orders them by relevance to the query. This improves answer quality significantly.

**How to get keys:**
1. Go to https://dashboard.cohere.com/api-keys
2. Sign up (free tier — no credit card required)
3. Copy the **Trial key** from the dashboard
4. Optionally create a second account for a second key
5. Paste comma-separated into `.env`:

   ```
   API_RERANKER=COHERE_KEY_1,COHERE_KEY_2
   ```

Free tier quota: 1,000 rerank calls/month per key.

**Alternative:** If you don't want Cohere, the pipeline also has a local fine-tuned reranker (`bhuy71/rerank_model`) used in `rerank_documents_finetune()`. The default path in `get_Article_Content_Results()` already uses the local reranker, so Cohere is actually optional for the main chat flow. But set the key anyway to avoid import errors.

---

### 2.3 Tavily — web search fallback (OPTIONAL)

```
TAVILY_API_KEY="tvly-..."
```

**What it does:** When Qdrant cannot find relevant legal documents for a question, the `/chatbot-with-search-web` endpoint falls back to Tavily to fetch external web pages. Tavily is used instead of Google Custom Search because Google closed CSE to new customers in 2025.

**How to get the key:**
1. Go to https://tavily.com/
2. Sign up (free tier — no credit card required). Supports Google / GitHub SSO.
3. After sign-in, the dashboard shows your API key (starts with `tvly-`)
4. Copy → paste into `.env`:
   ```
   TAVILY_API_KEY="tvly-..."
   ```

Free tier: 1,000 credits / month (≈ 1,000 searches). Plenty for a demo.

**Skip if:** You don't need web-search fallback. The main `/chatbot-with-gemini` endpoint works without Tavily — it only queries Qdrant.

**Legacy Google CSE variables** (`GOOGLE_SEARCH_API`, `TOOL_SEARCH_API`) are still in `.env` but unused; safe to delete.

---

### 2.4 Hardware — Torch device (VERIFY)

```
TORCH_DEVICE="mps"
```

**What it does:** Tells PyTorch which compute backend to use for embedding, reranking, and BERT extraction models.

**Values:**
- `mps` — Apple Silicon GPU (M1/M2/M3/M4). **Use this on your M4.**
- `cpu` — fallback; works on any Mac but ~5× slower
- `cuda` — NVIDIA GPU (not on Macs)

If you see MPS-related errors at startup, change to `cpu` and restart.

---

### 2.5 Chat history — SQLite (VERIFY)

```
DB_PATH="./source/data/chat_history.db"
```

**What it does:** Path where the SQLite file for chat sessions/messages/references is created. The schema is built automatically on first run.

Leave the default unless you want the DB somewhere else. The parent folder is auto-created.

---

### 2.6 Qdrant — vector DB (NO CHANGE)

```
URL_QDRANT_LOCAL="http://localhost:6333/"
EXIST_COLLECTION_NAME="Domain_Law"
```

Qdrant runs in Docker via `docker compose up -d qdrant`. The compose file mounts `./dataset/qdrant_data` (where your crawled/snapshot data lives) into the container. Collection name `Domain_Law` matches the snapshot you already have at `dataset/qdrant_data/collections/Domain_Law`.

**Do not change these** unless you rename the collection.

---

### 2.7 HuggingFace models (NO CHANGE)

```
GENERATE_MODEL_EXTRACT="bhuy71/bert_extract_full_fine-tuned"
GENERATE_MODEL_TOKENIZER="bhuy71/bert_extract_full_fine-tuned"
MODEL_RERANK="bhuy71/rerank_model"
MODEL_EMBEDDING="bhuy71/embedding_finetuned"
```

These are public HuggingFace models, auto-downloaded on first run (~2–3 GB total, cached in `~/.cache/huggingface/`). No credentials needed.

---

### 2.8 Prompt and information files (NO CHANGE)

```
PATH_PROMPT="gemini_prompting.yaml"
PATH_INFOR="information.json"
```

These resolve to `source/core/gemini_prompting.yaml` and `source/core/information.json`. Leave as-is.

---

## 3. Summary — What You Must Edit

Minimum changes for a working demo:

| Variable | Action |
|---|---|
| `LLM_PROVIDER` | Keep `bedrock` (recommended) or switch to `gemini` |
| `AWS_REGION` / `BEDROCK_MODEL_ID` | Already set for Claude Sonnet 4.5 in us-east-1 — run `aws configure` separately |
| `APIS_GEMINI_LIST` | Only needed if `LLM_PROVIDER=gemini` |
| `API_RERANKER` | Replace with your own 1–2 Cohere trial keys |
| `GOOGLE_SEARCH_API` | (Optional) replace if you want web-search fallback |
| `TOOL_SEARCH_API` | (Optional) replace if you want web-search fallback |
| `TORCH_DEVICE` | Keep `mps` on M4 |

Everything else is already set correctly.

---

## 4. Deploy and Run

```bash
cd "/Users/admin/đồ án tốt nghiệp/Retrieval_Information_System_With_Agentic_RAG-main"
./run_mac.sh
```

What the script does:
1. Creates a Python virtual environment `.venv`
2. Installs all dependencies from `requirements.txt`
3. Starts Qdrant in Docker (`docker compose up -d qdrant`)
4. Launches FastAPI on port 8000 with auto-reload

First launch: ~5–10 min (dependency install + model download).
Subsequent launches: ~30 sec.

When you see `Uvicorn running on http://0.0.0.0:8000`, open:

**http://127.0.0.1:8000**

The chat UI loads. Click **New chat** (top-left) and type a Vietnamese legal question.

---

## 5. Verify Each Component

Open a second terminal while the app runs:

**Qdrant:**
```bash
curl http://localhost:6333/collections/Domain_Law
```
Look for `"status":"green"` and non-zero `points_count`.

**FastAPI:**
```bash
curl http://127.0.0.1:8000/api/session/start-session -X POST
```
Should return `{"session_id": 1}`.

**Full pipeline:**
```bash
curl -X POST http://127.0.0.1:8000/api/chat/chatbot-with-gemini \
  -H 'Content-Type: application/json' \
  -d '{"query":"Điều kiện kết hôn theo pháp luật Việt Nam là gì?"}'
```
Should return an `answer` field with a Vietnamese legal response.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot connect to Docker daemon` | Docker Desktop not running | Open Docker Desktop app, wait for whale icon |
| `Connection refused localhost:6333` | Qdrant container not up | `docker compose up -d qdrant` |
| `429 Resource exhausted` (Gemini) | Gemini free-tier quota hit | Switch to `LLM_PROVIDER=bedrock` or use `models/gemini-2.0-flash` |
| `AccessDeniedException` (Bedrock) | Model access not granted | Bedrock console → Model access → enable Claude Sonnet 4.5 |
| `ValidationException: temperature and top_p cannot both be specified` | Old cached generate.py | Restart uvicorn to pick up the latest code |
| `ExpiredToken` / `NoCredentialsError` | AWS creds expired | Re-run `aws configure` |
| `MPS backend out of memory` | Apple GPU OOM on large batch | Set `TORCH_DEVICE=cpu` in `.env`, restart |
| `Address already in use :8080` | Previous run still running | `lsof -ti:8080 \| xargs kill -9` |
| Slow first query | Models downloading / MPS warmup | Normal; subsequent queries are 1–3s |
| `No module named 'bs4'` / `langchain_aws` | Dependencies missing | `pip install -r requirements.txt` inside `.venv` |

---

## 7. Stop the System

```bash
# in the terminal running uvicorn: Ctrl+C
docker compose down    # stop Qdrant
```

Your chat history in `source/data/chat_history.db` and the Qdrant snapshot in `dataset/qdrant_data/` persist across restarts.
