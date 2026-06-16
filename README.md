# Legal Retrieval System with Agentic RAG

A Vietnamese legal question-answering chatbot built on an **Agentic RAG** architecture. Instead of a single retrieve-then-generate pass, an **Orchestrator** drives a reflective loop that coordinates three specialized agents (**Planner**, **Grader**, **Hallucination Checker**) on top of three fine-tuned models (embedding, Cross-Encoder reranker, BERT answer extractor).

Reasoning and generation are powered by **AWS Bedrock (Claude Sonnet 4.5)** with a configurable Gemini fallback. The knowledge base covers five Vietnamese legal domains in **Qdrant**, conversations are stored in **SQLite**, realtime questions can be answered via **Tavily** web search, and every reasoning step is streamed to the UI over **Server-Sent Events (SSE)**.

---

## Features

- **Agentic reflective loop** — grade retrieval, rewrite & retry on weak context, verify the answer against its sources before returning.
- **Unified Planner** — one LLM call does intent classification, tool selection, query rewriting, and metadata-filter extraction.
- **Fine-tuned models** — domain-specific embedding, Cross-Encoder reranker, and BERT span extractor for Vietnamese law.
- **Web fallback** — Tavily search for realtime / out-of-corpus questions.
- **Two-tier memory** — rolling session summary + long-term user facts for multi-turn personalization.
- **Live streaming** — SSE shows each agent step (plan → retrieve → extract → grade → generate → verify) in real time.
- **5 legal domains** — Education, Transportation, Marriage & Family, Labor & Salary, Criminal Liability.

---

## Architecture

```
User (Web UI)
   │  POST /api/chat/chatbot-agentic  (SSE stream)
   ▼
Memory.build_context  ──► [long-term facts] + [session summary] + [last 5 turns]
   ▼
Planner Agent (1 LLM call) ──► intent + tool + rewritten_query + metadata filter
   ▼
 route by tool ─┬─ template   → chitchat reply
                ├─ ask_user   → clarifying question
                ├─ reject     → out-of-scope refusal
                ├─ web        → Tavily search → synthesize → hallucination check
                └─ rag        → Retrieve (multi-query + RRF) → Rerank → BERT extract
                                  → Grade ──(retry ≤ 2)── → Generate → Hallucination Check
   ▼
SSE "final" (answer + references) ──► background: persist to SQLite + update memory
```

---

## How It Works

The Orchestrator (`source/agent/orchestrator.py`) runs a reflective loop and emits SSE events for each step:

1. **Build context** — `Memory.build_context` assembles `[long-term facts] + [session summary] + [last 5 turns]`.
2. **Plan** — the Planner makes one LLM call to classify intent, select a tool, rewrite the query, and extract metadata filters (with enforced routing consistency).
3. **Route by tool**:
   - `template` → chitchat reply from `information.json`
   - `ask_user` → ask a clarifying question
   - `reject` → polite out-of-scope refusal (web search off)
   - `web` → Tavily search → synthesize → hallucination check
   - `rag` → the legal pipeline (below)
4. **Retrieve** (rag) — generate 3 query variants, tokenize with PyVi, search Qdrant under a single metadata filter, fuse with **RRF** (top ~50), then **Cross-Encoder rerank** to top 5.
5. **Extract** — fine-tuned BERT highlights the key span per document.
6. **Grade** — CRAG-inspired Grader scores relevance. Below `GRADER_THRESHOLD` (0.6) → rewrite query and retry (up to `MAX_ITERATIONS`, default 2), keeping the best iteration. Best score < 0.4 with web search on → fall back to Tavily.
7. **Generate** — Claude Sonnet 4.5 answers from full article text + BERT highlights and returns cited document indices.
8. **Verify** — Hallucination Checker validates claims against cited sources; triggers a single regeneration only on strong evidence (confidence ≥ 0.75 and ≥ 2 unsupported claims).
9. **Persist** — after the stream ends, a background task saves the turn to SQLite, refreshes the rolling summary, and extracts long-term user facts.

### Agents (`source/agent/`)

| Agent | Role | Output |
| ----- | ---- | ------ |
| **Planner** | Routing + query rewriting + metadata extraction in one call | `intent`, `tool`, `rewritten_query`, `entities` |
| **Grader** | Evaluate whether retrieved snippets answer the query | `relevance_score`, `decision`, `missing_aspects` |
| **Hallucination Checker** | Verify each claim against cited sources | `is_grounded`, `confidence`, `unsupported_claims` |
| **Orchestrator** | Wire agents into the reflective loop, stream SSE events | — |

### Memory subsystem

- **Short-term (per session)** — keeps the last 5 turns verbatim; older turns are compressed into a rolling summary in `Chat_Sessions.summary`.
- **Long-term (global)** — durable user facts (name, role, ongoing legal matter, preferences) extracted by the LLM and stored in `User_Facts`, injected into context every turn.

---

## Tech Stack

| Layer            | Technology                                          |
| ---------------- | --------------------------------------------------- |
| LLM reasoning    | AWS Bedrock — Claude Sonnet 4.5 (Gemini fallback)   |
| Agents           | Planner, Grader, Hallucination Checker (Orchestrator) |
| Embedding        | Sentence-BERT + Matryoshka — `bhuy71/embedding_finetuned` |
| Reranker         | Cross-Encoder + RRF — `bhuy71/rerank_model`         |
| Extractor        | Fine-tuned BERT — `bhuy71/bert_extract_full_fine-tuned` |
| Vector DB        | Qdrant (collection `Domain_Law`)                    |
| Web search       | Tavily                                              |
| Tokenization     | PyVi (ViTokenizer)                                  |
| Conversation DB  | SQLite                                              |
| Backend          | FastAPI + Uvicorn + SSE, Docker (Qdrant)            |

---

## Project Structure

```
.
├── main.py                  # FastAPI app entry (serves UI + mounts /api)
├── run_mac.sh               # macOS bootstrap: venv + deps + Qdrant + Uvicorn
├── docker-compose.yml       # Qdrant container (ports 6333/6334)
├── requirements.txt
├── .env.example             # configuration template
├── source/
│   ├── agent/               # planner, grader, hallucination_checker, orchestrator, schemas
│   ├── api/                 # router + endpoints (chat SSE, session CRUD)
│   ├── core/                # config, prompts, information.json
│   ├── data/                # SQLite (db + memory) and Qdrant client
│   ├── model/               # embedding, extract, generate, rerank wrappers
│   ├── search/ rerank/ extract/ generate/   # pipeline utilities
│   └── tool/                # Tavily web search
├── templates/ static/       # frontend
├── dataset/                 # legal documents + Qdrant storage
└── source_research/         # fine-tuning notebooks + evaluation scripts
```

---

## Getting Started

### Prerequisites
- Python 3.11
- Docker (for Qdrant)
- AWS Bedrock credentials (or a Gemini API key)

### Setup

1. Copy the env template and fill in your credentials:
   ```bash
   cp .env.example .env
   ```
   Set `LLM_PROVIDER` (`bedrock` or `gemini`), AWS Bedrock / Gemini keys, `TAVILY_API_KEY`, and the Qdrant URL. HuggingFace models are downloaded automatically on first run.

2. Start Qdrant and load the `Domain_Law` collection (data on the Google Drive link below).

3. Run the app:
   ```bash
   ./run_mac.sh
   ```
   or manually:
   ```bash
   docker compose up -d qdrant
   uvicorn main:app --host 0.0.0.0 --port 8080
   ```

4. Open <http://127.0.0.1:8080>.

### Configuration (`.env`)

| Variable                      | Description                                  | Default |
| ----------------------------- | -------------------------------------------- | ------- |
| `LLM_PROVIDER`                | `bedrock` or `gemini`                        | bedrock |
| `BEDROCK_MODEL_ID`            | Claude model id                              | claude-sonnet-4-5 |
| `MAX_ITERATIONS`              | max retrieval retries                        | 2       |
| `GRADER_THRESHOLD`            | relevance acceptance threshold               | 0.6     |
| `HALLUCINATION_CHECK_ENABLED` | toggle the fact-check stage                  | true    |
| `TORCH_DEVICE`                | `mps` / `cuda` / `cpu` (auto-detected)       | auto    |

---

## API

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| POST   | `/api/chat/chatbot-agentic` | Main agentic endpoint — streams reasoning via SSE (`query`, `session_id`, `use_web_fallback`) |
| GET    | `/api/chat/memory/facts` | List long-term user facts |
| DELETE | `/api/chat/memory/facts/{id}` | Delete a user fact |
| POST   | `/api/session/start-session` | Create a new chat session |
| GET    | `/api/session/get-sessions` | List sessions |
| GET    | `/api/session/get-chat-history/{id}` | Get a session's messages |
| DELETE | `/api/session/delete-session/{id}` | Delete a session |

---

## Evaluation

**Fine-tuned modules**

| Module         | Technique               | Score              |
| -------------- | ----------------------- | ------------------ |
| BERT Extractor | Full & LoRA Fine-Tuning | F1 0.89, EM 0.81   |
| Embedding      | SBERT + Matryoshka      | NDCG@10 0.92       |
| Reranker       | Cross-Encoder + RRF     | F1 0.9985          |

**Planner Agent** (1,000 samples): intent Macro-F1 **0.9427**, tool routing accuracy **0.958**, entity F1 0.9231, query rewriting cosine 0.8650, JSON parse 99.2%.

**Integrated retriever** (200 questions, LLM-as-judge): Hit@5 **0.815**, MRR@5 0.566, NDCG@5 0.642, Recall@50 0.865.

**End-to-end system** (722 legal questions): average cosine **0.8603**, ROUGE-L **0.3600**, Token-F1 **0.1911**.

Evaluation scripts: `source_research/Eval_System/` (end-to-end) and `source_research/Agentic_RAG/eval/` (Planner).

---

## Author

- Bùi Văn Huy — [bhuy71](https://github.com/bhuy71)
