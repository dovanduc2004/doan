from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import json

from source.search.utils_search import Qdrant_Utils
from source.rerank.utils_rerank import Rerank_Utils
from source.model.embedding_model import Sentences_Transformer_Embedding
from source.model.extract_model import Bert_Extract
from source.model.generate_model import Gemini
from source.model.rerank_model_finetune import RerankModelFinetune
from source.data.vectordb.qdrant import Qdrant_Vector
from source.core.config import Settings
from source.generate.generate import Gemini_Generate
from source.extract.utils_extract import Extract_Information
from source.schema.chatbot_querry import ChatbotQuery
from source.tool.utils_search import Utils_Search_Tools
from source.tool.tavily_search import TavilySearchTool
from source.data.db.db_connection import DBConnection
from source.data.db.utils_db import DB_Utils
from source.data.db.memory import Memory

from source.agent.planner import PlannerAgent
from source.agent.grader import GraderAgent
from source.agent.hallucination_checker import HallucinationChecker
from source.agent.orchestrator import Orchestrator


setting = Settings()
gemini = Gemini(setting)
bert = Bert_Extract(setting)
sentences_transformer_embedding = Sentences_Transformer_Embedding(setting)
qdrant = Qdrant_Vector(setting, sentences_transformer_embedding)
router = APIRouter()
model_finetune = RerankModelFinetune(setting)
rerank_Utils = Rerank_Utils(model_finetune)
extract_Utils = Extract_Information(bert)
generate_Utils = Gemini_Generate(gemini, setting)
qdrant_Utils = Qdrant_Utils(qdrant, generate_Utils)

# Web search
web_search_tool = TavilySearchTool(setting)
search_tools = Utils_Search_Tools(setting, generate_Utils, extract_Utils, web_search_tool)

# Memory (short-term rolling summary + long-term facts)
_db = DBConnection(setting)
_db_utils = DB_Utils(_db)
memory = Memory(_db, generator=generate_Utils)

# Agentic components
planner_agent = PlannerAgent(generate_Utils)
grader_agent = GraderAgent(generate_Utils, threshold=setting.GRADER_THRESHOLD)
hallucination_checker = HallucinationChecker(generate_Utils)
orchestrator = Orchestrator(
    planner=planner_agent,
    grader=grader_agent,
    checker=hallucination_checker,
    generator=generate_Utils,
    qdrant_utils=qdrant_Utils,
    rerank_utils=rerank_Utils,
    extract_utils=extract_Utils,
    search_tools=search_tools,
    memory=memory,
    settings=setting,
    embedding=sentences_transformer_embedding,
    max_iterations=setting.MAX_ITERATIONS,
)


def _persist_and_update_memory(session_id, user_msg, bot_msg, references=None):
    """Save the turn to SQLite and refresh rolling summary + long-term facts.

    Called from BackgroundTasks after the SSE stream completes so the user
    doesn't wait on memory housekeeping.
    """
    if session_id is None:
        return
    try:
        _db_utils.Insert_Message(session_id, "user", user_msg)
        bot_id = _db_utils.Insert_Message(session_id, "bot", bot_msg)
        if references:
            _db_utils.Insert_References(bot_id, references)
    except Exception as e:
        print(f"[persist_turn] {type(e).__name__}: {e}", flush=True)
    try:
        memory.refresh_summary_if_needed(session_id)
    except Exception as e:
        print(f"[refresh_summary] {type(e).__name__}: {e}", flush=True)
    try:
        memory.extract_and_save_facts(user_msg)
    except Exception as e:
        print(f"[extract_facts] {type(e).__name__}: {e}", flush=True)


@router.post("/chatbot-agentic")
def chatbot_agentic(query: ChatbotQuery, background_tasks: BackgroundTasks):
    """Single agentic endpoint — streams reasoning via SSE.

    Replaces /chatbot-with-gemini and /chatbot-with-search-web. The toggle
    now lives in `use_web_fallback`; the orchestrator itself decides when
    to hit Qdrant vs. Tavily based on planner intent.
    """
    user_input = query.query
    session_id = query.session_id
    web_search_enabled = query.use_web_fallback

    # Capture the final answer from the stream so we can persist it once done.
    captured = {"answer": "", "references": []}

    def event_stream():
        try:
            for frame in orchestrator.run(user_input, session_id, web_search_enabled):
                # Peek at every frame to capture the final answer. Frames are
                # SSE-formatted strings ("data: {...}\n\n") — parse the JSON
                # back out.
                try:
                    payload = frame[len("data: "):].strip()
                    if payload:
                        obj = json.loads(payload)
                        if obj.get("type") == "final":
                            captured["answer"] = obj.get("answer", "") or ""
                            captured["references"] = obj.get("references", []) or []
                except Exception:
                    pass
                yield frame
        finally:
            # Persist + memory refresh after the client is done streaming.
            if captured["answer"]:
                background_tasks.add_task(
                    _persist_and_update_memory,
                    session_id, user_input, captured["answer"], captured["references"],
                )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering if proxied
        },
    )


# ---- Long-term memory management ----

@router.get("/memory/facts")
def list_facts():
    return {"facts": memory.list_facts()}


@router.delete("/memory/facts/{fact_id}")
def delete_fact(fact_id: int):
    memory.delete_fact(fact_id)
    return {"status": "ok"}
