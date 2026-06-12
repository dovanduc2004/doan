"""Orchestrator — the agentic control loop.

Ties the four new pieces (planner, grader, hallucination checker,
iterative retrieval) together with the existing retrieval + rerank +
BERT-extract + generate stack. Emits Server-Sent Events so the frontend
can render an honest, step-by-step thinking panel instead of a fake
progress bar.

Event contract (payload is a JSON object with at minimum `type` and
`agent`, plus agent-specific fields):

    plan               — planner decision + reasoning
    template           — chitchat template reply
    clarify            — asks user for more info
    reject             — out-of-scope refusal
    retrieve_start     — about to query Qdrant (iteration N)
    retrieve_done      — docs retrieved, count
    extract            — BERT snippet extraction summary
    grade              — grader verdict
    iterate            — going to retry with new query
    web_start          — Tavily call starting
    web_done           — Tavily links found
    generate_start     — LLM synthesis starting
    generate_done      — answer produced
    check              — hallucination verdict
    regenerate         — about to regenerate with stricter prompt
    final              — final answer + references (terminal)
    error              — unrecoverable error (terminal)
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, Generator, List, Optional

from pyvi import ViTokenizer

from source.agent.grader import GraderAgent
from source.agent.hallucination_checker import HallucinationChecker
from source.agent.planner import PlannerAgent
from source.agent.schemas import Plan
from source.core.config import Settings
from source.data.db.memory import Memory
from source.extract.utils_extract import Extract_Information
from source.function.utils_shared import clean_code_fence_safe, parse_raw_json, load_information_from_json, search_from_json
from source.generate.generate import Gemini_Generate
from source.model.embedding_model import Sentences_Transformer_Embedding
from source.rerank.utils_rerank import Rerank_Utils
from source.search.utils_search import Qdrant_Utils
from source.tool.utils_search import Utils_Search_Tools


def _sse(event: Dict[str, Any]) -> str:
    """Format a dict as a Server-Sent Events frame."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


class Orchestrator:
    """Agentic RAG orchestrator with reflective reasoning.

    Level 3b in the ReAct/Self-Reflection taxonomy:
      Planner → (Retrieve → Grade → maybe retry)* → Generate → Check → Final
    """

    def __init__(
        self,
        planner: PlannerAgent,
        grader: GraderAgent,
        checker: HallucinationChecker,
        generator: Gemini_Generate,
        qdrant_utils: Qdrant_Utils,
        rerank_utils: Rerank_Utils,
        extract_utils: Extract_Information,
        search_tools: Utils_Search_Tools,
        memory: Memory,
        settings: Settings,
        embedding: Sentences_Transformer_Embedding,
        max_iterations: int = 2,
    ):
        self.planner = planner
        self.grader = grader
        self.checker = checker
        self.generator = generator
        self.qdrant_utils = qdrant_utils
        self.rerank_utils = rerank_utils
        self.extract_utils = extract_utils
        self.search_tools = search_tools
        self.memory = memory
        self.settings = settings
        self.embedding = embedding
        self.max_iterations = max_iterations
        # Lazy-loaded intro corpus (used for chitchat template replies)
        self._intro_corpus: Optional[List[str]] = None
        self._intro_embeddings: Optional[Any] = None

    # ---------- public entry point ----------

    def run(
        self,
        user_input: str,
        session_id: Optional[int],
        web_search_enabled: bool,
    ) -> Generator[str, None, None]:
        t0 = time.time()
        memory_context = self.memory.build_context(session_id) if session_id else ""

        # 1) PLAN ----------------------------------------------------------
        t_plan = time.time()
        try:
            plan = self.planner.plan(user_input, memory_context, web_search_enabled)
        except Exception as e:
            yield _sse({"type": "error", "agent": "planner", "message": f"{type(e).__name__}: {e}"})
            return
        yield _sse({
            "type": "plan",
            "agent": "planner",
            "elapsed_ms": int((time.time() - t_plan) * 1000),
            "intent": plan.intent,
            "tool": plan.tool,
            "rewritten_query": plan.rewritten_query,
            "search_query": plan.search_query,
            "entities": plan.entities,
            "reason": plan.reason,
        })

        # 2) BRANCH by tool -----------------------------------------------
        try:
            if plan.tool == "template":
                yield from self._run_template(plan, memory_context)
            elif plan.tool == "ask_user":
                yield from self._run_clarify(plan)
            elif plan.tool == "reject":
                yield from self._run_reject(plan, memory_context)
            elif plan.tool == "web":
                yield from self._run_web(plan, memory_context)
            elif plan.tool == "rag":
                yield from self._run_rag(plan, memory_context, web_search_enabled)
            else:
                yield _sse({"type": "error", "agent": "orchestrator",
                            "message": f"Unknown tool: {plan.tool}"})
        except Exception as e:
            yield _sse({"type": "error", "agent": "orchestrator",
                        "message": f"{type(e).__name__}: {e}"})
            return

        # Total elapsed trace (for debugging; frontend can ignore)
        yield _sse({"type": "done", "total_ms": int((time.time() - t0) * 1000)})

    # ---------- branches ----------

    def _run_template(self, plan: Plan, memory_context: str) -> Generator[str, None, None]:
        """Chitchat branch — cosine-sim over information.json + generate_information."""
        self._ensure_intro_loaded()
        t = time.time()
        context = search_from_json(
            self._intro_embeddings, self._intro_corpus,
            plan.rewritten_query, self.embedding,
        )
        yield _sse({"type": "generate_start", "agent": "generator",
                    "mode": "template", "reason": "Chào hỏi — trả lời bằng mẫu giới thiệu."})
        answer = self.generator.generate_information(
            plan.rewritten_query, context, memory_context=memory_context,
        )
        yield _sse({"type": "generate_done", "agent": "generator",
                    "elapsed_ms": int((time.time() - t) * 1000)})
        yield _sse({"type": "final", "answer": answer, "references": []})

    def _run_clarify(self, plan: Plan) -> Generator[str, None, None]:
        question = plan.clarifying_question or "Bạn có thể cung cấp thêm thông tin cụ thể không?"
        yield _sse({"type": "clarify", "agent": "planner", "question": question})
        yield _sse({"type": "final", "answer": question, "references": []})

    def _run_reject(self, plan: Plan, memory_context: str) -> Generator[str, None, None]:
        """Polite refusal — Web Search toggled off."""
        t = time.time()
        yield _sse({"type": "generate_start", "agent": "generator",
                    "mode": "reject", "reason": "Ngoài phạm vi — trả lời từ chối lịch sự và gợi ý bật Web Search."})
        answer = self.generator.invalid_query(plan.rewritten_query, memory_context=memory_context)
        yield _sse({"type": "generate_done", "agent": "generator",
                    "elapsed_ms": int((time.time() - t) * 1000)})
        yield _sse({"type": "final", "answer": answer, "references": []})

    def _run_web(self, plan: Plan, memory_context: str) -> Generator[str, None, None]:
        """Web search branch — Tavily + synthesize + hallucination check."""
        q = plan.search_query or plan.rewritten_query

        t = time.time()
        yield _sse({"type": "web_start", "agent": "web_researcher", "query": q})
        try:
            links = self.search_tools.search_backend.search(q)
        except Exception as e:
            yield _sse({"type": "error", "agent": "web_researcher",
                        "message": f"{type(e).__name__}: {e}"})
            return
        yield _sse({"type": "web_done", "agent": "web_researcher",
                    "elapsed_ms": int((time.time() - t) * 1000),
                    "n_links": len(links), "links": links})

        if not links:
            answer = "Xin lỗi, tôi không tìm thấy kết quả phù hợp trên web cho câu hỏi này."
            yield _sse({"type": "final", "answer": answer, "references": []})
            return

        tavily_answer = ""
        if hasattr(self.search_tools.search_backend, "get_last_answer"):
            tavily_answer = self.search_tools.search_backend.get_last_answer() or ""
        docs = self.search_tools.search_backend.extract_texts_from_links(links)
        excerpts = "\n\n".join(
            f"[Nguồn {i+1}: {links[i]}]\n{(doc or '')[:4000]}"
            for i, doc in enumerate(docs) if doc
        )

        t = time.time()
        yield _sse({"type": "generate_start", "agent": "generator",
                    "mode": "web_synthesize",
                    "reason": "Tổng hợp câu trả lời từ Tavily answer + trích đoạn các trang."})
        answer = self.search_tools._synthesize(q, tavily_answer, excerpts, links)
        yield _sse({"type": "generate_done", "agent": "generator",
                    "elapsed_ms": int((time.time() - t) * 1000)})

        # Hallucination check on web answer — sources are the trimmed excerpts.
        sources = [(d or "")[:4000] for d in docs if d]
        if tavily_answer:
            sources.append(f"[Tavily summary]\n{tavily_answer}")
        yield from self._maybe_regenerate_web(plan, answer, sources, excerpts, tavily_answer, links)

    def _maybe_regenerate_web(
        self, plan: Plan, answer: str, sources: List[str],
        excerpts: str, tavily_answer: str, links: List[str],
    ) -> Generator[str, None, None]:
        t = time.time()
        result = self.checker.check(answer, sources)
        yield _sse({"type": "check", "agent": "fact_checker",
                    "elapsed_ms": int((time.time() - t) * 1000),
                    "is_grounded": result.is_grounded,
                    "confidence": result.confidence,
                    "unsupported_claims": result.unsupported_claims,
                    "reason": result.reason})
        # Same strict threshold as RAG branch: only regen when we're really
        # sure the answer is wrong (≥ 2 unsupported claims, high confidence).
        if (
            (not result.is_grounded)
            and result.confidence >= 0.75
            and len(result.unsupported_claims) >= 2
        ):
            yield _sse({"type": "regenerate", "agent": "generator",
                        "reason": f"Phát hiện {len(result.unsupported_claims)} claim không có trong nguồn, tổng hợp lại với hướng dẫn cụ thể."})
            t = time.time()
            # NOTE: web path dùng `Utils_Search_Tools._synthesize`, không hỗ trợ
            # feedback injection như RAG — nếu cần chặt hơn có thể mở rộng sau.
            answer = self.search_tools._synthesize(
                plan.search_query or plan.rewritten_query,
                tavily_answer, excerpts, links,
            )
            yield _sse({"type": "generate_done", "agent": "generator",
                        "elapsed_ms": int((time.time() - t) * 1000), "regenerated": True})
        yield _sse({"type": "final", "answer": answer, "references": links})

    def _run_rag(
        self, plan: Plan, memory_context: str, web_search_enabled: bool,
    ) -> Generator[str, None, None]:
        """Legal RAG branch with iterative retrieval + grading.

        Iteration strategy:
          - Bug 1: exit early when score >= threshold (default 0.6), bất kể
            grader label "partial" hay "sufficient" — grader thường quá strict,
            liệt kê mọi gap nhỏ làm `missing_aspects` mà không phản ánh rằng
            câu trả lời vẫn OK. Score là nguồn sự thật đáng tin hơn label.
          - Bug 2: track BEST iteration (highest score) thay vì last —
            vòng retry có thể tệ hơn vòng 1, không được ghi đè mù.
          - Bug 3: web search fallback VẪN GIỮ, nhưng chỉ trigger khi BEST
            score thật sự thấp (<0.4), tức khi mọi iteration đều không đủ.
        """
        query = plan.rewritten_query
        entities = plan.entities or {}

        best_result = None
        best_score = -1.0
        accept_threshold = getattr(self.grader, "threshold", 0.6)
        WEB_FALLBACK_THRESHOLD = 0.4  # dưới mức này = mọi iter đều tệ

        for iteration in range(1, self.max_iterations + 1):
            t = time.time()
            yield _sse({"type": "retrieve_start", "agent": "retriever",
                        "iteration": iteration, "query": query,
                        "entities": entities or None})
            try:
                rerank_docs = self._retrieve_and_rerank(query, entities)
            except Exception as e:
                yield _sse({"type": "error", "agent": "retriever",
                            "message": f"{type(e).__name__}: {e}"})
                return
            yield _sse({"type": "retrieve_done", "agent": "retriever",
                        "iteration": iteration,
                        "elapsed_ms": int((time.time() - t) * 1000),
                        "n_docs": len(rerank_docs),
                        "docs": [self._format_doc_label(infor) for _, infor in rerank_docs]})

            if not rerank_docs:
                snippets: Dict[int, str] = {}
            else:
                t = time.time()
                doc_contents = [doc.replace("_", " ") for doc, _ in rerank_docs]
                snippets = self.extract_utils.predict(doc_contents, query)
                if isinstance(snippets, str):
                    snippets = {}
                yield _sse({"type": "extract", "agent": "extractor",
                            "elapsed_ms": int((time.time() - t) * 1000),
                            "n_snippets": len(snippets)})

            # Grade
            t = time.time()
            grade = self.grader.grade(query, snippets)
            yield _sse({"type": "grade", "agent": "grader",
                        "iteration": iteration,
                        "elapsed_ms": int((time.time() - t) * 1000),
                        "relevance_score": grade.relevance_score,
                        "decision": grade.decision,
                        "missing_aspects": grade.missing_aspects,
                        "reason": grade.reason})

            # Bug 2 fix: keep BEST iteration by score (not last)
            if grade.relevance_score > best_score:
                best_score = grade.relevance_score
                best_result = (rerank_docs, snippets, grade)

            # Bug 1 fix: exit early when score is high enough, regardless of
            # whether grader labeled 'sufficient' or 'partial'. A 75% score
            # means we have enough context to answer; pedantic 'missing'
            # aspects don't justify another retrieval round.
            if grade.decision == "sufficient" or grade.relevance_score >= accept_threshold:
                break

            if iteration == self.max_iterations:
                break

            next_query = grade.suggested_query or query
            yield _sse({"type": "iterate", "agent": "orchestrator",
                        "next_iteration": iteration + 1,
                        "next_query": next_query,
                        "reason": f"Điểm liên quan {int(grade.relevance_score*100)}% dưới ngưỡng {int(accept_threshold*100)}%, viết lại truy vấn và thử lại."})
            query = next_query

        rerank_docs, snippets, grade = best_result  # type: ignore[assignment]

        # Bug 3 fix: web fallback chỉ khi BEST score thật sự thấp.
        # Điểm trong khoảng [0.4, 0.6) vẫn sẽ generate bằng RAG (có thể có
        # warning "thông tin chưa đầy đủ" từ model) thay vì bỏ đi.
        if best_score < WEB_FALLBACK_THRESHOLD and web_search_enabled:
            yield _sse({"type": "iterate", "agent": "orchestrator",
                        "reason": f"Điểm liên quan tốt nhất chỉ {int(best_score*100)}%, CSDL thiếu nhiều thông tin quan trọng — chuyển sang Web Search."})
            fallback_plan = plan.model_copy(update={
                "tool": "web",
                "search_query": plan.search_query or plan.rewritten_query,
            })
            yield from self._run_web(fallback_plan, memory_context)
            return

        # Generate
        t = time.time()
        yield _sse({"type": "generate_start", "agent": "generator",
                    "mode": "rag",
                    "reason": "Tổng hợp câu trả lời từ các đoạn trích được chọn."})
        # Build hybrid input for generator: full doc content + BERT highlight.
        # BERT Extract trả về 1 span/doc tốt nhất theo nó — với câu hỏi có
        # nhiều ngưỡng/case (VD "mức phạt nồng độ cồn" có 3-4 ngưỡng trong
        # cùng 1 điều luật), BERT bỏ sót các span khác. Generator nhận full
        # doc sẽ thấy mọi mức phạt; BERT highlight vẫn giúp "focus attention"
        # vào đoạn quan trọng nhất.
        generator_input = self._build_generator_input(rerank_docs, snippets)
        answer, references, used_indices = self._generate_rag_answer(
            plan.rewritten_query, generator_input, rerank_docs, memory_context,
        )
        yield _sse({"type": "generate_done", "agent": "generator",
                    "elapsed_ms": int((time.time() - t) * 1000),
                    "n_citations": len(references)})

        if not references:
            # LLM returned key=[] — no citable source. If web is on, try web.
            if web_search_enabled:
                yield _sse({"type": "iterate", "agent": "orchestrator",
                            "reason": "LLM không chọn được văn bản trích dẫn, chuyển sang Web."})
                fallback_plan = plan.model_copy(update={
                    "tool": "web",
                    "search_query": plan.search_query or plan.rewritten_query,
                })
                yield from self._run_web(fallback_plan, memory_context)
                return
            # Otherwise: ship the message prompting the user to enable web.
            msg = (
                "Xin lỗi bạn, thông tin này không có trong cơ sở tri thức pháp luật "
                "của tôi. Nếu bạn muốn tôi tra cứu trên Internet, vui lòng bật chế "
                "độ **Web Search** ở thanh công cụ phía dưới rồi gửi lại câu hỏi."
            )
            yield _sse({"type": "final", "answer": msg, "references": []})
            return

        # Hallucination check — scope to the docs the generator actually
        # cited (used_indices), not the full top-5. This keeps the prompt
        # focused and reduces false positives from distractor docs the
        # generator never touched.
        t = time.time()
        if used_indices:
            source_texts = [
                rerank_docs[i][0].replace("_", " ")
                for i in used_indices if 0 <= i < len(rerank_docs)
            ]
        else:
            source_texts = [doc.replace("_", " ") for doc, _ in rerank_docs if doc]
        check = self.checker.check(answer, source_texts)
        yield _sse({"type": "check", "agent": "fact_checker",
                    "elapsed_ms": int((time.time() - t) * 1000),
                    "is_grounded": check.is_grounded,
                    "confidence": check.confidence,
                    "unsupported_claims": check.unsupported_claims,
                    "reason": check.reason})
        # Regenerate only when there's strong evidence of hallucination:
        #   - confidence ≥ 0.75 (checker really sure, not "maybe")
        #   - ≥ 2 unsupported claims (1 vague claim is noise, not a fix worth 10s)
        # When we do regenerate, we feed the unsupported_claims back to the
        # generator so the retry actually addresses the gap instead of
        # producing near-identical output.
        if (
            (not check.is_grounded)
            and check.confidence >= 0.75
            and len(check.unsupported_claims) >= 2
        ):
            yield _sse({"type": "regenerate", "agent": "generator",
                        "reason": f"Phát hiện {len(check.unsupported_claims)} claim không có trong văn bản, tổng hợp lại với hướng dẫn cụ thể."})
            t = time.time()
            # Regenerate cũng dùng full-doc input, không dùng BERT snippet đơn lẻ
            regen_input = self._build_generator_input(rerank_docs, snippets)
            answer, references, _ = self._generate_rag_answer(
                plan.rewritten_query, regen_input, rerank_docs, memory_context,
                feedback_claims=check.unsupported_claims,
            )
            yield _sse({"type": "generate_done", "agent": "generator",
                        "elapsed_ms": int((time.time() - t) * 1000),
                        "regenerated": True, "n_citations": len(references)})

        yield _sse({"type": "final", "answer": answer, "references": references})

    # ---------- helpers ----------

    def _retrieve_and_rerank(self, query: str, entities: Dict[str, str]):
        """Multi-query expansion + Qdrant search + RRF + CrossEncoder rerank.

        Unlike the legacy Qdrant_Utils.search_With_Similarity_Queries, we
        skip per-variant entity extraction — planner already gave us the
        filter once, which we apply to every variant.
        """
        # Generate 3 variants (+ the original rewritten query)
        queries = self.generator.generate_query(query)
        metadata_filter = self.qdrant_utils.build_metadata_filter(entities) if entities else None

        query_results = []
        for q in queries:
            tokenized = ViTokenizer.tokenize(q)
            query_results.append(
                self.qdrant_utils.search_documents(tokenized, filter=metadata_filter)
            )

        rrf = self.rerank_utils.reciprocal_rank_fusion(query_results)
        if not rrf:
            return []
        reranked = self.rerank_utils.rerank_documents_finetune(query, rrf)
        return reranked

    def _generate_rag_answer(
        self,
        query: str,
        snippets: Dict[int, str],
        rerank_docs,
        memory_context: str,
        feedback_claims: Optional[List[str]] = None,
    ):
        """Call generate_response, parse JSON, build reference blocks.

        `feedback_claims`: nếu set (trong regeneration path), chèn danh sách
        claim bị fact-checker flag vào query để LLM biết CHÍNH XÁC phải cải
        thiện gì. Nếu không truyền → gọi bình thường như lần đầu.

        Resilient to malformed LLM output: if parse fails, degrade to
        returning the raw text with no citations rather than crashing the
        whole stream.
        """
        effective_query = query
        if feedback_claims:
            feedback_block = (
                "\n\n[PHẢN HỒI TỪ FACT CHECKER — HÃY SỬA KHI VIẾT LẠI]\n"
                "Câu trả lời trước thiếu các chi tiết cụ thể sau (các claim này "
                "CÓ trong văn bản luật nhưng answer bỏ sót hoặc mơ hồ):\n"
                + "\n".join(f"- {c}" for c in feedback_claims)
                + "\n\nHƯỚNG DẪN:\n"
                "- Giữ nguyên SỐ LIỆU/MỨC PHẠT/ĐIỀU KHOẢN CỤ THỂ từ văn bản, "
                "KHÔNG paraphrase thành 'nghiêm khắc', 'mức cao hơn', 'đáng kể'.\n"
                "- Trả lời ĐẦY ĐỦ mọi ngưỡng/trường hợp có trong nguồn, không "
                "bỏ sót case nào.\n"
                "- KHÔNG bịa thông tin ngoài nguồn."
            )
            effective_query = query + feedback_block

        result = self.generator.generate_response(
            effective_query, snippets, memory_context=memory_context,
        )
        try:
            parsed = parse_raw_json(clean_code_fence_safe(result))
        except Exception as e:
            print(f"[_generate_rag_answer] parse error: {type(e).__name__}: {e}", flush=True)
            # Degrade: strip any visible JSON scaffolding so the user still
            # sees readable text.
            answer = (result or "").strip()
            if answer.startswith("{") and answer.endswith("}"):
                # Best-effort: drop braces and quote noise
                answer = re.sub(r'^\s*\{\s*"answer"\s*:\s*"?', '', answer)
                answer = re.sub(r'"?\s*,\s*"key".*$', '', answer, flags=re.DOTALL)
            return answer or "Xin lỗi, trợ lý không tạo được câu trả lời hợp lệ.", [], []
        answer = parsed.get("answer", "") or ""
        selected_keys = parsed.get("key", []) or []

        references: List[str] = []
        if selected_keys:
            # Filter invalid indices defensively.
            selected_documents = [
                rerank_docs[i] for i in selected_keys if 0 <= i < len(rerank_docs)
            ]
            references = [
                (
                    f"""\
                        Tài liệu tham khảo: {i+1}
                        Loại văn bản: {infor['doc_metadata'].get("LoaiVanBan", "")}
                        Nơi ban hành: {infor['doc_metadata'].get("NoiBanHanh", "")}
                        Số hiệu: {infor['doc_metadata'].get("SoHieu", "")}
                        Lĩnh vực - ngành: {infor['doc_metadata'].get("LinhVucNganh", "")}
                        Ngày ban hành: {infor['doc_metadata'].get("NgayBanHanh", "")}
                        Chủ đề: {infor['doc_metadata'].get("ChuDe", "")}
                        Chương: {infor['doc_metadata'].get("Chapter", "")}
                        Mục: {infor['doc_metadata'].get("Section", "")}
                        Tiểu mục: {infor['doc_metadata'].get("MiniSection", "")}
                        {infor['doc_metadata'].get("Article", "")}
                        <=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=>
                        {doc.replace("_", " ").replace(' .', '.').replace(' ,', ',').replace(' !', '!').replace(' ?', '?').replace(' :', ':').replace(' ;', ';')}
                        """
                )
                for i, (doc, infor) in enumerate(selected_documents)
            ]
        return answer, references, selected_keys

    def _ensure_intro_loaded(self):
        if self._intro_corpus is None:
            self._intro_corpus, self._intro_embeddings = load_information_from_json(
                self.settings, self.embedding,
            )

    @staticmethod
    def _build_generator_input(rerank_docs, snippets):
        """Gộp full doc + BERT highlight vào input cho Generator.

        BERT Extract trả về 1 span/doc — với câu hỏi multi-part (VD "mức phạt
        nồng độ cồn" có 3-4 ngưỡng trong cùng 1 điều luật), BERT chỉ bắt
        được 1 ngưỡng, các ngưỡng khác bị bỏ sót → Generator viết câu trả
        lời thiếu.

        Fix: đưa toàn bộ nội dung điều luật vào context (Claude Sonnet 4.5
        có 200k token context, dư sức), dùng BERT highlight làm gợi ý focus.
        """
        result: Dict[int, str] = {}
        snippets = snippets if isinstance(snippets, dict) else {}
        for idx, (doc, _) in enumerate(rerank_docs):
            full = doc.replace("_", " ").strip()
            hint = (snippets.get(idx) or "").strip()
            if hint and hint != "Không có câu trả lời":
                result[idx] = (
                    f"{full}\n"
                    f"[BERT highlight (đoạn BERT QA cho rằng relevant nhất, "
                    f"nhưng KHÔNG BỎ QUA nội dung khác trong điều luật): {hint}]"
                )
            else:
                result[idx] = full
        return result

    @staticmethod
    def _format_doc_label(infor) -> str:
        """Human-readable identifier for a retrieved legal document.

        Composed from metadata the UI can show as a chip: "Luật 52/2014/QH13
        — Điều 8 ...". Falls back to ChuDe, then to a short preview so the
        user always sees something meaningful, never an empty chip.
        """
        md = infor.get('doc_metadata', {}) if isinstance(infor, dict) else {}
        loai = (md.get('LoaiVanBan') or '').strip()
        sohieu = (md.get('SoHieu') or '').strip()
        article = (md.get('Article') or '').strip()
        chude = (md.get('ChuDe') or '').strip()

        head = ''
        if loai and sohieu:
            head = f"{loai} {sohieu}"
        elif loai:
            head = loai
        elif sohieu:
            head = sohieu

        tail = ''
        if article:
            tail = article.split('\n')[0][:80]
        elif chude:
            tail = chude[:80]

        if head and tail:
            return f"{head} — {tail}"
        return head or tail or "(văn bản)"
