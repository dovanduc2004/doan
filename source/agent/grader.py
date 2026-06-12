"""Grader agent — evaluates whether retrieved docs can answer the query.

Inspired by CRAG (Yan et al. 2024, arXiv 2401.15884): a lightweight
retrieval evaluator that tells the orchestrator whether to proceed with
generation, rewrite the query and retry, or fall back to web search.

We call this AFTER rerank + BERT extract, not after raw vector search —
because the extracted snippets are what the generator will actually see.
Grading on the raw top-50 would be wasteful and noisier.
"""
from __future__ import annotations

import json
import re
from typing import List, Tuple, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from source.agent.schemas import GraderResult
from source.generate.generate import Gemini_Generate


SYSTEM_PROMPT = """\
Bạn là **Grader Agent** trong pipeline Agentic RAG pháp luật Việt Nam.
Nhiệm vụ: đánh giá KHÁCH QUAN xem các đoạn trích (snippet) rút ra từ văn bản
luật có đủ để trả lời câu hỏi người dùng hay không.

BẠN KHÔNG trả lời câu hỏi. Bạn chỉ chấm điểm và gợi ý chiến lược tiếp theo.

## CÁC FIELD JSON:
- `relevance_score`: float 0.0-1.0. Xấp xỉ tỉ lệ thông tin trong snippet liên
  quan trực tiếp tới câu hỏi. 1.0 = cover đầy đủ; 0.5 = có liên quan nhưng
  thiếu chi tiết; 0.0 = lạc đề hoàn toàn.
- `decision`: một trong 3 giá trị:
    * `sufficient`: đủ thông tin để sinh câu trả lời (relevance_score ≥ 0.6
      VÀ không thiếu khía cạnh critical).
    * `partial`: có một phần, nhưng vẫn thiếu khía cạnh quan trọng mà câu hỏi
      yêu cầu. Nên rewrite query và retry.
    * `insufficient`: snippet không liên quan hoặc quá ít. Rewrite hoặc chuyển
      sang web search.
- `missing_aspects`: list các khía cạnh cụ thể còn thiếu. VD ["thời gian cụ thể",
  "điều khoản chính xác", "trường hợp ngoại lệ"]. Để [] nếu `sufficient`.
- `suggested_query`: khi `decision != sufficient`, viết lại truy vấn theo hướng
  bù đắp `missing_aspects`. Giữ ngắn gọn, tiếng Việt, keyword-heavy. Null nếu
  `sufficient`.
- `reason`: 1-2 câu tiếng Việt giải thích, sẽ hiển thị cho người dùng.

## VÍ DỤ:

### Ví dụ 1 — sufficient
Câu hỏi: "Điều kiện kết hôn theo pháp luật Việt Nam là gì"
Snippets:
[0] "nam đủ 20 tuổi trở lên, nữ đủ 18 tuổi trở lên, tự nguyện..."
[1] "không thuộc các trường hợp cấm kết hôn theo Điều 5..."
Output:
{{"relevance_score":0.9,"decision":"sufficient","missing_aspects":[],
"suggested_query":null,"reason":"Các snippet nêu đủ điều kiện tuổi, tự nguyện và các trường hợp cấm."}}

### Ví dụ 2 — partial
Câu hỏi: "Thời gian thử việc tối đa là bao lâu"
Snippets:
[0] "hợp đồng lao động gồm các loại..."
[1] "người sử dụng lao động có thể thỏa thuận về thử việc..."
Output:
{{"relevance_score":0.4,"decision":"partial",
"missing_aspects":["thời gian thử việc cụ thể theo loại hợp đồng","điều khoản chính xác"],
"suggested_query":"thời gian thử việc tối đa Bộ luật Lao động 2019 Điều 25",
"reason":"Đúng chủ đề lao động nhưng chưa nêu giới hạn thời gian cụ thể."}}

### Ví dụ 3 — insufficient
Câu hỏi: "Thủ tục đăng ký kết hôn với người nước ngoài"
Snippets:
[0] "Luật Doanh nghiệp quy định..."
[1] "Nghị định về đầu tư nước ngoài..."
Output:
{{"relevance_score":0.05,"decision":"insufficient",
"missing_aspects":["quy định về đăng ký kết hôn","yếu tố nước ngoài trong hôn nhân"],
"suggested_query":"thủ tục đăng ký kết hôn có yếu tố nước ngoài Luật Hôn nhân và Gia đình",
"reason":"Snippet lạc sang luật doanh nghiệp và đầu tư, không liên quan hôn nhân gia đình."}}

## QUY TẮC:
- Chỉ trả MỘT object JSON. Không giải thích thêm.
- Tiếng Việt cho mọi trường text."""


HUMAN_PROMPT = """\
Câu hỏi: {query}

Các đoạn trích ({n_snippets} snippet, mỗi snippet là kết quả BERT QA từ một văn bản luật):
{snippets}

JSON:"""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError(f"No JSON in grader output: {text[:200]!r}")
    return json.loads(match.group(0))


class GraderAgent:
    def __init__(self, generator: Gemini_Generate, threshold: float = 0.6):
        self.generator = generator
        self.threshold = threshold

    def grade(self, query: str, snippets: Dict[int, str]) -> GraderResult:
        """Grade a set of extracted snippets against the query.

        `snippets` is the dict returned by Extract_Information.predict —
        {doc_index: extracted_answer_span}. Empty or placeholder spans
        ("Không có câu trả lời") are kept so the LLM can see we truly
        retrieved nothing useful.
        """
        if not snippets:
            return GraderResult(
                relevance_score=0.0,
                decision="insufficient",
                missing_aspects=["không retrieve được văn bản nào"],
                suggested_query=query,
                reason="Không có snippet nào để đánh giá.",
            )
        lines = [f"[{i}] {(text or '(trống)').strip()[:600]}" for i, text in snippets.items()]
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]).format_messages(
            query=query,
            n_snippets=len(snippets),
            snippets="\n".join(lines),
        )
        model = self.generator._make_model(temperature=0.0, max_tokens=500)
        raw = (model | StrOutputParser()).invoke(prompt).strip()
        try:
            obj = _extract_json(raw)
            result = GraderResult(**obj)
        except Exception as e:
            print(f"[GraderAgent] parse error: {type(e).__name__}: {e}\nraw={raw!r}", flush=True)
            # Conservative default: assume sufficient so we don't trap in a loop.
            result = GraderResult(
                relevance_score=0.6,
                decision="sufficient",
                missing_aspects=[],
                suggested_query=None,
                reason="Không parse được đánh giá, mặc định chấp nhận để tiếp tục.",
            )
        # Normalise: enforce consistency between score and decision.
        if result.decision == "sufficient" and result.relevance_score < self.threshold:
            result.decision = "partial"
        return result
