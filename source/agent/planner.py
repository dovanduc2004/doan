"""Planner agent — single LLM call that decides the whole turn.

Replaces the previous per-turn chain of:
    classify_query + contextualize_query + rewrite_for_web_search
    + 4x extract_entities (one per generated variant).

The planner looks at the user message, memory, and the web_search toggle,
then produces a structured Plan (see schemas.py). The orchestrator uses
that Plan to route to RAG / web / template / clarification without any
further LLM call for routing.
"""
from __future__ import annotations

import json
import re
from datetime import date
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from source.agent.schemas import Plan
from source.generate.generate import Gemini_Generate


SYSTEM_PROMPT = """\
Bạn là **Planner Agent** của một hệ thống Agentic RAG chuyên pháp luật Việt Nam.
Nhiệm vụ của bạn: đọc tin nhắn người dùng + ngữ cảnh hội thoại, rồi trả về MỘT JSON duy nhất
mô tả cách xử lý turn này. Bạn KHÔNG trả lời câu hỏi — bạn chỉ định tuyến.

## 4 INTENT:
- `chitchat`: chào hỏi, hỏi về bot/khả năng của bot, cảm ơn, hội thoại xã giao.
- `legal`: câu hỏi pháp luật Việt Nam, có thể trả lời từ CSDL văn bản luật.
- `realtime`: câu hỏi phụ thuộc thời gian thực (thời tiết, tin tức, thể thao,
  giá cả/tỷ giá, lịch thi đấu, sự kiện mới...). Không có trong CSDL luật.
- `out_of_scope`: câu hỏi ngoài luật VN và không thuộc realtime (kiến thức khác,
  ngôn ngữ tục tĩu, câu vô nghĩa).

## 5 TOOL:
- `template`: trả lời bằng template cứng (dùng cho `chitchat`).
- `rag`: chạy pipeline RAG nội bộ (Qdrant + rerank + BERT + generate). Dùng cho `legal`.
- `web`: tìm trên Tavily rồi tổng hợp. Dùng cho `realtime`/`out_of_scope` khi
  người dùng đã bật Web Search.
- `reject`: trả lời lịch sự rằng câu hỏi ngoài phạm vi. Dùng cho `realtime`/
  `out_of_scope` khi Web Search tắt.
- `ask_user`: hỏi lại người dùng để có đủ thông tin. Dùng khi `realtime` nhưng
  thiếu yếu tố critical (vị trí, thời điểm, đối tượng cụ thể...).

## QUY TẮC ĐỊNH TUYẾN (tool ↔ intent × web_search_enabled):
| Intent        | Web OFF                 | Web ON                           |
|---------------|-------------------------|----------------------------------|
| chitchat      | template                | template                         |
| legal         | rag                     | rag                              |
| realtime      | reject                  | web (hoặc ask_user nếu thiếu info) |
| out_of_scope  | reject                  | web                              |

## CÁC FIELD JSON:
- `intent`: một trong 4 intent trên.
- `tool`: một trong 5 tool trên, THEO BẢNG ĐỊNH TUYẾN.
- `rewritten_query`: câu hỏi đầy đủ, tự chứa, đã giải tham chiếu hội thoại
  (VD "còn người nước ngoài thì sao" → "quy định thử việc đối với người lao
  động nước ngoài tại Việt Nam"). Nếu đã tự đủ, giữ nguyên.
- `search_query`: CHỈ điền khi tool = "web". Keyword-heavy, ngắn gọn, có mốc
  thời gian nếu cần (hôm nay: {today}). Các tool khác để null.
- `entities`: CHỈ điền khi tool = "rag" VÀ câu hỏi nhắc văn bản cụ thể. Trích
  từ `rewritten_query`:
    * `LoaiVanBanFilter`: một trong {{Luật, Nghị định, Quyết định, Thông tư, Văn bản hợp nhất}}
    * `NoibanHanhFilter`: một trong {{Bộ Công An, Bộ Giao thông vận tải,
      Bộ Giáo dục và Đào tạo, Bộ Lao động - Thương binh và Xã hội, Bộ Nội vụ,
      Bộ Quốc phòng, Bộ Tài chính, Bộ Tài nguyên và Môi trường, Bộ Tư pháp,
      Bộ Y tế, Chính phủ, Quốc hội, Thủ tướng Chính phủ}}
    * `NgayBanHanhFilter`: 'dd-mm-yyyy' hoặc 'yyyy'
  Nếu không trích được field nào → để null (KHÔNG để `{{}}`).
- `needs_clarification`: true khi tool = "ask_user". Chỉ set true cho `realtime`
  thiếu info critical — KHÔNG dùng cho câu hỏi legal mơ hồ (legal cứ chạy RAG).
- `clarifying_question`: câu hỏi lại người dùng, tiếng Việt lịch sự. Null nếu
  không cần.
- `reason`: 1-2 câu tiếng Việt giải thích quyết định. Sẽ HIỂN THỊ cho người dùng
  trong panel suy luận, nên viết dễ hiểu, không lộ prompt/tool name nội bộ.

## VÍ DỤ:

### Ví dụ 1 — chitchat
Input: "hello"
Context: (trống)
Output:
{{"intent":"chitchat","tool":"template","rewritten_query":"hello",
"search_query":null,"entities":null,"needs_clarification":false,
"clarifying_question":null,"reason":"Tin nhắn chào hỏi, trả lời bằng mẫu giới thiệu."}}

### Ví dụ 2 — legal không có filter
Input: "điều kiện kết hôn theo pháp luật Việt Nam là gì"
web_search_enabled: false
Output:
{{"intent":"legal","tool":"rag","rewritten_query":"Điều kiện kết hôn theo pháp luật Việt Nam hiện hành",
"search_query":null,"entities":null,"needs_clarification":false,
"clarifying_question":null,"reason":"Câu hỏi pháp luật hôn nhân gia đình, tra CSDL văn bản luật."}}

### Ví dụ 3 — legal có filter
Input: "Thông tư 57/2015/TT-BYT quy định điều kiện sức khỏe người làm IVF như thế nào"
web_search_enabled: false
Output:
{{"intent":"legal","tool":"rag",
"rewritten_query":"Thông tư 57/2015/TT-BYT của Bộ Y tế quy định điều kiện sức khỏe người thực hiện kỹ thuật thụ tinh trong ống nghiệm",
"search_query":null,
"entities":{{"LoaiVanBanFilter":"Thông tư","NoibanHanhFilter":"Bộ Y tế","NgayBanHanhFilter":"2015"}},
"needs_clarification":false,"clarifying_question":null,
"reason":"Hỏi cụ thể một văn bản, lọc metadata để thu hẹp truy vấn Qdrant."}}

### Ví dụ 4 — realtime, web OFF
Input: "thời tiết hôm nay thế nào"
web_search_enabled: false
Output:
{{"intent":"realtime","tool":"reject","rewritten_query":"thời tiết hôm nay",
"search_query":null,"entities":null,"needs_clarification":false,
"clarifying_question":null,
"reason":"Câu hỏi thời tiết cần Internet. Web Search đang tắt nên không tra được."}}

### Ví dụ 5 — realtime, web ON, thiếu vị trí
Input: "thời tiết hôm nay thế nào"
Context: (trống)
web_search_enabled: true
Output:
{{"intent":"realtime","tool":"ask_user","rewritten_query":"thời tiết hôm nay",
"search_query":null,"entities":null,"needs_clarification":true,
"clarifying_question":"Bạn đang ở thành phố/khu vực nào để mình tra thời tiết cho chính xác?",
"reason":"Chưa có vị trí trong ngữ cảnh, cần hỏi lại trước khi search."}}

### Ví dụ 6 — realtime, web ON, đã có vị trí từ context
Input: "hà nội việt nam"
Context: Trợ lý: "Bạn đang ở thành phố/khu vực nào để mình tra thời tiết cho chính xác?"
web_search_enabled: true
Output:
{{"intent":"realtime","tool":"web",
"rewritten_query":"Thời tiết hôm nay tại Hà Nội, Việt Nam",
"search_query":"thời tiết Hà Nội hôm nay {today}",
"entities":null,"needs_clarification":false,"clarifying_question":null,
"reason":"Đã có vị trí Hà Nội từ lượt trước, tra web với truy vấn kèm ngày."}}

### Ví dụ 7 — out_of_scope, web OFF
Input: "1 kg ổi bao nhiêu tiền"
web_search_enabled: false
Output:
{{"intent":"out_of_scope","tool":"reject","rewritten_query":"giá 1 kg ổi",
"search_query":null,"entities":null,"needs_clarification":false,
"clarifying_question":null,
"reason":"Ngoài phạm vi pháp luật VN và Web Search đang tắt."}}

### Ví dụ 8 — out_of_scope, web ON
Input: "kết quả trận U23 Việt Nam tối qua"
web_search_enabled: true
Output:
{{"intent":"out_of_scope","tool":"web",
"rewritten_query":"Kết quả trận đấu bóng đá của đội tuyển U23 Việt Nam tối qua",
"search_query":"kết quả U23 Việt Nam {today}",
"entities":null,"needs_clarification":false,"clarifying_question":null,
"reason":"Chủ đề thể thao, tra Tavily để lấy kết quả mới nhất."}}

## QUY TẮC TRẢ VỀ:
- Chỉ trả MỘT object JSON. Không markdown fence, không giải thích thêm.
- Tiếng Việt cho mọi trường text."""


HUMAN_PROMPT = """\
web_search_enabled: {web_search_enabled}
Ngữ cảnh hội thoại:
{memory_context}

Tin nhắn người dùng: {user_message}

JSON:"""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    """Strip code fences / prose and parse the first JSON object we find."""
    text = text.strip()
    # Remove a leading/trailing ``` fence if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError(f"No JSON object in planner output: {text[:200]!r}")
    return json.loads(match.group(0))


class PlannerAgent:
    def __init__(self, generator: Gemini_Generate):
        self.generator = generator

    def plan(
        self,
        user_message: str,
        memory_context: str = "",
        web_search_enabled: bool = False,
    ) -> Plan:
        today = date.today().isoformat()
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT.replace("{today}", today)),
            ("human", HUMAN_PROMPT),
        ]).format_messages(
            web_search_enabled=str(web_search_enabled).lower(),
            memory_context=memory_context or "(trống)",
            user_message=user_message,
        )
        model = self.generator._make_model(temperature=0.0, max_tokens=800)
        raw = (model | StrOutputParser()).invoke(prompt).strip()
        try:
            obj = _extract_json(raw)
            # LLM thỉnh thoảng trả entities = {"X": null, "Y": null} thay vì
            # omit key. Pydantic schema Dict[str, str] reject null → strip
            # None/empty values để giữ nguyên các entity có thật.
            ents = obj.get("entities")
            if isinstance(ents, dict):
                cleaned = {k: v for k, v in ents.items() if v is not None and str(v).strip()}
                obj["entities"] = cleaned if cleaned else None
            plan = Plan(**obj)
        except Exception as e:
            # Fallback — degrade gracefully instead of 500ing the whole turn.
            print(f"[PlannerAgent] parse error: {type(e).__name__}: {e}\nraw={raw!r}", flush=True)
            plan = self._fallback_plan(user_message, web_search_enabled)
        plan = self._enforce_routing(plan, web_search_enabled)
        return plan

    @staticmethod
    def _fallback_plan(user_message: str, web_search_enabled: bool) -> Plan:
        """Conservative default: treat as legal and let RAG decide.

        This preserves the old behaviour when the planner call fails.
        """
        return Plan(
            intent="legal",
            tool="rag",
            rewritten_query=user_message,
            search_query=None,
            entities=None,
            needs_clarification=False,
            clarifying_question=None,
            reason="Planner không parse được, mặc định chạy RAG nội bộ.",
        )

    @staticmethod
    def _enforce_routing(plan: Plan, web_search_enabled: bool) -> Plan:
        """Defensive guard: the model occasionally picks a tool that violates
        the toggle. Clamp it to the allowed set so the orchestrator doesn't
        have to handle every bad combination."""
        if plan.intent == "chitchat":
            plan.tool = "template"
        elif plan.intent == "legal":
            plan.tool = "rag"
        elif plan.intent in ("realtime", "out_of_scope"):
            if not web_search_enabled:
                plan.tool = "reject"
                plan.needs_clarification = False
                plan.clarifying_question = None
            else:
                if plan.tool not in ("web", "ask_user"):
                    plan.tool = "web"
        # Normalise empty entity dict → None
        if plan.entities is not None and not plan.entities:
            plan.entities = None
        return plan
