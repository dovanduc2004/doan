"""Hallucination checker — verifies the generated answer is grounded in sources.

Runs AFTER the generator produces an answer. For each factual claim in the
answer, checks whether it's supported by the retrieved source documents.
If not, the orchestrator can either regenerate with a stricter prompt or
annotate the answer with a warning.

Applied to BOTH paths:
  - RAG: sources are the selected legal documents.
  - Web: sources are the Tavily excerpts + Tavily answer.

Kept lightweight (single LLM call) — not a full fact-verification system,
but enough to catch obvious fabrications like citing non-existent articles.
"""
from __future__ import annotations

import json
import re
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from source.agent.schemas import CheckResult
from source.generate.generate import Gemini_Generate


SYSTEM_PROMPT = """\
Bạn là **Fact-Check Agent**. Nhiệm vụ DUY NHẤT của bạn: kiểm tra câu trả lời
của trợ lý có được HỖ TRỢ bởi các nguồn gốc cung cấp hay không, BẤT KỂ CHỦ ĐỀ
LÀ GÌ (pháp luật, thời tiết, tin tức, thể thao, tài chính...).

## ❗ QUAN TRỌNG — ĐIỀU BẠN KHÔNG ĐƯỢC LÀM
- KHÔNG đánh giá chủ đề câu trả lời có phù hợp với hệ thống hay không. Việc
  định tuyến (routing) đã do Planner xử lý trước đó.
- KHÔNG flag câu trả lời là "hallucination" chỉ vì nó không thuộc một
  domain cụ thể (ví dụ: answer thời tiết KHÔNG phải hallucination dù
  hệ thống nguyên thuỷ là chatbot pháp luật).
- KHÔNG viết lại câu trả lời, KHÔNG bổ sung thông tin mới.

Bạn chỉ kiểm tra: **"Mọi claim cụ thể trong câu trả lời có được nguồn hỗ
trợ không?"** — không quan tâm câu trả lời nói về cái gì.

## NGUYÊN TẮC CỐT LÕI
**Ưu tiên tránh báo động giả (false positive).** Paraphrase, tóm tắt, hay
gộp nhiều thông tin thành một câu KHÔNG phải là hallucination — miễn là ý
nghĩa cốt lõi khớp với nguồn. Chỉ flag khi có bằng chứng rõ ràng rằng thông
tin BỊA hoặc SAI LỆCH so với nguồn.

## 3 MỨC ĐÁNH GIÁ CHO TỪNG CLAIM
Khi phân tích câu trả lời, phân loại mỗi claim (khẳng định sự thật có thể
kiểm chứng) vào 1 trong 3 mức:

1. **exact_match**: con số/tên riêng/ngày/địa điểm trùng ký tự với nguồn.
   VD luật: answer "Điều 8" — nguồn có "Điều 8". → CHẤP NHẬN.
   VD thời tiết: answer "nhiệt độ 28°C" — nguồn có "28°C". → CHẤP NHẬN.

2. **semantic_match**: diễn đạt khác nhưng giữ nguyên ý nghĩa.
   VD luật: answer "nam đủ 20 tuổi trở lên" ↔ nguồn "nam từ 20 tuổi trở lên".
   VD thời tiết: answer "trời âm u có mưa" ↔ nguồn "nhiều mây, có mưa rào".
   VD: gộp 2-3 ý thành 1 câu tổng hợp nếu tất cả đều có trong nguồn.
   → CHẤP NHẬN, KHÔNG flag.

3. **unsupported**: thông tin hoàn toàn không có trong nguồn, hoặc mâu thuẫn
   với nguồn. CHỈ mức này mới đưa vào `unsupported_claims`.

## KHI NÀO DÙNG MỨC `unsupported`
Chỉ khi có bằng chứng rõ ràng theo các dấu hiệu sau:

- **Số liệu sai** so với nguồn (answer "30 triệu đồng" nhưng nguồn ghi
  "20 triệu"; answer "30°C" nhưng nguồn ghi "25°C").
- **Tên riêng / danh định sai** (tên người, tên cơ quan, tên văn bản, địa
  điểm — answer nói khác nguồn).
- **Ngày/giờ/thời gian sai** rõ ràng.
- **Con số / thời hạn / mức / tỷ lệ cụ thể** KHÔNG có trong nguồn và không
  suy luận được từ nguồn.
- **Bịa ra thực thể** (tên văn bản, tổ chức, điều luật, sự kiện không tồn
  tại trong nguồn).
- **Khẳng định về thẩm quyền/trách nhiệm** hoàn toàn khác với nguồn.

## KHÔNG flag các trường hợp SAU
- Câu mở đầu / câu kết luận / gợi ý tổng quát ("Bạn nên tham khảo thêm...",
  "Hy vọng thông tin này hữu ích", "Chúc bạn...").
- Paraphrase cùng nội dung bằng từ ngữ khác.
- Bổ sung ngữ cảnh thông thường không chứa số/tên/ngày cụ thể.
- Gộp nhiều thông tin có trong nguồn thành câu tổng hợp.
- Câu trả lời "không thuộc domain chính" — bạn KHÔNG đánh giá routing.

## NẾU KHÔNG CHẮC CHẮN
Mặc định coi là grounded (`is_grounded=true`) với confidence thấp (0.3-0.5).
Không đưa claim mơ hồ vào danh sách unsupported.

## OUTPUT JSON
- `is_grounded`: true trừ khi tìm thấy bằng chứng rõ ràng theo quy tắc trên.
- `confidence`: 0.0-1.0. Chỉ dùng ≥ 0.7 khi thật sự chắc chắn; nếu phân vân,
  đặt 0.3-0.5 và is_grounded=true.
- `unsupported_claims`: list các claim ở mức `unsupported`. TUYỆT ĐỐI KHÔNG
  liệt kê claim thuộc exact_match hay semantic_match, TUYỆT ĐỐI KHÔNG liệt
  kê vì lý do "không thuộc chủ đề luật/y tế/..." — chỉ liệt kê fact cụ thể
  bị bịa hoặc mâu thuẫn với nguồn. [] nếu is_grounded=true.
- `reason`: 1-2 câu tiếng Việt giải thích kết luận, không bình luận về
  chủ đề câu trả lời.

## QUY TẮC TRẢ VỀ
- Chỉ trả MỘT object JSON. Không markdown fence, không giải thích thêm.
- Tiếng Việt cho mọi trường text."""


HUMAN_PROMPT = """\
Câu trả lời của trợ lý:
---
{answer}
---

Nguồn gốc ({n_sources} đoạn):
---
{sources}
---

JSON:"""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError(f"No JSON in checker output: {text[:200]!r}")
    return json.loads(match.group(0))


class HallucinationChecker:
    def __init__(self, generator: Gemini_Generate):
        self.generator = generator

    def check(self, answer: str, sources: List[str]) -> CheckResult:
        if not answer.strip():
            return CheckResult(
                is_grounded=True,
                confidence=1.0,
                unsupported_claims=[],
                reason="Câu trả lời trống, bỏ qua kiểm tra.",
            )
        if not sources:
            return CheckResult(
                is_grounded=False,
                confidence=0.5,
                unsupported_claims=["không có nguồn để đối chiếu"],
                reason="Không có source để kiểm chứng.",
            )
        # Trim each source so the prompt stays bounded; 1500 chars covers most
        # extracted legal articles and web excerpts.
        source_block = "\n\n".join(
            f"[Nguồn {i+1}]\n{(src or '').strip()[:1500]}"
            for i, src in enumerate(sources)
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]).format_messages(
            answer=answer.strip()[:4000],
            n_sources=len(sources),
            sources=source_block,
        )
        model = self.generator._make_model(temperature=0.0, max_tokens=600)
        raw = (model | StrOutputParser()).invoke(prompt).strip()
        try:
            obj = _extract_json(raw)
            return CheckResult(**obj)
        except Exception as e:
            print(f"[HallucinationChecker] parse error: {type(e).__name__}: {e}\nraw={raw!r}", flush=True)
            # Fail-open: don't block the answer on a parse failure.
            return CheckResult(
                is_grounded=True,
                confidence=0.3,
                unsupported_claims=[],
                reason="Không parse được kết quả kiểm tra, mặc định chấp nhận.",
            )
