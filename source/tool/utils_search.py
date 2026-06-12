from source.extract.utils_extract import Extract_Information
from source.core.config import Settings
from source.generate.generate import Gemini_Generate


class Utils_Search_Tools:
    """Web-search synthesizer.

    After the agentic refactor this class keeps only the `_synthesize`
    helper — Tavily result → polished Vietnamese answer. The orchestrator
    calls the Tavily backend directly for the retrieval step so it can
    emit SSE events for each stage, then hands the materials to
    `_synthesize` for the final LLM call.

    The old `Search_Docs_From_Tools` (a blocking retrieve+synth combo)
    was removed; its logic is inlined in Orchestrator._run_web with
    per-step telemetry.
    """

    def __init__(self, setting: Settings, gemini_func: Gemini_Generate,
                 extract_func: Extract_Information, search_backend):
        self.setting = setting
        self.gemini_func = gemini_func
        self.extract_func = extract_func
        self.search_backend = search_backend

    def _synthesize(self, query, tavily_answer, excerpts, lst_links):
        """Ask Claude to polish the Tavily summary into a Vietnamese response.

        Falls back to the raw Tavily answer if the LLM call fails.
        """
        try:
            from datetime import date
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate

            today = date.today().isoformat()
            system_msg = (
                "Bạn là trợ lý thông tin đa lĩnh vực. Tổng hợp câu trả lời tiếng Việt "
                "ngắn gọn, rõ ràng từ HAI loại nguồn dưới đây:\n"
                "1) **Tóm tắt Tavily** — bản tổng hợp do công cụ tìm kiếm tạo trực tiếp "
                "từ các trang web hàng đầu. Đây là NGUỒN ĐÁNG TIN CẬY cho dữ liệu thời gian "
                "thực (thời tiết, tin tức, tỷ giá, giá cả...). ĐƯỢC PHÉP trích dẫn số liệu "
                "từ đây ngay cả khi không thấy trong trích đoạn web.\n"
                "2) **Trích đoạn từ trang web** — dùng để bổ sung chi tiết và ghi "
                "(Nguồn N) ở cuối các ý chính.\n\n"
                "QUY TẮC:\n"
                "- Ưu tiên Tavily cho con số/sự kiện thời gian thực; ưu tiên trích đoạn "
                "cho giải thích/ngữ cảnh.\n"
                "- Chỉ trả lời \"không đủ dữ liệu\" khi CẢ HAI nguồn đều không đề cập.\n"
                "- Không bịa thông tin ngoài hai nguồn này.\n"
                "- Nếu nội dung trái chiều giữa các nguồn, nêu cả hai."
            )
            human_msg = (
                f"Hôm nay là: {today}.\n\n"
                "Câu hỏi của người dùng:\n{query}\n\n"
                "Tóm tắt Tavily (nguồn đáng tin):\n{tavily_answer}\n\n"
                "Trích đoạn từ các trang web:\n{excerpts}\n\n"
                "Hãy viết câu trả lời cuối cùng cho người dùng. "
                "Định dạng: dùng bullet (•) cho danh sách, **bold** cho từ khóa quan trọng."
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_msg),
                ("human", human_msg),
            ]).format_messages(
                query=query,
                tavily_answer=tavily_answer or "(không có)",
                excerpts=excerpts or "(không có nội dung)",
            )
            model = self.gemini_func._make_model(temperature=0.2, max_tokens=4096)
            answer = (model | StrOutputParser()).invoke(prompt).strip()
            if answer:
                return answer
        except Exception as e:
            print(f"[_synthesize] LLM error: {type(e).__name__}: {e}", flush=True)
        return tavily_answer or "Không có đủ thông tin để trả lời câu hỏi này."
