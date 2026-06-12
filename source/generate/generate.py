from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock
from source.model.generate_model import Gemini
from source.function.utils_shared import load_prompt_from_yaml, clean_generated_queries
from source.core.config import Settings


class Gemini_Generate():
    """Unified LLM caller. Backed by AWS Bedrock (Claude) by default,
    falls back to Google Gemini if LLM_PROVIDER=gemini.

    After the agentic refactor this class only hosts generation primitives
    that the orchestrator still calls directly: multi-query expansion
    (`generate_query`), the legal synthesis prompt (`generate_response`),
    the chitchat self-introduction prompt (`generate_information`), and
    the polite-refusal prompt (`invalid_query`). Routing / classification /
    entity extraction moved into `source.agent.planner`.

    Class name kept as `Gemini_Generate` to avoid import churn across
    callers that haven't been refactored yet.
    """

    def __init__(self, gemini_model: Gemini, settings: Settings):
        self.gemini_model = gemini_model
        self.yaml_path = settings
        self.settings = settings
        self.provider = settings.LLM_PROVIDER

    def _make_model(self, temperature: float = 0.0, max_tokens: int = 4096, top_p: float = 1.0):
        if self.provider == "bedrock":
            # Claude on Bedrock rejects temperature + top_p together.
            model_kwargs = {"temperature": temperature}
            return ChatBedrock(
                model_id=self.gemini_model.bedrock_model_id,
                region_name=self.gemini_model.aws_region,
                model_kwargs=model_kwargs,
                max_tokens=max_tokens,
            )
        # Legacy Gemini path
        import importlib
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.caches import BaseCache
        from langchain_core.callbacks.base import Callbacks
        mod = importlib.import_module('langchain_google_genai')
        mod.BaseCache = BaseCache
        mod.Callbacks = Callbacks
        ChatGoogleGenerativeAI.model_rebuild()
        return ChatGoogleGenerativeAI(
            google_api_key=self.gemini_model.key_manager.get_next_key(),
            model=self.gemini_model.model_gemini,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

    def generate_query(self, original_query: str, memory_context: str = "") -> list[str]:
        prompt = load_prompt_from_yaml(self.yaml_path, 'query_generator')
        model = self._make_model(temperature=0)
        effective_query = (
            f"Ngữ cảnh hội thoại trước đó:\n{memory_context}\n\nCâu hỏi hiện tại: {original_query}"
            if memory_context else original_query
        )
        prompt = prompt.format_messages(original_query=effective_query)
        chain = model | StrOutputParser()
        result = chain.invoke(prompt)
        generated_queries = result.strip().split('\n')
        generated_queries = clean_generated_queries(generated_queries)
        return [original_query] + generated_queries

    def generate_response(self, query: str, docs: dict, memory_context: str = "") -> str:
        docs_str = "\n".join(f"{k}: {v}" for k, v in docs.items())
        if memory_context:
            docs_str = f"[NGỮ CẢNH HỘI THOẠI TRƯỚC ĐÓ]\n{memory_context}\n\n[VĂN BẢN LUẬT LIÊN QUAN]\n{docs_str}"
        prompt_template = load_prompt_from_yaml(self.yaml_path, "response")
        model = self._make_model(temperature=0.2, max_tokens=8192, top_p=0.6)
        prompt = prompt_template.format_messages(original_query=query, context=docs_str)
        chain = model | StrOutputParser()
        return chain.invoke(prompt).strip()

    def invalid_query(self, query: str, memory_context: str = "") -> str:
        prompt = load_prompt_from_yaml(self.yaml_path, 'invalid_query')
        model = self._make_model(temperature=0)
        prompt = prompt.format_messages(query=query)
        chain = model | StrOutputParser()
        return chain.invoke(prompt).strip()

    def generate_information(self, query: str, context, memory_context: str = "") -> str:
        prompt = load_prompt_from_yaml(self.yaml_path, 'information_query')
        model = self._make_model(temperature=0)
        combined_context = f"{memory_context}\n\n{context}" if memory_context else context
        prompt = prompt.format_messages(query=query, context=combined_context)
        chain = model | StrOutputParser()
        return chain.invoke(prompt).strip()
