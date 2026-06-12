from typing import List
from tavily import TavilyClient
from source.core.config import Settings


class TavilySearchTool:
    """Web search + content extraction via Tavily.

    Tavily's /search endpoint returns URLs, raw page content, and an
    optional LLM-generated answer summary in a single call. We cache the
    full response after `search()` so downstream code can:
      - read URLs (search() return)
      - read page text (extract_text_from_url)
      - read Tavily's pre-summarized answer (get_last_answer)
    """

    def __init__(self, setting: Settings):
        self.api_key = setting.TAVILY_API_KEY
        self.client = TavilyClient(api_key=self.api_key) if self.api_key else None
        self._cache: dict = {}
        self._last_answer: str = ""

    def search(self, query: str, num_results: int = 5) -> List[str]:
        if self.client is None:
            raise RuntimeError("TAVILY_API_KEY is not set in .env")
        response = self.client.search(
            query=query,
            max_results=num_results,
            include_raw_content=True,
            include_answer="advanced",
            search_depth="advanced",
        )
        self._last_answer = response.get("answer") or ""
        results = response.get("results", [])
        self._cache = {r["url"]: r for r in results}
        return [r["url"] for r in results]

    def get_last_answer(self) -> str:
        return self._last_answer

    def extract_text_from_url(self, url: str) -> str:
        entry = self._cache.get(url)
        if entry is None:
            return ""
        return entry.get("raw_content") or entry.get("content") or ""

    def extract_texts_from_links(self, links: List[str]) -> List[str]:
        return [self.extract_text_from_url(url) for url in links]
