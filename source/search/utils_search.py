from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from typing import Optional

from source.data.vectordb.qdrant import Qdrant_Vector


class Qdrant_Utils():
    """Thin wrapper over Qdrant: opens the collection and applies metadata
    filters built from planner-extracted entities.

    The legacy `search_With_Similarity_Queries` (which generated 3 query
    variants and ran 4 per-variant LLM calls to extract entities) was
    removed after the agentic refactor. The orchestrator now expands
    queries itself and applies a single filter coming from the planner.
    """

    def __init__(self, qdrant: Qdrant_Vector, gemini_util=None):
        self.qdrant = qdrant
        # `gemini_util` kept as an optional constructor arg to avoid breaking
        # callers that still pass it; no longer used internally.
        self.gemini = gemini_util

    def search_documents(self, query, top_k=25, filter: Optional[Filter] = None):
        connection = self.qdrant.Open_Qdrant()
        return connection.similarity_search_with_score(
            query=query,
            k=top_k,
            timeout=300,
            filter=filter,
        )

    def build_metadata_filter(self, entity_dict: dict) -> Optional[Filter]:
        conditions = []
        for key, value in (entity_dict or {}).items():
            if not value:
                continue
            if key == "NgayBanHanhFilter":
                conditions.append(
                    FieldCondition(
                        key="metadata." + key,
                        match=MatchAny(any=[value]),
                    )
                )
            else:
                conditions.append(
                    FieldCondition(
                        key="metadata." + key,
                        match=MatchValue(value=str(value).lower()),
                    )
                )
        if not conditions:
            return None
        return Filter(should=conditions)
