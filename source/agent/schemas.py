"""Pydantic schemas for agent outputs.

Each agent returns a structured JSON that we parse into one of these
models. Keeping them in one file makes the contract between planner,
grader, hallucination checker and the orchestrator easy to audit.
"""
from __future__ import annotations

from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field


Intent = Literal["chitchat", "legal", "realtime", "out_of_scope"]
Tool = Literal["template", "rag", "web", "reject", "ask_user"]


class Plan(BaseModel):
    """Output of the planner agent — decides the whole turn."""
    intent: Intent
    tool: Tool
    rewritten_query: str = Field(
        description="Self-contained question. Used by RAG retrieval and as the "
                    "canonical form of the user's ask."
    )
    search_query: Optional[str] = Field(
        default=None,
        description="Keyword-heavy query for Tavily. Includes today's date for "
                    "time-sensitive asks. Null if tool != 'web'."
    )
    entities: Optional[Dict[str, str]] = Field(
        default=None,
        description="Qdrant metadata filter extracted from the question: "
                    "{LoaiVanBanFilter, NoibanHanhFilter, NgayBanHanhFilter}. "
                    "Null or empty dict if no filter applies."
    )
    needs_clarification: bool = False
    clarifying_question: Optional[str] = None
    reason: str = Field(
        description="Short Vietnamese explanation of the routing decision — "
                    "shown to the user in the thinking panel."
    )


GraderDecision = Literal["sufficient", "partial", "insufficient"]


class GraderResult(BaseModel):
    """Output of the grader — evaluates whether retrieved docs answer the query."""
    relevance_score: float = Field(ge=0.0, le=1.0)
    decision: GraderDecision
    missing_aspects: List[str] = Field(
        default_factory=list,
        description="Concrete gaps the planner should address on retry."
    )
    suggested_query: Optional[str] = Field(
        default=None,
        description="Rewritten query for the next iteration. Null if "
                    "decision == 'sufficient'."
    )
    reason: str


class CheckResult(BaseModel):
    """Output of the hallucination checker — verifies answer against sources."""
    is_grounded: bool
    confidence: float = Field(ge=0.0, le=1.0)
    unsupported_claims: List[str] = Field(default_factory=list)
    reason: str
