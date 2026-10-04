"""
app/schemas/query.py
====================
Pydantic models for API query requests and responses.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator

from app.schemas.retrieval import RetrievedChunk


class AskRequest(BaseModel):
    """Standard request model for POST /ask."""

    question: str = Field(
        ...,
        description="User question to answer",
        json_schema_extra={"example": "What health insurance benefits are available to employees?"},
    )
    retriever: Optional[str] = Field(
        default=None,
        description="Retriever override ('tfidf', 'bm25', 'semantic', 'hybrid')",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of candidate chunks to retrieve",
    )
    relevance_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score threshold for context filtering",
    )
    department: Optional[str] = Field(
        default=None,
        description="Optional metadata filter by department (e.g. 'HR', 'IT')",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Question cannot be empty or whitespace only.")
        stripped = v.strip()
        if len(stripped) < 3:
            raise ValueError("Question must be at least 3 characters long.")
        if len(v) > 1000:
            raise ValueError("Question cannot exceed 1000 characters.")
        return stripped


class SourceDetail(BaseModel):
    document_id: str
    title: str
    relevance_score: float
    chunk_id: Optional[str] = None


class AskResponse(BaseModel):
    """Standard response model for POST /ask."""

    answer: str
    sources: list[str] = Field(description="Names/titles of verified sources used in the answer")
    source_details: Optional[list[SourceDetail]] = Field(
        default=None,
        description="Detailed metadata for each context source actually used",
    )
    groundedness_score: Optional[float] = None
    is_fallback: Optional[bool] = None
    retriever_used: Optional[str] = None
    latency_ms: Optional[float] = None


class QueryRequest(BaseModel):
    """Incoming user query to RAGOps API (backward compatible)."""

    question: str = Field(
        ...,
        description="User question to answer",
        json_schema_extra={"example": "How many days of annual leave are employees entitled to?"},
    )
    retriever: Optional[str] = Field(
        default=None,
        description="Retriever override ('tfidf', 'bm25', 'semantic', 'hybrid')",
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of context chunks to retrieve",
    )
    relevance_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score threshold for context filtering",
    )
    department: Optional[str] = Field(
        default=None,
        description="Optional metadata filter by department (e.g. 'HR', 'IT')",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Question cannot be empty or whitespace only.")
        stripped = v.strip()
        if len(stripped) < 3:
            raise ValueError("Question must be at least 3 characters long.")
        if len(v) > 1000:
            raise ValueError("Question cannot exceed 1000 characters.")
        return stripped


class QueryResponse(BaseModel):
    """Outgoing RAGOps answer with citations and quality scores (backward compatible)."""

    question: str
    answer: str
    sources: list[RetrievedChunk]
    groundedness_score: float
    is_fallback: bool
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    retriever_used: str
    prompt_version: str


class FeedbackRequest(BaseModel):
    """User feedback on an answer for monitoring and drift tracking."""

    query: str
    answer: str
    rating: int = Field(..., ge=1, le=5, description="1 (poor) to 5 (excellent)")
    comments: Optional[str] = None
    retriever_used: Optional[str] = None
