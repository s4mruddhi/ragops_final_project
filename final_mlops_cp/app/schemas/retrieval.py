"""
app/schemas/retrieval.py
========================
Data schemas for the retrieval layer.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """A single context chunk returned by a retriever."""

    chunk_id: str = Field(description="Unique identifier for the chunk, e.g. HR-LEAVE-001_chunk_001")
    text: str = Field(description="Text content of the chunk")
    score: float = Field(description="Relevance or similarity score")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata associated with the chunk")
    document_id: Optional[str] = Field(default=None, description="Parent document identifier")
    retriever: Optional[str] = Field(default=None, description="Name of the retriever that produced this result")

    def model_post_init(self, __context: Any) -> None:
        if not self.document_id and "document_id" in self.metadata:
            self.document_id = str(self.metadata["document_id"])
        elif not self.document_id and "_" in self.chunk_id:
            # Fallback extraction from chunk_id like HR-LEAVE-001_chunk_001
            self.document_id = self.chunk_id.split("_chunk_")[0]


class RetrievalResult(BaseModel):
    """Output of a retrieval request."""

    query: str = Field(description="User query string")
    chunks: list[RetrievedChunk] = Field(default_factory=list, description="List of retrieved chunks")
    retriever: str = Field(description="Retriever method used")
    latency_ms: float = Field(default=0.0, description="Retrieval latency in milliseconds")
    top_k: int = Field(default=5, description="Number of chunks requested")
