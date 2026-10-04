"""
app/api/endpoints.py
====================
FastAPI router endpoints for RAGOps.
Supports:
  - POST /ask       (Standard RAG endpoint with source attribution)
  - POST /query     (Advanced RAG endpoint with full chunk metadata)
  - GET  /health    (System & vector store health check)
  - GET  /model-info (Model & configuration metadata)
  - GET  /metrics   (Prometheus metrics)
  - POST /feedback  (User feedback logging)
  - GET  /drift     (Data drift PSI/KS status)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Response
import jsonlines
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.core.config import settings
from app.generation.generator import AnswerGenerator, LLMUnavailableError
from app.monitoring import cloudwatch as cw
from app.monitoring import drift as drift_mod
from app.monitoring.metrics import (
    RAG_CHUNKS_INDEXED,
    RAG_CITATION_CORRECTNESS,
    RAG_DRIFT_DETECTED,
    RAG_FEEDBACK_MEAN,
    RAG_FEEDBACK_RATING,
    RAG_GROUNDEDNESS_SCORE,
    RAG_LATENCY_SECONDS,
    RAG_PSI,
    RAG_QUERY_LENGTH,
    RAG_REQUEST_COUNT,
    RAG_TOP_SCORE,
)
from app.retrieval.factory import get_retriever
from app.retrieval.pipeline import retrieve_and_filter_candidates
from app.schemas.query import (
    AskRequest,
    AskResponse,
    FeedbackRequest,
    QueryRequest,
    QueryResponse,
    SourceDetail,
)
from app.vectorstore.chromadb_store import ChromaDBStore

logger = logging.getLogger(__name__)
router = APIRouter()

# Singletons for service lifetime
_generator: AnswerGenerator | None = None
_store: ChromaDBStore | None = None


def get_generator() -> AnswerGenerator:
    global _generator
    if _generator is None:
        import os
        allow_mock = settings.allow_test_mock or os.getenv("ALLOW_TEST_MOCK", "0").lower() in ("1", "true", "yes")
        _generator = AnswerGenerator(allow_test_mock=allow_mock)
    return _generator




CHAMPION_FILE = Path("data/evaluation/reports/champion.json")


def champion_retriever() -> str:
    """Retriever promoted by the MLflow registry step (falls back to params default)."""
    try:
        return json.loads(CHAMPION_FILE.read_text(encoding="utf-8"))["retriever"]
    except Exception:
        return settings.retrieval_method or "semantic"


def get_store() -> ChromaDBStore:
    global _store
    if _store is None:
        _store = ChromaDBStore()
    return _store


@router.post("/ask", response_model=AskResponse, tags=["RAG Inference"])
async def ask_rag(request: AskRequest):
    """
    Standard RAG inference endpoint.
    Performs candidate retrieval -> relevance filtering -> context construction -> LLM answer -> source attribution.
    """
    total_start = time.perf_counter()
    retriever_name = request.retriever or champion_retriever()

    try:
        retriever = get_retriever(retriever_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    where = {"department": request.department} if request.department else None

    # Step 1 & 2: Candidate retrieval and relevance filtering
    t0 = time.perf_counter()
    try:
        final_context, candidates = retrieve_and_filter_candidates(
            query=request.question,
            retriever=retriever,
            candidate_k=request.top_k,
            relevance_threshold=request.relevance_threshold,
            where=where,
        )
    except Exception as exc:
        logger.error(f"Retrieval error: {exc}", exc_info=True)
        RAG_REQUEST_COUNT.labels(retriever=retriever_name, status="error", is_fallback="false").inc()
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}")
    ret_latency = (time.perf_counter() - t0) * 1000.0

    # Step 3: Generation & Quality Check
    gen = get_generator()
    try:
        gen_out = gen.generate_answer(request.question, final_context)
    except LLMUnavailableError as exc:
        logger.error(f"LLM unavailable: {exc}")
        RAG_REQUEST_COUNT.labels(retriever=retriever_name, status="error", is_fallback="false").inc()
        raise HTTPException(
            status_code=503,
            detail={"error": str(exc), "provider": settings.llm_provider},
        )

    total_latency_ms = (time.perf_counter() - total_start) * 1000.0

    # Format source attribution (only the context actually used)
    used_sources: list[str] = []
    source_details: list[SourceDetail] = []
    for c in final_context:
        title = (c.metadata.get("title") if c.metadata else None) or c.document_id or c.chunk_id
        if title not in used_sources:
            used_sources.append(title)
        source_details.append(
            SourceDetail(
                document_id=c.document_id or (c.metadata.get("document_id", "") if c.metadata else ""),
                title=title,
                relevance_score=round(float(c.score), 4),
                chunk_id=c.chunk_id,
            )
        )

    # Record Prometheus metrics
    is_fb_str = "true" if gen_out["is_fallback"] else "false"
    RAG_REQUEST_COUNT.labels(retriever=retriever_name, status="success", is_fallback=is_fb_str).inc()
    RAG_LATENCY_SECONDS.labels(phase="retrieval").observe(ret_latency / 1000.0)
    RAG_LATENCY_SECONDS.labels(phase="generation").observe(gen_out["generation_latency_ms"] / 1000.0)
    RAG_LATENCY_SECONDS.labels(phase="total").observe(total_latency_ms / 1000.0)
    RAG_GROUNDEDNESS_SCORE.observe(gen_out["groundedness_score"])
    RAG_CITATION_CORRECTNESS.observe(gen_out["citation_correctness"])

    return AskResponse(
        answer=gen_out["answer"],
        sources=used_sources,
        source_details=source_details,
        groundedness_score=gen_out["groundedness_score"],
        is_fallback=gen_out["is_fallback"],
        retriever_used=retriever_name,
        latency_ms=round(total_latency_ms, 2),
    )


@router.post("/query", response_model=QueryResponse, tags=["RAG Inference"])
async def query_rag(request: QueryRequest):
    """
    RAG inference endpoint with full chunk payloads.
    Retrieves filtered context, generates verified answer with citations,
    and returns latency and quality metrics.
    """
    total_start = time.perf_counter()
    retriever_name = request.retriever or champion_retriever()

    try:
        retriever = get_retriever(retriever_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    where = {"department": request.department} if request.department else None

    # Step 1: Retrieval with relevance filtering
    t0 = time.perf_counter()
    try:
        final_context, candidates = retrieve_and_filter_candidates(
            query=request.question,
            retriever=retriever,
            candidate_k=request.top_k,
            relevance_threshold=request.relevance_threshold,
            where=where,
        )
    except Exception as exc:
        logger.error(f"Retrieval error: {exc}", exc_info=True)
        RAG_REQUEST_COUNT.labels(retriever=retriever_name, status="error", is_fallback="false").inc()
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}")
    ret_latency = (time.perf_counter() - t0) * 1000.0

    # Step 2: Generation & Quality Check
    gen = get_generator()
    try:
        gen_out = gen.generate_answer(request.question, final_context)
    except LLMUnavailableError as exc:
        logger.error(f"LLM unavailable: {exc}")
        RAG_REQUEST_COUNT.labels(retriever=retriever_name, status="error", is_fallback="false").inc()
        raise HTTPException(
            status_code=503,
            detail={"error": str(exc), "provider": settings.llm_provider},
        )

    total_latency_ms = (time.perf_counter() - total_start) * 1000.0

    # Record Prometheus metrics
    is_fb_str = "true" if gen_out["is_fallback"] else "false"
    RAG_REQUEST_COUNT.labels(retriever=retriever_name, status="success", is_fallback=is_fb_str).inc()
    RAG_LATENCY_SECONDS.labels(phase="retrieval").observe(ret_latency / 1000.0)
    RAG_LATENCY_SECONDS.labels(phase="generation").observe(gen_out["generation_latency_ms"] / 1000.0)
    RAG_LATENCY_SECONDS.labels(phase="total").observe(total_latency_ms / 1000.0)
    RAG_GROUNDEDNESS_SCORE.observe(gen_out["groundedness_score"])
    RAG_CITATION_CORRECTNESS.observe(gen_out["citation_correctness"])
    top_score = float(final_context[0].score) if final_context else 0.0
    n_words = len(request.question.split())
    RAG_QUERY_LENGTH.observe(n_words)
    RAG_TOP_SCORE.observe(max(0.0, min(1.0, top_score)))
    cw.put_metrics({  # no-op unless ENABLE_CLOUDWATCH=1
        "TotalLatency": (total_latency_ms, "Milliseconds"),
        "Groundedness": (gen_out["groundedness_score"], "None"),
        "Fallback": (1.0 if gen_out["is_fallback"] else 0.0, "None"),
    })
    try:
        drift_mod.record_query({
            "ts": time.time(),
            "query_length": n_words,
            "top_retrieval_score": top_score,
            "groundedness_score": gen_out["groundedness_score"],
            "fallback": 1.0 if gen_out["is_fallback"] else 0.0,
            "retriever": retriever_name,
        })
    except Exception as exc:  # monitoring must never break inference
        logger.warning(f"query log write failed: {exc}")

    return QueryResponse(
        question=request.question,
        answer=gen_out["answer"],
        sources=final_context,
        groundedness_score=gen_out["groundedness_score"],
        is_fallback=gen_out["is_fallback"],
        retrieval_latency_ms=round(ret_latency, 2),
        generation_latency_ms=round(gen_out["generation_latency_ms"], 2),
        total_latency_ms=round(total_latency_ms, 2),
        retriever_used=retriever_name,
        prompt_version=gen_out["prompt_version"],
    )


@router.get("/health", tags=["System"])
async def health():
    """Service health check reporting vectorstore chunks and environment."""
    store = get_store()
    try:
        count = store.count()
        RAG_CHUNKS_INDEXED.set(count)
        status = "healthy"
    except Exception as exc:
        logger.warning(f"Vector store health check issue: {exc}")
        count = 0
        status = "degraded"

    return {
        "status": status,
        "version": settings.app_version,
        "environment": settings.app_env,
        "vectorstore_chunks_count": count,
    }


@router.get("/model-info", tags=["System"])
async def model_info():
    """Model information and operational configuration."""
    champ_ver = None
    champ_data = {}
    if CHAMPION_FILE.exists():
        try:
            champ_data = json.loads(CHAMPION_FILE.read_text(encoding="utf-8"))
            champ_ver = champ_data.get("version")
        except Exception:
            pass

    chunking_cfg = champ_data.get("chunking") or settings.params.get("chunking", {
        "strategy": "recursive",
        "chunk_size": 512,
        "chunk_overlap": 64,
    })

    return {
        "app_version": settings.app_version,
        "environment": settings.app_env,
        "retrieval_method": champion_retriever(),
        "champion_retriever": champion_retriever(),
        "default_retriever": settings.retrieval_method,
        "registry_champion_version": champ_ver,
        "champion": champ_data,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "chunking": chunking_cfg,
        "top_k": settings.top_k,
        "relevance_threshold": settings.relevance_threshold,
        "final_context_k": settings.final_context_k,
        "groundedness_threshold": settings.groundedness_threshold,
    }




@router.get("/version", tags=["System"])
async def version():
    """Version and model configuration metadata."""
    return await model_info()


@router.get("/stats", tags=["Monitoring"])
async def stats():
    """Vector database statistics and metadata distribution."""
    store = get_store()
    return store.collection_stats()


@router.get("/retrievers", tags=["Retrieval"])
async def list_retrievers():
    """List supported retrievers and champion configuration."""
    return {
        "supported_retrievers": ["tfidf", "bm25", "semantic", "hybrid"],
        "default_retriever": champion_retriever(),
        "champion_retriever": champion_retriever(),
    }


@router.post("/feedback", tags=["Monitoring"])
async def log_feedback(feedback: FeedbackRequest):
    """Log user feedback for drift detection and quality improvement."""
    feedback_file = Path("data/monitoring/feedback.jsonl")
    feedback_file.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": time.time(),
        "query": feedback.query,
        "answer": feedback.answer,
        "rating": feedback.rating,
        "comments": feedback.comments,
        "retriever_used": feedback.retriever_used,
    }

    with jsonlines.open(feedback_file, mode="a") as writer:
        writer.write(record)
    RAG_FEEDBACK_RATING.observe(feedback.rating)
    ratings = [json.loads(l)["rating"] for l in feedback_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    if ratings:
        RAG_FEEDBACK_MEAN.set(sum(ratings) / len(ratings))

    return {"status": "logged", "timestamp": record["timestamp"]}


@router.get("/drift", tags=["Monitoring"])
async def drift_status(window: int = 100):
    """Compare the last `window` logged queries against the stored reference distribution."""
    cfg = settings
    ref = drift_mod.load_reference()
    cur = drift_mod.load_features(drift_mod.QUERY_LOG, last_n=window)
    report = drift_mod.detect_drift(
        ref, cur, psi_threshold=cfg.drift_psi_threshold, ks_alpha=cfg.drift_ks_p_value, min_samples=30
    )
    for f, r in report["features"].items():
        if r.get("psi") is not None:
            RAG_PSI.labels(feature=f).set(r["psi"])
            RAG_DRIFT_DETECTED.labels(feature=f).set(1 if r["drift"] else 0)
    return report


@router.get("/metrics", tags=["Monitoring"], include_in_schema=False)
async def metrics():
    """Prometheus metrics scrape endpoint."""
    store = get_store()
    try:
        RAG_CHUNKS_INDEXED.set(store.count())
    except Exception:
        pass
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
