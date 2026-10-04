"""
app/main.py
===========
FastAPI application entry point.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import settings
from app.api.endpoints import router as api_router

app = FastAPI(
    title="RAGOps: Enterprise Knowledge Assistant",
    description=(
        "Monitored, versioned RAG system for answering questions "
        "grounded in verified organizational documents."
    ),
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount endpoints at root (/ask, /health, /model-info) and at /api/v1 prefix
app.include_router(api_router)
app.include_router(api_router, prefix="/api/v1")


@app.middleware("http")
async def count_requests(request: Request, call_next):
    from app.monitoring.metrics import RAG_HTTP_REQUESTS

    path = request.url.path
    try:
        response = await call_next(request)
    except Exception:  # unhandled -> 500, still counted
        RAG_HTTP_REQUESTS.labels(path=path, status="500").inc()
        raise
    if path not in ("/metrics", "/api/v1/metrics"):
        RAG_HTTP_REQUESTS.labels(path=path, status=str(response.status_code)).inc()
    return response


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    logging.getLogger(__name__).error("Unhandled error", exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/", tags=["System"], include_in_schema=False)
async def root():
    """Root endpoint redirecting to docs."""
    return RedirectResponse(url="/docs")
