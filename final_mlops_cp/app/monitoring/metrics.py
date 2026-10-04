"""
app/monitoring/metrics.py
=========================
Prometheus metrics registry and definitions for RAGOps observability.
"""

from prometheus_client import Counter, Gauge, Histogram

# Request counter
RAG_REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total count of RAG query requests received",
    ["retriever", "status", "is_fallback"],
)

# Latency histograms
RAG_LATENCY_SECONDS = Histogram(
    "rag_latency_seconds",
    "Latency breakdown in seconds",
    ["phase"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# Quality metrics histograms
RAG_GROUNDEDNESS_SCORE = Histogram(
    "rag_groundedness_score",
    "Distribution of groundedness verification scores",
    buckets=(0.1, 0.25, 0.5, 0.7, 0.75, 0.85, 0.95, 1.0),
)

RAG_CITATION_CORRECTNESS = Histogram(
    "rag_citation_correctness",
    "Distribution of citation correctness scores",
    buckets=(0.2, 0.4, 0.6, 0.8, 0.9, 1.0),
)

# Gauges
RAG_CHUNKS_INDEXED = Gauge(
    "rag_chunks_indexed_total",
    "Total number of document chunks currently in vector database",
)

RAG_DRIFT_DETECTED = Gauge(
    "rag_drift_detected",
    "1 if statistical drift was detected in query or retrieval distribution, 0 otherwise",
    ["feature"],
)

RAG_PSI = Gauge("rag_feature_psi", "Population Stability Index per monitored feature", ["feature"])
RAG_QUERY_LENGTH = Histogram(
    "rag_query_length_words", "Words per incoming query (input drift)",
    buckets=(2, 4, 6, 8, 10, 15, 20, 40, 80),
)
RAG_TOP_SCORE = Histogram(
    "rag_top_retrieval_score", "Top-1 retrieval score (prediction distribution / drift)",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
RAG_FEEDBACK_RATING = Histogram(
    "rag_feedback_rating", "User ratings 1-5 (ground-truth quality signal)",
    buckets=(1, 2, 3, 4, 5),
)
RAG_FEEDBACK_MEAN = Gauge("rag_feedback_rating_mean", "Mean user rating over logged feedback")
RAG_HTTP_REQUESTS = Counter(
    "rag_http_requests_total", "All HTTP requests by path and status code", ["path", "status"]
)
