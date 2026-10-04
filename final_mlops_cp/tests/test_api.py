"""API integration tests (FastAPI TestClient) - health, query, ask, model-info, validation, errors, metrics, drift."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import endpoints
from app.generation.generator import AnswerGenerator
from app.main import app

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / "data" / "vectorstore" / "index_stats.json").exists(),
    reason="vector index not built - run `python scripts/run_dag_pipeline.py` first",
)

client = TestClient(app, raise_server_exceptions=False)
BASE = "/api/v1"


def test_health_endpoint():
    for url in ("/health", f"{BASE}/health"):
        r = client.get(url)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert body["vectorstore_chunks_count"] > 0


def test_model_info_endpoint():
    for url in ("/model-info", f"{BASE}/model-info"):
        r = client.get(url)
        assert r.status_code == 200
        body = r.json()
        assert "retrieval_method" in body
        assert "embedding_model" in body
        assert "chunking" in body
        assert "champion" in body


def test_ask_missing_credentials_returns_503(monkeypatch):
    monkeypatch.setattr(endpoints, "_generator", AnswerGenerator(allow_test_mock=False))
    r = client.post("/ask", json={"question": "How many days of annual leave are employees entitled to?"})
    assert r.status_code == 503
    body = r.json()
    assert "error" in body
    assert "LLM service unavailable" in body["error"]


def test_ask_with_mock_generator(monkeypatch):
    monkeypatch.setattr(endpoints, "_generator", AnswerGenerator(allow_test_mock=True))
    r = client.post("/ask", json={"question": "How many days of annual leave are employees entitled to?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert len(body["sources"]) > 0
    assert isinstance(body["sources"][0], str)
    assert len(body["source_details"]) > 0
    assert "title" in body["source_details"][0]
    assert "relevance_score" in body["source_details"][0]



def test_ask_whitespace_question_rejected():
    r = client.post("/ask", json={"question": "     "})
    assert r.status_code == 422


def test_query_returns_grounded_answer_with_sources(monkeypatch):
    monkeypatch.setattr(endpoints, "_generator", AnswerGenerator(allow_test_mock=True))
    r = client.post(f"{BASE}/query", json={"question": "How many days of annual leave are employees entitled to?",
                                           "retriever": "tfidf"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert body["sources"], "must return cited sources"
    assert 0.0 <= body["groundedness_score"] <= 1.0
    assert body["total_latency_ms"] >= 0


def test_query_missing_credentials_returns_503(monkeypatch):
    monkeypatch.setattr(endpoints, "_generator", AnswerGenerator(allow_test_mock=False))
    r = client.post(f"{BASE}/query", json={"question": "What is the leave policy?"})
    assert r.status_code == 503
    body = r.json()
    assert "error" in body


@pytest.mark.parametrize("payload", [{}, {"question": "x"}, {"question": "ok question", "top_k": 0},
                                     {"question": "ok question", "top_k": 99}, {"question": "a" * 1001}])
def test_input_schema_validation_rejects_bad_requests(payload):
    assert client.post(f"{BASE}/query", json=payload).status_code == 422


def test_unknown_retriever_is_400():
    r = client.post(f"{BASE}/query", json={"question": "What is the VPN policy?", "retriever": "nope"})
    assert r.status_code == 400


def test_metrics_endpoint_exposes_prometheus_series(monkeypatch):
    monkeypatch.setattr(endpoints, "_generator", AnswerGenerator(allow_test_mock=True))
    client.post(f"{BASE}/query", json={"question": "What is the password policy?", "retriever": "bm25"})
    text = client.get(f"{BASE}/metrics").text
    for series in ("rag_requests_total", "rag_latency_seconds", "rag_groundedness_score", "rag_http_requests_total"):
        assert series in text


def test_feedback_is_logged_and_validated():
    ok = client.post(f"{BASE}/feedback", json={"query": "q?", "answer": "a", "rating": 5})
    assert ok.status_code == 200
    assert client.post(f"{BASE}/feedback", json={"query": "q?", "answer": "a", "rating": 9}).status_code == 422


def test_drift_endpoint_runs():
    r = client.get(f"{BASE}/drift")
    assert r.status_code == 200
    assert "features" in r.json()


def test_retrievers_lists_champion():
    body = client.get(f"{BASE}/retrievers").json()
    assert body["champion_retriever"] in body["supported_retrievers"]

