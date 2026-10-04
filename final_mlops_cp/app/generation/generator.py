"""
app/generation/generator.py
===========================
RAG Answer Generator with prompt versioning, LLM provider abstraction,
groundedness checking, and strict real-LLM enforcement.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Optional, Sequence

import yaml

from app.core.config import settings
from app.generation.citations import extract_cited_document_ids, verify_citations
from app.generation.groundedness import compute_groundedness
from app.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


class LLMUnavailableError(RuntimeError):
    """Raised when the configured LLM is unavailable and mock fallback is disabled."""
    pass


def _load_generation_params(params_path: str = "params.yaml") -> dict:
    p = Path(params_path)
    if not p.is_absolute():
        p = ROOT / p
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("generation", {})
    return {}


class AnswerGenerator:
    """
    Orchestrates prompt assembly, LLM generation, groundedness scoring,
    citation verification, and fallback handling.
    """

    def __init__(
        self,
        prompt_version: Optional[str] = None,
        fallback_prompt_version: Optional[str] = None,
        groundedness_threshold: Optional[float] = None,
        params_path: str = "params.yaml",
        allow_test_mock: Optional[bool] = None,
    ):
        cfg = _load_generation_params(params_path)
        self.prompt_version = prompt_version or cfg.get("prompt_version", "answer_v1")
        self.fallback_prompt_version = fallback_prompt_version or cfg.get("fallback_prompt_version", "fallback_v1")
        self.groundedness_threshold = (
            groundedness_threshold
            if groundedness_threshold is not None
            else float(cfg.get("groundedness_threshold", 0.75))
        )
        self.max_context_chunks = int(cfg.get("max_context_chunks", 5))

        # Test isolation: only allows mock if explicitly enabled for unit tests
        if allow_test_mock is not None:
            self.allow_test_mock = allow_test_mock
        else:
            self.allow_test_mock = os.environ.get("ENABLE_TEST_MOCK_LLM", "0") == "1"

        self.prompt_template = self._load_prompt(self.prompt_version)
        self.fallback_template = self._load_prompt(self.fallback_prompt_version)

    def _load_prompt(self, version_name: str) -> str:
        """Load prompt file from prompts/ directory."""
        candidates = [
            ROOT / "prompts" / f"{version_name}.txt",
            ROOT / "prompts" / version_name,
        ]
        for p in candidates:
            if p.exists():
                return p.read_text(encoding="utf-8")
        logger.warning(f"Prompt template '{version_name}' not found. Using minimal fallback.")
        return "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

    def format_context(self, chunks: Sequence[Any]) -> str:
        """Format retrieved chunks into a structured context block for the prompt."""
        formatted_chunks = []
        for i, c in enumerate(chunks[: self.max_context_chunks], start=1):
            if isinstance(c, dict):
                text = c.get("text", "")
                cid = c.get("chunk_id", f"chunk_{i}")
                meta = c.get("metadata", {})
            else:
                text = getattr(c, "text", "")
                cid = getattr(c, "chunk_id", f"chunk_{i}")
                meta = getattr(c, "metadata", {}) or {}

            doc_id = meta.get("document_id") or getattr(c, "document_id", None) or cid.split("_chunk_")[0]
            title = meta.get("title", "Policy Document")
            ver = meta.get("version", "1.0")

            formatted_chunks.append(
                f"[Chunk {i} | Document: {doc_id} | Title: {title} | Version: {ver}]\n{text}"
            )
        return "\n\n".join(formatted_chunks)

    def generate_raw_llm_response(self, prompt: str) -> str:
        """
        Call the configured real LLM provider (OpenAI, Ollama, or HuggingFace).
        If the real LLM is not configured/available, raises LLMUnavailableError.
        Never silently switches to fake or extractive answers in production.
        """
        provider = settings.llm_provider.lower().strip()

        # 1. OpenAI provider
        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY") or settings.openai_api_key
            if not api_key or api_key.startswith("sk-your") or len(api_key.strip()) < 10:
                if self.allow_test_mock:
                    return self._extractive_generate(prompt)
                raise LLMUnavailableError(
                    "LLM service unavailable. Configure the required LLM credentials/model."
                )
            try:
                from openai import OpenAI

                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful enterprise knowledge assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=512,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                if self.allow_test_mock:
                    logger.warning(f"OpenAI error ({exc}) in test mode; using test mock.")
                    return self._extractive_generate(prompt)
                raise LLMUnavailableError(f"LLM service unavailable: {exc}")

        # 2. Ollama local provider (Active default)
        elif provider == "ollama":
            import urllib.request
            import urllib.error
            import json

            base_url = settings.ollama_base_url.rstrip("/")
            url = f"{base_url}/api/generate"
            payload = json.dumps({
                "model": settings.llm_model,
                "prompt": prompt,
                "system": (
                    "You are an enterprise knowledge assistant for Acme Corp.\n"
                    "Answer the user's question using the provided verified organizational context.\n"
                    "Do not invent facts, numbers, dates, or policies not explicitly stated in the context.\n"
                    "Cite sources using [Document ID] — [Title] format.\n"
                    "If the retrieved context does not contain enough information to answer the question, "
                    "clearly state: 'Based on available documents, I cannot find sufficient information to answer this question.'"
                ),
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 512,
                },
            }).encode("utf-8")

            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data.get("response", "")
            except urllib.error.HTTPError as exc:
                err_body = ""
                try:
                    err_body = exc.read().decode("utf-8")
                except Exception:
                    pass
                if exc.code == 404:
                    raise LLMUnavailableError(
                        f"Ollama model '{settings.llm_model}' is not available. "
                        f"Run 'ollama pull {settings.llm_model}' first."
                    )
                raise LLMUnavailableError(f"Ollama HTTP error {exc.code}: {err_body or exc.reason}")
            except Exception as exc:
                logger.error(f"Ollama connection error: {exc}")
                raise LLMUnavailableError(
                    f"Ollama service unavailable. Could not connect to Ollama at {base_url}. "
                    f"Ensure Ollama is running (`ollama serve`). Details: {exc}"
                )

        # 3. HuggingFace local pipeline
        elif provider == "huggingface":
            try:
                from transformers import pipeline

                generator = pipeline("text-generation", model=settings.llm_model, max_new_tokens=256)
                out = generator(prompt)
                return out[0]["generated_text"]
            except Exception as exc:
                if self.allow_test_mock:
                    return self._extractive_generate(prompt)
                raise LLMUnavailableError(f"LLM service unavailable: HuggingFace error: {exc}")

        # 4. Google Gemini provider (Free tier supported)
        elif provider == "gemini":
            api_key = (
                os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
                or settings.gemini_api_key
            )
            if not api_key or api_key.startswith("your_") or len(api_key.strip()) < 10:
                if self.allow_test_mock:
                    return self._extractive_generate(prompt)
                raise LLMUnavailableError(
                    "LLM service unavailable. Configure GEMINI_API_KEY in .env."
                )

            # Use configured model, defaulting to gemini-2.5-flash (active standard)
            raw_model = (settings.llm_model or "").strip()
            model_name = raw_model if raw_model else "gemini-2.5-flash"
            # Strip any leading 'models/' if user included it
            if model_name.startswith("models/"):
                model_name = model_name[len("models/"):]

            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                gemini_model = genai.GenerativeModel(model_name)
                resp = gemini_model.generate_content(prompt)
                return resp.text or ""
            except ImportError:
                import urllib.request
                import urllib.error
                import json

                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
                payload = json.dumps({
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512},
                }).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                try:
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        res_data = json.loads(resp.read().decode("utf-8"))
                        candidates = res_data.get("candidates", [])
                        if candidates:
                            content = candidates[0].get("content", {})
                            parts = content.get("parts", [])
                            if parts:
                                return parts[0].get("text", "")
                        return ""
                except Exception as exc:
                    detail_msg = str(exc)
                    if hasattr(exc, "read"):
                        try:
                            err_body = exc.read().decode("utf-8")
                            parsed = json.loads(err_body)
                            detail_msg = parsed.get("error", {}).get("message") or err_body
                        except Exception:
                            pass
                    logger.error(f"Gemini API request failed ({exc}): {detail_msg}")
                    if self.allow_test_mock:
                        logger.warning("Gemini error in test mode; using test mock.")
                        return self._extractive_generate(prompt)
                    raise LLMUnavailableError(f"Gemini API error ({exc}): {detail_msg}")
            except Exception as exc:
                logger.error(f"Gemini SDK error: {exc}")
                if self.allow_test_mock:
                    logger.warning("Gemini SDK error in test mode; using test mock.")
                    return self._extractive_generate(prompt)
                raise LLMUnavailableError(f"Gemini error: {exc}")


        # Test mock fallback allowed ONLY if explicitly requested in unit tests
        if self.allow_test_mock:
            return self._extractive_generate(prompt)

        raise LLMUnavailableError("LLM service unavailable. Configure the required LLM credentials/model.")

    def _extractive_generate(self, prompt: str) -> str:
        """
        Isolated test mock for deterministic unit-test assertions without external network.
        NEVER used in the production path.
        """
        context_match = re.search(
            r"CONTEXT \(verified organizational documents\):\s*(.*?)\s*---\s*EMPLOYEE QUESTION:\s*(.*?)\s*---",
            prompt,
            flags=re.DOTALL,
        )
        if not context_match:
            return "Based on available documents, I cannot find sufficient information to answer this question."

        context_text = context_match.group(1).strip()
        question = context_match.group(2).strip()

        # Find document IDs in context
        doc_matches = re.findall(r"Document:\s*([A-Z0-9\-]+)\s*\|\s*Title:\s*([^|\n]+)", context_text)
        sources_list = []
        for doc_id, title in doc_matches:
            entry = f"[{doc_id}] — {title.strip()}"
            if entry not in sources_list:
                sources_list.append(entry)

        # Extract sentences from context
        sentences = [
            s.strip()
            for s in re.split(r"[.\n]+", context_text)
            if len(s.strip()) > 20 and not s.startswith("[Chunk")
        ]

        q_words = {w.lower() for w in re.findall(r"\w+", question) if len(w) > 3}
        best_sentence = ""
        best_score = -1

        for sent in sentences:
            s_words = {w.lower() for w in re.findall(r"\w+", sent)}
            score = len(q_words.intersection(s_words))
            if score > best_score:
                best_score = score
                best_sentence = sent

        min_overlap = max(2, int(round(0.2 * len(q_words))))
        if not best_sentence or best_score < min_overlap:
            return "Based on available documents, I cannot find sufficient information to answer this question."

        ans = f"{best_sentence.strip()}."
        if sources_list:
            ans += "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources_list[:2])
        return ans

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: Sequence[Any],
    ) -> dict[str, Any]:
        """
        Complete RAG generation pipeline:
        1. Context formatting
        2. Prompt assembly
        3. Real LLM generation
        4. Groundedness computation
        5. Citation verification
        6. Fallback enforcement
        """
        start_time = time.perf_counter()

        if not retrieved_chunks:
            # No context retrieved or all filtered out by relevance threshold
            fallback_text = self.fallback_template.replace("{question}", question)
            return {
                "answer": fallback_text,
                "raw_answer": fallback_text,
                "is_fallback": True,
                "is_grounded": False,
                "groundedness_score": 0.0,
                "citation_correctness": 1.0,
                "cited_sources": [],
                "generation_latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "prompt_version": self.fallback_prompt_version,
                "llm_used": False,
            }

        context_str = self.format_context(retrieved_chunks)
        prompt = self.prompt_template.replace("{context}", context_str).replace("{question}", question)

        raw_answer = self.generate_raw_llm_response(prompt)

        # Compute groundedness and citation correctness
        groundedness_res = compute_groundedness(
            answer=raw_answer,
            context_chunks=retrieved_chunks,
            threshold=self.groundedness_threshold,
        )
        citation_res = verify_citations(
            answer=raw_answer,
            context_chunks=retrieved_chunks,
        )

        groundedness_score = groundedness_res["score"]
        is_fallback = False
        final_answer = raw_answer

        # Enforce quality threshold: if unsupported -> fallback
        if groundedness_score < self.groundedness_threshold:
            logger.warning(
                f"Groundedness score {groundedness_score} is below threshold {self.groundedness_threshold}. Triggering fallback."
            )
            final_answer = self.fallback_template.replace("{question}", question)
            is_fallback = True

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "answer": final_answer,
            "raw_answer": raw_answer,
            "is_fallback": is_fallback,
            "groundedness_score": groundedness_score,
            "is_grounded": groundedness_res["is_grounded"],
            "citation_correctness": citation_res["citation_correctness"],
            "cited_sources": citation_res["cited_document_ids"],
            "generation_latency_ms": round(latency_ms, 2),
            "prompt_version": self.fallback_prompt_version if is_fallback else self.prompt_version,
            "llm_used": True,
        }
