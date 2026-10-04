"""
app/core/config.py
==================
Central configuration using Pydantic BaseSettings.
All values are read from environment variables (or .env file).
Never hardcode secrets or keys anywhere else in the codebase.

Usage:
    from app.core.config import settings
    print(settings.s3_bucket)
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root — the directory containing this file's grandparent
ROOT_DIR = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT_DIR / "params.yaml"


def _load_params() -> dict:
    """Load params.yaml once and return as dict."""
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


# ---------------------------------------------------------------------------
# Main settings class
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables / .env file.
    Pydantic validates types and raises clear errors on misconfiguration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -----------------------------------------------------------------------
    # Application
    # -----------------------------------------------------------------------
    app_env: Literal["local", "aws"] = Field(default="local", description="Deployment environment")
    app_version: str = Field(default="1.0.0")
    log_level: str = Field(default="INFO")

    # -----------------------------------------------------------------------
    # AWS
    # -----------------------------------------------------------------------
    aws_region: str = Field(default="us-east-1")
    aws_s3_bucket: str = Field(default="")
    aws_access_key_id: str = Field(default="")
    aws_secret_access_key: str = Field(default="")

    # -----------------------------------------------------------------------
    # MLflow
    # -----------------------------------------------------------------------
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    mlflow_artifact_root: str = Field(default="./mlruns")

    # -----------------------------------------------------------------------
    # LLM
    # -----------------------------------------------------------------------
    llm_provider: Literal["openai", "ollama", "huggingface", "gemini"] = Field(default="ollama")
    llm_model: str = Field(default="llama3.2:latest")
    gemini_api_key: str = Field(default="")
    openai_api_key: str = Field(default="")
    ollama_base_url: str = Field(default="http://localhost:11434")
    allow_test_mock: bool = Field(default=False)



    # -----------------------------------------------------------------------
    # Embedding
    # -----------------------------------------------------------------------
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_device: str = Field(default="cpu")

    # -----------------------------------------------------------------------
    # Server / Host
    # -----------------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # -----------------------------------------------------------------------
    # Retrieval & Relevance Filtering
    # -----------------------------------------------------------------------
    retrieval_method: str = Field(default="semantic")
    top_k: int = Field(default=5)
    final_context_k: int = Field(default=3)
    relevance_threshold: float = Field(default=0.2)
    hybrid_alpha: float = Field(default=0.5)

    # -----------------------------------------------------------------------
    # ChromaDB / Vector Store
    # -----------------------------------------------------------------------
    chroma_persist_dir: str = Field(default="./data/vectorstore")
    vector_db_path: str = Field(default="./data/vectorstore")
    chroma_collection_name: str = Field(default="ragops_v1")

    # -----------------------------------------------------------------------
    # Generation / RAG
    # -----------------------------------------------------------------------
    groundedness_threshold: float = Field(default=0.75)

    # -----------------------------------------------------------------------
    # Drift
    # -----------------------------------------------------------------------
    drift_psi_threshold: float = Field(default=0.2)
    drift_ks_p_value: float = Field(default=0.05)

    # -----------------------------------------------------------------------
    # CloudWatch
    # -----------------------------------------------------------------------
    cloudwatch_log_group: str = Field(default="/ragops/api")
    cloudwatch_log_stream: str = Field(default="application")

    # -----------------------------------------------------------------------
    # SageMaker
    # -----------------------------------------------------------------------
    sagemaker_role_arn: str = Field(default="")
    sagemaker_instance_type: str = Field(default="ml.t3.medium")

    # -----------------------------------------------------------------------
    # Derived properties
    # -----------------------------------------------------------------------

    @property
    def is_aws_mode(self) -> bool:
        return self.app_env == "aws"

    @property
    def params(self) -> dict:
        """Return parsed params.yaml as a dict."""
        return _load_params()

    @property
    def chroma_persist_path(self) -> Path:
        p = Path(self.chroma_persist_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v.upper()


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns the cached Settings singleton.
    Use this everywhere instead of constructing Settings() directly.
    """
    return Settings()


# Convenience alias — import `settings` directly where needed
settings: Settings = get_settings()
