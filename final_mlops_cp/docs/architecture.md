# RAGOps: Monitored Enterprise Knowledge Assistant — Architecture

> **Version:** 1.0  
> **Date:** 2026-10-02  
> **Author:** Student Project — B.Tech AI & DS  

---

## 1. Overall RAGOps Architecture

```mermaid
flowchart TD
    A["📄 Verified Organizational Documents\n(PDF / DOCX / TXT)"] --> B["DVC Versioning\n(data/raw/)"]
    B --> C["AWS S3\ns3://bucket/raw-documents/"]
    C --> D["Apache Airflow DAG\nragops_pipeline"]
    D --> E["Document Validation\n(type · readability · metadata · duplicates)"]
    E --> F["Text Extraction\n(PyMuPDF / python-docx)"]
    F --> G["Text Cleaning & Chunking\n(configurable size + overlap)"]
    G --> H["Embedding Generation\n(sentence-transformers)"]
    H --> I["ChromaDB Vector Store\n(persistent · versioned)"]

    I --> J["Retrieval Layer\nTF-IDF | BM25 | Semantic | Hybrid"]
    J --> K["Reranker (optional)"]
    K --> L["Top-K Context Chunks"]
    L --> M["Prompt Assembly\n(versioned prompts/)"]
    M --> N["LLM\n(OpenAI / Ollama / HuggingFace)"]
    N --> O["Answer + Citations\n+ Groundedness Score"]

    O --> P["FastAPI Service\n(Dockerised)"]
    P --> Q["Prometheus\n/metrics"]
    Q --> R["Grafana Dashboard"]
    P --> S["AWS CloudWatch\n(logs + alarms)"]

    H --> T["MLflow Experiment Tracking\n(metrics · params · artifacts)"]
    T --> U["MLflow Model Registry\n(retrieval config versioning)"]
    U --> V["Quality Gate\n(Recall@K · Groundedness · Latency)"]
    V -->|"PASS"| I
    V -->|"FAIL"| W["Block & Alert"]

    I --> X["Drift Detector\n(PSI / KS test)"]
    X --> Y["Re-indexing Decision"]

    subgraph AWS_CLOUD ["AWS Cloud"]
        C
        S
        Z["AWS SageMaker\nRetrieval Evaluation Job"]
        AA["AWS S3\nMLflow Artifacts / DVC Remote"]
    end
```

---

## 2. Document Ingestion Pipeline

```mermaid
flowchart LR
    A["raw docs\ndata/raw/"] --> B["ingest_documents\n(list + checksum)"]
    B --> C["validate_documents\n(type/size/metadata/duplicates)"]
    C -->|invalid| D["validation_report.json\nto S3"]
    C -->|valid| E["extract_text\n(PDF/DOCX/TXT)"]
    E --> F["clean_text\n(normalise whitespace)"]
    F --> G["chunk_documents\n(chunk_size/overlap/metadata)"]
    G --> H["generate_embeddings\n(sentence-transformer batch)"]
    H --> I["update_vector_db\n(ChromaDB upsert)"]
    I --> J["evaluate_retrieval\n(Recall@K/Precision@K/MRR)"]
    J --> K["evaluate_rag\n(groundedness/citation)"]
    K --> L["quality_gate\n(threshold enforcement)"]
    L -->|pass| M["register_version\n(MLflow Registry)"]
    L -->|fail| N["BLOCK and log failure"]
```

---

## 3. Retrieval Pipeline

```mermaid
flowchart TD
    Q["User Question"] --> QE["Query Embedding\n(same model as indexing)"]
    QE --> R1["TF-IDF Retriever"]
    QE --> R2["BM25 Retriever"]
    QE --> R3["Semantic Retriever\n(ChromaDB cosine similarity)"]
    R1 & R2 & R3 --> RR["Reranker / Score Fusion\n(configurable)"]
    RR --> TK["Top-K Chunks\n(document_id/chunk_id/text/metadata)"]
    TK --> PA["Prompt Assembly\n(versioned template)"]
    PA --> LLM["LLM Generation"]
    LLM --> ANS["Answer + Citations + Groundedness"]
```

---

## 4. Airflow DAG

```mermaid
flowchart LR
    T1["ingest_documents"] --> T2["validate_documents"]
    T2 --> T3["extract_text"]
    T3 --> T4["clean_text"]
    T4 --> T5["chunk_documents"]
    T5 --> T6["generate_embeddings"]
    T6 --> T7["update_vector_database"]
    T7 --> T8["evaluate_retrieval"]
    T8 --> T9["evaluate_rag"]
    T9 --> T10["quality_gate"]
    T10 --> T11["register_version"]
```

---

## 5. MLflow Experiment Lifecycle

```mermaid
flowchart TD
    A["Airflow evaluate_retrieval task"] --> B["MLflow Run Start\n(run_name/tags/params)"]
    B --> C["Log Parameters\n(embedding_model/chunk_size/chunk_overlap\ntop_k/retrieval_method/prompt_version)"]
    C --> D["Execute Retrieval Experiment\n(TF-IDF / BM25 / Semantic)"]
    D --> E["Log Metrics\n(Recall@K/Precision@K/MRR/latency\ngroundedness/citation_correctness)"]
    E --> F["Log Artifacts\n(eval_report.json/plots/config)"]
    F --> G["Quality Gate Check\nthresholds from params.yaml"]
    G -->|pass| H["Register to MLflow Model Registry\n(alias: candidate or champion)"]
    G -->|fail| I["Log Failure\n(do NOT register)"]
```

---

## 6. AWS Architecture

```mermaid
flowchart TD
    subgraph LOCAL ["Local / CI"]
        L1["Docker Compose\n(ragops-api / prometheus / grafana)"]
        L2["ChromaDB local persistent"]
        L3["Airflow local"]
    end

    subgraph AWS ["AWS"]
        S3["S3 Bucket\nraw-docs / processed / dvc / mlflow\nembeddings / vector-index / evaluation\nmodels / reports"]
        SM["SageMaker\nRetrieval Evaluation Job"]
        EC2["EC2 or ECS\nFastAPI Docker Container"]
        CW["CloudWatch\nLogs / Metrics / Alarms"]
        IAM["IAM Roles\n(no hardcoded keys)"]
    end

    L1 -->|"dvc push/pull"| S3
    L1 -->|"mlflow artifacts"| S3
    L3 -->|"document artifacts"| S3
    SM -->|"read versioned data"| S3
    SM -->|"write metrics/artifacts"| S3
    EC2 --> CW
    IAM --> S3
    IAM --> SM
    IAM --> EC2
```

---

## 7. Monitoring Architecture

```mermaid
flowchart LR
    API["FastAPI /metrics\n(prometheus_client)"] --> PROM["Prometheus\nScrape :9090"]
    PROM --> GRAF["Grafana Dashboard\n:3000"]
    API --> CW["AWS CloudWatch\n(boto3 structured logs)"]
    CW --> AL["CloudWatch Alarms\n(error_rate / latency)"]
```

---

## 8. Drift Detection Architecture

```mermaid
flowchart TD
    REF["Reference Window\n(production queries Day 1-7)"] --> DD["DriftDetector\n(app/monitoring/drift.py)"]
    CUR["Current Window\n(recent queries last 24h)"] --> DD
    DD --> F1["Query Length Distribution\n(KS test)"]
    DD --> F2["Query Vocabulary Distribution\n(PSI)"]
    DD --> F3["Retrieval Score Distribution\n(PSI)"]
    DD --> F4["Groundedness Distribution\n(PSI)"]
    DD --> F5["Retrieved Doc Distribution\n(PSI)"]
    F1 & F2 & F3 & F4 & F5 --> RPT["drift_report.md\n(detected/not-detected / metric / threshold)"]
    RPT -->|"drift detected"| ALERT["Alert: Consider Re-indexing"]
    RPT --> PROM["Prometheus Gauge\nrag_drift_detected"]
```

---

## Component Ownership Matrix

| Component | Technology | Environment | Purpose |
|---|---|---|---|
| Raw Doc Storage | Local FS + DVC + S3 | Both | Document versioning |
| Vector Store | ChromaDB | Both | Embedding search |
| Retrieval | TF-IDF / BM25 / Semantic | Both | Candidate generation |
| Reranker | Cross-encoder (optional) | Both | Result reranking |
| LLM | OpenAI / Ollama | Configurable | Answer generation |
| Experiment Tracking | MLflow | Both | Metric/param logging |
| Pipeline Orchestration | Apache Airflow | Both | Reproducible pipeline |
| API Service | FastAPI + Uvicorn | Both | Inference endpoint |
| Containerisation | Docker + Compose | Both | Reproducible deploy |
| App Monitoring | Prometheus + Grafana | Both | ML observability |
| Cloud Monitoring | AWS CloudWatch | AWS | Infra/log observability |
| ML Cloud Workload | AWS SageMaker | AWS | Retrieval eval job |
| Artifact Storage | AWS S3 | AWS | Remote artifacts |
| CI/CD | GitHub Actions | CI | Automated testing |
| Drift Detection | PSI / KS (custom) | Both | Query/retrieval drift |
