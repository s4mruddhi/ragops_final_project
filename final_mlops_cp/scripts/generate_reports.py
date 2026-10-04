"""
Regenerate the evidence documents from the REAL result files so docs can never drift from reality.

    python scripts/generate_reports.py

Writes: docs/performance_report.md, docs/drift_report.md, docs/model_card.md, docs/data_card.md
(responsible_ai.md, aws_deployment.md, demo_script.md are hand-written and not overwritten).
"""
from __future__ import annotations

import collections
import json
from datetime import date
from pathlib import Path

import jsonlines
import yaml

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "data" / "evaluation" / "reports"
DOCS = ROOT / "docs"


def jl(p, default=None):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
ret = {k: v for k, v in jl(R / "retrieval_metrics.json", {}).items() if not k.startswith("_")}
rag = jl(R / "rag_metrics.json", {}).get("summary", {})
gate = jl(R / "quality_gate.json", {})
champ = jl(R / "champion.json", {})
bench = jl(DOCS / "benchmark_results.json")
drift = jl(ROOT / "data/monitoring/drift_report.json")
idx = jl(ROOT / "data/vectorstore/index_stats.json", {})
manifest = jl(ROOT / "data/processed/ingest_manifest.json", {})
backend = idx.get("embedding_backend", "unknown")
today = date.today().isoformat()
champ_name = champ.get("retriever", "n/a")

banner = ""
if backend == "hashing":
    banner = ("> **WARNING - fallback embeddings.** This report was produced where the HuggingFace model could not be "
              "downloaded, so the *semantic/hybrid* rows use deterministic hashing embeddings, **not** "
              "`all-MiniLM-L6-v2`. Re-run `python scripts/run_dag_pipeline.py` with internet access and regenerate.\n\n")


def row(m, s):
    return (f"| {m} | {s['mean_recall_at_1']:.4f} | {s['mean_recall_at_5']:.4f} | {s['mean_precision_at_5']:.4f} | "
            f"{s['mean_mrr']:.4f} | {s['mean_ndcg_at_5']:.4f} | {s['avg_latency_ms']:.2f} | {s['p95_latency_ms']:.2f} |")


# ------------------------------------------------------------------ performance report
lines = [f"# Performance Report\n", f"> Generated {today} by `scripts/generate_reports.py` from "
         f"`data/evaluation/reports/*.json` and `docs/benchmark_results.json`.\n\n", banner,
         "## 1. Experiment comparison (32 gold questions, retrieval on document-level labels)\n\n",
         "| Retriever | Recall@1 | Recall@5 | Precision@5 | MRR | NDCG@5 | Mean latency ms | p95 latency ms |\n",
         "|---|---|---|---|---|---|---|---|\n"]
for m, s in ret.items():
    tag = f"{m} (baseline)" if m == params["mlflow"]["baseline_method"] else m
    tag = f"**{tag} - champion**" if m == champ_name else tag
    lines.append(row(tag, s) + "\n")
lines.append(f"\nBaseline = `{params['mlflow']['baseline_method']}`; candidates = the other three. Champion is chosen by "
             f"(Recall@5, MRR), registered in MLflow as `{champ.get('name', params['mlflow']['registered_model_name'])}` "
             f"**v{champ.get('version', '?')}** with alias `{champ.get('alias', 'champion')}`.\n\n")
lines.append("## 2. Business + model metrics (required by the project brief)\n\n| Metric | Value | Meaning |\n|---|---|---|\n")
ch = ret.get(champ_name, {})
lines += [
    f"| **Model metric** - Recall@5 / MRR | {ch.get('mean_recall_at_5', 'n/a')} / {ch.get('mean_mrr', 'n/a')} | Correct policy document is in the top 5 chunks |\n",
    f"| **Business metric** - answerable questions answered (1 - false-abstention) | {1 - rag.get('false_abstention_rate_answerable', 0):.2%} | Employee gets a usable cited answer instead of being sent to HR |\n",
    f"| **Business metric** - unanswerable questions correctly refused | {rag.get('abstention_rate_unanswerable', 0):.2%} | Hallucination avoided (n=3 questions - small sample) |\n",
    f"| Groundedness / citation correctness | {rag.get('mean_groundedness', 'n/a')} / {rag.get('mean_citation_correctness', 'n/a')} | Lexical-overlap groundedness (see limitations) |\n",
]
if bench:
    lines += [f"| **p95 inference latency** (API, c={bench['concurrency']}) | {bench['latency_ms']['p95']} ms | End-to-end `/query` |\n",
              f"| **Throughput** | {bench['throughput_rps']} req/s | {bench['requests']} requests, {bench['concurrency']} concurrent clients |\n",
              f"| **Error rate** | {bench['error_rate']:.2%} | {bench['errors']} failed of {bench['requests']} |\n"]
else:
    lines.append("| p95 latency / throughput / error rate | *run `make load-test`* | `scripts/benchmark/load_test.py` writes docs/benchmark_results.json |\n")
if drift:
    dd = drift["drifted_window"]
    top = max((v for v in dd["features"].values() if v.get("psi") is not None), key=lambda v: v["psi"], default=None)
    lines.append(f"| **Drift metric** - max PSI on simulated shift | {top['psi'] if top else 'n/a'} (threshold {params['drift']['psi_threshold']}) | See drift_report.md |\n")
lines.append("\n## 3. Quality gate (promotion threshold)\n\n| Check | Value | Threshold | Result |\n|---|---|---|---|\n")
for k, v in gate.get("checks", {}).items():
    op = "<=" if "latency" in k else ">="
    lines.append(f"| {k} | {v['value']} | {op} {v['threshold']} | {'PASS' if v['pass'] else 'FAIL'} |\n")
lines.append(f"\n**Overall: {'PASSED - model promoted to alias `champion`' if gate.get('passed') else 'FAILED - promotion blocked'}**\n\n")
lines.append("""## 4. Honest limitations (say these in the viva before the examiner does)

* **Gold set is small** (32 questions, 3 unanswerable) - differences of a few points between retrievers are within noise.
* **Default generator is extractive** (no LLM key): it returns the best-matching sentence, so groundedness is naturally high.
  Groundedness is a *lexical-overlap* heuristic, not an NLI/LLM judge.
* **Abstention** on unanswerable questions is the weakest point; the relevance rule trades false-abstention against refusal.
  Plugging in an LLM (`OPENAI_API_KEY`) is the intended fix.
* **Benchmark** numbers depend on the machine; rerun `make load-test` on the demo machine and regenerate this report.
""")
(DOCS / "performance_report.md").write_text("".join(lines), encoding="utf-8")

# ------------------------------------------------------------------ drift report
if drift:
    d0, d1 = drift["no_drift_window"], drift["drifted_window"]
    L = [f"# Drift Report\n\n> Generated {today} from `data/monitoring/drift_report.json` (`scripts/monitoring/simulate_drift.py`).\n\n", banner,
         f"**Setup.** Reference = {drift['reference_queries']} gold questions through the `{drift['retriever']}` retriever. "
         f"Window A = {drift['window_size']} resampled normal queries. Window B = {drift['window_size']} *shifted* queries "
         "(1/3 off-topic, 1/3 very long, 1/3 typo-heavy). Method: PSI (threshold "
         f"{drift['thresholds']['psi_threshold']}) and two-sample KS test (alpha {drift['thresholds']['ks_alpha']}).\n\n",
         "| Feature | PSI normal | PSI shifted | KS p (shifted) | Drift flagged (shifted) |\n|---|---|---|---|---|\n"]
    for f in d1["features"]:
        a, b = d0["features"][f], d1["features"][f]
        L.append(f"| {f} | {a['psi']} | {b['psi']} | {b['ks_p']} | {'YES' if b['drift'] else 'no'} |\n")
    L.append(f"\n**Result:** normal window flagged = **{d0['drift_detected']}** (no false alarm); shifted window flagged = **{d1['drift_detected']}**.\n\n")
    L.append("""**Action policy.** Prometheus alert `InputDriftDetected` (PSI > 0.2) -> review new queries -> add/refresh documents in
`data/raw` (DVC version bump) -> re-run the Airflow DAG (`ragops_pipeline`) -> quality gate -> new MLflow model version.
Live view: `GET /api/v1/drift`, Grafana panels *Feature PSI* and *Drift detected*.

**Caveat.** PSI is biased upward on small samples; bins are therefore adaptive (>=10 reference points per bin) and the
detector needs >= 30 samples per window.
""")
    (DOCS / "drift_report.md").write_text("".join(L), encoding="utf-8")

# ------------------------------------------------------------------ data card
qs = list(jsonlines.open(ROOT / params["evaluation"]["dataset_path"]))
cat = collections.Counter(q["category"] for q in qs)
(DOCS / "data_card.md").write_text(f"""# Data Card

**Name:** RAGOps synthetic enterprise policy corpus ({params['dataset']['current_version']})  **Generated:** {today}

## Composition
* **{manifest.get('num_documents', '?')} documents** (plain text) in `data/raw/` across HR, IT, Product, Travel, plus one *adversarial* prompt-injection
  test document (`IT-ADVERSARIAL-TEST`). Each has a JSON metadata sidecar (id, title, department, version, effective date, access level).
* **{idx.get('total_chunks', '?')} chunks** after cleaning + chunking (size {params['chunking']['chunk_size']} chars, overlap {params['chunking']['chunk_overlap']}).
  Department split: {idx.get('department_distribution', {})}.
* **Gold evaluation set:** {len(qs)} questions - {dict(cat)} - each with expected document ids and a reference answer.

## Provenance and versioning
* **Synthetic**: written by `scripts/generate_synthetic_docs.py`; contains **no real personal or company data**. Policy numbers are fictional.
* Versioned with **DVC** (`data/raw.dvc`, md5 `{manifest.get('dvc_md5', 'n/a')}`); `scripts/evaluation/ingest_documents.py` writes a SHA-256 manifest per run.
  `scripts/generate_synthetic_docs_v2.py` adds documents to demonstrate a v1 -> v2 dataset change.

## Known limitations / biases
* Synthetic text is cleaner and more consistent than real policies; real-world performance will be lower.
* Single language (English); documents are short, so chunk-level retrieval is easier than on long PDFs.
* Evaluation questions were written by the same author as the documents (lexical overlap is optimistic).
* One adversarial document is deliberately included to test prompt-injection resistance; it must never be treated as an instruction.

## Privacy
No PII. If real HR documents are ever used: restrict by `access_level`, redact personal data before indexing, and log queries without user identifiers.
""", encoding="utf-8")

# ------------------------------------------------------------------ model card
(DOCS / "model_card.md").write_text(f"""# Model Card - RAGOps retrieval + answer pipeline

**Registered model:** `{champ.get('name', params['mlflow']['registered_model_name'])}` v{champ.get('version', '?')} (alias `{champ.get('alias', 'champion')}`)  |  **Date:** {today}

## What the "model" is
RAG is a pipeline, not one network. The MLflow-registered artifact is the **champion configuration**:
retriever = **{champ_name}**, top_k = {params['retrieval']['top_k']}, embedding model = `{params['embedding']['model']}`,
chunking = {params['chunking']['chunk_size']}/{params['chunking']['chunk_overlap']}, prompt = `{params['generation']['prompt_version']}`.
Generation is extractive by default; an OpenAI/Ollama LLM can be enabled via `.env`.

## Intended use
Answer employee questions **only from verified organizational documents**, with citations; abstain when the context does not support an answer.
Decision support for staff - **not** an authority on legal/HR decisions.

## Out of scope
Medical/legal advice, questions outside the document set, any use on confidential data without access controls.

## Experiments (retrieval, 32 questions)
| Retriever | Recall@5 | MRR |
|---|---|---|
""" + "".join(f"| {m}{' (baseline)' if m == params['mlflow']['baseline_method'] else ''} | {s['mean_recall_at_5']} | {s['mean_mrr']} |\n" for m, s in ret.items()) + f"""
## Quality gate (blocks promotion)
{'PASSED' if gate.get('passed') else 'FAILED'} - recall@5, precision@5, MRR, groundedness, citation correctness, latency, abstention (thresholds in `params.yaml`).

## Limitations
Small gold set; lexical groundedness proxy; extractive answers are terse; weak refusal on out-of-scope questions (see performance_report.md section 4);
semantic quality depends on `all-MiniLM-L6-v2`. Prompt injection inside documents is tested with `IT-ADVERSARIAL-TEST` and must be handled as data.

## Monitoring & retraining
Prometheus/Grafana (latency, errors, fallback share, groundedness, PSI drift); `/feedback` ratings are the label signal. PSI > {params['drift']['psi_threshold']}
raises an alert -> re-run the Airflow DAG.

## Reproducibility
Seed {params['seed']}; DVC-versioned data; params in `params.yaml`; MLflow run ids for every experiment.
""", encoding="utf-8")
print("reports written")
