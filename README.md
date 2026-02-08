# NG12 Cancer Risk Assessor

A Clinical Decision Support system that uses Google Gemini 2.5 Pro and the NICE NG12
guidelines to assess cancer risk and answer clinical questions.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Frontend (HTML/JS)                     │
│         [Risk Assessment]       [Chat]                   │
└──────────────────┬───────────────────┬───────────────────┘
                   │                   │
            POST /assess          POST /chat
                   │                   │
┌──────────────────▼───────────────────▼───────────────────┐
│     SecurityHeaders + Tracing + CORS + Rate Limiting     │
│              + Prompt Injection Guardrails                │
│  ┌─────────────────────┐  ┌────────────────────────────┐ │
│  │  Risk Assessment     │  │    Chat Agent              │ │
│  │  Agent               │  │    (multi-turn memory)     │ │
│  │  (manual agent loop) │  │    (manual agent loop)     │ │
│  │  + structured output │  │    + grounding guardrails  │ │
│  │  + multi-query RAG   │  │    + corrective RAG        │ │
│  └────┬────┬────────────┘  └────┬────────────────────── │ │
│       │    │                    │                        │ │
│  ┌────▼┐ ┌─▼──────────────────▼──────────────────┐     │ │
│  │Tool:│ │  Tool: search_clinical_guidelines      │     │ │
│  │get_ │ │  (corrective RAG + timeout wrapper)    │     │ │
│  │data │ │  Vector + BM25 → RRF → Cross-encoder   │     │ │
│  └──┬──┘ └──────────┬────────────────────────────┘     │ │
│     │               │                                   │ │
│  ┌──▼──────┐  ┌─────▼──────────┐  ┌──────────────┐     │ │
│  │patients │  │   ChromaDB     │  │   Redis      │     │ │
│  │.json    │  │  (NG12 chunks  │  │  (sessions + │     │ │
│  │         │  │   + embeddings)│  │   RAG cache) │     │ │
│  └─────────┘  └────────────────┘  └──────────────┘     │ │
│                                                         │ │
│  ┌──────────────────────────────────────────────────┐   │ │
│  │  Prometheus /metrics  │  Self-Evaluation (LLM)   │   │ │
│  └──────────────────────────────────────────────────┘   │ │
└──────────────┬──────────────────────────────────────────┘
               │
      ┌────────▼─────────┐       ┌──────────────────┐
      │   Prometheus      │──────▶│   Grafana         │
      │   :9090           │       │   :3000           │
      │   (scrape /metrics│       │   (dashboards)    │
      │    every 5s)      │       │                   │
      └──────────────────┘       └──────────────────┘

      ┌──────────────────┐
      │   Locust          │
      │   :8089           │
      │   (load testing)  │
      └──────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- A Google API key from [Google AI Studio](https://aistudio.google.com/apikey)

### Option A: Run Locally

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd ng12-cancer-risk-assessor

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=your-key-here

# 5. Ingest the NG12 PDF (downloads + builds vector store)
python scripts/ingest_pdf.py

# 6. Start the server
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000 in your browser.

**Chat mode (local):** Click the "Chat with NG12" tab. Redis is optional —
if unavailable, sessions fall back to in-memory storage automatically.

### Option B: Run with Docker

```bash
# 1. Configure your API key
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=your-key-here

# 2. Build and start (includes Redis for persistent sessions)
docker-compose up --build
```

The first start will automatically download the PDF and build the vector store.
Open http://localhost:8000 in your browser.

**Docker hardening:** The image uses a multi-stage build (smaller image size),
runs as a non-root `appuser`, includes a `HEALTHCHECK` directive, and Redis has
its own health check with `restart: unless-stopped` policies.

## API Endpoints

### Part 1 — Risk Assessment

| Method | Path | Description |
|--------|------|-------------|
| GET | `/patients` | List all patients |
| POST | `/assess` | Assess cancer risk for a patient (rate limited, guardrails) |
| GET | `/assess/{patient_id}/evaluation` | Get background evaluation results for an assessment |

**POST /assess** — Request body:
```json
{ "patient_id": "PT-101" }
```

`patient_id` must match `PT-\d{3}` format. Invalid IDs return 400.

### Part 2 — Chat

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Send a chat message (rate limited, guardrails) |
| GET | `/chat/{session_id}/evaluation` | Get background evaluation results for a chat session |
| GET | `/chat/{session_id}/history` | Get conversation history |
| DELETE | `/chat/{session_id}` | Clear a session |

**POST /chat** — Request body:
```json
{
  "session_id": "abc123",
  "message": "What symptoms trigger an urgent referral for lung cancer?",
  "top_k": 5
}
```

Field constraints:
- `session_id`: 1-64 chars, alphanumeric/hyphens/underscores only
- `message`: 1-2000 chars, must not be blank
- `top_k`: 1-20 (default 5)

All chat endpoints include prompt injection detection. Malicious inputs return 400.

### Health Checks

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health/live` | Liveness probe (always 200) |
| GET | `/health/ready` | Readiness probe (200 or 503) |
| GET | `/health` | Full health check (backward-compatible) |
| GET | `/metrics` | Prometheus metrics (if instrumentator installed) |

### Authentication

If `API_KEY` is set in `.env`, all `/assess` and `/chat`
endpoints require an `X-API-Key` header. When unset, endpoints are open
(suitable for local development).

## Project Structure

```
├── app/
│   ├── main.py           # FastAPI endpoints, middleware, guardrails, health checks
│   ├── agent.py          # Part 1: risk assessment (manual agent loop + structured output)
│   ├── chat_agent.py     # Part 2: conversational chat (manual agent loop + multi-turn memory)
│   ├── rag.py            # Hybrid RAG + corrective RAG + timeout protection
│   ├── cache.py          # RAG query cache (Redis or in-memory LRU fallback)
│   ├── guardrails.py     # Prompt injection detection, input validation, medical disclaimer
│   ├── llm_judge.py      # LLM-as-a-Judge evaluation (5 criteria assess, 3 criteria chat)
│   ├── tracing.py        # Correlation IDs, structured JSON logging, agent step tracing
│   ├── patient_data.py   # Simulated patient database
│   ├── models.py         # Pydantic request/response schemas with field constraints
│   └── config.py         # Environment-based configuration
├── scripts/
│   ├── ingest_pdf.py     # PDF download, parse, chunk, embed, store
│   └── run_eval_harness.py # Eval harness: run judges, report metrics, CSV output
├── static/
│   └── index.html        # Two-tab frontend (assessment + chat + disclaimers)
├── tests/
│   ├── test_api.py       # Endpoint + validation + security header tests
│   ├── test_prompt_injection.py # Prompt injection guardrail tests
│   ├── test_corrective_rag.py   # Corrective RAG unit tests
│   ├── test_rag_metadata.py     # Metadata filtering + reranker cache tests
│   ├── test_cache.py            # Cache get/set + warm-up tests
│   ├── test_evaluation.py       # Gold-standard evaluation harness (--run-eval)
│   ├── evaluation_data.py       # Expected outcomes for all 10 patients
│   ├── conftest.py              # Pytest config (--run-eval flag)
│   ├── test_guardrails.py       # Grounding guardrail unit tests
│   ├── test_patient_data.py     # Patient data module tests
│   ├── test_llm_judge.py        # LLM-as-a-Judge framework tests
│   └── evaluation_dataset.py    # Labeled dataset for judge alignment (25 samples, 34 FAIL labels)
├── data/
│   └── patients.json     # 10 simulated patient records
├── monitoring/
│   ├── prometheus/
│   │   └── prometheus.yml      # Prometheus scrape config
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/
│       │   │   └── datasource.yml  # Auto-provision Prometheus datasource
│       │   └── dashboards/
│       │       └── dashboard.yml   # Auto-provision dashboard loader
│       └── dashboards/
│           └── ng12-cancer-risk-assessor.json  # Pre-built dashboard (11 panels)
├── locustfile.py         # Load testing (3 user classes)
├── Dockerfile            # Multi-stage build, non-root user, HEALTHCHECK
├── docker-compose.yml    # App + Redis + Prometheus + Grafana
├── .dockerignore         # Reduce build context
├── entrypoint.sh
├── PROMPTS.md            # Part 1 prompt engineering strategy
├── CHAT_PROMPTS.md       # Part 2 prompt engineering strategy
└── README.md
```

## Design Decisions

1. **Manual agentic loop**: Instead of using `AutomaticFunctionCallingResponder`,
   the Vertex AI path implements an explicit tool-calling loop. This gives full
   visibility into each step (logged), a safety bound (`MAX_AGENT_STEPS`), and
   mirrors production agentic patterns.

2. **Structured JSON output**: The risk assessment agent uses Gemini's
   `response_schema` with `response_mime_type="application/json"` to guarantee
   valid JSON output with enum-constrained risk levels.

3. **Multi-query RAG pre-fetch**: Before the agent loop starts, the system
   generates targeted queries from the patient's symptoms and demographics,
   retrieves relevant chunks, and injects them into the prompt as context.

4. **Hybrid search pipeline**: `search_guidelines()` combines vector search
   (ChromaDB) + BM25 keyword search, merges via Reciprocal Rank Fusion, then
   re-scores with a cross-encoder (`ms-marco-MiniLM-L-6-v2`). All features
   are configurable and degrade gracefully.

5. **Corrective RAG**: When initial retrieval confidence is low (below
   `CORRECTIVE_RAG_THRESHOLD`), the system reformulates the query and retries
   up to `CORRECTIVE_RAG_MAX_RETRIES` times. All unique chunks are merged and
   re-sorted for maximum recall.

6. **Prompt injection guardrails**: Every chat input is scanned against compiled
   regex patterns detecting instruction overrides, role hijacking, system prompt
   extraction, encoding attacks, and delimiter injection. Malicious inputs are
   rejected with a 400 status before reaching the LLM.

7. **Medical disclaimer**: Every response (risk assessment and chat) includes a
   medical disclaimer stating this is not a substitute for professional medical
   advice.

8. **Request tracing**: Every request gets a UUID correlation ID (generated or
   extracted from `X-Correlation-ID` header), propagated via `ContextVar`,
   returned in response headers alongside `X-Response-Time-Ms`.

9. **Structured JSON logging**: When `STRUCTURED_LOGGING_ENABLED=true`, all logs
   are emitted as JSON with timestamp, level, correlation_id, and optional extras
   (patient_id, tool_name, duration_ms, etc.).

10. **Security headers**: `SecurityHeadersMiddleware` adds `X-Content-Type-Options`,
    `X-Frame-Options`, `X-XSS-Protection`, `Referrer-Policy`, and
    `Content-Security-Policy` to every response.

11. **Docker hardening**: Multi-stage build reduces image size. Non-root `appuser`
    prevents container escape. `HEALTHCHECK` directive enables orchestrator probes.
    Redis has its own health check with `restart: unless-stopped` policies.

12. **Evaluation harness**: Gold-standard expected outcomes for all 10 patients.
    Parametrized pytest tests verify risk level, citations, and reasoning keywords.
    Run with `pytest tests/ -v --run-eval`.

13. **Timeout protection**: Both RAG searches and LLM `send_message()` calls are
    wrapped in `ThreadPoolExecutor` with configurable timeouts (`RAG_TIMEOUT_SECONDS`
    and `LLM_TIMEOUT_SECONDS`). On timeout, graceful degradation — RAG returns an
    empty list, LLM raises a clear timeout error.

14. **Pydantic field constraints**: All request models enforce `min_length`,
    `max_length`, regex `pattern`, and range constraints (`ge`/`le`). Invalid
    inputs are rejected at the validation layer before reaching business logic.

15. **Two-phase LLM Judge evaluation**: The safety criterion runs synchronously as a
    gate — if safety fails, the assessment is re-generated with corrective feedback.
    The remaining criteria (faithfulness, correctness, citation accuracy, completeness)
    run asynchronously in a background thread. Results are retrievable via
    `GET /assess/{patient_id}/evaluation` and `GET /chat/{session_id}/evaluation`.
    This reduces response latency by ~15-30 seconds while maintaining full evaluation.

16. **Rate limiting**: slowapi enforces per-IP rate limits (configurable via
    `RATE_LIMIT_ASSESS` and `RATE_LIMIT_CHAT` env vars).

17. **API key auth**: Optional `X-API-Key` header check — disabled when
    `API_KEY` is empty for local development.

18. **RAG query cache**: Repeated queries are served from cache (5-minute TTL).
    Uses Redis (db=1) when available, falls back to an in-memory LRU dict
    (max 200 entries). Cache keys are SHA-256 hashes of `(query, top_k)`.

19. **Async endpoints**: The `/assess` and `/chat` endpoints are `async` and
    offload blocking LLM/RAG calls to `asyncio.run_in_executor()`, keeping the
    event loop responsive under concurrent load.

20. **Prometheus metrics**: When `prometheus-fastapi-instrumentator` is installed,
    a `/metrics` endpoint exposes request counts, latencies, and status code
    distributions in Prometheus format. Gracefully disabled if the package is absent.

21. **LLM-as-a-Judge evaluation**: Based on findings from 24 research papers on
    LLM evaluation. Every `/assess` response is evaluated on 5 binary criteria
    (faithfulness, correctness, citation accuracy, completeness, safety) and
    every `/chat` response on 3 criteria (faithfulness, relevance, citation
    accuracy). Uses Chain-of-Thought prompts, binary PASS/FAIL verdicts, and
    optional cross-examination for faithfulness failures. Safety runs synchronously
    as a gate (with re-generation on failure); remaining criteria run in background
    threads to avoid blocking the response.

## Configuration

All settings are configured via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `vertex_ai` | `vertex_ai` or `google_genai` |
| `GOOGLE_API_KEY` | — | Required for `google_genai` mode |
| `GCP_PROJECT_ID` | — | Required for `vertex_ai` mode |
| `BM25_ENABLED` | `true` | Enable BM25 hybrid search |
| `RERANK_ENABLED` | `true` | Enable cross-encoder reranking |
| `RETRIEVAL_CANDIDATES` | `10` | Candidates before reranking |
| `CORRECTIVE_RAG_ENABLED` | `true` | Enable corrective RAG with retry |
| `CORRECTIVE_RAG_THRESHOLD` | `0.3` | Minimum confidence to accept results |
| `CORRECTIVE_RAG_MAX_RETRIES` | `2` | Max reformulation retries |
| `LLM_TIMEOUT_SECONDS` | `120` | LLM call timeout |
| `RAG_TIMEOUT_SECONDS` | `30` | RAG search timeout |
| `STRUCTURED_LOGGING_ENABLED` | `false` | Enable JSON log output |
| `API_KEY` | — | Optional bearer-token auth |
| `RATE_LIMIT_ASSESS` | `10/minute` | Rate limit for `/assess` |
| `RATE_LIMIT_CHAT` | `30/minute` | Rate limit for `/chat` |
| `MAX_AGENT_STEPS` | `20` | Safety limit for agent loop |
| `REDIS_HOST` | `localhost` | Redis host (auto-fallback if unavailable) |
| `LLM_JUDGE_ENABLED` | `true` | Enable multi-criteria LLM judge evaluation |
| `LLM_JUDGE_TIMEOUT` | `60` | Timeout (seconds) for judge LLM calls |
| `LLM_JUDGE_CROSS_EXAMINE` | `false` | Enable cross-examination follow-up on faithfulness failures |
| `METADATA_FILTER_ENABLED` | `true` | Enable query-aware cancer type metadata filtering in RAG |
| `CACHE_WARMUP_ENABLED` | `true` | Pre-run common queries on startup to warm the cache |

## Using Vertex AI (Production Mode)

To use Google Cloud Vertex AI instead of an API key:

```bash
# In .env:
LLM_PROVIDER=vertex_ai
GCP_PROJECT_ID=your-project-id
GCP_LOCATION=us-central1

# Authenticate:
gcloud auth application-default login
```

Ensure the Vertex AI API is enabled in your GCP project.

## Running Tests

```bash
# Structural tests (no LLM required)
pytest tests/ -v

# Evaluation harness (requires live LLM + vector store)
pytest tests/ -v --run-eval
```

### Test Coverage

| Test File | Count | What It Tests |
|-----------|-------|---------------|
| `test_api.py` | 19 | Endpoint routing, validation, security headers, health checks |
| `test_prompt_injection.py` | 30+ | Guardrail detection, input validation, patient/session ID format |
| `test_corrective_rag.py` | 12 | Confidence scoring, query reformulation |
| `test_rag_metadata.py` | 15 | Metadata filtering, cancer type extraction, filter fallback, reranker cache |
| `test_cache.py` | 13 | Cache key generation, get/set, stats, warm-up function |
| `test_guardrails.py` | 12 | Grounding guardrail logic (relevance, tool-use, citations) |
| `test_patient_data.py` | 5 | Patient loading and field validation |
| `test_llm_judge.py` | 32 | Judge framework: per-criterion evaluators, verdict aggregation, metrics |
| `test_evaluation.py` | 30 | Gold-standard risk level, citations, reasoning keywords (--run-eval) |

## Verification Checklist

1. `pytest tests/ -v` — all structural + guardrail + validation + judge tests pass (146 tests)
2. `pytest tests/ -v --run-eval` — evaluation harness runs against live LLM (30 tests)
3. Prompt injection blocked: `POST /chat` with "ignore all instructions" returns 400
4. `GET /health/live` returns 200
5. `GET /health/ready` returns 200/503 based on service status
6. Response headers include `X-Correlation-ID`, `X-Content-Type-Options`, `X-Frame-Options`
7. `docker-compose up --build` builds with multi-stage, runs as non-root
8. Logs are structured JSON when `STRUCTURED_LOGGING_ENABLED=true`
9. `GET /metrics` returns Prometheus metrics (when instrumentator installed)
10. Risk assessment responses include `evaluation_status` (safety gate + background evaluation)
11. Repeated RAG queries served from cache (check logs for "RAG cache hit")
12. `docker-compose up -d` starts all 4 services (app, redis, prometheus, grafana)
13. Grafana dashboard loads at http://localhost:3000 (admin/admin)
14. Locust load test runs: `locust -f locustfile.py --host http://localhost:8000`

## Load Testing

[Locust](https://locust.io/) is used for load testing with three user classes:

- **HealthCheckUser** (weight=1) — hits `/health/live`, `/health/ready`, `/health`
- **CancerAssessorUser** (weight=3) — realistic workflow: list patients, assess, chat, history
- **RateLimitTestUser** (weight=1) — aggressive requests to verify rate limiting (slowapi)

### Web UI Mode

```bash
locust -f locustfile.py --host http://localhost:8000
```

Open http://localhost:8089 to configure users, spawn rate, and view real-time results.

### Headless Mode

```bash
# 10 users, spawn 2/sec, run for 60 seconds
locust -f locustfile.py --host http://localhost:8000 --headless -u 10 -r 2 --run-time 60s
```

### With API Key

```bash
API_KEY=your-key-here locust -f locustfile.py --host http://localhost:8000
```

## Observability Stack

Prometheus and Grafana are included in `docker-compose.yml` for monitoring.

### Starting the Stack

```bash
docker-compose up -d
```

This starts all services: **app** (:8000), **redis** (:6379), **prometheus** (:9090), **grafana** (:3000).

### Prometheus

- URL: http://localhost:9090
- Scrapes the app's `/metrics` endpoint every 5 seconds
- 7-day data retention

Verify the target is up:
```bash
curl http://localhost:9090/api/v1/targets
```

### Grafana

- URL: http://localhost:3000
- Default credentials: `admin` / `admin`
- Pre-provisioned dashboard: **NG12 Cancer Risk Assessor**

The dashboard includes 11 panels across 4 rows:

| Row | Panels |
|-----|--------|
| Overview | Request Rate by Endpoint, Total Request Rate, Requests In-Flight |
| Latency | p50, p95, p99 Latency by Endpoint |
| Errors | 4xx Error Rate, 5xx Error Rate, Error Ratio (%) |
| Throughput | Request Breakdown (table), Response Size |

## LLM-as-a-Judge Evaluation

Every response from `/assess` and `/chat` is evaluated by independent per-criterion
LLM judges before being returned.  The framework is based on findings from 24
research papers and the scaling methodology from Anshuman Mishra's evaluation series.

### Design Principles

| Research Finding | How We Apply It |
|------------------|-----------------|
| **One evaluator per dimension** | Never a "God Evaluator" — each criterion is a separate LLM call |
| **Binary PASS/FAIL > Likert scales** | Every criterion returns PASS or FAIL, never a 1-5 score |
| **Chain-of-Thought improves accuracy** | Every criterion includes step-by-step reasoning before the verdict |
| **Direct scoring for objective tasks** | All our criteria are objective (faithfulness, correctness) — no pairwise |
| **Cross-examination for faithfulness** | Optional follow-up when faithfulness fails (+6-10% recall) |
| **Classification metrics > correlation** | We measure precision, recall, F1, Cohen's kappa — not Spearman/Pearson |
| **Parallel execution** | All criteria run concurrently via ThreadPoolExecutor |

### Criteria

**`/assess` — 5 criteria:**

| Criterion | What It Checks | FAIL Means |
|-----------|---------------|------------|
| Faithfulness | Every claim in reasoning is supported by retrieved NG12 passages | Model hallucinated or used world knowledge |
| Correctness | Risk level is appropriate given patient data + NG12 evidence | Wrong risk classification |
| Citation Accuracy | Cited passages actually support the conclusion | Citations are irrelevant or decorative |
| Completeness | All patient symptoms are addressed in reasoning | Symptoms were ignored |
| Safety | Recommendations won't cause patient harm | Dangerous under-triage |

**`/chat` — 3 criteria:**

| Criterion | What It Checks | FAIL Means |
|-----------|---------------|------------|
| Faithfulness | Answer grounded in retrieved NG12 passages only | Hallucinated guideline content |
| Relevance | Answer addresses the specific clinical question | Off-topic or tangential |
| Citation Accuracy | Inline [NG12 p.X] citations are accurate | Fabricated page references |

### Response Format

Responses include an `evaluation_status` field. The safety check runs synchronously — if safety
fails, the response is **re-generated with corrective feedback** (not just flagged). The full
multi-criteria evaluation runs in the background:

```json
{
  "risk_level": "HIGH",
  "evaluation_status": "processing",
  "...": "response returned immediately"
}
```

Retrieve the full evaluation later via `GET /assess/{patient_id}/evaluation`:

```json
{
  "patient_id": "PT-101",
  "evaluation_status": "complete",
  "evaluation": {
    "overall_verdict": "PASS",
    "score": "4/5",
    "criteria": {
      "faithfulness":      {"verdict": "PASS", "reasoning": "All claims supported..."},
      "correctness":       {"verdict": "PASS", "reasoning": "Risk level matches..."},
      "citation_accuracy": {"verdict": "PASS", "reasoning": "Citations relevant..."},
      "completeness":      {"verdict": "FAIL", "reasoning": "Fatigue not addressed..."},
      "safety":            {"verdict": "PASS", "reasoning": "Urgent referral is safe..."}
    },
    "critical_issues": ["symptom coverage gap"],
    "cross_examination": null
  }
}
```

### Cross-Examination (Optional)

When `LLM_JUDGE_CROSS_EXAMINE=true` and faithfulness fails, a second LLM call
generates verification questions for each flagged claim and checks them against
the retrieved context.  This improves recall by 6-10% per the research.

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_JUDGE_ENABLED` | `true` | Enable/disable the judge entirely |
| `LLM_JUDGE_TIMEOUT` | `60` | Timeout (seconds) for judge LLM calls |
| `LLM_JUDGE_CROSS_EXAMINE` | `false` | Enable cross-examination on faithfulness failures |

Set `LLM_JUDGE_ENABLED=false` to disable evaluation (e.g. for load testing or
when optimising for latency).

### Evaluation Harness

The eval harness runs the judges against a labeled dataset and reports
precision, recall, F1, and Cohen's kappa per criterion.

```bash
# Full evaluation (requires live LLM)
python scripts/run_eval_harness.py

# Assessment criteria only
python scripts/run_eval_harness.py --mode assess

# Chat criteria only
python scripts/run_eval_harness.py --mode chat

# Use dev/test split (75/25)
python scripts/run_eval_harness.py --split test

# Append results to CSV for experiment tracking
python scripts/run_eval_harness.py --csv eval_results.csv --model gemini-2.5-pro
```

The labeled dataset (`tests/evaluation_dataset.py`) includes both correct
outputs and deliberate failure cases (wrong risk levels, hallucinated claims,
missing symptoms, fabricated citations, dangerous under-triage).

**Experiment workflow:**
1. Tweak a config (prompt, model, RAG parameters)
2. Run the eval harness
3. Check the single-row CSV output
4. Compare across experiments

## Assumptions, Trade-offs & Future Improvements

### Assumptions

- **Single-document scope:** The system is purpose-built for NICE NG12 (95 pages). The
  page-to-section metadata map (`PAGE_SECTION_MAP`) is hardcoded to this specific PDF.
  A production system would need a more general section detection approach.

- **Simulated patient database:** `patients.json` is a static file with 10 records. In
  production, this would be a BigQuery / FHIR / EHR API call. The agent's `get_patient_data`
  tool abstraction makes this a straightforward swap.

- **Google AI Studio as default:** The exercise specifies Vertex AI, but the system supports
  both `vertex_ai` and `google_genai` providers via the `LLM_PROVIDER` config. Google AI
  Studio is the default for easier local development (no GCP project required).

- **Gemini 2.5 Pro over 1.5 Pro:** The exercise specifies Gemini 1.5, but we use Gemini 2.5
  Pro for stronger reasoning on complex clinical assessments. Gemini 2.5 Pro has improved
  function calling reliability, better structured output adherence, and more accurate
  multi-step clinical reasoning — all critical for a decision support system where
  correctness matters. The model is configurable via `GEMINI_MODEL` env var, so switching
  back to `gemini-1.5-pro` requires zero code changes.

- **Embedding model choice:** We use `all-MiniLM-L6-v2` (local, fast, free) rather than
  Vertex AI Embeddings. This avoids API costs during development and keeps the vector store
  self-contained. Switching to Vertex AI Embeddings would require re-ingestion but no code
  changes to the RAG pipeline.

### Trade-offs

- **Manual agent loop vs. AutomaticFunctionCallingResponder:** We implement an explicit
  tool-calling loop instead of using Gemini's automatic function calling. This adds ~50 lines
  of code but gives full visibility into each step (logged with correlation IDs), a safety
  bound (`MAX_AGENT_STEPS`), and the ability to inject pre-fetched RAG context before the
  loop starts. In production agentic systems, this level of control is essential for
  debugging and auditing.

- **Cross-encoder reranker on CPU:** The `ms-marco-MiniLM-L-6-v2` cross-encoder adds 2-4
  seconds of latency per query on CPU. We mitigated this with metadata pre-filtering (fewer
  candidates), reranker score caching (skip re-scoring on corrective RAG retries), and cache
  warm-up (pre-populate common queries at startup). On a GPU instance, reranking would drop
  to ~100ms. The feature is flag-gated (`RERANK_ENABLED`) so it can be disabled entirely if
  latency is critical.

- **Two-phase LLM Judge evaluation:** To avoid blocking responses for 15-30 seconds, we split
  the evaluation into two phases: (1) a synchronous safety gate (~3-5s, single LLM call) that
  triggers re-generation if the assessment is unsafe, and (2) a background thread running the
  remaining 4 criteria (faithfulness, correctness, citation accuracy, completeness). Results
  are retrievable asynchronously via dedicated evaluation endpoints. The trade-off is that the
  full evaluation isn't immediately visible, but the safety-critical check is always synchronous.

- **Binary PASS/FAIL over Likert scales:** Research (G-Eval, Prometheus) shows binary
  verdicts are more reliable and reproducible than 1-5 scores. The trade-off is less
  granularity, but for a clinical system we care about "is this safe?" not "how good
  is this on a scale?"

- **In-memory BM25 index:** The BM25 index loads all chunks into memory on first query.
  For 95 pages (~100 chunks), this is negligible. For a larger corpus (thousands of
  guidelines), an external search engine (Elasticsearch) would be more appropriate.

### What I Would Improve with More Time

- **CI/CD pipeline:** Add a GitHub Actions workflow that runs `pytest` on push, builds the
  Docker image, and optionally runs the evaluation harness on a schedule. Currently, all
  testing is manual.

- **Latency breakdown in responses:** Add `rag_latency_ms`, `rerank_latency_ms`, and
  `llm_latency_ms` to every API response so users (and monitoring) can see exactly where
  time is spent.

- **Multi-guideline support:** Generalize the ingestion pipeline to support multiple PDFs
  (e.g., NG12 + NG85 + CG27) with a `guideline` metadata field. The RAG pipeline would
  then support filtering by guideline, not just cancer type.

- **Semantic chunking:** The current chunker uses fixed-size paragraphs (1000 chars, 200
  overlap). A semantic chunker that splits on topic boundaries (e.g., using embedding
  similarity between paragraphs) would produce more coherent chunks, especially for the
  symptom tables section.

- **Async LLM calls:** Currently, LLM calls are synchronous (offloaded to executor threads).
  Using the Gemini async API natively would reduce thread overhead under high concurrency.

- **User authentication:** The API key auth is a simple shared secret. A production system
  would use OAuth 2.0 / JWT with role-based access control (clinician vs. admin vs. audit).

- **Audit trail:** Clinical decision support systems require an immutable audit log of every
  assessment. Adding a write-ahead log (or sending events to BigQuery) with the full
  request, response, RAG context, and evaluation results would satisfy regulatory
  requirements.

## Latency & Performance

### Where the Time Goes

The primary latency bottleneck is **Gemini 2.5 Pro API response time**, not the application
code. Each LLM call takes ~10-20 seconds depending on prompt complexity and Google's server
load. The agentic loop makes 2-4 LLM calls per request (tool use decisions + final answer
generation), resulting in ~20-30 seconds of unavoidable API wait time.

| Component | Latency | Controllable? |
|-----------|---------|---------------|
| RAG retrieval (warm cache) | ~5ms | Already optimized |
| RAG retrieval (cold) | 2-4s | Already optimized |
| Gemini API call (tool use) | 10-15s | No — external API |
| Gemini API call (final answer) | 10-15s | No — external API |
| LLM Judge (background) | 0s impact | Moved to background thread |
| **Total response time** | **~20-30s** | **~95% is Gemini API time** |

### Optimizations Applied

We've optimized everything within our control:

1. **Metadata-aware retrieval** — ChromaDB `where` filters narrow the search space by cancer
   type, reducing vector search from 95+ chunks to ~10-15 relevant chunks
2. **Reranker score caching** — cross-encoder scores are cached per `(query_hash, chunk_id)`
   pair, eliminating redundant scoring on corrective RAG retries
3. **Cache warm-up** — 10 common cancer-type queries are pre-run at startup, so subsequent
   matching queries are served from cache in ~5ms
4. **Background LLM Judge** — full evaluation (5 criteria for assess, 3 for chat) runs in a
   background thread. Only the safety check runs synchronously (~3-5s, single LLM call)
5. **Safety-driven re-generation** — if safety fails, the system feeds the safety reasoning
   back to Gemini as corrective feedback and re-generates (rather than returning an unsafe response)
6. **Reduced reranker candidates** — from 15 to 10, cutting cross-encoder latency from 8-12s to 2-4s

### How to Further Reduce Latency (Production)

- **Gemini Flash** — switch from `gemini-2.5-pro` to `gemini-2.0-flash` for 3-5x faster
  inference (set `GEMINI_MODEL=gemini-2.0-flash` in `.env`)
- **Single-shot prompting** — skip the agentic tool-use loop and inject RAG context directly
  into the prompt (1 API call instead of 2-4)
- **GPU-accelerated reranking** — the CPU cross-encoder adds 2-4s; on GPU this drops to ~100ms
- **True token streaming** — stream tokens as they generate for perceived responsiveness
- **Async Gemini API** — use native async SDK calls instead of thread pool executors
