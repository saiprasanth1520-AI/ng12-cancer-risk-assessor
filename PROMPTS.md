# Prompt Engineering Strategy — Risk Assessment Agent

## Overview

The risk assessment agent (Part 1) uses a carefully structured system prompt with
Gemini 2.5 Pro's function-calling capability to produce deterministic, evidence-based
cancer risk assessments.

## System Prompt Design

### 1. Role Definition
The prompt opens with a clear identity:
> "You are a Clinical Decision Support Agent that evaluates cancer risk using
> the NICE NG12 guidelines."

This anchors the model in the clinical domain and prevents it from acting as a
general-purpose assistant.

### 2. Explicit Workflow
The prompt prescribes a step-by-step procedure:
1. **Retrieve patient data** via `get_patient_data` tool
2. **Search guidelines per symptom** — the prompt instructs the model to search
   for *each* symptom individually, improving retrieval relevance
3. **Cross-reference** patient context (age, gender, smoking, duration) with
   guideline text
4. **Decide** on risk level

This enforced sequence ensures the agent gathers evidence before reasoning.

### 3. Constrained Output Vocabulary
The prompt defines exactly four risk levels:
- "Urgent Referral (2-week wait)"
- "Urgent Investigation"
- "Non-urgent"
- "No cancer indicators"

By limiting choices, we make the output predictable and parseable.

### 4. Structured JSON Output (Gemini `response_schema`)
Instead of relying on prompt instructions alone, the Vertex AI path uses
`GenerationConfig(response_mime_type="application/json", response_schema=...)`
to enforce a strict JSON schema at the API level. The schema mirrors the
`RiskAssessmentResponse` model and includes an `enum` constraint on `risk_level`.

This guarantees valid JSON output and eliminates the need for markdown-fence
stripping — though a fallback parser is retained for robustness on the
`google_genai` path.

### 5. Grounding Rules
Key instructions:
- *"Base your assessment ONLY on retrieved guideline text."*
- *"Never invent criteria."*
- *"If the guidelines are unclear, state that."*
- *"Also consider the pre-fetched guideline passages provided in the initial context."*

These reduce hallucination risk for clinical content.

## Tool Definitions

Two tools are defined with explicit `FunctionDeclaration` schemas for full
control over parameter descriptions and types:

| Tool | Purpose |
|------|---------|
| `get_patient_data(patient_id)` | Fetches structured patient record |
| `search_clinical_guidelines(query)` | RAG search over NG12 vector store |

## Manual Agentic Loop

The Vertex AI path uses a **manual tool-calling loop** instead of
`AutomaticFunctionCallingResponder`. This provides:

1. **Full visibility** — every function call (name, args) and response is logged
2. **Safety bound** — `MAX_AGENT_STEPS` (default 20) prevents runaway loops
3. **Custom dispatch** — a `tool_map` dict routes function calls to Python callables
4. **Production readiness** — the pattern mirrors how real agentic systems work

The loop checks each response for `function_call` parts. If found, it executes
the function, sends `Part.from_function_response()` back, and repeats. When the
model returns text (no function calls), the loop terminates.

The `google_genai` fallback retains `enable_automatic_function_calling=True`
for simplicity.

## Multi-Query RAG (Pre-fetch)

Before the agentic loop starts, the system performs **deterministic
multi-query RAG** based on the patient's record:

1. **Per-symptom queries**: `"{symptom} cancer referral criteria NG12"` for each
   symptom in the patient record
2. **Demographic query**: `"{age}yo {gender} {smoking_status} cancer risk NG12"`

All unique chunks from these queries are:
- Collected as tracked citations
- Formatted as numbered passages and injected into the user prompt

This means the model has relevant guideline context from the very first turn,
even before it calls any tools. The prompt instructs the model to still call
`search_clinical_guidelines` for thorough coverage.

## Hybrid Search Pipeline

The `search_guidelines()` function in `app/rag.py` supports three modes:

1. **Vector-only** (fallback) — standard ChromaDB semantic search
2. **Hybrid: BM25 + Vector** — keyword and semantic search merged via
   Reciprocal Rank Fusion (RRF): `score = weight * 1/(rank + k)`
3. **Hybrid + Cross-encoder reranking** — a `CrossEncoder` model re-scores
   the merged candidates for maximum relevance

All features are gated behind config flags (`BM25_ENABLED`, `RERANK_ENABLED`)
for graceful degradation.

## Citation Tracking

Citations are tracked via a closure pattern: the `search_clinical_guidelines`
function appends every retrieved chunk to a list captured in the calling scope.
After the agent finishes, these chunks are de-duplicated and returned alongside
the model's response — even if the model omits some in its JSON output.

## Code-Level Guardrails

Three guardrails run in application code (not just the prompt):

### Guardrail 1: Relevance Gate (`_filter_relevant_chunks`)
- Drops chunks with ChromaDB L2 distance > 1.5
- Prevents the model from citing irrelevant passages

### Guardrail 2: Tool-Use Check (`_check_tool_was_called`)
- Flags if `tracked_citations` is empty (model never searched guidelines)
- Appends a grounding warning to the reasoning

### Guardrail 3: Citation Check (`_check_citations_in_json`)
- Verifies the JSON output contains at least one citation
- Falls back to checking for page number references in the reasoning text

## Corrective RAG

When initial retrieval produces low-confidence results, the system automatically
reformulates the query and retries:

1. **Confidence scoring** (`_compute_confidence`): Normalizes cross-encoder
   rerank scores or inverse L2 distances to a 0.0-1.0 scale.
2. **Threshold check**: If confidence < `CORRECTIVE_RAG_THRESHOLD` (default 0.3),
   the query is reformulated.
3. **Reformulation strategy**:
   - Attempt 1: Prepends "NICE NG12 suspected cancer referral pathway criteria for:"
   - Attempt 2: Broadens to "cancer recognition symptoms {keyword} referral"
4. **Merge and re-sort**: All unique chunks from all attempts are merged and
   sorted by best available score.

This is gated behind `CORRECTIVE_RAG_ENABLED` and protected by
`RAG_TIMEOUT_SECONDS` for graceful degradation.

## Prompt Injection Guardrails

All user-facing inputs are scanned before reaching the LLM:

- **Instruction overrides**: "ignore all previous instructions", "disregard rules"
- **Role hijacking**: "you are now a pirate", "act as a financial advisor"
- **System prompt extraction**: "show your system prompt", "what are your rules"
- **Encoding attacks**: base64, hex escapes, HTML entities
- **Delimiter injection**: `<|system|>`, `[INST]`, markdown code fences with role names

Matched inputs return HTTP 400 before any LLM call is made.

## Medical Disclaimer

Every response includes a constant disclaimer: "This is an AI-powered clinical
decision support tool. It is NOT a substitute for professional medical advice..."

This is appended at the application layer (not the prompt), so it cannot be
overridden by the model.

## Request Tracing

Each request receives a UUID correlation ID (generated or propagated from
`X-Correlation-ID` header). The ID flows through:
- `ContextVar` propagation across async boundaries
- `AgentStepTracer` context manager wrapping each tool call with timing
- Structured JSON log output (when enabled) with correlation_id field
- Response headers: `X-Correlation-ID` and `X-Response-Time-Ms`

## Evaluation Harness

Gold-standard expected outcomes for all 10 patients enable systematic
quality measurement:

- **Risk level matching**: Exact string match against expected risk levels
- **Citation coverage**: Minimum citation count per patient
- **Reasoning keywords**: Required clinical terms in the reasoning text

Run with `pytest tests/ -v --run-eval` (requires live LLM).

## Trade-offs and Considerations

- **Manual loop vs. automatic**: the manual loop adds ~20 lines of code but gives
  full control over execution, logging, and error handling. This is the pattern
  used in production agentic systems.
- **Structured output vs. prompt-only JSON**: `response_schema` guarantees valid
  JSON but is only available on Vertex AI. The `google_genai` path relies on
  prompt instructions with a fallback parser.
- **Multi-query pre-fetch** adds latency (multiple RAG calls upfront) but
  significantly improves recall for multi-symptom patients.
- **Hybrid search** (BM25 + vector) captures both keyword-exact and semantic
  matches, improving retrieval for clinical terminology.
- **Corrective RAG** adds up to 2 extra retrieval rounds but significantly
  improves recall for edge cases where the initial query underperforms.
- **Prompt injection scanning** uses regex patterns (fast, no ML model needed)
  which may occasionally produce false positives for unusual clinical queries.
  The pattern set is conservative to minimize this risk.
