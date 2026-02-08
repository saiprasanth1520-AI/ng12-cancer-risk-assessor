# Prompt Engineering Strategy — Chat Agent (Part 2)

## Overview

The chat agent extends the RAG pipeline from Part 1 into a multi-turn
conversational interface over the NG12 guidelines.  It reuses the same ChromaDB
vector store, hybrid search pipeline, and embedding model.

## System Prompt Design

### 1. Role & Scope
> "You are a clinical knowledge assistant specialising in the NICE NG12
> guidelines for suspected cancer recognition and referral."

This scopes the agent strictly to NG12 content — it will not answer questions
outside this domain.

### 2. Mandatory Tool Use
> "Call [search_clinical_guidelines] at least once per question."

This ensures every answer is backed by a fresh retrieval, even for follow-up
questions that might seem answerable from context alone.

### 3. Citation Format
> "Cite your sources inline as [NG12 p.X]."

Inline citations make the response immediately verifiable and match the UI's
rendering format.

### 4. Failure Behaviour (Guardrails)
> "If the retrieved passages do not contain enough information, respond:
> 'I couldn't find sufficient evidence in the NG12 guidelines to answer that.'"

This hard-coded refusal template prevents the model from speculating when
evidence is missing.

### 5. Anti-Hallucination Rule
> "NEVER invent thresholds, age cut-offs, or referral criteria."

Explicit prohibition of the most dangerous hallucination category in a clinical
context.

## Manual Agentic Loop

Like the risk assessment agent, the chat agent's Vertex AI path uses a **manual
tool-calling loop** (`_run_vertex_chat_loop`) instead of
`AutomaticFunctionCallingResponder`. Benefits:

1. **Full logging** of each tool call (function name, args) per step
2. **Safety bound** — `MAX_AGENT_STEPS` (capped at 10 for chat) prevents
   runaway loops
3. **Consistent architecture** — same pattern as the assessment agent

The `google_genai` fallback retains `enable_automatic_function_calling=True`.

## Streaming Support (SSE)

The chat agent provides a `chat_stream()` generator for Server-Sent Events:

1. **Tool calls run non-streamed** — the full agent loop executes normally,
   resolving all function calls and applying guardrails
2. **Final answer is streamed** — once complete, the buffered response is
   yielded in character chunks as SSE events: `data: {"token": "..."}\n\n`
3. **Metadata event** — a final event with `{"done": true, "citations": [...],
   "grounding_flags": [...]}` signals completion

This approach ensures guardrails run on the complete answer *before* any
tokens reach the client. The frontend uses `fetch` + `ReadableStream` to
parse SSE events and append tokens to the chat bubble in real-time.

## Code-Level Guardrails (beyond the prompt)

The system prompt alone is not reliable — the model can ignore instructions.
We therefore enforce three guardrails in **application code** that run
before and after every model response:

### Guardrail 1: Relevance Gate (`_filter_relevant_chunks`)
- **When**: Before the model sees any retrieved chunks.
- **How**: ChromaDB returns an L2 distance score with each result.  Chunks
  with `distance > 1.5` are dropped.
- **Effect**: If a user asks something unrelated to cancer (e.g. "What's the
  weather?"), all chunks will score poorly and most get filtered out.  The
  model receives little or no context and is far more likely to refuse.

### Guardrail 2: Tool-Use Check (`_check_tool_was_called`)
- **When**: After the model responds.
- **How**: If `tracked_citations` is empty, the model answered without ever
  calling `search_clinical_guidelines` — meaning it used its own knowledge.
- **Effect**: A disclaimer is appended to the answer and `grounding_flags`
  includes `"no_retrieval"`.

### Guardrail 3: Citation Check (`_check_citations_present`)
- **When**: After the model responds (if guardrail 2 passed).
- **How**: A regex checks for at least one `[NG12 p.X]` pattern in the text.
- **Effect**: If no citations found, a disclaimer is appended and
  `grounding_flags` includes `"missing_citations"`.

The `grounding_flags` array is returned in the API response so the frontend
can display a visible yellow warning banner to the user.

## Hybrid Search Pipeline

The chat agent benefits from the same hybrid retrieval as the assessment agent:

1. **Vector search** (ChromaDB semantic) — always active
2. **BM25 keyword search** — captures exact clinical terminology matches
3. **Reciprocal Rank Fusion** — merges both ranked lists with configurable weights
4. **Cross-encoder reranking** — re-scores merged candidates for maximum relevance

All features are configurable via environment variables (`BM25_ENABLED`,
`RERANK_ENABLED`) and degrade gracefully when disabled.

## Multi-turn Conversation Memory

- Session history is stored in **Redis** (with automatic fallback to an
  in-memory dict if Redis is not available).
- When starting a new Gemini chat, the stored history is converted to
  Vertex AI `Content` objects (or `genai.protos.Content`) and passed as
  the `history` parameter.
- Only the *text* portions of each turn are stored (tool calls and responses
  are not persisted).  This simplifies storage while preserving conversational
  coherence.

## Rate Limiting & Auth

The API layer adds:
- **Rate limiting** (slowapi): `/chat` and `/chat/stream` are limited to
  30 requests/minute per IP (configurable via `RATE_LIMIT_CHAT`)
- **Optional API key auth**: when `API_KEY` is set, all endpoints require
  an `X-API-Key` header

## Trade-offs

- **Text-only history**: we do not persist function-call turns, so the model
  cannot reference *which* chunks it retrieved in previous turns.  This is
  acceptable because each turn performs a fresh search.
- **Buffer-then-stream**: the streaming endpoint buffers the full response
  before streaming, adding latency equal to a non-streaming call.  This is
  the trade-off for running guardrails on the complete answer.
- **Redis sessions**: sessions persist across restarts via Redis.  If Redis is
  unavailable, the app falls back to an in-memory dict gracefully.
- **top_k is configurable** per request, defaulting to 5.  Higher values
  improve recall at the cost of latency and prompt length.
