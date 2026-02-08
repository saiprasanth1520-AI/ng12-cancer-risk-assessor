"""Conversational Chat Agent — multi-turn Q&A grounded in the NG12 guidelines.

Reuses the same ChromaDB vector store built in Part 1.
Session memory is stored in Redis (with automatic fallback to in-memory if
Redis is unavailable, e.g. during local development without Docker).

Supports two LLM providers (set LLM_PROVIDER env var):
  - "google_genai"  — Google AI Studio via google-generativeai SDK (API key)
  - "vertex_ai"     — Google Cloud Vertex AI via google-cloud-aiplatform SDK

Architecture improvements:
  - Manual agentic loop (no AutomaticFunctionCallingResponder)
  - Streaming support via chat_stream() generator (SSE)

Grounding guardrails are enforced at the application level:
  1. Relevance gate  — low-relevance chunks are filtered before the model sees them
  2. Tool-use check  — if the model skipped the search tool, the response is flagged
  3. Citation check   — if the answer lacks inline citations, a disclaimer is appended
"""

import json
import logging
import re
from typing import Optional, Generator
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import redis

from app.config import (
    GOOGLE_API_KEY, GEMINI_MODEL, LLM_PROVIDER,
    GCP_PROJECT_ID, GCP_LOCATION,
    REDIS_HOST, REDIS_PORT, REDIS_TTL_SECONDS,
    MAX_AGENT_STEPS, LLM_TIMEOUT_SECONDS,
)
from app.rag import search_guidelines, search_guidelines_with_timeout
from app.tracing import AgentStepTracer
from app.guardrails import MEDICAL_DISCLAIMER
from app.llm_judge import judge_chat, store_evaluation, get_stored_evaluation

logger = logging.getLogger(__name__)

# ── Initialise (safe to call even if agent.py already called it) ─────────
if LLM_PROVIDER == "vertex_ai":
    import vertexai
    from vertexai.generative_models import (
        GenerativeModel, Content, Part, Tool,
        FunctionDeclaration,
    )
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
else:
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)

# ── Redis session store (falls back to in-memory dict) ───────────────────
_KEY_PREFIX = "chat_session:"
_redis_client: Optional[redis.Redis] = None
_fallback_sessions: dict[str, list[dict]] = {}  # used only when Redis is down

try:
    _redis_client = redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, db=0,
        decode_responses=True, socket_connect_timeout=2,
    )
    _redis_client.ping()
    logger.info("Connected to Redis at %s:%s", REDIS_HOST, REDIS_PORT)
except (redis.ConnectionError, redis.TimeoutError, OSError):
    _redis_client = None
    logger.warning(
        "Redis not available at %s:%s — falling back to in-memory sessions",
        REDIS_HOST, REDIS_PORT,
    )

# ── System prompt ────────────────────────────────────────────────────────
CHAT_SYSTEM_PROMPT = """\
You are a clinical knowledge assistant specialising in the NICE NG12 guidelines
for suspected cancer recognition and referral.

## Rules
1. Answer ONLY using evidence retrieved from the NG12 guideline via the
   search_clinical_guidelines tool.  Call it at least once per question.
2. Cite your sources inline as [NG12 p.X] where X is the page number.
3. If the retrieved passages do not contain enough information, respond:
   "I couldn't find sufficient evidence in the NG12 guidelines to answer that."
4. NEVER invent thresholds, age cut-offs, or referral criteria.
5. For follow-up questions, use the conversation history for context but
   always verify facts against a fresh guideline search.
6. Keep answers clear, structured, and suitable for a clinical audience.

## How to search
Use search_clinical_guidelines with specific clinical terms.
You may call it multiple times in one turn for thorough coverage.
"""

# ChromaDB L2 distance threshold.
RELEVANCE_DISTANCE_THRESHOLD = 1.5

_NO_EVIDENCE_DISCLAIMER = (
    "\n\nNote: This response could not be fully grounded in the NG12 guidelines. "
    "The retrieved evidence was insufficient or the model did not cite specific "
    "guideline passages. Please verify independently."
)


# ── Guardrail helpers ────────────────────────────────────────────────────

def _filter_relevant_chunks(chunks: list[dict]) -> list[dict]:
    """Guardrail 1 — Remove chunks whose distance exceeds the threshold."""
    return [c for c in chunks if c.get("distance", 0) <= RELEVANCE_DISTANCE_THRESHOLD]


def _check_tool_was_called(tracked_citations: list[dict]) -> bool:
    """Guardrail 2 — Return True if the search tool was actually called."""
    return len(tracked_citations) > 0


def _check_citations_present(answer: str) -> bool:
    """Guardrail 3 — Return True if the answer contains at least one inline
    citation like [NG12 p.X] or [NG12 p.XX].
    """
    return bool(re.search(r"\[NG12\s+p\.\d+\]", answer))


# ── Session helpers (Redis-backed, with in-memory fallback) ──────────────

def _redis_key(session_id: str) -> str:
    return f"{_KEY_PREFIX}{session_id}"


def get_session_history(session_id: str) -> list[dict]:
    if _redis_client:
        raw = _redis_client.get(_redis_key(session_id))
        return json.loads(raw) if raw else []
    return _fallback_sessions.get(session_id, [])


def add_to_history(session_id: str, role: str, content: str):
    history = get_session_history(session_id)
    history.append({"role": role, "content": content})
    if _redis_client:
        _redis_client.set(
            _redis_key(session_id),
            json.dumps(history),
            ex=REDIS_TTL_SECONDS,
        )
    else:
        _fallback_sessions[session_id] = history


def clear_session(session_id: str) -> bool:
    if _redis_client:
        return _redis_client.delete(_redis_key(session_id)) > 0
    if session_id in _fallback_sessions:
        del _fallback_sessions[session_id]
        return True
    return False


def list_sessions() -> list[str]:
    if _redis_client:
        keys = _redis_client.keys(f"{_KEY_PREFIX}*")
        return [k.replace(_KEY_PREFIX, "") for k in keys]
    return list(_fallback_sessions.keys())


# ── LLM timeout wrapper ────────────────────────────────────────────────

def _send_message_with_timeout(chat, message, timeout=LLM_TIMEOUT_SECONDS):
    """Wrap chat.send_message with a timeout using ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(chat.send_message, message)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            logger.error("LLM call timed out after %ds", timeout)
            raise TimeoutError(f"LLM call timed out after {timeout}s")


# ── Manual agent loop (Vertex AI) ───────────────────────────────────────

def _run_vertex_chat_loop(
    message: str,
    history: list[dict],
    search_clinical_guidelines,
    max_steps: int = 10,
) -> str:
    """Manual tool-calling loop for the chat agent (Vertex AI)."""

    tool_map = {"search_clinical_guidelines": search_clinical_guidelines}

    search_func_decl = FunctionDeclaration(
        name="search_clinical_guidelines",
        description="Search the NICE NG12 cancer referral guidelines for relevant passages.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Clinical search terms about symptoms, referral criteria, or cancer types.",
                },
            },
            "required": ["query"],
        },
    )
    tool = Tool(function_declarations=[search_func_decl])

    model = GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=CHAT_SYSTEM_PROMPT,
        tools=[tool],
    )

    # Convert stored history to Vertex AI Content objects
    vertex_history = []
    for msg in history:
        vertex_history.append(
            Content(
                role=msg["role"],
                parts=[Part.from_text(msg["content"])],
            )
        )

    chat_session = model.start_chat(history=vertex_history)
    response = _send_message_with_timeout(chat_session, message)

    for step in range(max_steps):
        candidate = response.candidates[0]
        function_calls = [
            part.function_call
            for part in candidate.content.parts
            if part.function_call and part.function_call.name
        ]

        if not function_calls:
            text_parts = [
                part.text for part in candidate.content.parts if part.text
            ]
            return "".join(text_parts).strip()

        fn_response_parts = []
        for fc in function_calls:
            fn_name = fc.name
            fn_args = dict(fc.args) if fc.args else {}
            logger.info("  Chat step %d: calling %s(%s)", step + 1, fn_name, fn_args)

            with AgentStepTracer(fn_name, step=step + 1):
                if fn_name in tool_map:
                    fn_result = tool_map[fn_name](**fn_args)
                else:
                    fn_result = {"error": f"Unknown function: {fn_name}"}

            fn_response_parts.append(
                Part.from_function_response(
                    name=fn_name,
                    response={"result": fn_result},
                )
            )

        response = _send_message_with_timeout(chat_session, fn_response_parts)

    logger.warning("Chat agent loop hit max steps (%d)", max_steps)
    text_parts = [
        part.text for part in response.candidates[0].content.parts if part.text
    ]
    return "".join(text_parts).strip() if text_parts else ""


# ── Main chat function ───────────────────────────────────────────────────

def chat(session_id: str, message: str, top_k: int = 5) -> dict:
    """Process a single chat turn and return a grounded response with citations."""

    tracked_citations: list[dict] = []

    def search_clinical_guidelines(query: str) -> list:
        """Search the NICE NG12 suspected cancer referral guidelines."""
        raw_results = search_guidelines_with_timeout(query, top_k=top_k)

        # Guardrail 1: filter out low-relevance chunks
        relevant = _filter_relevant_chunks(raw_results)
        if len(relevant) < len(raw_results):
            logger.info(
                "Relevance filter: kept %d/%d chunks (threshold=%.2f)",
                len(relevant), len(raw_results), RELEVANCE_DISTANCE_THRESHOLD,
            )

        tracked_citations.extend(relevant)
        return relevant

    # ── Build model and inject conversation history ──────────────────────
    history = get_session_history(session_id)

    if LLM_PROVIDER == "vertex_ai":
        logger.info("Chat [%s] user (Vertex AI): %s", session_id, message[:80])
        answer = _run_vertex_chat_loop(
            message, history, search_clinical_guidelines,
            max_steps=min(MAX_AGENT_STEPS, 10),
        )
    else:
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            tools=[search_clinical_guidelines],
            system_instruction=CHAT_SYSTEM_PROMPT,
        )

        # Convert stored history to GenAI Content objects
        gemini_history = []
        for msg in history:
            gemini_history.append(
                genai.protos.Content(
                    role=msg["role"],
                    parts=[genai.protos.Part(text=msg["content"])],
                )
            )

        chat_session = model.start_chat(
            history=gemini_history,
            enable_automatic_function_calling=True,
        )

        logger.info("Chat [%s] user (Google AI): %s", session_id, message[:80])
        response = _send_message_with_timeout(chat_session, message)
        answer = response.text.strip()

    logger.info("Chat [%s] response: %d chars, %d citations",
                session_id, len(answer), len(tracked_citations))

    # ── Guardrail 2: detect if model skipped the search tool ─────────
    grounding_flags: list[str] = []

    if not _check_tool_was_called(tracked_citations):
        logger.warning("Chat [%s] model did NOT call the search tool", session_id)
        grounding_flags.append("no_retrieval")
        answer += _NO_EVIDENCE_DISCLAIMER

    # ── Guardrail 3: check the answer contains inline citations ──────
    elif not _check_citations_present(answer):
        logger.warning("Chat [%s] answer lacks inline [NG12 p.X] citations", session_id)
        grounding_flags.append("missing_citations")
        answer += _NO_EVIDENCE_DISCLAIMER

    # Save the text parts of this turn to session memory
    add_to_history(session_id, "user", message)
    add_to_history(session_id, "model", answer)

    # De-duplicate citations
    seen_ids: set[str] = set()
    citations = []
    for c in tracked_citations:
        cid = c.get("chunk_id", "")
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        citations.append(
            {
                "source": c.get("source", "NG12 PDF"),
                "page": c.get("page", 0),
                "chunk_id": cid,
                "excerpt": c.get("text", "")[:300],
            }
        )

    # LLM Judge: run in background thread so it doesn't block the response
    import threading

    def _run_chat_evaluation():
        try:
            evaluation = judge_chat(message, answer, tracked_citations)
            store_evaluation(f"chat:{session_id}", evaluation)
        except Exception:
            logger.exception("Background chat evaluation failed for session %s", session_id)

    eval_thread = threading.Thread(target=_run_chat_evaluation, daemon=True)
    eval_thread.start()

    return {
        "session_id": session_id,
        "answer": answer,
        "citations": citations,
        "grounding_flags": grounding_flags,
        "disclaimer": MEDICAL_DISCLAIMER,
        "evaluation_status": "processing",
    }


# ── Streaming chat ──────────────────────────────────────────────────────

def chat_stream(
    session_id: str, message: str, top_k: int = 5,
) -> Generator[str, None, None]:
    """Process a chat turn and yield SSE-formatted events.

    Strategy: run the full agent loop (non-streamed) to resolve tool calls and
    apply guardrails, then stream the final answer token-by-token as SSE events.
    """
    # Get the complete response first (tool calls happen here)
    result = chat(session_id, message, top_k)
    answer = result.get("answer", "")
    citations = result.get("citations", [])
    grounding_flags = result.get("grounding_flags", [])

    # Stream the answer in chunks (simulate token-level streaming)
    chunk_size = 12  # characters per SSE event
    for i in range(0, len(answer), chunk_size):
        token = answer[i : i + chunk_size]
        event_data = json.dumps({"token": token})
        yield f"data: {event_data}\n\n"

    # Send final metadata event
    final = json.dumps({
        "done": True,
        "citations": citations,
        "grounding_flags": grounding_flags,
    })
    yield f"data: {final}\n\n"
