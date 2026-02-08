"""FastAPI application — serves both the Risk Assessment and Chat APIs,
plus a minimal frontend UI.

Includes:
  - Security headers middleware (CSP, X-Frame-Options, etc.)
  - Request tracing with correlation IDs
  - Rate limiting (slowapi)
  - Optional API key auth (X-API-Key header)
  - Prompt injection guardrails
  - SSE streaming endpoint for chat
  - Split health endpoints (/health/live + /health/ready)
"""

import asyncio
import logging
import os
import time
from functools import partial
from fastapi import FastAPI, HTTPException, Request, Depends, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.models import AssessmentRequest, ChatRequest, ChatStreamRequest
from app.patient_data import load_patients, get_patient, list_patients
from app.config import (
    GOOGLE_API_KEY, CHROMA_PERSIST_DIR, API_KEY,
    RATE_LIMIT_ASSESS, RATE_LIMIT_CHAT,
    STRUCTURED_LOGGING_ENABLED,
)
from app.guardrails import (
    check_prompt_injection, check_input_length,
    validate_session_id, validate_patient_id,
    MEDICAL_DISCLAIMER,
)
from app.tracing import (
    generate_correlation_id, get_correlation_id, set_correlation_id,
    setup_structured_logging,
)
from app import agent, chat_agent

logging.basicConfig(level=logging.INFO)

# Enable structured JSON logging if configured
if STRUCTURED_LOGGING_ENABLED:
    setup_structured_logging()

# ── Rate limiter ─────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="NG12 Cancer Risk Assessor",
    description="Clinical Decision Support Agent powered by NICE NG12 Guidelines and Gemini 2.5 Pro",
    version="3.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Security Headers Middleware ─────────────────────────────────────────

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to every response."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'"
        )
        return response


# ── Tracing Middleware ──────────────────────────────────────────────────

class TracingMiddleware(BaseHTTPMiddleware):
    """Generate/extract correlation ID, log request, add response headers."""

    async def dispatch(self, request: Request, call_next):
        # Generate or extract correlation ID
        cid = request.headers.get("X-Correlation-ID", "")
        if not cid:
            cid = generate_correlation_id()
        set_correlation_id(cid)

        start_time = time.monotonic()

        response = await call_next(request)

        duration_ms = (time.monotonic() - start_time) * 1000

        # Add tracing headers
        response.headers["X-Correlation-ID"] = cid
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.1f}"

        # Log the request
        logging.getLogger("app.http").info(
            "%s %s -> %d (%.1fms)",
            request.method, request.url.path,
            response.status_code, duration_ms,
            extra={
                "http_method": request.method,
                "http_path": request.url.path,
                "http_status": response.status_code,
                "duration_ms": round(duration_ms, 1),
            },
        )

        return response


# Add middleware in order: security headers first, then tracing, then CORS
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(TracingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Prometheus metrics ────────────────────────────────────────────────────

try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/health/live", "/metrics"],
    ).instrument(app).expose(app, endpoint="/metrics")
    logging.getLogger(__name__).info("Prometheus metrics enabled at /metrics")
except ImportError:
    logging.getLogger(__name__).info("prometheus-fastapi-instrumentator not installed, /metrics disabled")


# ── Optional API key auth ────────────────────────────────────────────────

def verify_api_key(request: Request):
    """If API_KEY is configured, require it via X-API-Key header."""
    if not API_KEY:
        return  # No auth required in local dev
    provided = request.headers.get("X-API-Key", "")
    if provided != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.on_event("startup")
async def startup():
    load_patients()

    # Warm the RAG cache in the background so it doesn't block startup
    from app.cache import warm_cache
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, warm_cache)


# ═══════════════════════════════════════════════════════════════════════════
# Part 1 — Risk Assessment
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/patients")
def get_all_patients():
    """List all patients in the simulated database."""
    return list_patients()


@app.post("/assess")
@limiter.limit(RATE_LIMIT_ASSESS)
async def assess_patient_endpoint(request: Request, body: AssessmentRequest, _=Depends(verify_api_key)):
    """Assess cancer risk for a patient against the NG12 guidelines."""
    # Guardrail: validate patient ID format
    pid_check = validate_patient_id(body.patient_id)
    if not pid_check.passed:
        raise HTTPException(status_code=400, detail=pid_check.reason)

    patient = get_patient(body.patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {body.patient_id} not found")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, partial(agent.assess_patient, body.patient_id)
        )
        # Ensure disclaimer is always present
        if isinstance(result, dict):
            result.setdefault("disclaimer", MEDICAL_DISCLAIMER)
        return result
    except Exception as e:
        logging.exception("Assessment failed for %s", body.patient_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/assess/{patient_id}/evaluation")
def get_assessment_evaluation(patient_id: str):
    """Retrieve the background evaluation results for a patient assessment.

    Returns the full 5-criteria LLM judge evaluation once processing completes.
    If still running, returns evaluation_status: 'processing'.
    """
    from app.llm_judge import get_stored_evaluation

    evaluation = get_stored_evaluation(patient_id)
    if evaluation is None:
        return {
            "patient_id": patient_id,
            "evaluation_status": "processing",
            "evaluation": None,
        }
    return {
        "patient_id": patient_id,
        "evaluation_status": "complete",
        "evaluation": evaluation,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Part 2 — Conversational Chat
# ═══════════════════════════════════════════════════════════════════════════

def _validate_chat_input(session_id: str, message: str):
    """Run guardrail checks on chat input. Raises HTTPException on failure."""
    # Session ID validation
    sid_check = validate_session_id(session_id)
    if not sid_check.passed:
        raise HTTPException(status_code=400, detail=sid_check.reason)

    # Input length
    length_check = check_input_length(message)
    if not length_check.passed:
        raise HTTPException(status_code=400, detail=length_check.reason)

    # Prompt injection detection
    injection_check = check_prompt_injection(message)
    if not injection_check.passed:
        raise HTTPException(status_code=400, detail=injection_check.reason)


@app.post("/chat")
@limiter.limit(RATE_LIMIT_CHAT)
async def chat_endpoint(request: Request, body: ChatRequest, _=Depends(verify_api_key)):
    """Send a message to the NG12 knowledge-base chat agent."""
    _validate_chat_input(body.session_id, body.message)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, partial(chat_agent.chat, body.session_id, body.message, body.top_k)
        )
        # Ensure disclaimer is always present
        if isinstance(result, dict):
            result.setdefault("disclaimer", MEDICAL_DISCLAIMER)
        return result
    except Exception as e:
        logging.exception("Chat failed for session %s", body.session_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
@limiter.limit(RATE_LIMIT_CHAT)
async def chat_stream_endpoint(request: Request, body: ChatStreamRequest, _=Depends(verify_api_key)):
    """Stream a chat response as Server-Sent Events."""
    _validate_chat_input(body.session_id, body.message)

    return StreamingResponse(
        chat_agent.chat_stream(body.session_id, body.message, body.top_k),
        media_type="text/event-stream",
    )


@app.get("/chat/{session_id}/evaluation")
def get_chat_evaluation(session_id: str):
    """Retrieve background evaluation results for a chat session."""
    from app.llm_judge import get_stored_evaluation

    evaluation = get_stored_evaluation(f"chat:{session_id}")
    if evaluation is None:
        return {"session_id": session_id, "evaluation_status": "processing", "evaluation": None}
    return {"session_id": session_id, "evaluation_status": "complete", "evaluation": evaluation}


@app.get("/chat/{session_id}/history")
def get_chat_history(session_id: str):
    """Return full conversation history for a session."""
    history = chat_agent.get_session_history(session_id)
    return {"session_id": session_id, "messages": history}


@app.delete("/chat/{session_id}")
def delete_chat_session(session_id: str):
    """Clear a chat session."""
    deleted = chat_agent.clear_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


# ═══════════════════════════════════════════════════════════════════════════
# Health Check Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health/live")
def health_live():
    """Lightweight liveness probe — always returns 200."""
    return {"status": "alive"}


@app.get("/health/ready")
def health_ready():
    """Readiness probe — returns 200 if all critical services are available, 503 otherwise."""
    from app.config import BM25_ENABLED, RERANK_ENABLED

    checks = {}
    ready = True

    # Google API key
    checks["google_api_key"] = "configured" if GOOGLE_API_KEY else "MISSING"
    if not GOOGLE_API_KEY:
        ready = False

    # Vector store
    chroma_ok = os.path.isdir(CHROMA_PERSIST_DIR) and bool(os.listdir(CHROMA_PERSIST_DIR))
    checks["vector_store"] = "ready" if chroma_ok else "NOT BUILT"
    if not chroma_ok:
        ready = False

    # Redis
    checks["redis"] = (
        "connected" if chat_agent._redis_client else "unavailable (in-memory fallback)"
    )

    # Patients
    patients = list_patients()
    checks["patients"] = f"{len(patients)} loaded" if patients else "NONE"
    if not patients:
        ready = False

    status_code = 200 if ready else 503
    return JSONResponse(
        content={"status": "ready" if ready else "not_ready", "checks": checks},
        status_code=status_code,
    )


@app.get("/health")
def health_check():
    """Full health check (backward-compatible)."""
    from app.rag import is_bm25_ready, is_reranker_ready
    from app.config import BM25_ENABLED, RERANK_ENABLED

    status = {"status": "ok", "checks": {}}

    # 1. Is the Google API key configured?
    status["checks"]["google_api_key"] = "configured" if GOOGLE_API_KEY else "MISSING"

    # 2. Is the ChromaDB vector store built?
    chroma_ok = os.path.isdir(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR)
    status["checks"]["vector_store"] = "ready" if chroma_ok else "NOT BUILT — run: python scripts/ingest_pdf.py"

    # 3. Is Redis reachable?
    status["checks"]["redis"] = (
        "connected" if chat_agent._redis_client else "unavailable (using in-memory fallback)"
    )

    # 4. Are patients loaded?
    patients = list_patients()
    status["checks"]["patients"] = f"{len(patients)} loaded" if patients else "NONE"

    # 5. BM25 index
    if BM25_ENABLED:
        status["checks"]["bm25_index"] = "ready" if is_bm25_ready() else "not yet built (lazy — built on first query)"
    else:
        status["checks"]["bm25_index"] = "disabled"

    # 6. Cross-encoder reranker
    if RERANK_ENABLED:
        status["checks"]["reranker"] = "loaded" if is_reranker_ready() else "not yet loaded (lazy — loaded on first query)"
    else:
        status["checks"]["reranker"] = "disabled"

    # 7. API key auth
    status["checks"]["api_key_auth"] = "enabled" if API_KEY else "disabled (open access)"

    # Overall status
    if not GOOGLE_API_KEY or not chroma_ok:
        status["status"] = "degraded"

    return status


# ═══════════════════════════════════════════════════════════════════════════
# Static files & UI
# ═══════════════════════════════════════════════════════════════════════════

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")
