"""Tests for FastAPI endpoints (no LLM calls — only structural tests)."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_returns_html():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_list_patients():
    resp = client.get("/patients")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 10
    assert data[0]["patient_id"] == "PT-101"


def test_assess_unknown_patient_returns_404():
    resp = client.post("/assess", json={"patient_id": "PT-999"})
    assert resp.status_code == 404


def test_assess_missing_body_returns_422():
    resp = client.post("/assess", json={})
    assert resp.status_code == 422


def test_chat_missing_fields_returns_422():
    resp = client.post("/chat", json={"message": "hello"})
    assert resp.status_code == 422


def test_chat_history_empty_session():
    resp = client.get("/chat/nonexistent_session/history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["messages"] == []


def test_delete_nonexistent_session_returns_404():
    resp = client.delete("/chat/nonexistent_session")
    assert resp.status_code == 404


# ── Pydantic field constraint validation ──────────────────────────────

def test_assess_invalid_patient_id_format():
    """Patient ID must match PT-\\d{3} pattern."""
    resp = client.post("/assess", json={"patient_id": "INVALID"})
    assert resp.status_code == 422


def test_chat_message_too_long():
    """Message must be <= 2000 characters."""
    resp = client.post("/chat", json={
        "session_id": "test-session",
        "message": "a" * 2001,
    })
    assert resp.status_code == 422


def test_chat_session_id_invalid_chars():
    """Session ID must be alphanumeric + hyphens/underscores."""
    resp = client.post("/chat", json={
        "session_id": "invalid session id!",
        "message": "hello",
    })
    assert resp.status_code == 422


def test_chat_blank_message():
    """Whitespace-only messages should be rejected."""
    resp = client.post("/chat", json={
        "session_id": "test-session",
        "message": "   ",
    })
    assert resp.status_code == 422


def test_chat_top_k_too_low():
    """top_k must be >= 1."""
    resp = client.post("/chat", json={
        "session_id": "test-session",
        "message": "hello",
        "top_k": 0,
    })
    assert resp.status_code == 422


def test_chat_top_k_too_high():
    """top_k must be <= 20."""
    resp = client.post("/chat", json={
        "session_id": "test-session",
        "message": "hello",
        "top_k": 21,
    })
    assert resp.status_code == 422


# ── Security headers ──────────────────────────────────────────────────

def test_security_headers_present():
    """All security headers should be present on every response."""
    resp = client.get("/patients")
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("X-Frame-Options") == "DENY"
    assert resp.headers.get("X-XSS-Protection") == "1; mode=block"
    assert resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert "default-src" in resp.headers.get("Content-Security-Policy", "")


def test_correlation_id_header():
    """X-Correlation-ID should be returned on every response."""
    resp = client.get("/patients")
    assert "X-Correlation-ID" in resp.headers
    assert len(resp.headers["X-Correlation-ID"]) > 0


def test_response_time_header():
    """X-Response-Time-Ms should be returned on every response."""
    resp = client.get("/patients")
    assert "X-Response-Time-Ms" in resp.headers


# ── Split health endpoints ────────────────────────────────────────────

def test_health_live_endpoint():
    """Liveness probe should always return 200."""
    resp = client.get("/health/live")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "alive"


def test_health_ready_endpoint():
    """Readiness probe returns 200 or 503 with checks."""
    resp = client.get("/health/ready")
    assert resp.status_code in (200, 503)
    data = resp.json()
    assert "checks" in data
    assert data["status"] in ("ready", "not_ready")


def test_health_endpoint_backward_compat():
    """Original /health endpoint still works."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "checks" in data
