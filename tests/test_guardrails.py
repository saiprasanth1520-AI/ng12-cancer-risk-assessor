"""Tests for grounding guardrails (both agents share the same logic)."""

from app.chat_agent import (
    _filter_relevant_chunks,
    _check_tool_was_called,
    _check_citations_present,
)
from app.agent import _check_citations_in_json


# ── Relevance filter ─────────────────────────────────────────────────────

def test_filter_keeps_relevant_chunks():
    chunks = [
        {"chunk_id": "a", "distance": 0.5},
        {"chunk_id": "b", "distance": 1.2},
    ]
    result = _filter_relevant_chunks(chunks)
    assert len(result) == 2


def test_filter_removes_irrelevant_chunks():
    chunks = [
        {"chunk_id": "a", "distance": 0.5},
        {"chunk_id": "b", "distance": 2.0},
        {"chunk_id": "c", "distance": 1.8},
    ]
    result = _filter_relevant_chunks(chunks)
    assert len(result) == 1
    assert result[0]["chunk_id"] == "a"


def test_filter_empty_list():
    assert _filter_relevant_chunks([]) == []


# ── Tool-use check ───────────────────────────────────────────────────────

def test_tool_was_called_true():
    assert _check_tool_was_called([{"chunk_id": "x"}]) is True


def test_tool_was_called_false():
    assert _check_tool_was_called([]) is False


# ── Citation presence (chat) ─────────────────────────────────────────────

def test_citation_present():
    assert _check_citations_present("Refer urgently [NG12 p.45] per guidelines.") is True


def test_citation_missing():
    assert _check_citations_present("This patient should be referred.") is False


def test_citation_various_formats():
    assert _check_citations_present("[NG12 p.3]") is True
    assert _check_citations_present("[NG12 p.123]") is True
    assert _check_citations_present("See [NG12 p.7] for details") is True


# ── Citation presence (risk assessment JSON) ─────────────────────────────

def test_json_citations_present():
    result = {"citations": [{"page": 10, "excerpt": "..."}]}
    assert _check_citations_in_json(result) is True


def test_json_citations_empty():
    result = {"citations": [], "reasoning": "No evidence found."}
    assert _check_citations_in_json(result) is False


def test_json_citations_in_reasoning_text():
    result = {"citations": [], "reasoning": "Per NG12 p.45, refer urgently."}
    assert _check_citations_in_json(result) is True
