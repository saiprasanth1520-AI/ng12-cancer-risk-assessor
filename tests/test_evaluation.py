"""Evaluation harness — parametrized tests for all 10 patients.

Run with: pytest tests/test_evaluation.py -v --run-eval

These tests require a live LLM connection (Google API key or Vertex AI)
and the ChromaDB vector store to be built. They are skipped by default.
"""

import pytest
from app.agent import assess_patient
from tests.evaluation_data import EXPECTED_OUTCOMES


PATIENT_IDS = list(EXPECTED_OUTCOMES.keys())


@pytest.mark.eval
@pytest.mark.parametrize("patient_id", PATIENT_IDS)
def test_patient_risk_level(patient_id):
    """Assert that the risk level matches the gold-standard expected outcome."""
    expected = EXPECTED_OUTCOMES[patient_id]
    result = assess_patient(patient_id)

    actual_risk = result.get("risk_level", "")
    expected_risk = expected["expected_risk_level"]

    assert actual_risk == expected_risk, (
        f"Patient {patient_id}: expected '{expected_risk}', got '{actual_risk}'"
    )


@pytest.mark.eval
@pytest.mark.parametrize("patient_id", PATIENT_IDS)
def test_patient_has_citations(patient_id):
    """Assert the assessment returns at least the minimum expected citations."""
    expected = EXPECTED_OUTCOMES[patient_id]
    result = assess_patient(patient_id)

    citations = result.get("citations", [])
    min_expected = expected["min_citations"]

    assert len(citations) >= min_expected, (
        f"Patient {patient_id}: expected >= {min_expected} citations, got {len(citations)}"
    )


@pytest.mark.eval
@pytest.mark.parametrize("patient_id", PATIENT_IDS)
def test_patient_reasoning_keywords(patient_id):
    """Assert the reasoning contains expected clinical keywords."""
    expected = EXPECTED_OUTCOMES[patient_id]
    result = assess_patient(patient_id)

    reasoning = result.get("reasoning", "").lower()
    missing = []
    for keyword in expected["reasoning_must_mention"]:
        if keyword.lower() not in reasoning:
            missing.append(keyword)

    assert not missing, (
        f"Patient {patient_id}: reasoning missing keywords: {missing}"
    )
