"""Tests for the patient data module."""

import pytest
from app.patient_data import get_patient, list_patients, load_patients


@pytest.fixture(autouse=True)
def _load():
    load_patients()


def test_get_existing_patient():
    patient = get_patient("PT-101")
    assert patient is not None
    assert patient["name"] == "John Doe"
    assert patient["age"] == 55
    assert "unexplained hemoptysis" in patient["symptoms"]


def test_get_nonexistent_patient():
    assert get_patient("PT-999") is None


def test_list_patients_returns_all():
    patients = list_patients()
    assert len(patients) == 10
    ids = [p["patient_id"] for p in patients]
    assert "PT-101" in ids
    assert "PT-110" in ids


def test_patient_fields():
    """Every patient record must have the required fields."""
    required = {"patient_id", "name", "age", "gender", "smoking_history",
                "symptoms", "symptom_duration_days"}
    for patient in list_patients():
        assert required.issubset(patient.keys()), f"{patient['patient_id']} missing fields"
