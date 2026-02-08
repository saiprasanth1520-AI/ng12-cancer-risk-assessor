import json
from typing import Optional
from app.config import PATIENTS_JSON_PATH

_patients_db: dict = {}


def load_patients():
    global _patients_db
    with open(PATIENTS_JSON_PATH) as f:
        patients = json.load(f)
    _patients_db = {p["patient_id"]: p for p in patients}


def get_patient(patient_id: str) -> Optional[dict]:
    if not _patients_db:
        load_patients()
    return _patients_db.get(patient_id)


def list_patients() -> list[dict]:
    if not _patients_db:
        load_patients()
    return list(_patients_db.values())
