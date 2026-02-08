"""
Load testing for NG12 Cancer Risk Assessor.

Three user classes with different behaviors:
- HealthCheckUser: lightweight health endpoint checks
- CancerAssessorUser: realistic clinical workflow (assess, chat, stream)
- RateLimitTestUser: aggressive requests to verify rate limiting

Usage:
  # Web UI mode (opens at http://localhost:8089)
  locust -f locustfile.py --host http://localhost:8000

  # Headless mode
  locust -f locustfile.py --host http://localhost:8000 --headless -u 10 -r 2 --run-time 60s
"""

import os
import random
import string

from locust import HttpUser, between, task


API_KEY = os.environ.get("API_KEY", "")

PATIENT_IDS = [f"PT-{i}" for i in range(101, 111)]
INVALID_PATIENT_IDS = ["PT-999", "PT-000"]

CHAT_MESSAGES = [
    "What are the NICE NG12 referral criteria for suspected lung cancer?",
    "Which symptoms should prompt an urgent referral for breast cancer?",
    "What is the two-week wait pathway for colorectal cancer?",
    "How should a GP assess unexplained weight loss in the context of cancer?",
    "What blood tests are recommended when cancer is suspected?",
    "What are the red flag symptoms for upper gastrointestinal cancer?",
    "When should a patient with persistent cough be referred for investigation?",
    "What is the role of safety netting in cancer diagnosis?",
    "How does the NG12 guideline define unexplained lymphadenopathy?",
    "What imaging is recommended for suspected pancreatic cancer?",
]


def _headers():
    """Return request headers, including API key if configured."""
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h


def _random_session_id():
    """Generate a random alphanumeric session ID."""
    return "load-test-" + "".join(random.choices(string.ascii_lowercase + string.digits, k=8))


class HealthCheckUser(HttpUser):
    """Lightweight user that only hits health endpoints."""

    weight = 1
    wait_time = between(1, 3)

    @task(3)
    def health_live(self):
        self.client.get("/health/live", name="/health/live")

    @task(2)
    def health_ready(self):
        self.client.get("/health/ready", name="/health/ready")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")


class CancerAssessorUser(HttpUser):
    """Primary realistic user simulating clinical workflows."""

    weight = 3
    wait_time = between(2, 5)

    @task(3)
    def list_patients(self):
        with self.client.get("/patients", name="/patients", catch_response=True) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if not data:
                    resp.failure("Empty patient list")
            else:
                resp.failure(f"Unexpected status: {resp.status_code}")

    @task(5)
    def assess_patient(self):
        patient_id = random.choice(PATIENT_IDS)
        with self.client.post(
            "/assess",
            json={"patient_id": patient_id},
            headers=_headers(),
            name="/assess",
            catch_response=True,
        ) as resp:
            if resp.status_code in (200, 429):
                resp.success()
            else:
                resp.failure(f"Unexpected status: {resp.status_code}")

    @task(1)
    def assess_invalid_patient(self):
        patient_id = random.choice(INVALID_PATIENT_IDS)
        with self.client.post(
            "/assess",
            json={"patient_id": patient_id},
            headers=_headers(),
            name="/assess [404]",
            catch_response=True,
        ) as resp:
            if resp.status_code in (404, 429):
                resp.success()
            else:
                resp.failure(f"Expected 404 or 429, got {resp.status_code}")

    @task(4)
    def chat(self):
        session_id = _random_session_id()
        message = random.choice(CHAT_MESSAGES)
        with self.client.post(
            "/chat",
            json={"session_id": session_id, "message": message},
            headers=_headers(),
            name="/chat",
            catch_response=True,
        ) as resp:
            if resp.status_code in (200, 429):
                resp.success()
            else:
                resp.failure(f"Unexpected status: {resp.status_code}")

    @task(3)
    def chat_stream(self):
        session_id = _random_session_id()
        message = random.choice(CHAT_MESSAGES)
        with self.client.post(
            "/chat/stream",
            json={"session_id": session_id, "message": message},
            headers=_headers(),
            name="/chat/stream",
            stream=True,
            catch_response=True,
        ) as resp:
            if resp.status_code in (200, 429):
                if resp.status_code == 200:
                    # Consume the SSE stream
                    for _ in resp.iter_content(chunk_size=1024):
                        pass
                resp.success()
            else:
                resp.failure(f"Unexpected status: {resp.status_code}")

    @task(2)
    def chat_history(self):
        session_id = _random_session_id()
        self.client.get(
            f"/chat/{session_id}/history",
            name="/chat/{session_id}/history",
        )


class RateLimitTestUser(HttpUser):
    """Aggressive user designed to trigger rate limits and verify slowapi works."""

    weight = 1
    wait_time = between(0.1, 0.5)

    @task(5)
    def hammer_assess(self):
        patient_id = random.choice(PATIENT_IDS)
        with self.client.post(
            "/assess",
            json={"patient_id": patient_id},
            headers=_headers(),
            name="/assess [rate-limit]",
            catch_response=True,
        ) as resp:
            if resp.status_code in (200, 429):
                resp.success()
            else:
                resp.failure(f"Unexpected status: {resp.status_code}")

    @task(5)
    def hammer_chat(self):
        session_id = _random_session_id()
        message = random.choice(CHAT_MESSAGES)
        with self.client.post(
            "/chat",
            json={"session_id": session_id, "message": message},
            headers=_headers(),
            name="/chat [rate-limit]",
            catch_response=True,
        ) as resp:
            if resp.status_code in (200, 429):
                resp.success()
            else:
                resp.failure(f"Unexpected status: {resp.status_code}")
