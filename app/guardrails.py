"""Prompt injection detection, input validation, and medical disclaimer.

Provides security guardrails for all user-facing endpoints:
  - Prompt injection pattern matching (instruction overrides, role hijacking, etc.)
  - Input length validation
  - Session ID and patient ID format validation
  - Medical disclaimer constant for every response
"""

import re
from dataclasses import dataclass

# ── Medical disclaimer ──────────────────────────────────────────────────

MEDICAL_DISCLAIMER = (
    "DISCLAIMER: This is an AI-powered clinical decision support tool based on "
    "the NICE NG12 guidelines. It is NOT a substitute for professional medical "
    "advice, diagnosis, or treatment. Always consult a qualified healthcare "
    "professional for clinical decisions. This tool is intended to assist, not "
    "replace, clinical judgement."
)

# ── Prompt injection patterns ───────────────────────────────────────────

_INJECTION_PATTERNS = [
    # Instruction overrides
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules)", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules)", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules)", re.IGNORECASE),
    re.compile(r"override\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules)", re.IGNORECASE),
    re.compile(r"do\s+not\s+follow\s+(your|the)\s+(instructions|rules|guidelines)", re.IGNORECASE),
    re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),

    # Role hijacking
    re.compile(r"you\s+are\s+now\s+(a|an|the)\s+", re.IGNORECASE),
    re.compile(r"act\s+as\s+(a|an|the)\s+", re.IGNORECASE),
    re.compile(r"pretend\s+(to\s+be|you\s+are)\s+", re.IGNORECASE),
    re.compile(r"switch\s+to\s+\w+\s+mode", re.IGNORECASE),
    re.compile(r"enter\s+(developer|admin|debug|god)\s+mode", re.IGNORECASE),

    # System prompt extraction
    re.compile(r"(show|reveal|print|output|repeat|display)\s+(your|the)\s+(system\s+)?prompt", re.IGNORECASE),
    re.compile(r"(show|reveal|print|output|repeat|display)\s+(your|the)\s+(system\s+)?instructions", re.IGNORECASE),
    re.compile(r"what\s+(is|are)\s+your\s+(system\s+)?(instructions|prompt|rules)", re.IGNORECASE),
    re.compile(r"(show|reveal|print|output)\s+(your|the)\s+(initial|original)\s+(instructions|prompt)", re.IGNORECASE),

    # Encoding / obfuscation attacks
    re.compile(r"base64\s*(encode|decode)", re.IGNORECASE),
    re.compile(r"\\x[0-9a-fA-F]{2}", re.IGNORECASE),
    re.compile(r"&#x?[0-9a-fA-F]+;", re.IGNORECASE),

    # Delimiter injection
    re.compile(r"```\s*(system|assistant|user)\s*\n", re.IGNORECASE),
    re.compile(r"<\|?(system|im_start|im_end|endoftext)\|?>", re.IGNORECASE),
    re.compile(r"\[INST\]|\[/INST\]|\[SYS\]|\[/SYS\]", re.IGNORECASE),

    # Direct command injection
    re.compile(r"(sudo|rm\s+-rf|exec|eval|import\s+os|subprocess)", re.IGNORECASE),
]


# ── Result container ────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    """Result of a guardrail check."""
    passed: bool
    reason: str = ""


# ── Check functions ─────────────────────────────────────────────────────

def check_prompt_injection(text: str) -> GuardrailResult:
    """Scan text against known prompt injection patterns.

    Returns GuardrailResult(passed=True) if clean, or
    GuardrailResult(passed=False, reason=...) if a pattern matched.
    """
    if not text:
        return GuardrailResult(passed=True)

    for pattern in _INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return GuardrailResult(
                passed=False,
                reason=f"Potential prompt injection detected: '{match.group()}'",
            )

    return GuardrailResult(passed=True)


def check_input_length(text: str, max_length: int = 2000) -> GuardrailResult:
    """Validate that input text does not exceed the maximum length."""
    if len(text) > max_length:
        return GuardrailResult(
            passed=False,
            reason=f"Input too long: {len(text)} characters (max {max_length})",
        )
    return GuardrailResult(passed=True)


def validate_session_id(session_id: str) -> GuardrailResult:
    """Validate session ID format: alphanumeric + hyphens/underscores, max 64 chars."""
    if not session_id:
        return GuardrailResult(passed=False, reason="Session ID is required")

    if len(session_id) > 64:
        return GuardrailResult(
            passed=False,
            reason=f"Session ID too long: {len(session_id)} characters (max 64)",
        )

    if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
        return GuardrailResult(
            passed=False,
            reason="Session ID contains invalid characters (allowed: alphanumeric, hyphens, underscores)",
        )

    return GuardrailResult(passed=True)


def validate_patient_id(patient_id: str) -> GuardrailResult:
    """Validate patient ID format: must match PT-\\d{3} pattern."""
    if not patient_id:
        return GuardrailResult(passed=False, reason="Patient ID is required")

    if not re.match(r'^PT-\d{3}$', patient_id):
        return GuardrailResult(
            passed=False,
            reason=f"Invalid patient ID format: '{patient_id}' (expected PT-XXX where X is a digit)",
        )

    return GuardrailResult(passed=True)
