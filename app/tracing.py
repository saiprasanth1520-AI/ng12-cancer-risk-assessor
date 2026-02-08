"""Request tracing with correlation IDs and structured JSON logging.

Provides:
  - ContextVar-based correlation ID propagation
  - StructuredLogFormatter for JSON log output
  - AgentStepTracer context manager for timing tool calls
"""

import json
import logging
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone

# ── Correlation ID context var ──────────────────────────────────────────

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def generate_correlation_id() -> str:
    """Generate a new UUID-based correlation ID."""
    return str(uuid.uuid4())


def get_correlation_id() -> str:
    """Get the current correlation ID from context."""
    return _correlation_id.get()


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID in the current context."""
    _correlation_id.set(cid)


# ── Structured JSON log formatter ──────────────────────────────────────

class StructuredLogFormatter(logging.Formatter):
    """Format log records as JSON with correlation ID and optional extras.

    Output fields: timestamp, level, logger, message, correlation_id,
    and any extras passed via the `extra` dict (e.g. patient_id, session_id,
    agent_step, tool_name, duration_ms).
    """

    EXTRA_KEYS = frozenset({
        "patient_id", "session_id", "agent_step",
        "tool_name", "duration_ms", "http_method",
        "http_path", "http_status",
    })

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": get_correlation_id(),
        }

        # Include any recognized extra keys
        for key in self.EXTRA_KEYS:
            value = getattr(record, key, None)
            if value is not None:
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


# ── Agent step tracer ──────────────────────────────────────────────────

class AgentStepTracer:
    """Context manager that wraps a tool call with timing and structured logging.

    Usage:
        with AgentStepTracer("search_clinical_guidelines", step=1) as tracer:
            result = search_fn(query)
        # Automatically logs duration on exit
    """

    def __init__(self, tool_name: str, step: int = 0, **extra):
        self.tool_name = tool_name
        self.step = step
        self.extra = extra
        self.start_time = 0.0
        self.duration_ms = 0.0
        self.logger = logging.getLogger("app.tracing")

    def __enter__(self):
        self.start_time = time.monotonic()
        self.logger.info(
            "Tool call started: %s (step %d)",
            self.tool_name, self.step,
            extra={
                "tool_name": self.tool_name,
                "agent_step": self.step,
                **self.extra,
            },
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration_ms = (time.monotonic() - self.start_time) * 1000
        level = logging.ERROR if exc_type else logging.INFO
        msg = (
            f"Tool call {'failed' if exc_type else 'completed'}: "
            f"{self.tool_name} (step {self.step}, {self.duration_ms:.1f}ms)"
        )
        self.logger.log(
            level, msg,
            extra={
                "tool_name": self.tool_name,
                "agent_step": self.step,
                "duration_ms": round(self.duration_ms, 1),
                **self.extra,
            },
        )
        return False  # Don't suppress exceptions


def setup_structured_logging():
    """Replace the root logger handler with StructuredLogFormatter."""
    root = logging.getLogger()
    # Remove existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(StructuredLogFormatter())
    root.addHandler(handler)
    root.setLevel(logging.INFO)
