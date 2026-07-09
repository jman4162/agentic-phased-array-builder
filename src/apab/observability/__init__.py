"""OpenTelemetry observability for APAB agent runs.

Tracing is a soft dependency: with the ``observability`` extra missing
or ``observability.enabled`` false, every function here is a no-op, so
core modules can import this package unconditionally.
"""

from apab.observability.redaction import capture_args, capture_text
from apab.observability.tracing import (
    current_trace_ids,
    init_observability,
    is_enabled,
    shutdown_observability,
    span,
)

__all__ = [
    "capture_args",
    "capture_text",
    "current_trace_ids",
    "init_observability",
    "is_enabled",
    "shutdown_observability",
    "span",
]
