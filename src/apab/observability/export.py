"""Span exporters and processor construction.

Only imported once OpenTelemetry is known to be installed
(from :func:`apab.observability.tracing.init_observability`).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan

    from apab.core.schemas import ObservabilitySpec
    from apab.core.workspace import RunContext

logger = logging.getLogger(__name__)


class JsonlSpanExporter(SpanExporter):
    """Write one JSON object per span to a .jsonl file.

    Kept dependency-light so trace.jsonl in the run bundle can be read
    without any OpenTelemetry tooling.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def export(self, spans: Any) -> SpanExportResult:
        try:
            with self._path.open("a") as fh:
                for s in spans:
                    fh.write(json.dumps(_span_to_dict(s), default=str) + "\n")
        except OSError:
            logger.exception("Failed to write %s", self._path)
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def _span_to_dict(s: ReadableSpan) -> dict[str, Any]:
    return {
        "name": s.name,
        "trace_id": format(s.context.trace_id, "032x"),
        "span_id": format(s.context.span_id, "016x"),
        "parent_span_id": (
            format(s.parent.span_id, "016x") if s.parent else None
        ),
        "start_time_unix_nano": s.start_time,
        "end_time_unix_nano": s.end_time,
        "status": s.status.status_code.name,
        "attributes": dict(s.attributes or {}),
        "events": [
            {"name": e.name, "attributes": dict(e.attributes or {})}
            for e in s.events
        ],
    }


def build_processors(
    spec: ObservabilitySpec,
    run_ctx: RunContext | None,
) -> list[SimpleSpanProcessor]:
    """Build span processors for the configured exporters.

    Spans are low-volume here (one per turn/tool call), so simple
    processors are used throughout: they export synchronously, which
    keeps trace.jsonl complete even on a crash.
    """
    processors: list[SimpleSpanProcessor] = []

    if spec.trace_jsonl and run_ctx is not None:
        processors.append(SimpleSpanProcessor(
            JsonlSpanExporter(run_ctx.run_dir / "trace.jsonl"),
        ))

    if spec.console_exporter:
        processors.append(SimpleSpanProcessor(ConsoleSpanExporter()))

    endpoint = spec.otlp_endpoint or os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            processors.append(SimpleSpanProcessor(
                OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces"),
            ))
        except ImportError:
            logger.warning(
                "otlp_endpoint is set but the OTLP exporter is not "
                "installed; run: pip install 'apab[observability]'"
            )

    return processors
