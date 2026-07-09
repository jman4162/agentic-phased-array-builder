"""Tracer lifecycle and span helpers with a soft OpenTelemetry dependency."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from apab.core.schemas import ObservabilitySpec
    from apab.core.workspace import RunContext

logger = logging.getLogger(__name__)

_provider: Any = None
_tracer: Any = None
_enabled = False


class _NoopSpan:
    """Stand-in span used when tracing is disabled."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def record_exception(self, exception: BaseException) -> None:
        pass

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        pass


_NOOP_SPAN = _NoopSpan()


def is_enabled() -> bool:
    """Whether tracing is currently active."""
    return _enabled


def init_observability(
    spec: ObservabilitySpec,
    run_ctx: RunContext | None = None,
    extra_processors: list[Any] | None = None,
) -> bool:
    """Set up the APAB tracer provider from *spec*.

    Returns True when tracing is active. Safe to call when the
    ``opentelemetry`` packages are absent: logs a warning and stays
    disabled. The ``APAB_OBSERVABILITY=1`` env var forces ``enabled``.

    ``extra_processors`` is a hook for tests to inject an in-memory
    span processor.
    """
    global _provider, _tracer, _enabled

    enabled = spec.enabled or os.environ.get("APAB_OBSERVABILITY") == "1"
    if not enabled:
        return False
    if _enabled:
        # Already initialised (e.g. a second session in one process).
        return True

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
    except ImportError:
        logger.warning(
            "observability is enabled but opentelemetry is not installed; "
            "run: pip install 'apab[observability]'"
        )
        return False

    from apab.observability.export import build_processors

    provider = TracerProvider(
        resource=Resource.create({"service.name": spec.service_name}),
    )
    for processor in build_processors(spec, run_ctx):
        provider.add_span_processor(processor)
    for processor in extra_processors or []:
        provider.add_span_processor(processor)

    if spec.set_global:
        trace.set_tracer_provider(provider)

    _provider = provider
    _tracer = provider.get_tracer("apab")
    _enabled = True
    logger.info("Observability enabled (service=%s)", spec.service_name)
    return True


def shutdown_observability() -> None:
    """Flush exporters and disable tracing."""
    global _provider, _tracer, _enabled
    if _provider is not None:
        try:
            _provider.shutdown()
        except Exception:
            logger.exception("Tracer provider shutdown failed")
    _provider = None
    _tracer = None
    _enabled = False


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Open a child span, or yield a no-op span when tracing is off.

    Attribute values of ``None`` are skipped.
    """
    if not _enabled or _tracer is None:
        yield _NOOP_SPAN
        return

    with _tracer.start_as_current_span(name) as s:
        for key, value in attributes.items():
            if value is not None:
                s.set_attribute(key, value)
        yield s


def current_trace_ids() -> tuple[str, str] | None:
    """Return (trace_id, span_id) of the current span as hex strings."""
    if not _enabled:
        return None
    from opentelemetry import trace

    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return None
    return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")


def set_span_error(s: Any, exception: BaseException) -> None:
    """Record *exception* on span *s* and mark its status as error."""
    try:
        s.record_exception(exception)
        from opentelemetry.trace import Status, StatusCode

        s.set_status(Status(StatusCode.ERROR, str(exception)))
    except ImportError:
        pass
