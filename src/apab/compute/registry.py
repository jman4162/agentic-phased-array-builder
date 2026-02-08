"""Discovery of installed compute backend plugins."""

from __future__ import annotations

import importlib.metadata
import logging
from typing import Any

logger = logging.getLogger(__name__)

_ENTRY_POINT_GROUP = "apab.compute_backends"


def discover_compute_backends() -> dict[str, type]:
    """Discover installed compute backend plugins via entry points.

    Looks up all entry points registered under the ``apab.compute_backends``
    group and returns a mapping of backend name to backend class.

    Returns
    -------
    dict[str, type]
        Mapping from plugin name to the backend class.
    """
    backends: dict[str, type] = {}

    eps = importlib.metadata.entry_points()
    if hasattr(eps, "select"):
        selected: Any = eps.select(group=_ENTRY_POINT_GROUP)
    else:
        selected = eps.get(_ENTRY_POINT_GROUP, [])

    for ep in selected:
        try:
            backend_cls = ep.load()
            backends[ep.name] = backend_cls
        except Exception:
            logger.warning(
                "Failed to load compute backend plugin '%s'", ep.name, exc_info=True
            )

    return backends
