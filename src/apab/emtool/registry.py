"""Discovery of installed external EM adapter plugins."""

from __future__ import annotations

import importlib.metadata
import logging

logger = logging.getLogger(__name__)

_ENTRY_POINT_GROUP = "apab.em_adapters"


def discover_em_adapters() -> dict[str, type]:
    """Discover installed EM adapter plugins via entry points.

    Looks up all entry points registered under the ``apab.em_adapters`` group
    and returns a mapping of adapter name to adapter class.

    Returns
    -------
    dict[str, type]
        Mapping from plugin name to the adapter class.
    """
    adapters: dict[str, type] = {}

    # entry_points(group=...) is available on all supported versions (3.10+).
    selected = importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP)

    for ep in selected:
        try:
            adapter_cls = ep.load()
            adapters[ep.name] = adapter_cls
        except Exception:
            logger.warning("Failed to load EM adapter plugin '%s'", ep.name, exc_info=True)

    return adapters
