"""Result caching for APAB simulations (stub for Phase 0)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ResultCache:
    """Simple disk-based result cache wrapping diskcache."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self._cache: Any = None

    def _ensure_cache(self) -> Any:
        if self._cache is None:
            import diskcache

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache = diskcache.Cache(str(self.cache_dir))
        return self._cache

    @staticmethod
    def make_key(operation: str, **params: Any) -> str:
        """Create a deterministic cache key from operation name and parameters."""
        raw = json.dumps({"op": operation, **params}, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        """Retrieve a cached result, or None if not found."""
        cache = self._ensure_cache()
        return cache.get(key, default=None)

    def set(self, key: str, value: Any) -> None:
        """Store a result in the cache."""
        cache = self._ensure_cache()
        cache.set(key, value)

    def has(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        cache = self._ensure_cache()
        return key in cache
