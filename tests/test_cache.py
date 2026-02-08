"""Tests for the result cache."""

from __future__ import annotations

from apab.core.cache import ResultCache


class TestResultCache:
    def test_set_and_get(self, tmp_path):
        cache = ResultCache(tmp_path / "cache")
        key = cache.make_key("pattern", nx=4, ny=4, freq=10e9)
        cache.set(key, {"directivity": 17.5})
        assert cache.get(key) == {"directivity": 17.5}

    def test_miss_returns_none(self, tmp_path):
        cache = ResultCache(tmp_path / "cache")
        assert cache.get("nonexistent_key") is None

    def test_has(self, tmp_path):
        cache = ResultCache(tmp_path / "cache")
        key = cache.make_key("test_op", x=1)
        assert cache.has(key) is False
        cache.set(key, "value")
        assert cache.has(key) is True

    def test_key_changes_with_params(self, tmp_path):
        cache = ResultCache(tmp_path / "cache")
        k1 = cache.make_key("pattern", nx=4, ny=4)
        k2 = cache.make_key("pattern", nx=8, ny=8)
        assert k1 != k2

    def test_deterministic_key(self, tmp_path):
        cache = ResultCache(tmp_path / "cache")
        k1 = cache.make_key("op", a=1, b=2)
        k2 = cache.make_key("op", a=1, b=2)
        assert k1 == k2

    def test_overwrite(self, tmp_path):
        cache = ResultCache(tmp_path / "cache")
        key = cache.make_key("op")
        cache.set(key, "v1")
        cache.set(key, "v2")
        assert cache.get(key) == "v2"
