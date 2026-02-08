"""Tests for provenance tracking."""

from __future__ import annotations

from apab.core.provenance import (
    build_manifest,
    hash_config,
    hash_geometry,
    hash_sweep,
)


class TestHashing:
    def test_deterministic_config_hash(self):
        config = {"project": {"name": "test"}, "array": {"size": [4, 4]}}
        h1 = hash_config(config)
        h2 = hash_config(config)
        assert h1 == h2
        assert len(h1) == 16

    def test_different_configs_different_hashes(self):
        h1 = hash_config({"a": 1})
        h2 = hash_config({"a": 2})
        assert h1 != h2

    def test_deterministic_geometry_hash(self):
        geom = {"kind": "patch", "params": {"er": 3.5}}
        h1 = hash_geometry(geom)
        h2 = hash_geometry(geom)
        assert h1 == h2

    def test_deterministic_sweep_hash(self):
        sweep = {"freq_hz": {"start": 1e9, "stop": 2e9, "n": 10}}
        h1 = hash_sweep(sweep)
        h2 = hash_sweep(sweep)
        assert h1 == h2

    def test_key_order_irrelevant(self):
        h1 = hash_config({"a": 1, "b": 2})
        h2 = hash_config({"b": 2, "a": 1})
        assert h1 == h2


class TestBuildManifest:
    def test_basic_manifest(self):
        m = build_manifest("run_001", timestamp="2024-01-15T12:00:00Z")
        assert m["run_id"] == "run_001"
        assert m["timestamp"] == "2024-01-15T12:00:00Z"
        assert m["status"] == "created"
        assert isinstance(m["dependency_versions"], dict)

    def test_manifest_with_config(self):
        config = {
            "project": {"name": "test"},
            "solver": {"backend": "edgefem"},
            "llm": {"provider": "ollama", "model": "test-model"},
        }
        m = build_manifest("run_002", config=config)
        assert m["config_hash"] != ""
        assert m["provider_name"] == "ollama"
        assert m["model_name"] == "test-model"

    def test_manifest_with_geometry(self):
        config = {
            "unit_cell": {
                "geometry": {"kind": "patch", "params": {"er": 3.5}},
            },
        }
        m = build_manifest("run_003", config=config)
        assert "geometry_hash" in m

    def test_manifest_with_sweep(self):
        config = {
            "sweep": {"freq_hz": {"start": 1e9, "stop": 2e9, "n": 10}},
        }
        m = build_manifest("run_004", config=config)
        assert "sweep_hash" in m

    def test_dependency_versions_collected(self):
        m = build_manifest("run_005")
        deps = m["dependency_versions"]
        # numpy should always be installed in our test env
        assert "numpy" in deps

    def test_auto_timestamp(self):
        m = build_manifest("run_006")
        assert m["timestamp"] != ""
