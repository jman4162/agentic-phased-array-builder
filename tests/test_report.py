"""Tests for the report builder."""

from __future__ import annotations

import json

import pytest

from apab.report.build_report import ReportBuilder


@pytest.fixture()
def run_dir(tmp_path):
    """Create a fixture run directory with sample artifacts."""
    run = tmp_path / "test_run"
    run.mkdir()

    # Write manifest
    manifest = {
        "run_id": "test_run",
        "timestamp": "2024-01-15T12:00:00Z",
        "config_hash": "abc123",
        "geometry_hash": "def456",
        "sweep_hash": "ghi789",
        "solver_version": "edgefem 1.0.0",
        "artifacts": ["patterns/pattern.json", "plots/cuts.png"],
        "status": "completed",
    }
    (run / "manifest.json").write_text(json.dumps(manifest))

    # Write pattern result
    (run / "patterns").mkdir()
    pattern = {
        "directivity_dbi": 17.5,
        "sidelobe_level_db": -13.2,
        "metadata": {
            "e_plane_hpbw_deg": 10.5,
            "h_plane_hpbw_deg": 11.2,
        },
    }
    (run / "patterns" / "pattern.json").write_text(json.dumps(pattern))

    # Write system result
    (run / "system").mkdir()
    system = {"eirp_dbw": 45.2, "snr_db": 15.3}
    (run / "system" / "metrics.json").write_text(json.dumps(system))

    return run


class TestReportBuilder:
    def test_build_markdown(self, run_dir):
        builder = ReportBuilder(run_dir)
        report = builder.build_markdown()

        assert "test_run" in report
        assert "17.50 dBi" in report
        assert "-13.20 dB" in report
        assert "10.50" in report

    def test_markdown_includes_system(self, run_dir):
        builder = ReportBuilder(run_dir)
        report = builder.build_markdown()

        assert "System Analysis" in report
        assert "eirp_dbw" in report

    def test_empty_run_dir(self, tmp_path):
        empty = tmp_path / "empty_run"
        empty.mkdir()

        builder = ReportBuilder(empty)
        report = builder.build_markdown()

        assert "No artifacts found" in report

    def test_manifest_loaded(self, run_dir):
        builder = ReportBuilder(run_dir)
        report = builder.build_markdown()

        assert "abc123" in report
        assert "edgefem 1.0.0" in report

    def test_missing_artifacts_handled(self, tmp_path):
        """Even with a manifest referencing missing files, report shouldn't crash."""
        run = tmp_path / "partial_run"
        run.mkdir()
        manifest = {
            "run_id": "partial_run",
            "timestamp": "2024-01-15T12:00:00Z",
            "artifacts": ["missing/file.json"],
        }
        (run / "manifest.json").write_text(json.dumps(manifest))

        builder = ReportBuilder(run)
        report = builder.build_markdown()
        assert "partial_run" in report
