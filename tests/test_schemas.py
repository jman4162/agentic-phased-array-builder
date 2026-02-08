"""Tests for APAB Pydantic schemas."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from apab.core.schemas import (
    CouplingResult,
    PatternResult,
    ProjectConfig,
    ScanPoint,
    SimRequest,
    SimResult,
)


class TestProjectConfig:
    def test_validates_full_spec_example(self, sample_config_dict: dict) -> None:
        config = ProjectConfig.model_validate(sample_config_dict)
        assert config.project.name == "dualpol_ucell_28ghz"
        assert config.llm.provider == "ollama"
        assert config.llm.model == "qwen2.5-coder:14b"
        assert config.llm.redaction_mode.value == "none"
        assert config.mcp.server_port == 8765
        assert config.compute.backend == "local"
        assert config.solver.backend == "edgefem"
        assert config.unit_cell is not None
        assert config.unit_cell.lattice.dx_m == 0.005
        assert config.unit_cell.geometry.kind.value == "patch"
        assert config.unit_cell.ports.count == 2
        assert config.sweep is not None
        assert config.sweep.freq_hz.start == 26.0e9
        assert config.sweep.freq_hz.n == 41
        assert config.sweep.scan.n_theta == 13
        assert config.array.size == [16, 16]
        assert config.array.taper == "taylor"
        assert config.array.steer.theta_deg == 45
        assert config.outputs.export_touchstone is True
        assert "arc_scan" in config.outputs.plots

    def test_required_project_name(self) -> None:
        with pytest.raises(ValidationError):
            ProjectConfig.model_validate({"project": {}})

    def test_missing_project_field(self) -> None:
        with pytest.raises(ValidationError):
            ProjectConfig.model_validate({})

    def test_defaults_applied(self) -> None:
        config = ProjectConfig.model_validate(
            {"project": {"name": "test"}}
        )
        assert config.llm.provider == "ollama"
        assert config.compute.backend == "local"
        assert config.unit_cell is None
        assert config.sweep is None

    def test_json_roundtrip(self, sample_config_dict: dict) -> None:
        config = ProjectConfig.model_validate(sample_config_dict)
        json_str = config.model_dump_json()
        reloaded = ProjectConfig.model_validate_json(json_str)
        assert reloaded.project.name == config.project.name
        assert reloaded.llm.provider == config.llm.provider


class TestSimRequestResult:
    def test_sim_request_roundtrip(self) -> None:
        req = SimRequest(
            unit_cell={
                "lattice": {"type": "rect", "dx_m": 0.005, "dy_m": 0.005},
                "geometry": {"kind": "patch", "params": {"patch_w_m": 0.003}},
            },
            sweep={
                "freq_hz": {"start": 1e9, "stop": 2e9, "n": 5},
                "polarization": ["H"],
            },
        )
        data = json.loads(req.model_dump_json())
        req2 = SimRequest.model_validate(data)
        assert req2.unit_cell.lattice.dx_m == 0.005
        assert req2.sweep.freq_hz.n == 5

    def test_sim_result_roundtrip(self) -> None:
        res = SimResult(
            freq_hz=[1e9, 2e9],
            scan_points=[ScanPoint(theta_deg=0, phi_deg=0)],
            polarizations=["H"],
            s_params=[[1.0, 0.0], [0.0, 1.0]],
            metadata={"solver": "edgefem"},
        )
        data = json.loads(res.model_dump_json())
        res2 = SimResult.model_validate(data)
        assert res2.freq_hz == [1e9, 2e9]
        assert res2.metadata["solver"] == "edgefem"


class TestCouplingResult:
    def test_roundtrip(self) -> None:
        cr = CouplingResult(
            freq_hz=[1e9],
            scan_points=[ScanPoint(theta_deg=0, phi_deg=0)],
            polarizations=["H"],
            gamma_active=[0.1 + 0.2j],
        )
        data = json.loads(cr.model_dump_json())
        cr2 = CouplingResult.model_validate(data)
        assert cr2.freq_hz == [1e9]


class TestPatternResult:
    def test_roundtrip(self) -> None:
        pr = PatternResult(
            theta_deg=[0, 10, 20],
            phi_deg=[0],
            pattern_db=[-3.0, 0.0, -3.0],
            directivity_dbi=15.0,
            sidelobe_level_db=-20.0,
        )
        data = json.loads(pr.model_dump_json())
        pr2 = PatternResult.model_validate(data)
        assert pr2.directivity_dbi == 15.0
        assert pr2.sidelobe_level_db == -20.0
