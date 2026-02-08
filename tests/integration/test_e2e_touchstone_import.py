"""End-to-end: Touchstone import → coupling → array pattern → report."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture()
def touchstone_file(tmp_path):
    """Create a minimal 2-port Touchstone fixture."""
    s2p = tmp_path / "test.s2p"
    lines = [
        "! Test 2-port S-parameters",
        "# GHz S RI R 50",
        "10.0  0.1 -0.2  0.01 0.005  0.01 0.005  0.1 -0.2",
        "10.5  0.12 -0.18  0.02 0.008  0.02 0.008  0.12 -0.18",
        "11.0  0.15 -0.15  0.03 0.01  0.03 0.01  0.15 -0.15",
    ]
    s2p.write_text("\n".join(lines))
    return s2p


class TestTouchstoneToReport:
    def test_import_and_analyse(self, touchstone_file, tmp_path):
        from apab.coupling.active_impedance import (
            active_impedance,
            active_reflection_coefficient,
            detect_scan_blindness,
        )
        from apab.emtool.importers import import_touchstone
        from apab.pattern.wrappers_pam import PAMPatternEngine
        from apab.report.build_report import ReportBuilder

        # 1. Import Touchstone
        data = import_touchstone(str(touchstone_file))
        assert data["n_ports"] == 2
        assert len(data["freqs"]) == 3

        # 2. Compute coupling metrics at first frequency
        s_matrix = data["s_params"][0]  # shape (2, 2)
        excitation = np.ones(2, dtype=complex)
        gamma = active_reflection_coefficient(s_matrix, excitation)
        z_active = active_impedance(gamma)
        blindness = detect_scan_blindness(gamma)

        assert gamma.shape == (2,)
        assert z_active.shape == (2,)
        assert isinstance(blindness, list)

        # 3. Compute array pattern (small array for speed)
        from apab.core.schemas import ArraySpec, ScanPoint

        spec = ArraySpec(
            size=[4, 4],
            spacing_m=[0.005, 0.005],
            taper="uniform",
            steer=ScanPoint(theta_deg=0, phi_deg=0),
        )
        engine = PAMPatternEngine()
        result = engine.full_pattern(spec, 10e9, 0.0, 0.0)
        assert result.directivity_dbi > 0

        # 4. Build report
        run_dir = tmp_path / "run_001"
        run_dir.mkdir()
        (run_dir / "patterns").mkdir()
        (run_dir / "patterns" / "pattern.json").write_text(
            json.dumps({
                "directivity_dbi": result.directivity_dbi,
                "sidelobe_level_db": result.sidelobe_level_db,
                "metadata": result.metadata,
            }, default=str)
        )

        builder = ReportBuilder(run_dir)
        report = builder.build_markdown()
        assert "dBi" in report


class TestTouchstoneImportMCPTool:
    async def test_via_mcp_tool(self, touchstone_file):
        from apab.mcp.tools_io import io_import_touchstone

        result = await io_import_touchstone(filepath=str(touchstone_file))
        assert result["status"] == "imported"
        assert result["n_ports"] == 2
        assert result["n_freqs"] == 3
