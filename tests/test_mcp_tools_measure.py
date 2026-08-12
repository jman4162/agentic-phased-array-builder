"""Tests for measured-data import persistence and compare_sim_measured."""

from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import pytest

from apab.mcp.tools_emtool import emtool_import_results
from apab.mcp.tools_io import io_import_touchstone
from apab.mcp.tools_measure import compare_sim_measured

FIXTURE = Path(__file__).parent / "fixtures" / "patch_28ghz_synthetic.s2p"


class TestImportPersistence:
    async def test_metadata_only_without_run(self):
        result = await io_import_touchstone(str(FIXTURE))
        assert result["status"] == "imported"
        assert result["n_ports"] == 2
        assert "artifact_path" not in result

    async def test_arrays_persist_with_run(self, tmp_path):
        result = await io_import_touchstone(
            str(FIXTURE), run_id="run1", workspace=str(tmp_path)
        )
        assert result["status"] == "imported"
        artifact = Path(result["artifact_path"])
        assert artifact.exists()
        assert artifact.parent.name == "emtool"

        with h5py.File(artifact, "r") as fh:
            freqs = fh["freqs_hz"][:]
            s = fh["s_params"][:]
            assert fh.attrs["n_ports"] == 2
            assert fh.attrs["z0"] == 50.0
        assert len(freqs) == 101
        # Hand-pinned fixture value: S11 at exactly 28 GHz is -1/9
        idx = int(abs(freqs - 28e9).argmin())
        assert s[idx][0][0] == pytest.approx(-1 / 9, abs=1e-6)

    async def test_emtool_import_persists_too(self, tmp_path):
        result = await emtool_import_results(
            str(FIXTURE), run_id="run1", workspace=str(tmp_path)
        )
        assert result["status"] == "imported"
        assert Path(result["artifact_path"]).exists()


class TestCompareSimMeasured:
    async def test_identical_files_compare_to_zero(self, tmp_path):
        result = await compare_sim_measured(
            sim_path=str(FIXTURE),
            measured_path=str(FIXTURE),
            run_id="run1",
            workspace=str(tmp_path),
        )
        assert result["status"] == "completed"
        assert result["rmse_db"] == pytest.approx(0.0, abs=1e-12)
        assert result["synthetic_measurement"] is True

        report = json.loads(Path(result["report_path"]).read_text())
        assert report["measured_provenance"]["synthetic"] is True
        assert report["n_points"] == 101

    async def test_perturbed_sim_reports_known_deviation(self, tmp_path):
        """Scale every measured S11 magnitude by 10^(1/20): exactly 1 dB
        deviation at every point, so RMSE = max = 1 dB."""
        lines = FIXTURE.read_text().splitlines()
        out = []
        scale = 10 ** (1 / 20)
        for line in lines:
            if line.startswith(("!", "#")):
                out.append(line)
                continue
            vals = [float(x) for x in line.split()]
            vals[1] *= scale
            vals[2] *= scale
            vals[7] *= scale
            vals[8] *= scale
            out.append(" ".join(f"{v:.9f}" for v in vals))
        perturbed = tmp_path / "perturbed.s2p"
        perturbed.write_text("\n".join(out) + "\n")

        # measured side stays the contract fixture (has a sidecar)
        result = await compare_sim_measured(
            sim_path=str(perturbed),
            measured_path=str(FIXTURE),
            run_id="run1",
            workspace=str(tmp_path),
        )
        assert result["status"] == "completed"
        assert result["rmse_db"] == pytest.approx(1.0, abs=1e-6)
        assert result["max_abs_deviation_db"] == pytest.approx(1.0, abs=1e-6)

    async def test_missing_sidecar_refused(self, tmp_path):
        bare = tmp_path / "bare.s2p"
        bare.write_text(FIXTURE.read_text())
        result = await compare_sim_measured(
            sim_path=str(FIXTURE),
            measured_path=str(bare),
            run_id="run1",
            workspace=str(tmp_path),
        )
        assert result["status"] == "failed"
        assert "sidecar" in result["error"]

    async def test_fixture_dip_is_where_the_sidecar_says(self):
        """Independent check of the fixture itself: the S11 dip sits at
        28 GHz with |S11| = 1/9 (-19.085 dB)."""
        from apab.emtool.importers import import_touchstone, load_provenance

        prov = load_provenance(FIXTURE)
        assert prov.synthetic is True
        data = import_touchstone(str(FIXTURE))
        mags = [abs(s[0][0]) for s in data["s_params"]]
        i = mags.index(min(mags))
        assert data["freqs"][i] == pytest.approx(28e9, rel=1e-9)
        assert 20 * math.log10(mags[i]) == pytest.approx(-19.085, abs=0.01)
