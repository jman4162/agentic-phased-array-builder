"""Tests for EM tool importers (Touchstone and far-field CSV)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apab.emtool.importers import import_farfield_csv, import_touchstone

# ── Touchstone fixtures and tests ──


@pytest.fixture
def s2p_file(tmp_path: Path) -> Path:
    """Create a minimal 2-port Touchstone (.s2p) file."""
    content = (
        "! Test 2-port touchstone\n"
        "# GHZ S RI R 50.0\n"
        "1.0  0.5 0.1  0.01 -0.02  0.01 -0.02  0.5 0.1\n"
        "2.0  0.4 0.15  0.02 -0.01  0.02 -0.01  0.4 0.15\n"
    )
    fpath = tmp_path / "test.s2p"
    fpath.write_text(content)
    return fpath


class TestImportTouchstone:
    """Tests for import_touchstone."""

    def test_frequencies_converted_to_hz(self, s2p_file: Path) -> None:
        result = import_touchstone(s2p_file)
        assert len(result["freqs"]) == 2
        assert result["freqs"][0] == pytest.approx(1.0e9)
        assert result["freqs"][1] == pytest.approx(2.0e9)

    def test_n_ports(self, s2p_file: Path) -> None:
        result = import_touchstone(s2p_file)
        assert result["n_ports"] == 2

    def test_s_params_shape(self, s2p_file: Path) -> None:
        result = import_touchstone(s2p_file)
        assert len(result["s_params"]) == 2
        for mat in result["s_params"]:
            assert mat.shape == (2, 2)

    def test_z0(self, s2p_file: Path) -> None:
        result = import_touchstone(s2p_file)
        assert result["z0"] == pytest.approx(50.0)

    def test_s11_first_freq(self, s2p_file: Path) -> None:
        result = import_touchstone(s2p_file)
        s11 = result["s_params"][0][0, 0]
        assert s11.real == pytest.approx(0.5, abs=1e-10)
        assert s11.imag == pytest.approx(0.1, abs=1e-10)

    def test_s21_first_freq(self, s2p_file: Path) -> None:
        result = import_touchstone(s2p_file)
        s21 = result["s_params"][0][1, 0]
        assert s21.real == pytest.approx(0.01, abs=1e-10)
        assert s21.imag == pytest.approx(-0.02, abs=1e-10)

    def test_comments_parsed(self, s2p_file: Path) -> None:
        result = import_touchstone(s2p_file)
        assert len(result["comments"]) >= 1
        assert "Test 2-port touchstone" in result["comments"][0]

    def test_s_params_are_complex(self, s2p_file: Path) -> None:
        result = import_touchstone(s2p_file)
        for mat in result["s_params"]:
            assert mat.dtype == np.complex128 or np.issubdtype(mat.dtype, np.complexfloating)


# ── 1-port Touchstone ──


@pytest.fixture
def s1p_file(tmp_path: Path) -> Path:
    """Create a minimal 1-port Touchstone (.s1p) file."""
    content = (
        "! 1-port test\n"
        "# MHZ S MA R 50.0\n"
        "1000.0  0.9 -45.0\n"
        "2000.0  0.8 -90.0\n"
    )
    fpath = tmp_path / "test.s1p"
    fpath.write_text(content)
    return fpath


class TestImportTouchstoneS1P:
    """Tests for 1-port Touchstone parsing."""

    def test_n_ports_is_one(self, s1p_file: Path) -> None:
        result = import_touchstone(s1p_file)
        assert result["n_ports"] == 1

    def test_freq_unit_mhz(self, s1p_file: Path) -> None:
        result = import_touchstone(s1p_file)
        assert result["freqs"][0] == pytest.approx(1.0e9)
        assert result["freqs"][1] == pytest.approx(2.0e9)

    def test_ma_format_conversion(self, s1p_file: Path) -> None:
        result = import_touchstone(s1p_file)
        s11 = result["s_params"][0][0, 0]
        # MA: mag=0.9, angle=-45 deg
        expected = 0.9 * np.exp(1j * np.radians(-45.0))
        assert s11.real == pytest.approx(expected.real, abs=1e-10)
        assert s11.imag == pytest.approx(expected.imag, abs=1e-10)


# ── Far-field CSV fixtures and tests ──


@pytest.fixture
def farfield_csv(tmp_path: Path) -> Path:
    """Create a minimal far-field CSV file."""
    content = (
        "theta_deg,phi_deg,gain_dB\n"
        "0,0,10.0\n"
        "30,0,8.5\n"
        "60,0,3.0\n"
        "90,0,-5.0\n"
    )
    fpath = tmp_path / "farfield.csv"
    fpath.write_text(content)
    return fpath


class TestImportFarfieldCSV:
    """Tests for import_farfield_csv."""

    def test_correct_number_of_points(self, farfield_csv: Path) -> None:
        result = import_farfield_csv(farfield_csv)
        assert len(result["theta_deg"]) == 4
        assert len(result["phi_deg"]) == 4
        assert len(result["gain_db"]) == 4

    def test_theta_values(self, farfield_csv: Path) -> None:
        result = import_farfield_csv(farfield_csv)
        assert result["theta_deg"] == pytest.approx([0.0, 30.0, 60.0, 90.0])

    def test_phi_values(self, farfield_csv: Path) -> None:
        result = import_farfield_csv(farfield_csv)
        assert result["phi_deg"] == pytest.approx([0.0, 0.0, 0.0, 0.0])

    def test_gain_values(self, farfield_csv: Path) -> None:
        result = import_farfield_csv(farfield_csv)
        assert result["gain_db"] == pytest.approx([10.0, 8.5, 3.0, -5.0])

    def test_metadata_from_comments(self, tmp_path: Path) -> None:
        content = (
            "# source: HFSS export\n"
            "# frequency: 28 GHz\n"
            "theta_deg,phi_deg,gain_dB\n"
            "0,0,10.0\n"
        )
        fpath = tmp_path / "ff_meta.csv"
        fpath.write_text(content)
        result = import_farfield_csv(fpath)
        assert result["metadata"]["source"] == "HFSS export"
        assert result["metadata"]["frequency"] == "28 GHz"
