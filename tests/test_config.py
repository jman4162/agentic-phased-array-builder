"""Tests for APAB configuration loading and saving."""

from __future__ import annotations

from pathlib import Path

import pytest

from apab.core.config import load_config, save_config
from apab.core.schemas import ProjectConfig


class TestLoadConfig:
    def test_load_from_yaml(self, sample_config_yaml: Path) -> None:
        config = load_config(sample_config_yaml)
        assert isinstance(config, ProjectConfig)
        assert config.project.name == "dualpol_ucell_28ghz"

    def test_load_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_load_invalid_yaml(self, tmp_path: Path) -> None:
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("just a string, not a mapping")
        with pytest.raises(ValueError, match="expected a YAML mapping"):
            load_config(bad_yaml)


class TestSaveConfig:
    def test_save_and_reload(self, tmp_path: Path, sample_config_dict: dict) -> None:
        config = ProjectConfig.model_validate(sample_config_dict)
        out_path = tmp_path / "output" / "apab.yaml"
        save_config(config, out_path)
        assert out_path.exists()
        reloaded = load_config(out_path)
        assert reloaded.project.name == config.project.name
        assert reloaded.llm.provider == config.llm.provider
        assert reloaded.solver.backend == config.solver.backend
