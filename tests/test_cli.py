"""Tests for the APAB CLI."""

from __future__ import annotations

import pytest

from apab.cli import build_parser, main


class TestBuildParser:
    def test_subcommands_present(self):
        parser = build_parser()
        # Parse each subcommand with --help shouldn't fail
        for cmd in ["init", "design", "run", "report"]:
            with pytest.raises(SystemExit) as exc_info:
                parser.parse_args([cmd, "--help"])
            assert exc_info.value.code == 0

    def test_version_flag(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0


class TestCmdInit:
    def test_creates_scaffold(self, tmp_path):
        project_dir = tmp_path / "test_project"
        main(["init", "--name", "test_project", "--dir", str(project_dir)])

        assert (project_dir / "apab.yaml").exists()
        assert (project_dir / "workspace" / "runs").exists()
        assert (project_dir / "workspace" / "cache").exists()
        assert (project_dir / "workspace" / "logs").exists()

        # Config should be valid YAML
        import yaml

        config = yaml.safe_load((project_dir / "apab.yaml").read_text())
        assert config["project"]["name"] == "test_project"

    def test_no_overwrite(self, tmp_path):
        project_dir = tmp_path / "existing"
        project_dir.mkdir()
        (project_dir / "apab.yaml").write_text("existing: true")

        main(["init", "--name", "test", "--dir", str(project_dir)])

        # Should not overwrite
        content = (project_dir / "apab.yaml").read_text()
        assert "existing: true" in content
