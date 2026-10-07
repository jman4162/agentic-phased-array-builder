"""Tests for APAB workspace management."""

from __future__ import annotations

from pathlib import Path

import pytest

from apab.core.workspace import RunContext, Workspace


class TestWorkspace:
    def test_ensure_dirs_creates_tree(self, tmp_workspace: Path) -> None:
        ws = Workspace(tmp_workspace)
        ws.ensure_dirs()
        assert ws.runs_dir.exists()
        assert ws.cache_dir.exists()

    def test_new_run_creates_structure(self, tmp_workspace: Path) -> None:
        ws = Workspace(tmp_workspace)
        ws.ensure_dirs()
        ctx = ws.new_run()
        assert isinstance(ctx, RunContext)
        assert ctx.run_dir.exists()
        assert ctx.coupling_dir.exists()
        assert ctx.patterns_dir.exists()
        assert ctx.system_dir.exists()
        assert ctx.emtool_dir.exists()
        assert ctx.plots_dir.exists()
        assert ctx.report_dir.exists()
        assert ctx.run_id in str(ctx.run_dir)

    def test_new_run_custom_id(self, tmp_workspace: Path) -> None:
        ws = Workspace(tmp_workspace)
        ws.ensure_dirs()
        ctx = ws.new_run(run_id="test_run_001")
        assert ctx.run_id == "test_run_001"
        assert ctx.run_dir.name == "test_run_001"

    def test_is_within_workspace_accepts_inside(self, tmp_workspace: Path) -> None:
        ws = Workspace(tmp_workspace)
        ws.ensure_dirs()
        assert ws.is_within_workspace(tmp_workspace / "runs" / "test")

    def test_is_within_workspace_rejects_outside(self, tmp_workspace: Path) -> None:
        ws = Workspace(tmp_workspace)
        assert not ws.is_within_workspace("/tmp/outside")
        assert not ws.is_within_workspace(tmp_workspace.parent / "other")

    def test_multiple_runs_unique(self, tmp_workspace: Path) -> None:
        ws = Workspace(tmp_workspace)
        ws.ensure_dirs()
        ctx1 = ws.new_run()
        ctx2 = ws.new_run()
        assert ctx1.run_id != ctx2.run_id
        assert ctx1.run_dir != ctx2.run_dir


class TestPathContainment:
    def test_sibling_prefix_is_outside(self, tmp_path):
        # "/x/ws2/f" starts with "/x/ws" as a string but is not inside it
        from apab.core.workspace import validate_path_within

        root = tmp_path / "ws"
        root.mkdir()
        with pytest.raises(ValueError, match="outside"):
            validate_path_within(tmp_path / "ws2" / "f.png", root)
        assert not Workspace(root).is_within_workspace(tmp_path / "ws2" / "f.png")


class TestOutputContext:
    def test_relative_goes_to_default_dir(self, tmp_path):
        from apab.core.workspace import resolve_output_path, set_output_context

        set_output_context(tmp_path, tmp_path / "runs" / "r1" / "artifacts")
        out = resolve_output_path("plots/a.png")
        assert out == (tmp_path / "runs" / "r1" / "artifacts" / "plots" / "a.png").resolve()
        assert out.parent.is_dir()

    def test_absolute_inside_root_allowed(self, tmp_path):
        from apab.core.workspace import resolve_output_path

        assert resolve_output_path(tmp_path / "x.png") == (tmp_path / "x.png").resolve()

    def test_outside_and_traversal_refused(self, tmp_path):
        from apab.core.workspace import resolve_output_path

        with pytest.raises(ValueError, match="outside"):
            resolve_output_path("/etc/apab_test.png")
        with pytest.raises(ValueError, match="traversal"):
            resolve_output_path("../escape.png")

    def test_workspace_arg_cannot_widen_root(self, tmp_path):
        from apab.core.workspace import resolve_workspace_arg

        assert resolve_workspace_arg(tmp_path / "sub") == (tmp_path / "sub").resolve()
        with pytest.raises(ValueError, match="outside"):
            resolve_workspace_arg("/")

    def test_default_dir_must_be_inside_root(self, tmp_path):
        from apab.core.workspace import set_output_context

        with pytest.raises(ValueError, match="outside root"):
            set_output_context(tmp_path / "a", tmp_path / "b")

    def test_env_fallback(self, tmp_path, monkeypatch):
        from apab.core.workspace import output_root, set_output_context

        set_output_context(None)
        monkeypatch.setenv("APAB_WORKSPACE", str(tmp_path / "envws"))
        assert output_root() == (tmp_path / "envws").resolve()


class TestProjectInit:
    async def test_scaffold_and_config_stay_in_workspace(self, tmp_path, monkeypatch):
        from apab.mcp.tools_io import project_init

        monkeypatch.chdir(tmp_path.parent)  # a cwd outside the root
        result = await project_init(name="demo", workspace=str(tmp_path / "proj"))
        assert result["status"] == "initialized"
        assert result["config_path"] == str((tmp_path / "proj" / "apab.yaml").resolve())
        assert (tmp_path / "proj" / "apab.yaml").exists()
        assert not (tmp_path.parent / "apab.yaml").exists()

    async def test_workspace_outside_root_refused(self, tmp_path):
        from apab.mcp.tools_io import project_init

        result = await project_init(name="demo", workspace="/path/to/workspace")
        assert result["status"] == "failed"
