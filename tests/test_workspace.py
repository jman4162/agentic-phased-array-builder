"""Tests for APAB workspace management."""

from __future__ import annotations

from pathlib import Path

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
