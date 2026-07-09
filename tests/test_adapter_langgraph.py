"""Tests for the LangGraph pipeline adapter."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("langgraph")

from apab.adapters.langgraph_pipeline import (  # noqa: E402
    Constraints,
    Scenario,
    build_pipeline,
    run_pipeline,
)
from apab.core.schemas import ProjectConfig, ProjectMeta  # noqa: E402


@pytest.fixture()
def config(tmp_path):
    return ProjectConfig(
        project=ProjectMeta(name="pipe", workspace=str(tmp_path / "ws")),
    )


def _thread(run_ctx):
    return {"configurable": {"thread_id": run_ctx.run_id}}


class TestBuildPipeline:
    def test_compiles(self, config):
        graph, run_ctx, state = build_pipeline(config, checkpoint=False)
        assert graph is not None
        assert state["run_id"] == run_ctx.run_id

    def test_accepts_dict_config(self, tmp_path):
        graph, _, state = build_pipeline(
            {"project": {"name": "d", "workspace": str(tmp_path)}},
            checkpoint=False,
        )
        assert state["config"]["project"]["name"] == "d"


class TestFullRun:
    def test_node_order_without_edgefem(self, config, monkeypatch):
        import apab.adapters.langgraph_pipeline as mod

        monkeypatch.setattr(mod, "_edgefem_available", lambda: False)
        graph, run_ctx, state = build_pipeline(config, checkpoint=False)

        visited = []
        for update in graph.stream(state, stream_mode="updates"):
            visited.extend(update.keys())

        assert visited == [
            "validate_config",
            "pattern",
            "system_eval",
            "constraint_check",
            "plots",
            "report",
        ]

    def test_final_state_and_run_bundle(self, config):
        result = run_pipeline(
            config,
            constraints=Constraints(min_directivity_dbi=5.0),
            checkpoint=False,
        )

        assert result["pattern"]["directivity_dbi"] > 5.0
        assert result["violations"] == []
        assert result["errors"] == []

        run_dir = pytest.importorskip("pathlib").Path(result["run_dir"])
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["status"] == "success"

        report = (run_dir / "artifacts" / "report" / "pipeline_report.md").read_text()
        assert "All constraints satisfied" in report

    def test_constraint_violation_recorded(self, config):
        result = run_pipeline(
            config,
            constraints=Constraints(min_directivity_dbi=99.0),
            checkpoint=False,
        )

        assert result["violations"], "expected a directivity violation"
        run_dir = pytest.importorskip("pathlib").Path(result["run_dir"])
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["status"] == "constraint_violation"

    def test_invalid_config_routes_to_report(self, config):
        graph, run_ctx, state = build_pipeline(config, checkpoint=False)
        state["config"] = {"project": {}}  # missing required name

        visited = []
        for update in graph.stream(state, stream_mode="updates"):
            visited.extend(update.keys())

        assert visited == ["validate_config", "report"]

    def test_scenario_parameters_reach_system_eval(self, config):
        result = run_pipeline(
            config,
            scenario=Scenario(scenario_type="comms", range_m=250.0),
            checkpoint=False,
        )
        assert "system" in result


class TestCheckpointing:
    def test_checkpoint_file_created_and_resumable(self, config):
        graph, run_ctx, state = build_pipeline(config)
        thread = _thread(run_ctx)

        graph.invoke(state, config=thread)

        assert (run_ctx.run_dir / "checkpoint.sqlite").exists()

        # Resuming the same thread returns the persisted final state
        snapshot = graph.get_state(thread)
        assert snapshot.values["report_path"]
        assert snapshot.next == ()
