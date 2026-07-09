"""Deterministic LangGraph pipeline over APAB's engineering tools.

Where the agent orchestrator lets an LLM pick tools turn by turn, this
pipeline runs a fixed engineering sequence with explicit state,
checkpointing, and streaming progress:

    validate_config
    -> unit_cell (only with EdgeFEM installed and configured)
    -> pattern
    -> system_eval
    -> constraint_check
    -> plots
    -> report

Nodes are plain Python callables that dispatch APAB's MCP tools
in-process; no LLM is involved. Results land in a normal APAB run
bundle (manifest.json, artifacts/), and each node is wrapped in an
``apab.node.<name>`` span when observability is enabled.

Requires the ``langgraph`` extra::

    pip install "apab[langgraph]"

Note: langgraph depends on langchain-core; APAB uses no LangChain
model wrappers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from apab.agent.tool_dispatch import ToolDispatcher
from apab.core.schemas import ProjectConfig
from apab.core.workspace import RunContext, Workspace
from apab.observability import span

logger = logging.getLogger(__name__)

NODE_ORDER = [
    "validate_config",
    "unit_cell",
    "pattern",
    "system_eval",
    "constraint_check",
    "plots",
    "report",
]


class PipelineState(TypedDict, total=False):
    """State threaded through the pipeline graph."""

    config: dict[str, Any]
    run_id: str
    run_dir: str
    unit_cell: dict[str, Any]
    pattern: dict[str, Any]
    system: dict[str, Any]
    violations: list[str]
    plots: list[str]
    report_path: str
    errors: list[str]


@dataclass
class Scenario:
    """System-evaluation scenario parameters for the system_eval node."""

    bandwidth_hz: float = 100e6
    range_m: float = 1000.0
    tx_power_w_per_elem: float = 1.0
    scenario_type: str = "comms"
    required_snr_db: float = 10.0


@dataclass
class Constraints:
    """Metric thresholds checked by the constraint_check node."""

    min_directivity_dbi: float | None = None
    max_sidelobe_level_db: float | None = None
    extra: dict[str, float] = field(default_factory=dict)


def _edgefem_available() -> bool:
    try:
        import edgefem  # noqa: F401
        return True
    except ImportError:
        return False


def _dispatch(dispatcher: ToolDispatcher, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """Call an MCP tool and parse its JSON result."""
    result = json.loads(dispatcher.dispatch(tool, args))
    if not isinstance(result, dict):
        return {"result": result}
    return result


def _array_args(config: dict[str, Any]) -> dict[str, Any]:
    """Extract pattern/system tool arguments from a config dump."""
    array = config["array"]
    nx, ny = array["size"]
    dx_m, dy_m = array["spacing_m"]

    sweep = config.get("sweep")
    if sweep:
        freqs = sweep["freq_hz"]
        freq_hz = (freqs["start"] + freqs["stop"]) / 2
    else:
        # Half-wave spacing sets the design frequency.
        from scipy.constants import c as speed_of_light

        freq_hz = speed_of_light / (2 * dx_m)

    return {
        "nx": nx,
        "ny": ny,
        "dx_m": dx_m,
        "dy_m": dy_m,
        "freq_hz": freq_hz,
        "taper": array.get("taper", "uniform"),
    }


class _Nodes:
    """Pipeline node implementations bound to a dispatcher and run context."""

    def __init__(
        self,
        dispatcher: ToolDispatcher,
        run_ctx: RunContext,
        scenario: Scenario,
        constraints: Constraints,
    ) -> None:
        self.dispatcher = dispatcher
        self.run_ctx = run_ctx
        self.scenario = scenario
        self.constraints = constraints

    def validate_config(self, state: PipelineState) -> dict[str, Any]:
        with span("apab.node.validate_config"):
            try:
                config = ProjectConfig.model_validate(state["config"])
                # mode="json" keeps checkpointed state msgpack-friendly
                # (no enum objects).
                return {"config": config.model_dump(mode="json"), "errors": []}
            except Exception as exc:
                return {"errors": [f"validate_config: {exc}"]}

    def unit_cell(self, state: PipelineState) -> dict[str, Any]:
        with span("apab.node.unit_cell"):
            uc = state["config"].get("unit_cell")
            if not uc:
                return {}
            lattice = uc["lattice"]
            params = (uc.get("geometry") or {}).get("params") or {}
            sweep = state["config"].get("sweep") or {}
            freqs = sweep.get("freq_hz") or {}
            args = {
                "period_x": lattice["dx_m"],
                "period_y": lattice["dy_m"],
                # GeometryParams names: substrate_h_m, er, patch_w_m, patch_l_m
                "substrate_height": params.get("substrate_h_m", 0.000508),
                "substrate_eps_r": params.get("er", 2.2),
                "freq_start": freqs.get("start", 26e9),
                "freq_stop": freqs.get("stop", 30e9),
                "n_freq": freqs.get("n", 11),
                "patch_w_m": params.get("patch_w_m"),
                "patch_l_m": params.get("patch_l_m"),
            }
            result = _dispatch(self.dispatcher, "edgefem_run_unit_cell", args)
            if "error" in result:
                return {"errors": state.get("errors", []) + [f"unit_cell: {result['error']}"]}
            return {"unit_cell": result}

    def pattern(self, state: PipelineState) -> dict[str, Any]:
        with span("apab.node.pattern"):
            result = _dispatch(
                self.dispatcher, "pattern_compute", _array_args(state["config"]),
            )
            if "error" in result:
                return {"errors": state.get("errors", []) + [f"pattern: {result['error']}"]}
            return {"pattern": result}

    def system_eval(self, state: PipelineState) -> dict[str, Any]:
        with span("apab.node.system_eval"):
            if "pattern" not in state:
                return {}
            args = _array_args(state["config"])
            args.update({
                "bandwidth_hz": self.scenario.bandwidth_hz,
                "range_m": self.scenario.range_m,
                "tx_power_w_per_elem": self.scenario.tx_power_w_per_elem,
                "scenario_type": self.scenario.scenario_type,
                "required_snr_db": self.scenario.required_snr_db,
            })
            result = _dispatch(self.dispatcher, "system_evaluate", args)
            if "error" in result:
                return {"errors": state.get("errors", []) + [f"system_eval: {result['error']}"]}
            return {"system": result}

    def constraint_check(self, state: PipelineState) -> dict[str, Any]:
        with span("apab.node.constraint_check"):
            violations: list[str] = []
            pattern = state.get("pattern", {})

            c = self.constraints
            directivity = pattern.get("directivity_dbi")
            if c.min_directivity_dbi is not None and directivity is not None:
                if directivity < c.min_directivity_dbi:
                    violations.append(
                        f"directivity {directivity:.2f} dBi < "
                        f"required {c.min_directivity_dbi:.2f} dBi"
                    )
            sll = pattern.get("sidelobe_level_db")
            if c.max_sidelobe_level_db is not None and sll is not None:
                if sll > c.max_sidelobe_level_db:
                    violations.append(
                        f"sidelobe level {sll:.2f} dB > "
                        f"allowed {c.max_sidelobe_level_db:.2f} dB"
                    )
            return {"violations": violations}

    def plots(self, state: PipelineState) -> dict[str, Any]:
        with span("apab.node.plots"):
            if "pattern" not in state:
                return {}
            out = self.run_ctx.plots_dir / "pattern_cuts.png"
            args = _array_args(state["config"])
            args["output_path"] = str(out)
            result = _dispatch(self.dispatcher, "pattern_plot_cuts", args)
            if "error" in result:
                return {"errors": state.get("errors", []) + [f"plots: {result['error']}"]}
            return {"plots": [str(out)]}

    def report(self, state: PipelineState) -> dict[str, Any]:
        with span("apab.node.report"):
            report_path = self.run_ctx.report_dir / "pipeline_report.md"
            report_path.write_text(_render_report(state))

            from apab.core.provenance import build_manifest

            manifest = build_manifest(
                self.run_ctx.run_id,
                config=state.get("config"),
                artifacts=sorted(
                    str(p.relative_to(self.run_ctx.run_dir))
                    for p in self.run_ctx.artifacts_dir.rglob("*")
                    if p.is_file()
                ),
            )
            if state.get("errors"):
                manifest["status"] = "error"
            elif state.get("violations"):
                manifest["status"] = "constraint_violation"
            else:
                manifest["status"] = "success"
            (self.run_ctx.run_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, default=str)
            )
            return {"report_path": str(report_path)}


def _render_report(state: PipelineState) -> str:
    lines = ["# APAB Pipeline Report", ""]
    lines += [f"Run: `{state.get('run_id', '')}`", ""]

    pattern = state.get("pattern")
    if pattern:
        lines += [
            "## Pattern",
            "",
            f"- Directivity: {pattern.get('directivity_dbi', 'n/a'):.2f} dBi"
            if isinstance(pattern.get("directivity_dbi"), float)
            else f"- Directivity: {pattern.get('directivity_dbi', 'n/a')}",
            f"- Sidelobe level: {pattern.get('sidelobe_level_db', 'n/a')} dB",
            "",
        ]

    system = state.get("system")
    if system:
        lines += ["## System metrics", ""]
        lines += [f"- {k}: {v}" for k, v in system.items() if not isinstance(v, dict)]
        lines += [""]

    violations = state.get("violations") or []
    lines += ["## Constraints", ""]
    if violations:
        lines += [f"- VIOLATION: {v}" for v in violations]
    else:
        lines += ["- All constraints satisfied."]
    lines += [""]

    errors = state.get("errors") or []
    if errors:
        lines += ["## Errors", ""]
        lines += [f"- {e}" for e in errors]
        lines += [""]

    plots = state.get("plots") or []
    if plots:
        lines += ["## Plots", ""]
        lines += [f"- {p}" for p in plots]
        lines += [""]

    return "\n".join(lines)


def build_pipeline(
    config: ProjectConfig | dict[str, Any],
    *,
    scenario: Scenario | None = None,
    constraints: Constraints | None = None,
    workspace: Workspace | None = None,
    checkpoint: bool = True,
) -> tuple[Any, RunContext, PipelineState]:
    """Compile the pipeline graph.

    Returns ``(graph, run_ctx, initial_state)``. Invoke with::

        graph.invoke(initial_state, config={"configurable": {"thread_id": run_ctx.run_id}})

    or stream node-by-node with ``graph.stream(..., stream_mode="updates")``.
    With ``checkpoint`` true, state persists to
    ``<run_dir>/checkpoint.sqlite`` keyed by ``thread_id``, so a rerun
    with the same thread id resumes rather than recomputes.
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise ImportError(
            "The LangGraph pipeline requires the langgraph package. "
            "Install it with: pip install 'apab[langgraph]'"
        ) from exc

    if isinstance(config, dict):
        config = ProjectConfig.model_validate(config)

    workspace = workspace or Workspace(Path(config.project.workspace))
    workspace.ensure_dirs()
    run_ctx = workspace.new_run()

    mode = config.llm.redaction_mode
    dispatcher = ToolDispatcher(
        redaction_mode=mode.value if hasattr(mode, "value") else str(mode),
    )
    nodes = _Nodes(
        dispatcher,
        run_ctx,
        scenario or Scenario(),
        constraints or Constraints(),
    )

    graph = StateGraph(PipelineState)
    graph.add_node("validate_config", nodes.validate_config)
    graph.add_node("unit_cell", nodes.unit_cell)
    graph.add_node("pattern", nodes.pattern)
    graph.add_node("system_eval", nodes.system_eval)
    graph.add_node("constraint_check", nodes.constraint_check)
    graph.add_node("plots", nodes.plots)
    graph.add_node("report", nodes.report)

    def route_after_validate(state: PipelineState) -> str:
        if state.get("errors"):
            return "report"
        cfg = state["config"]
        wants_edgefem = (
            (cfg.get("solver") or {}).get("backend") == "edgefem"
            and cfg.get("unit_cell") is not None
        )
        if wants_edgefem and _edgefem_available():
            return "unit_cell"
        return "pattern"

    graph.add_edge(START, "validate_config")
    graph.add_conditional_edges(
        "validate_config",
        route_after_validate,
        {"unit_cell": "unit_cell", "pattern": "pattern", "report": "report"},
    )
    graph.add_edge("unit_cell", "pattern")
    graph.add_edge("pattern", "system_eval")
    graph.add_edge("system_eval", "constraint_check")
    graph.add_edge("constraint_check", "plots")
    graph.add_edge("plots", "report")
    graph.add_edge("report", END)

    checkpointer = None
    if checkpoint:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(
            run_ctx.run_dir / "checkpoint.sqlite", check_same_thread=False,
        )
        checkpointer = SqliteSaver(conn)

    compiled = graph.compile(checkpointer=checkpointer)

    initial_state: PipelineState = {
        "config": config.model_dump(mode="json"),
        "run_id": run_ctx.run_id,
        "run_dir": str(run_ctx.run_dir),
        "errors": [],
    }
    return compiled, run_ctx, initial_state


def run_pipeline(
    config: ProjectConfig | dict[str, Any],
    *,
    scenario: Scenario | None = None,
    constraints: Constraints | None = None,
    workspace: Workspace | None = None,
    checkpoint: bool = True,
) -> PipelineState:
    """Build and run the pipeline; returns the final state."""
    from apab.observability import init_observability, shutdown_observability

    graph, run_ctx, initial_state = build_pipeline(
        config,
        scenario=scenario,
        constraints=constraints,
        workspace=workspace,
        checkpoint=checkpoint,
    )

    cfg = config if isinstance(config, ProjectConfig) else ProjectConfig.model_validate(config)
    init_observability(cfg.observability, run_ctx=run_ctx)
    try:
        with span("apab.pipeline", **{"apab.run_id": run_ctx.run_id}):
            result: PipelineState = graph.invoke(
                initial_state,
                config={"configurable": {"thread_id": run_ctx.run_id}},
            )
        return result
    finally:
        shutdown_observability()
