#!/usr/bin/env python3
"""Example 08: Deterministic LangGraph pipeline.

Runs the fixed engineering sequence (validate -> pattern -> system ->
constraints -> plots -> report) as a LangGraph graph with SQLite
checkpointing, streaming node-by-node progress. No LLM involved: this
is the reproducible counterpart to the agent-driven examples.

Requirements:
    pip install "apab[langgraph]"

EdgeFEM is optional; without it the unit_cell node is skipped.
"""

import sys
import tempfile


def main() -> int:
    print("=" * 50)
    print("APAB Example 08: LangGraph Golden Pipeline")
    print("=" * 50)

    try:
        from apab.adapters.langgraph_pipeline import (
            Constraints,
            Scenario,
            build_pipeline,
        )
    except ImportError as exc:
        print(f"LangGraph is not installed ({exc}).")
        print('Install with: pip install "apab[langgraph]"')
        return 1

    from apab.core.schemas import ProjectConfig

    workspace = tempfile.mkdtemp(prefix="apab_pipeline_")
    config = ProjectConfig.model_validate({
        "project": {"name": "pipeline_demo", "workspace": workspace},
        "array": {
            "size": [8, 8],
            "spacing_m": [0.0054, 0.0054],  # half-wave at 28 GHz
            "taper": "taylor",
        },
    })

    graph, run_ctx, initial_state = build_pipeline(
        config,
        scenario=Scenario(bandwidth_hz=200e6, range_m=500.0),
        constraints=Constraints(
            min_directivity_dbi=20.0,
            max_sidelobe_level_db=-15.0,
        ),
    )
    thread = {"configurable": {"thread_id": run_ctx.run_id}}

    print(f"\nRun bundle: {run_ctx.run_dir}\n")
    final_state = initial_state
    for update in graph.stream(initial_state, config=thread, stream_mode="updates"):
        for node, delta in update.items():
            keys = ", ".join(delta.keys()) if delta else "no state change"
            print(f"  [{node}] -> {keys}")
        final_state = graph.get_state(thread).values

    pattern = final_state.get("pattern", {})
    print(f"\nDirectivity: {pattern.get('directivity_dbi', float('nan')):.2f} dBi")
    print(f"Sidelobe level: {pattern.get('sidelobe_level_db')} dB")
    violations = final_state.get("violations") or []
    print(f"Constraint violations: {violations or 'none'}")
    print(f"Report: {final_state.get('report_path')}")

    # Checkpointing: re-invoking the same thread id resumes from the
    # saved state instead of recomputing.
    resumed = graph.get_state(thread)
    print(f"Checkpointed steps: {len(list(graph.get_state_history(thread)))}")
    print(f"Resumable thread id: {resumed.config['configurable']['thread_id']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
