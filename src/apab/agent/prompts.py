"""System prompts for the APAB agent."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """\
You are an expert phased-array antenna design assistant.  You have access to \
a suite of engineering tools for:

1. **Unit-cell simulation** (EdgeFEM) — full-wave frequency sweeps, surface \
   impedance, S-parameter extraction, and Touchstone export.
2. **Array pattern computation** (phased-array-modeling) — full 2-D patterns, \
   E/H-plane cuts, multi-beam, null steering, and taper/window functions.
3. **System-level analysis** (phased-array-systems) — link budgets, radar \
   detection scenarios, architecture evaluation, and DOE trade studies with \
   Pareto analysis.
4. **Import/Export** — Touchstone (.sNp) import, far-field CSV import, \
   project init/validate, and data persistence.
5. **Visualization** — pattern cut plots, 3-D surface plots, and quick-look \
   summaries.

When the user asks you to design, analyse, or optimise a phased-array antenna:
- Break the problem into steps and explain your reasoning.
- **You MUST use the tool-calling function interface to invoke tools.** Always \
  prefer the structured tool/function calling mechanism. If you include a \
  tool call as JSON in your text, the system will attempt to parse it, but \
  structured calls are more reliable.
- Call tools one at a time, wait for results, then proceed to the next step.
- Present results clearly with key metrics (directivity, sidelobe level, \
  beamwidth, EIRP, etc.).
- Suggest improvements or trade-offs when appropriate.

Always call tools with physically realistic parameters. When uncertain about \
a parameter, state your assumptions before proceeding.

If a tool call returns an error, read the error message carefully and fix \
the arguments before retrying. Do not retry the same failing call more than \
twice. If still stuck, explain what went wrong and suggest next steps.
"""

# Prefix → display group name mapping for tool listing.
_TOOL_GROUPS: list[tuple[str, str]] = [
    ("edgefem_", "Unit-cell (EdgeFEM)"),
    ("pattern_", "Array patterns"),
    ("system_", "System analysis"),
    ("project_", "Project"),
    ("io_", "Import/Export"),
    ("plot_", "Visualization"),
    ("emtool_", "EM tool adapters"),
]


def _group_tool_names(tool_names: list[str]) -> str:
    """Group tool names by prefix and format as a Markdown section."""
    groups: dict[str, list[str]] = {}
    ungrouped: list[str] = []

    for name in sorted(tool_names):
        matched = False
        for prefix, label in _TOOL_GROUPS:
            if name.startswith(prefix):
                groups.setdefault(label, []).append(name)
                matched = True
                break
        if not matched:
            ungrouped.append(name)

    lines = ["\n## Available Tools"]
    for _, label in _TOOL_GROUPS:
        if label in groups:
            lines.append(f"- **{label}:** {', '.join(groups[label])}")
    if ungrouped:
        lines.append(f"- **Other:** {', '.join(ungrouped)}")

    return "\n".join(lines)


def build_system_prompt(
    config: dict[str, Any] | None = None,
    tool_names: list[str] | None = None,
) -> str:
    """Build a system prompt, optionally incorporating project config context."""
    parts = [SYSTEM_PROMPT]

    if tool_names:
        parts.append(_group_tool_names(tool_names))

    if config is not None:
        project_name = config.get("project", {}).get("name", "unnamed")
        parts.append(f"\nYou are working on project: **{project_name}**.")

        # Include array spec if present
        array = config.get("array")
        if array:
            size = array.get("size", [])
            if size:
                parts.append(
                    f"Default array: {size[0]}×{size[1]}, "
                    f"spacing {array.get('spacing_m', [])}, "
                    f"taper '{array.get('taper', 'uniform')}'."
                )

    return "\n".join(parts)


OPTIMIZE_PROMPT = """\
You are an autonomous phased-array antenna optimization agent. Your goal \
is to {direction} **{metric}** while satisfying all constraints.

{objective}

## Current Best
{best_summary}

## Recent Experiment History
{history}

## Strategy
{strategy}

## Your Task
Propose ONE design change that you think will improve {metric}. \
Call the appropriate tool (e.g. ``pattern_compute`` or \
``system_evaluate``) with your proposed parameters. \
Briefly explain your reasoning before calling the tool.

## Rules
- Change only one or two variables at a time.
- Build on the current best design, not the baseline.
- If the last 3 experiments were all discarded, try a completely \
  different region of the design space.
- Keep your explanation to 2-3 sentences.
"""


def build_optimize_prompt(
    protocol: Any,
    tracker: Any,
    is_baseline: bool = False,
) -> str:
    """Build the system prompt for an optimization experiment."""
    if is_baseline:
        return (
            "You are evaluating a baseline phased-array antenna design. "
            "Call system_evaluate with the baseline parameters to "
            "establish the starting point. Use reasonable defaults "
            "for any parameters not specified."
        )

    best = tracker.best
    if best is not None:
        best_summary = (
            f"{protocol.metric}: {best.metrics.get(protocol.metric, 0):.1f} "
            f"(experiment #{best.experiment_id:03d}, {best.description})"
        )
    else:
        best_summary = "No experiments yet."

    history = tracker.format_history(n=5, metric=protocol.metric)
    if not history:
        history = "(no experiments yet)"

    return OPTIMIZE_PROMPT.format(
        direction=protocol.direction,
        metric=protocol.metric,
        objective=protocol.objective,
        best_summary=best_summary,
        history=history,
        strategy=protocol.strategy or "Use engineering judgment.",
    )
