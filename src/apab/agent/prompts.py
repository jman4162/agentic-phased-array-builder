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
