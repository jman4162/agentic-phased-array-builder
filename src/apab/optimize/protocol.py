"""Parse a research protocol (research.md) into structured config."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Constraint:
    metric: str
    op: str  # "<", ">", "<=", ">="
    value: float


@dataclass
class ResearchProtocol:
    """Parsed research protocol for the optimization loop."""

    objective: str = ""
    metric: str = "eirp_dbw"
    direction: str = "maximize"  # "maximize" or "minimize"
    constraints: list[Constraint] = field(default_factory=list)
    strategy: str = ""
    raw_text: str = ""


def load_protocol(path: Path) -> ResearchProtocol:
    """Load and parse a research.md protocol file."""
    text = path.read_text()
    proto = ResearchProtocol(raw_text=text)

    # Extract objective section
    obj_match = re.search(
        r"##\s*Objective\s*\n(.*?)(?=\n##|\Z)", text, re.DOTALL
    )
    if obj_match:
        proto.objective = obj_match.group(1).strip()

    # Extract metric
    metric_match = re.search(
        r"Primary:\s*(\w+)\s*\((\w+)\)", text
    )
    if metric_match:
        proto.metric = metric_match.group(1)
        proto.direction = metric_match.group(2)

    # Extract constraints: "Constraint: cost_usd < 10000, snr_db > 10"
    constraint_match = re.search(
        r"Constraint:\s*(.+)", text
    )
    if constraint_match:
        parts = constraint_match.group(1).split(",")
        for part in parts:
            m = re.match(
                r"\s*(\w+)\s*([<>]=?)\s*([\d.eE+\-]+)", part.strip()
            )
            if m:
                proto.constraints.append(Constraint(
                    metric=m.group(1),
                    op=m.group(2),
                    value=float(m.group(3)),
                ))

    # Extract strategy section
    strat_match = re.search(
        r"##\s*Strategy\s*\n(.*?)(?=\n##|\Z)", text, re.DOTALL
    )
    if strat_match:
        proto.strategy = strat_match.group(1).strip()

    return proto
