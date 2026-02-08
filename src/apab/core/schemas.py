"""Pydantic schemas for APAB configuration and data models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ── Enums ──


class RedactionMode(str, Enum):
    none = "none"
    metadata_only = "metadata_only"
    strict = "strict"


class LatticeType(str, Enum):
    rect = "rect"
    triangular = "triangular"


class GeometryKind(str, Enum):
    patch = "patch"
    slot = "slot"
    dipole = "dipole"
    custom = "custom"


# ── LLM / MCP / Compute specs ──


class LLMSpec(BaseModel):
    provider: str = "ollama"
    model: str = "qwen2.5-coder:14b"
    base_url: str = "http://localhost:11434"
    api_key_env: str | None = None
    redaction_mode: RedactionMode = RedactionMode.none
    extra: dict[str, Any] = Field(default_factory=dict)


class MCPSpec(BaseModel):
    mode: str = "local"
    server_host: str = "127.0.0.1"
    server_port: int = 8765


class ComputeSpec(BaseModel):
    backend: str = "local"
    settings: dict[str, Any] = Field(default_factory=dict)


# ── Solver ──


class SolverSpec(BaseModel):
    backend: str = "edgefem"
    settings: dict[str, Any] = Field(default_factory=dict)


# ── Unit cell ──


class LatticeSpec(BaseModel):
    type: LatticeType = LatticeType.rect
    dx_m: float
    dy_m: float


class GeometryParams(BaseModel):
    patch_w_m: float | None = None
    patch_l_m: float | None = None
    substrate_h_m: float | None = None
    er: float | None = None
    model_config = {"extra": "allow"}


class GeometrySpec(BaseModel):
    kind: GeometryKind = GeometryKind.patch
    params: dict[str, Any] = Field(default_factory=dict)


class PortSpec(BaseModel):
    count: int = 1
    description: str = ""


class PolarizationSpec(BaseModel):
    basis: list[str] = Field(default_factory=lambda: ["H", "V"])


class UnitCellSpec(BaseModel):
    lattice: LatticeSpec
    geometry: GeometrySpec
    ports: PortSpec = Field(default_factory=PortSpec)
    polarization: PolarizationSpec = Field(default_factory=PolarizationSpec)


# ── Sweep ──


class FreqRange(BaseModel):
    start: float
    stop: float
    n: int


class ScanSpec(BaseModel):
    theta_deg: list[float] = Field(default_factory=lambda: [0.0, 60.0])
    phi_deg: list[float] = Field(default_factory=lambda: [0.0, 90.0])
    n_theta: int = 13
    n_phi: int = 7


class SweepSpec(BaseModel):
    freq_hz: FreqRange
    scan: ScanSpec = Field(default_factory=ScanSpec)
    polarization: list[str] = Field(default_factory=lambda: ["H", "V"])


class ScanPoint(BaseModel):
    theta_deg: float
    phi_deg: float


# ── Array ──


class ArraySpec(BaseModel):
    size: list[int] = Field(default_factory=lambda: [16, 16])
    spacing_m: list[float] = Field(default_factory=lambda: [0.005, 0.005])
    taper: str = "uniform"
    steer: ScanPoint = Field(default_factory=lambda: ScanPoint(theta_deg=0, phi_deg=0))


# ── Outputs ──


class OutputsSpec(BaseModel):
    export_touchstone: bool = False
    export_hdf5: bool = True
    plots: list[str] = Field(default_factory=list)


# ── Project config ──


class ProjectMeta(BaseModel):
    name: str
    workspace: str = "./workspace"


class ProjectConfig(BaseModel):
    project: ProjectMeta
    llm: LLMSpec = Field(default_factory=LLMSpec)
    mcp: MCPSpec = Field(default_factory=MCPSpec)
    compute: ComputeSpec = Field(default_factory=ComputeSpec)
    solver: SolverSpec = Field(default_factory=SolverSpec)
    unit_cell: UnitCellSpec | None = None
    sweep: SweepSpec | None = None
    array: ArraySpec = Field(default_factory=ArraySpec)
    outputs: OutputsSpec = Field(default_factory=OutputsSpec)


# ── Simulation request / result ──


class SimRequest(BaseModel):
    unit_cell: UnitCellSpec
    sweep: SweepSpec
    solver: SolverSpec = Field(default_factory=SolverSpec)


class SimResult(BaseModel):
    freq_hz: list[float]
    scan_points: list[ScanPoint]
    polarizations: list[str]
    s_params: Any = None  # complex ndarray serialized as list
    metadata: dict[str, Any] = Field(default_factory=dict)


class CouplingResult(BaseModel):
    freq_hz: list[float]
    scan_points: list[ScanPoint]
    polarizations: list[str]
    gamma_active: Any = None  # complex ndarray serialized as list
    z_active: Any = None  # complex ndarray serialized as list
    scan_blindness_flags: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PatternResult(BaseModel):
    theta_deg: list[float] = Field(default_factory=list)
    phi_deg: list[float] = Field(default_factory=list)
    pattern_db: Any = None  # ndarray serialized as list
    directivity_dbi: float | None = None
    sidelobe_level_db: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Finite-array (reserved for future) ──


class FiniteArraySimRequest(BaseModel):
    array_size: tuple[int, int]
    element_map: dict[str, Any] = Field(default_factory=dict)
    excitations: dict[str, Any] = Field(default_factory=dict)
    freq_hz: list[float] = Field(default_factory=list)
    boundary: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)


# ── Run bundle manifest ──


class RunBundleManifest(BaseModel):
    run_id: str
    timestamp: str
    config_hash: str = ""
    geometry_hash: str = ""
    sweep_hash: str = ""
    solver_version: str = ""
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    provider_name: str = ""
    model_name: str = ""
    artifacts: list[str] = Field(default_factory=list)
    status: str = "created"
