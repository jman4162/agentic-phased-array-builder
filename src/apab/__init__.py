"""APAB — Agentic Phased Array Builder."""

from __future__ import annotations

__version__ = "0.3.0"

__all__ = [
    "__version__",
    # Schemas
    "ArraySpec",
    "CouplingResult",
    "GeometryKind",
    "LatticeSpec",
    "PatternResult",
    "ProjectConfig",
    "ScanPoint",
    "SimResult",
    "SweepSpec",
    "UnitCellSpec",
    # Engines
    "PAMPatternEngine",
    "PASSystemEngine",
    # Config
    "load_config",
    # Agent
    "AgentOrchestrator",
    "LLMProvider",
    "get_provider",
]

_LAZY_IMPORTS: dict[str, str] = {
    "ArraySpec": "apab.core.schemas",
    "CouplingResult": "apab.core.schemas",
    "GeometryKind": "apab.core.schemas",
    "LatticeSpec": "apab.core.schemas",
    "PatternResult": "apab.core.schemas",
    "ProjectConfig": "apab.core.schemas",
    "ScanPoint": "apab.core.schemas",
    "SimResult": "apab.core.schemas",
    "SweepSpec": "apab.core.schemas",
    "UnitCellSpec": "apab.core.schemas",
    "PAMPatternEngine": "apab.pattern.wrappers_pam",
    "PASSystemEngine": "apab.system.wrappers_pas",
    "load_config": "apab.core.config",
    "AgentOrchestrator": "apab.agent.orchestrator",
    "LLMProvider": "apab.agent.provider_registry",
    "get_provider": "apab.agent.provider_registry",
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'apab' has no attribute {name!r}")
