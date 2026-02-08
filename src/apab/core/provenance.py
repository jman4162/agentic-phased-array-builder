"""Provenance tracking for APAB run bundles."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from typing import Any


def hash_config(config_dict: dict[str, Any]) -> str:
    """Deterministic hash of a config dictionary."""
    raw = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def hash_geometry(geometry_dict: dict[str, Any]) -> str:
    """Deterministic hash of unit-cell geometry parameters."""
    raw = json.dumps(geometry_dict, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def hash_sweep(sweep_dict: dict[str, Any]) -> str:
    """Deterministic hash of sweep parameters."""
    raw = json.dumps(sweep_dict, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _get_dependency_versions() -> dict[str, str]:
    """Collect installed versions of key APAB dependencies."""
    packages = [
        "apab",
        "edgefem",
        "phased-array-modeling",
        "phased-array-systems",
        "numpy",
        "scipy",
        "pydantic",
        "mcp",
    ]
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def build_manifest(
    run_id: str,
    timestamp: str | None = None,
    config: dict[str, Any] | None = None,
    artifacts: list[str] | None = None,
) -> dict[str, Any]:
    """Build a run-bundle manifest dictionary with full provenance."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": timestamp,
        "config_hash": hash_config(config) if config else "",
        "dependency_versions": _get_dependency_versions(),
        "artifacts": artifacts or [],
        "status": "created",
    }

    # Extract sub-hashes if config sections are present
    if config:
        geom = config.get("unit_cell", {}).get("geometry")
        if geom:
            manifest["geometry_hash"] = hash_geometry(geom)

        sweep = config.get("sweep")
        if sweep:
            manifest["sweep_hash"] = hash_sweep(sweep)

        solver = config.get("solver", {})
        solver_backend = solver.get("backend", "")
        solver_version = ""
        if solver_backend:
            try:
                solver_version = importlib.metadata.version(solver_backend)
            except importlib.metadata.PackageNotFoundError:
                pass
        manifest["solver_version"] = f"{solver_backend} {solver_version}".strip()

        llm = config.get("llm", {})
        manifest["provider_name"] = llm.get("provider", "")
        manifest["model_name"] = llm.get("model", "")

    return manifest
