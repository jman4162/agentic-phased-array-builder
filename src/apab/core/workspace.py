"""Workspace and run-bundle management for APAB."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path


class RunContext:
    """Context for a single APAB run within a workspace."""

    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.artifacts_dir = run_dir / "artifacts"
        self.coupling_dir = self.artifacts_dir / "coupling"
        self.patterns_dir = self.artifacts_dir / "patterns"
        self.system_dir = self.artifacts_dir / "system"
        self.emtool_dir = self.artifacts_dir / "emtool"
        self.plots_dir = self.artifacts_dir / "plots"
        self.report_dir = self.artifacts_dir / "report"

    def ensure_dirs(self) -> None:
        """Create all artifact subdirectories."""
        for d in [
            self.coupling_dir,
            self.patterns_dir,
            self.system_dir,
            self.emtool_dir,
            self.plots_dir,
            self.report_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


class Workspace:
    """Manages the APAB workspace directory tree."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.runs_dir = self.root / "runs"
        self.cache_dir = self.root / "cache"

    def ensure_dirs(self) -> None:
        """Create the workspace directory structure."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def new_run(self, run_id: str | None = None) -> RunContext:
        """Create a new run context with unique ID and directory structure."""
        if run_id is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            short = uuid.uuid4().hex[:8]
            run_id = f"{ts}_{short}"
        run_dir = self.runs_dir / run_id
        ctx = RunContext(run_dir, run_id)
        ctx.ensure_dirs()
        return ctx

    def is_within_workspace(self, path: str | Path) -> bool:
        """Check if a path is within the workspace root."""
        try:
            resolved = Path(path).resolve()
            return str(resolved).startswith(str(self.root))
        except (OSError, ValueError):
            return False


def validate_path_within(path: str | Path, root: str | Path) -> Path:
    """Resolve *path* and verify it is inside *root*.

    Raises :class:`ValueError` if the path escapes the root directory
    (e.g. via ``../`` traversal).
    """
    resolved = Path(path).resolve()
    root_resolved = Path(root).resolve()
    if not str(resolved).startswith(str(root_resolved)):
        raise ValueError(
            f"Path {path!r} resolves to {resolved}, "
            f"which is outside the allowed root {root_resolved}"
        )
    return resolved


def reject_path_traversal(path: str | Path) -> Path:
    """Reject paths containing ``..`` components that could escape directories.

    Unlike :func:`validate_path_within`, this does not require a root
    directory — it simply rejects any path with parent-directory traversal.

    Raises :class:`ValueError` if the path contains ``..`` components.
    """
    p = Path(path)
    if ".." in p.parts:
        raise ValueError(
            f"Path {path!r} contains '..' traversal and is not allowed"
        )
    return p
