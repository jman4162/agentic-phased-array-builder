"""Workspace and run-bundle management for APAB."""

from __future__ import annotations

import os
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
            return Path(path).resolve().is_relative_to(self.root)
        except (OSError, ValueError):
            return False


def validate_path_within(path: str | Path, root: str | Path) -> Path:
    """Resolve *path* and verify it is inside *root*.

    Raises :class:`ValueError` if the path escapes the root directory
    (e.g. via ``../`` traversal).
    """
    resolved = Path(path).resolve()
    root_resolved = Path(root).resolve()
    # is_relative_to, not str.startswith: "/ws2/x" must not pass for root "/ws".
    if not resolved.is_relative_to(root_resolved):
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


# ── output root for tool-written files ───────────────────────────────────
#
# Tools that write files take paths from the model. Those paths must land
# inside the workspace: relative paths go to a default output directory,
# absolute paths must already be inside the root, and ``..`` is refused.
# The root is set by whoever owns the session -- the agent orchestrator per
# run, ``create_server`` from its config -- and otherwise falls back to
# ``$APAB_WORKSPACE`` or ``./workspace``. A ``workspace`` argument supplied
# by the model can narrow the root but never widen it.

_output_root: Path | None = None
_output_dir: Path | None = None


def set_output_context(root: str | Path | None, default_dir: str | Path | None = None) -> None:
    """Set the root tool outputs must stay in, and where relative paths go.

    ``default_dir`` defaults to ``<root>/artifacts`` and must be inside
    ``root``. Pass ``root=None`` to clear the context.
    """
    global _output_root, _output_dir
    if root is None:
        _output_root = _output_dir = None
        return
    root_path = Path(root).resolve()
    out_dir = Path(default_dir).resolve() if default_dir is not None else root_path / "artifacts"
    if not out_dir.is_relative_to(root_path):
        raise ValueError(f"default output dir {out_dir} is outside root {root_path}")
    _output_root, _output_dir = root_path, out_dir


def output_root() -> Path:
    """The directory every tool-written file must stay inside."""
    if _output_root is not None:
        return _output_root
    return Path(os.environ.get("APAB_WORKSPACE", "./workspace")).resolve()


def output_dir() -> Path:
    """Where tool outputs given as bare relative paths are written."""
    return _output_dir if _output_dir is not None else output_root() / "artifacts"


def resolve_output_path(path: str | Path) -> Path:
    """Resolve a model-supplied output path to a location inside the root.

    Relative paths are placed under :func:`output_dir`; absolute paths must
    already be inside :func:`output_root`. The parent directory is created.
    Raises :class:`ValueError` for ``..`` components or paths outside the root.
    """
    p = reject_path_traversal(path)
    target = p if p.is_absolute() else output_dir() / p
    resolved = validate_path_within(target, output_root())
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_workspace_arg(workspace: str | Path) -> Path:
    """Resolve a model-supplied ``workspace`` argument inside the root.

    Relative values resolve against the current directory, as before, and
    the result must be the root or a directory under it.
    """
    reject_path_traversal(workspace)
    return validate_path_within(workspace, output_root())
