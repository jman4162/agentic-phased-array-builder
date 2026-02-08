"""Report builder: generates markdown reports from run bundles."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ReportBuilder:
    """Build reports from an APAB run bundle directory.

    Parameters
    ----------
    run_dir:
        Path to the run bundle directory (e.g. ``workspace/runs/<run_id>/``).
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self._manifest: dict[str, Any] | None = None

    def _load_manifest(self) -> dict[str, Any]:
        """Load the run manifest if available."""
        if self._manifest is not None:
            return self._manifest

        manifest_path = self.run_dir / "manifest.json"
        if manifest_path.exists():
            self._manifest = json.loads(manifest_path.read_text())
        else:
            self._manifest = {}
        return self._manifest

    def _load_json_artifact(self, *parts: str) -> dict[str, Any] | None:
        """Load a JSON artifact from the run directory."""
        path = self.run_dir.joinpath(*parts)
        if path.exists():
            try:
                result: dict[str, Any] = json.loads(path.read_text())
                return result
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def _list_artifacts(self, subdir: str, suffix: str = "") -> list[Path]:
        """List artifact files in a subdirectory."""
        d = self.run_dir / subdir
        if not d.exists():
            return []
        files = sorted(d.iterdir())
        if suffix:
            files = [f for f in files if f.suffix == suffix]
        return files

    def build_markdown(self) -> str:
        """Build a Markdown report and return it as a string."""
        manifest = self._load_manifest()
        sections = []

        # Title
        run_id = manifest.get("run_id", self.run_dir.name)
        sections.append(f"# APAB Run Report: {run_id}\n")
        sections.append(
            f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        )

        if manifest.get("timestamp"):
            sections.append(f"**Run timestamp**: {manifest['timestamp']}\n")

        # Config summary
        sections.append("## Configuration\n")
        if manifest.get("config_hash"):
            sections.append(f"- Config hash: `{manifest['config_hash']}`")
        if manifest.get("geometry_hash"):
            sections.append(f"- Geometry hash: `{manifest['geometry_hash']}`")
        if manifest.get("sweep_hash"):
            sections.append(f"- Sweep hash: `{manifest['sweep_hash']}`")
        if manifest.get("solver_version"):
            sections.append(f"- Solver: {manifest['solver_version']}")
        sections.append("")

        has_content = False

        # Coupling results
        coupling_files = self._list_artifacts("coupling", ".json")
        if coupling_files:
            has_content = True
            sections.append("## Coupling Analysis\n")
            for f in coupling_files:
                data = self._load_json_artifact("coupling", f.name)
                if data:
                    sections.append(f"### {f.stem}\n")
                    if "freq_hz" in data:
                        sections.append(f"- Frequencies: {len(data['freq_hz'])} points")
                    if "scan_blindness_flags" in data:
                        flags = data["scan_blindness_flags"]
                        if flags:
                            sections.append(f"- Scan blindness detected: {len(flags)} points")
                        else:
                            sections.append("- No scan blindness detected")
                    sections.append("")

        # Pattern results
        pattern_files = self._list_artifacts("patterns", ".json")
        if pattern_files:
            has_content = True
            sections.append("## Array Pattern\n")
            for f in pattern_files:
                data = self._load_json_artifact("patterns", f.name)
                if data:
                    sections.append(f"### {f.stem}\n")
                    if "directivity_dbi" in data:
                        sections.append(f"- Directivity: {data['directivity_dbi']:.2f} dBi")
                    if "sidelobe_level_db" in data:
                        sll = data["sidelobe_level_db"]
                        if sll is not None:
                            sections.append(f"- Sidelobe level: {sll:.2f} dB")
                    meta = data.get("metadata", {})
                    if "e_plane_hpbw_deg" in meta:
                        sections.append(f"- E-plane HPBW: {meta['e_plane_hpbw_deg']:.2f}°")
                    if "h_plane_hpbw_deg" in meta:
                        sections.append(f"- H-plane HPBW: {meta['h_plane_hpbw_deg']:.2f}°")
                    sections.append("")

        # Pattern plots
        plot_files = self._list_artifacts("plots", ".png")
        if plot_files:
            has_content = True
            sections.append("## Plots\n")
            for f in plot_files:
                rel = f.relative_to(self.run_dir)
                sections.append(f"![{f.stem}]({rel})\n")

        # System results
        system_files = self._list_artifacts("system", ".json")
        if system_files:
            has_content = True
            sections.append("## System Analysis\n")
            for f in system_files:
                data = self._load_json_artifact("system", f.name)
                if data:
                    sections.append(f"### {f.stem}\n")
                    for key, val in sorted(data.items()):
                        if isinstance(val, (int, float, str, bool)):
                            sections.append(f"- {key}: {val}")
                    sections.append("")

        # Artifacts list
        artifacts = manifest.get("artifacts", [])
        if artifacts:
            sections.append("## Artifacts\n")
            for a in artifacts:
                sections.append(f"- `{a}`")
            sections.append("")

        # If nothing was found, add a note
        if not has_content:
            sections.append(
                "No artifacts found in this run bundle. "
                "Run the analysis first with `apab run` or `apab design`.\n"
            )

        return "\n".join(sections)
