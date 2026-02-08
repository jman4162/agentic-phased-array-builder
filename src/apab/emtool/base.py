"""Protocol definition for external EM tool adapters."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ExternalEMToolAdapter(Protocol):
    """Protocol for adapters that bridge APAB to external EM simulation tools.

    Implementations wrap commercial solvers (HFSS, CST, FEKO, etc.) and expose
    a uniform interface for opening projects, running sweeps, and exporting
    results back into the APAB data pipeline.
    """

    name: str

    def capabilities(self) -> dict[str, Any]:
        """Return a dictionary describing the adapter's capabilities.

        Expected keys include ``"solver"``, ``"supported_formats"``,
        ``"supports_remote"``, etc.
        """
        ...

    def open_project(self, project_ref: str) -> None:
        """Open or attach to an existing EM project.

        Parameters
        ----------
        project_ref:
            Path or URI to the project file (e.g. ``*.aedt``, ``*.cst``).
        """
        ...

    def run_sweep(self, request: Any) -> str:
        """Launch a parameter/frequency sweep on the external tool.

        Parameters
        ----------
        request:
            Solver-specific sweep configuration (frequencies, scan angles, etc.).

        Returns
        -------
        str
            A unique run-handle ID that can be used to query status and
            retrieve results.
        """
        ...

    def export_results(self, run_handle: str, out_dir: str) -> dict[str, Any]:
        """Export completed results to *out_dir*.

        Parameters
        ----------
        run_handle:
            The ID returned by :meth:`run_sweep`.
        out_dir:
            Directory where exported files (Touchstone, CSV, etc.) are written.

        Returns
        -------
        dict
            Mapping with at least ``"files"`` (list of exported paths) and
            ``"format"`` (export format description).
        """
        ...
