"""Protocol definition for compute backends."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ComputeBackend(Protocol):
    """Protocol for compute backends that execute simulation tasks.

    The default implementation is :class:`apab.compute.local.LocalBackend`
    which runs tasks locally using a thread pool.  Future implementations may
    dispatch work to cloud services (AWS Batch, GCP Cloud Run, etc.).
    """

    name: str

    def submit(self, task: dict[str, Any]) -> str:
        """Submit a task for execution.

        Parameters
        ----------
        task:
            A dictionary describing the work to perform.  If a ``"callable"``
            key is present, its value should be a callable that will be
            invoked.  Additional keys may carry arguments or metadata.

        Returns
        -------
        str
            A unique task ID that can be used with :meth:`status` and
            :meth:`fetch_artifacts`.
        """
        ...

    def status(self, task_id: str) -> dict[str, Any]:
        """Query the status of a previously submitted task.

        Returns
        -------
        dict
            Must contain a ``"status"`` key with one of:
            ``"pending"``, ``"running"``, ``"completed"``, ``"failed"``,
            or ``"unknown"``.  On failure, an ``"error"`` key should be
            present.
        """
        ...

    def fetch_artifacts(self, task_id: str) -> dict[str, Any]:
        """Retrieve artifacts produced by a completed task.

        Returns
        -------
        dict
            Must contain an ``"artifacts"`` key (list).  If the task is
            not yet finished, ``"status"`` should be ``"not_ready"``.
        """
        ...
