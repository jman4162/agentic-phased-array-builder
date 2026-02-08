"""Local compute backend using a thread pool."""

from __future__ import annotations

import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any


class LocalBackend:
    """Execute tasks locally via :class:`~concurrent.futures.ThreadPoolExecutor`.

    This is the default compute backend shipped with APAB.  It stores
    submitted futures in memory and exposes a uniform submit/status/fetch
    interface matching the :class:`~apab.compute.base.ComputeBackend` protocol.
    """

    name: str = "local"

    def __init__(self, max_workers: int = 4) -> None:
        self._max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: dict[str, dict[str, Any]] = {}

    # ── Protocol methods ──

    def submit(self, task: dict[str, Any]) -> str:
        """Submit a task for local execution.

        If *task* contains a ``"callable"`` key, that callable is submitted to
        the thread pool.  An optional ``"args"`` key (tuple) and ``"kwargs"``
        key (dict) are forwarded to the callable.

        Returns a unique task ID string.
        """
        task_id = uuid.uuid4().hex

        callable_fn = task.get("callable")
        if callable_fn is None:
            # Nothing to execute -- mark as completed with no result
            self._tasks[task_id] = {
                "future": None,
                "task": task,
                "status": "completed",
                "result": None,
            }
            return task_id

        args = task.get("args", ())
        kwargs = task.get("kwargs", {})
        future: Future[Any] = self._executor.submit(callable_fn, *args, **kwargs)

        self._tasks[task_id] = {
            "future": future,
            "task": task,
        }
        return task_id

    def status(self, task_id: str) -> dict[str, Any]:
        """Return the current status of *task_id*."""
        entry = self._tasks.get(task_id)
        if entry is None:
            return {"status": "unknown"}

        # Tasks without a callable are pre-resolved
        if entry.get("future") is None:
            return {"status": entry.get("status", "completed")}

        future: Future[Any] = entry["future"]
        if future.running():
            return {"status": "running"}
        if not future.done():
            return {"status": "pending"}

        # Future is done -- check for exception
        exc = future.exception()
        if exc is not None:
            return {"status": "failed", "error": str(exc)}

        return {"status": "completed"}

    def fetch_artifacts(self, task_id: str) -> dict[str, Any]:
        """Return artifacts once the task has completed."""
        entry = self._tasks.get(task_id)
        if entry is None:
            return {"artifacts": [], "status": "unknown"}

        # Pre-resolved (no callable)
        if entry.get("future") is None:
            return {"artifacts": [entry.get("result")]}

        future: Future[Any] = entry["future"]
        if not future.done():
            return {"artifacts": [], "status": "not_ready"}

        exc = future.exception()
        if exc is not None:
            return {"artifacts": [], "status": "failed", "error": str(exc)}

        return {"artifacts": [future.result()]}

    # ── Lifecycle helpers ──

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the thread pool."""
        self._executor.shutdown(wait=wait)
