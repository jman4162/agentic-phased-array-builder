"""Tests for the local compute backend."""

from __future__ import annotations

import time

import pytest

from apab.compute.local import LocalBackend


class TestLocalBackendSubmit:
    """Test task submission."""

    def test_submit_returns_string_task_id(self) -> None:
        backend = LocalBackend(max_workers=2)
        task_id = backend.submit({"callable": lambda: 42})
        assert isinstance(task_id, str)
        assert len(task_id) > 0
        backend.shutdown()

    def test_submit_without_callable(self) -> None:
        backend = LocalBackend(max_workers=2)
        task_id = backend.submit({"description": "no-op task"})
        assert isinstance(task_id, str)
        status = backend.status(task_id)
        assert status["status"] == "completed"
        backend.shutdown()


class TestLocalBackendStatus:
    """Test status queries."""

    def test_completed_for_simple_task(self) -> None:
        backend = LocalBackend(max_workers=2)
        task_id = backend.submit({"callable": lambda: 42})
        # Wait for completion
        _wait_for_completion(backend, task_id)
        status = backend.status(task_id)
        assert status["status"] == "completed"
        backend.shutdown()

    def test_failed_for_raising_task(self) -> None:
        def _boom() -> None:
            raise RuntimeError("intentional failure")

        backend = LocalBackend(max_workers=2)
        task_id = backend.submit({"callable": _boom})
        _wait_for_completion(backend, task_id)
        status = backend.status(task_id)
        assert status["status"] == "failed"
        assert "intentional failure" in status["error"]
        backend.shutdown()

    def test_unknown_task_id(self) -> None:
        backend = LocalBackend(max_workers=2)
        status = backend.status("nonexistent_id")
        assert status["status"] == "unknown"
        backend.shutdown()


class TestLocalBackendFetchArtifacts:
    """Test artifact retrieval."""

    def test_fetch_result_after_completion(self) -> None:
        backend = LocalBackend(max_workers=2)
        task_id = backend.submit({"callable": lambda: 42})
        _wait_for_completion(backend, task_id)
        artifacts = backend.fetch_artifacts(task_id)
        assert artifacts["artifacts"] == [42]
        backend.shutdown()

    def test_fetch_returns_not_ready_while_running(self) -> None:
        def _slow() -> int:
            time.sleep(2)
            return 99

        backend = LocalBackend(max_workers=1)
        task_id = backend.submit({"callable": _slow})
        # Immediately check before completion
        artifacts = backend.fetch_artifacts(task_id)
        # The task may or may not have started, but should not be done
        if "status" in artifacts:
            assert artifacts["status"] in ("not_ready", "pending", "running")
        backend.shutdown(wait=True)

    def test_fetch_failed_task(self) -> None:
        def _boom() -> None:
            raise ValueError("broken")

        backend = LocalBackend(max_workers=2)
        task_id = backend.submit({"callable": _boom})
        _wait_for_completion(backend, task_id)
        artifacts = backend.fetch_artifacts(task_id)
        assert artifacts["artifacts"] == []
        assert artifacts["status"] == "failed"
        backend.shutdown()

    def test_fetch_unknown_task(self) -> None:
        backend = LocalBackend(max_workers=2)
        artifacts = backend.fetch_artifacts("does_not_exist")
        assert artifacts["artifacts"] == []
        backend.shutdown()

    def test_callable_with_args(self) -> None:
        def _add(a: int, b: int) -> int:
            return a + b

        backend = LocalBackend(max_workers=2)
        task_id = backend.submit({"callable": _add, "args": (3, 7)})
        _wait_for_completion(backend, task_id)
        artifacts = backend.fetch_artifacts(task_id)
        assert artifacts["artifacts"] == [10]
        backend.shutdown()

    def test_callable_with_kwargs(self) -> None:
        def _greet(name: str = "world") -> str:
            return f"hello {name}"

        backend = LocalBackend(max_workers=2)
        task_id = backend.submit({"callable": _greet, "kwargs": {"name": "apab"}})
        _wait_for_completion(backend, task_id)
        artifacts = backend.fetch_artifacts(task_id)
        assert artifacts["artifacts"] == ["hello apab"]
        backend.shutdown()


# ── Helpers ──


def _wait_for_completion(backend: LocalBackend, task_id: str, timeout: float = 5.0) -> None:
    """Poll until a task reaches a terminal state or timeout is reached."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st = backend.status(task_id)
        if st["status"] in ("completed", "failed"):
            return
        time.sleep(0.05)
    pytest.fail(f"Task {task_id} did not complete within {timeout}s")
