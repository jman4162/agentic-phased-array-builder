"""Tests for the apab doctor command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from apab.commands.doctor import (
    _check_core_deps,
    _check_edgefem,
    _check_model,
    _check_ollama_package,
    _check_ollama_server,
    _check_python,
)


class TestCheckPython:
    def test_passes_on_current_python(self):
        status, name, detail = _check_python()
        assert status == "pass"
        assert "Python" in name


class TestCheckCoreDeps:
    def test_passes_when_all_present(self):
        status, name, detail = _check_core_deps()
        assert status == "pass"

    def test_fails_when_missing(self):
        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            status, _, detail = _check_core_deps()
            assert status == "fail"
            assert "numpy" in detail


class TestCheckEdgefem:
    def test_returns_warn_when_missing(self):
        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "edgefem":
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            status, _, _ = _check_edgefem()
            assert status == "warn"


class TestCheckOllamaPackage:
    def test_passes_when_present(self):
        # ollama is installed in test env
        status, _, _ = _check_ollama_package()
        assert status == "pass"

    def test_fails_when_missing(self):
        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "ollama":
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            status, _, detail = _check_ollama_package()
            assert status == "fail"
            assert "pip install" in detail


class TestCheckOllamaServer:
    @patch("httpx.get")
    @patch("httpx.Timeout")
    def test_passes_when_reachable(self, mock_timeout, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": []}
        mock_get.return_value = mock_resp

        status, _, _ = _check_ollama_server("http://localhost:11434")
        assert status == "pass"

    @patch("httpx.get")
    @patch("httpx.Timeout")
    def test_fails_when_unreachable(self, mock_timeout, mock_get):
        import httpx

        mock_get.side_effect = httpx.ConnectError("refused")

        status, _, detail = _check_ollama_server("http://localhost:11434")
        assert status == "fail"
        assert "ollama serve" in detail


class TestCheckModel:
    @patch("httpx.get")
    @patch("httpx.Timeout")
    def test_passes_when_model_found(self, mock_timeout, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "models": [{"name": "qwen2.5-coder:14b"}]
        }
        mock_get.return_value = mock_resp

        status, _, _ = _check_model("http://localhost:11434", "qwen2.5-coder:14b")
        assert status == "pass"

    @patch("httpx.get")
    @patch("httpx.Timeout")
    def test_fails_when_model_missing(self, mock_timeout, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "models": [{"name": "llama3:latest"}]
        }
        mock_get.return_value = mock_resp

        status, _, detail = _check_model("http://localhost:11434", "qwen2.5-coder:14b")
        assert status == "fail"
        assert "ollama pull" in detail
