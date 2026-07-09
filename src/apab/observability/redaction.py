"""Redaction of captured tool arguments and results for span attributes.

Mirrors the RedactionMode semantics used by the tool-dispatch audit log:
``none`` captures values, ``metadata_only`` captures shape only,
``strict`` captures nothing beyond a content hash.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from apab.core.schemas import RedactionMode


def args_hash(arguments: dict[str, Any]) -> str:
    """Deterministic 16-char hash of a tool-call argument dict."""
    raw = json.dumps(arguments, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def capture_args(
    arguments: dict[str, Any],
    mode: RedactionMode,
) -> dict[str, Any]:
    """Return span attributes describing *arguments* under *mode*.

    Always includes ``args_hash`` so identical calls can be correlated
    across runs without exposing values.
    """
    attrs: dict[str, Any] = {"args_hash": args_hash(arguments)}
    if mode == RedactionMode.none:
        attrs["args_json"] = json.dumps(arguments, default=str)[:2000]
    elif mode == RedactionMode.metadata_only:
        attrs["arg_keys"] = sorted(arguments.keys())
    # strict: hash only
    return attrs


def capture_text(
    text: str,
    mode: RedactionMode,
    max_len: int = 200,
) -> str | None:
    """Return *text* truncated per *mode*, or None if it must be dropped."""
    if mode == RedactionMode.strict:
        return None
    if mode == RedactionMode.metadata_only:
        return f"<{len(text)} chars>"
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
