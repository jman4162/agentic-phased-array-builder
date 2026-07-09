"""Shared per-call usage tracking for LLM providers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderUsage:
    """Token and cost tracking for a single LLM call.

    Providers expose the most recent call's usage via a ``last_usage``
    property. Local providers (e.g. Ollama) report a zero cost estimate.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cost_estimate_usd: float = 0.0
