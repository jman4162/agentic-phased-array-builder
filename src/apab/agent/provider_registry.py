"""LLM provider protocol and registry."""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import os
from typing import Any, Protocol, runtime_checkable

from apab.providers.usage import ProviderUsage

logger = logging.getLogger(__name__)

__all__ = [
    "LLMProvider",
    "ProviderUsage",
    "discover_providers",
    "get_provider",
    "validate_provider",
]


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM provider backends.

    Providers may additionally expose an optional ``last_usage`` property
    returning an :class:`apab.providers.usage.ProviderUsage` (or ``None``)
    for the most recent ``chat()`` call. It is not part of the protocol so
    that third-party providers registered via entry points keep working;
    consumers must read it with ``getattr(provider, "last_usage", None)``.
    """

    @property
    def name(self) -> str:
        """Short identifier for the provider (e.g. ``"ollama"``)."""
        ...

    def supports_tool_calling(self) -> bool:
        """Whether this provider supports tool-calling."""
        ...

    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming responses."""
        ...

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a chat request and return a normalised response.

        The returned dict has at minimum:
        - ``"role"`` — always ``"assistant"``
        - ``"content"`` — text content (may be ``None`` if tool calls)
        - ``"tool_calls"`` — list of ``{"name": str, "arguments": dict}``
          dicts, or ``None`` if no tool calls.
        """
        ...


def discover_providers() -> dict[str, type]:
    """Discover LLM provider classes registered via entry points."""
    providers: dict[str, type] = {}
    # entry_points(group=...) is available on all supported versions (3.10+).
    eps = importlib.metadata.entry_points(group="apab.llm_providers")

    for ep in eps:
        try:
            cls = ep.load()
            providers[ep.name] = cls
        except Exception:
            pass
    return providers


_BUILTINS: dict[str, str] = {
    "ollama": "apab.providers.ollama",
    "openai": "apab.providers.openai",
    "anthropic": "apab.providers.anthropic",
    "gemini": "apab.providers.gemini",
    "openai_compatible": "apab.providers.openai_compat",
}


def get_provider(provider_name: str, **kwargs: Any) -> LLMProvider:
    """Instantiate and return an LLM provider by name."""
    discovered = discover_providers()
    if provider_name in discovered:
        return discovered[provider_name](**kwargs)  # type: ignore[no-any-return]

    module_path = _BUILTINS.get(provider_name)
    if module_path is None:
        raise ValueError(
            f"Unknown LLM provider '{provider_name}'. "
            f"Available: {sorted(set(discovered) | set(_BUILTINS))}"
        )

    mod = importlib.import_module(module_path)
    for attr_name in dir(mod):
        if attr_name.startswith("_"):
            continue
        attr = getattr(mod, attr_name)
        if isinstance(attr, type) and attr is not LLMProvider:
            # Check if it looks like a provider (has name property and chat method)
            if hasattr(attr, "chat") and hasattr(attr, "name"):
                return attr(**kwargs)  # type: ignore[no-any-return]

    raise ValueError(f"No LLMProvider found in module '{module_path}'")


def provider_from_spec(llm: Any) -> LLMProvider:
    """Instantiate the provider described by an ``LLMSpec`` (``config.llm``).

    When ``llm.api_key_env`` names an environment variable, its value is
    passed as ``api_key``; a named variable that is unset or empty is an
    error, so a configured key is never silently replaced by a provider's
    default variable. Without ``api_key_env`` providers read their own
    defaults (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ...).
    """
    kwargs: dict[str, Any] = {"model": llm.model, "base_url": llm.base_url}
    if llm.api_key_env:
        key = os.environ.get(llm.api_key_env)
        if not key:
            raise ValueError(
                f"llm.api_key_env is {llm.api_key_env!r}, but that environment "
                "variable is not set"
            )
        kwargs["api_key"] = key
    return get_provider(llm.provider, **kwargs)


def validate_provider(provider_name: str) -> bool:
    """Check if a provider name is available without instantiating it.

    Returns ``True`` if the provider can be found; logs a warning and
    returns ``False`` otherwise.
    """
    discovered = discover_providers()
    if provider_name in discovered:
        return True
    if provider_name in _BUILTINS:
        return True
    logger.warning(
        "LLM provider %r not found. Available: %s",
        provider_name,
        sorted(set(discovered) | set(_BUILTINS)),
    )
    return False
