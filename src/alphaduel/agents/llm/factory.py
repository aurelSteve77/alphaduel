"""Build chat-model clients for LLM agents.

Instantiate models here (or in notebooks/scripts), then pass the object into
:class:`~alphaduel.agents.llm.vanilla.VanillaLLMAgent`::

    from alphaduel.agents.llm import LLMHandler, VanillaLLMAgent

    llm = LLMHandler.create("qwen3.5:2b", provider="ollama", temperature=0.0)
    llm = LLMHandler.create("gpt-5.4-mini", provider="openai", temperature=0.0)
    llm = LLMHandler.create(
        "claude-haiku-4-5-20251001", provider="anthropic", temperature=0.0
    )
    llm = LLMHandler.create(
        "meta/meta-llama-3-8b-instruct", provider="replicate", temperature=0.0
    )
    agent = VanillaLLMAgent(llm=llm)
"""

from __future__ import annotations

from typing import Any

SUPPORTED_PROVIDERS = ("ollama", "openai", "anthropic", "replicate")

# Sensible defaults shown in the dashboard when switching providers.
DEFAULT_MODELS: dict[str, str] = {
    "ollama": "qwen3.5:2b",
    "openai": "gpt-5.4-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "replicate": "meta/meta-llama-3-8b-instruct",
}

# OpenAI reasoning models (o-series / gpt-5.*): default effort when not overridden.
DEFAULT_OPENAI_REASONING_EFFORT = "low"


class LLMHandler:
    """Factory for LangChain chat models used by AlphaDuel LLM agents."""

    @classmethod
    def create(
        cls,
        name: str,
        *,
        provider: str = "ollama",
        temperature: float = 0.0,
        reasoning: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Create a chat model from a model name / provider.

        Parameters
        ----------
        name:
            Model identifier.
            - Ollama: e.g. ``\"qwen3.5:2b\"``
            - OpenAI: e.g. ``\"gpt-5.4-mini\"`` / ``\"gpt-4o-mini\"``
            - Anthropic: e.g. ``\"claude-haiku-4-5-20251001\"`` /
              ``\"claude-sonnet-4-5\"``
            - Replicate: e.g. ``\"meta/meta-llama-3-8b-instruct\"`` or
              ``\"owner/name:version\"``
        provider:
            ``\"ollama\"``, ``\"openai\"`` (ChatOpenAI), ``\"anthropic\"``
            (ChatAnthropic), or ``\"replicate\"`` (ChatReplicate).
        temperature:
            Sampling temperature. For Replicate this is forwarded via
            ``model_kwargs`` (Replicate OpenAPI inputs).
        reasoning:
            Ollama-only: disable/enable thinking traces. For OpenAI, use
            ``reasoning_effort`` (default ``\"low\"``).
        **kwargs:
            Extra provider-specific kwargs
            (``base_url``, ``api_key``, ``replicate_api_token``, ``model_kwargs``,
            ``reasoning_effort``, ``max_tokens``, ``timeout``, …).
        """
        provider = (provider or "ollama").lower().strip()
        if not name or not str(name).strip():
            raise ValueError("LLM model name is required")

        if provider == "ollama":
            kwargs.pop("reasoning_effort", None)
            return cls._create_ollama(
                str(name).strip(),
                temperature=temperature,
                reasoning=reasoning,
                **kwargs,
            )
        if provider == "openai":
            return cls._create_openai(
                str(name).strip(),
                temperature=temperature,
                **kwargs,
            )
        if provider == "anthropic":
            kwargs.pop("reasoning_effort", None)
            return cls._create_anthropic(
                str(name).strip(),
                temperature=temperature,
                **kwargs,
            )
        if provider == "replicate":
            kwargs.pop("reasoning_effort", None)
            return cls._create_replicate(
                str(name).strip(),
                temperature=temperature,
                **kwargs,
            )
        raise ValueError(
            f"Unknown LLM provider {provider!r}. Supported: {SUPPORTED_PROVIDERS}."
        )

    @staticmethod
    def _create_ollama(
        name: str, *, temperature: float, reasoning: bool = False, **kwargs: Any
    ) -> Any:
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ImportError(
                "Ollama chat models require the `llm` extra: "
                "`uv sync --extra llm` (langchain-ollama)."
            ) from exc
        return ChatOllama(
            model=name, temperature=temperature, reasoning=reasoning, **kwargs
        )

    @staticmethod
    def _create_openai(name: str, *, temperature: float, **kwargs: Any) -> Any:
        """Build :class:`langchain_openai.ChatOpenAI`.

        Auth: set ``OPENAI_API_KEY`` in the environment / ``.env``, or pass
        ``api_key=...``. Optional ``base_url`` / ``OPENAI_API_BASE`` for proxies.
        Defaults ``reasoning_effort=\"low\"`` (override or pass ``None`` to omit).
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI chat models require the `llm` extra: "
                "`uv sync --extra llm` (langchain-openai)."
            ) from exc

        api_key = kwargs.pop("api_key", None)
        if api_key is None:
            api_key = kwargs.pop("openai_api_key", None)

        # Accept either base_url or openai_api_base (LangChain aliases).
        base_url = kwargs.pop("base_url", None)
        if base_url is None:
            base_url = kwargs.pop("openai_api_base", None)

        # Default low reasoning effort; explicit None removes the knob entirely.
        if "reasoning_effort" not in kwargs:
            kwargs["reasoning_effort"] = DEFAULT_OPENAI_REASONING_EFFORT
        elif kwargs["reasoning_effort"] is None:
            kwargs.pop("reasoning_effort")

        init: dict[str, Any] = {
            "model": name,
            "temperature": temperature,
        }
        if api_key is not None:
            init["api_key"] = api_key
        if base_url is not None:
            init["base_url"] = base_url
        init.update(kwargs)
        return ChatOpenAI(**init)

    @staticmethod
    def _create_anthropic(name: str, *, temperature: float, **kwargs: Any) -> Any:
        """Build :class:`langchain_anthropic.ChatAnthropic`.

        Auth: set ``ANTHROPIC_API_KEY`` in the environment / ``.env``, or pass
        ``api_key=...``. See https://docs.langchain.com/oss/python/integrations/chat/anthropic
        """
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic chat models require the `llm` extra: "
                "`uv sync --extra llm` (langchain-anthropic)."
            ) from exc

        api_key = kwargs.pop("api_key", None)
        if api_key is None:
            api_key = kwargs.pop("anthropic_api_key", None)

        init: dict[str, Any] = {
            "model": name,
            "temperature": temperature,
        }
        if api_key is not None:
            init["api_key"] = api_key
        init.update(kwargs)
        return ChatAnthropic(**init)

    @staticmethod
    def _create_replicate(name: str, *, temperature: float, **kwargs: Any) -> Any:
        """Build :class:`langchain_replicate.ChatReplicate`.

        Auth: set ``REPLICATE_API_TOKEN`` in the environment / ``.env``, or pass
        ``replicate_api_token=...``. See https://github.com/replicate/replicate-langchain
        """
        try:
            from langchain_replicate import ChatReplicate
        except ImportError as exc:
            raise ImportError(
                "Replicate chat models require the `llm` extra: "
                "`uv sync --extra llm` (langchain-replicate from GitHub)."
            ) from exc

        model_kwargs = dict(kwargs.pop("model_kwargs", {}) or {})
        # ChatReplicate takes generation knobs via model_kwargs (OpenAPI inputs).
        if "temperature" not in model_kwargs:
            model_kwargs["temperature"] = temperature

        api_token = kwargs.pop("replicate_api_token", None)
        if api_token is None:
            api_token = kwargs.pop("api_token", None)

        streaming = bool(kwargs.pop("streaming", False))
        return ChatReplicate(
            model=name,
            model_kwargs=model_kwargs,
            replicate_api_token=api_token,
            streaming=streaming,
            **kwargs,
        )


def create_llm(
    name: str,
    *,
    provider: str = "ollama",
    temperature: float = 0.0,
    reasoning: bool = False,
    **kwargs: Any,
) -> Any:
    """Shortcut for :meth:`LLMHandler.create`."""
    return LLMHandler.create(
        name,
        provider=provider,
        temperature=temperature,
        reasoning=reasoning,
        **kwargs,
    )


def _inject_secrets(cfg: dict[str, Any]) -> None:
    """Fill API tokens from ``Secrets`` / ``.env`` when not passed explicitly."""
    provider = cfg.get("provider", "ollama")
    try:
        from alphaduel.config.schema import Secrets

        secrets = Secrets()
    except Exception:  # noqa: BLE001 — secrets optional at import time
        return

    if provider == "replicate" and not cfg.get("replicate_api_token"):
        if secrets.replicate_api_token:
            cfg["replicate_api_token"] = secrets.replicate_api_token
    elif provider == "openai" and not cfg.get("api_key") and not cfg.get("openai_api_key"):
        if secrets.openai_api_key:
            cfg["api_key"] = secrets.openai_api_key
    elif (
        provider == "anthropic"
        and not cfg.get("api_key")
        and not cfg.get("anthropic_api_key")
    ):
        if secrets.anthropic_api_key:
            cfg["api_key"] = secrets.anthropic_api_key


def resolve_llm(params: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Pop LLM construction keys from agent params and return ``(llm, agent_params)``.

    Accepts either:
    - an already-built ``llm`` object,
    - a nested ``llm: {name/model, provider, temperature, ...}`` mapping,
    - or flat ``model`` / ``provider`` / ``temperature`` keys (YAML convenience).
    """
    out = dict(params)
    llm = out.pop("llm", None)

    if llm is not None and not isinstance(llm, dict):
        return llm, out

    cfg: dict[str, Any] = dict(llm) if isinstance(llm, dict) else {}
    # Flat YAML keys fill any gaps in a nested llm block.
    for key in (
        "provider",
        "model",
        "name",
        "temperature",
        "base_url",
        "openai_api_base",
        "reasoning",
        "api_key",
        "openai_api_key",
        "anthropic_api_key",
        "replicate_api_token",
        "streaming",
        "model_kwargs",
        "max_retries",
        "timeout",
        "max_tokens",
        "reasoning_effort",
        "max_completion_tokens",
        "inference_geo",
    ):
        if key in out and key not in cfg:
            cfg[key] = out.pop(key)
        elif key in out and key in cfg:
            out.pop(key)

    name = cfg.pop("name", None) or cfg.pop("model", None)
    if name is None:
        raise ValueError(
            "LLM agent requires an `llm` object or model name "
            "(params.llm / params.model / params.llm.model)."
        )
    _inject_secrets(cfg)
    return create_llm(str(name), **cfg), out
