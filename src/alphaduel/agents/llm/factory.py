"""Build chat-model clients for LLM agents.

Instantiate models here (or in notebooks/scripts), then pass the object into
:class:`~alphaduel.agents.llm.vanilla.VanillaLLMAgent`::

    from alphaduel.agents.llm import LLMHandler, VanillaLLMAgent

    llm = LLMHandler.create("qwen3.5:2b", temperature=0.0)
    agent = VanillaLLMAgent(llm=llm)
"""

from __future__ import annotations

from typing import Any


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
            Model identifier (e.g. ``\"qwen3.5:2b\"`` for Ollama).
        provider:
            Backend to use. Currently ``\"ollama\"`` (via ``langchain-ollama``).
        temperature:
            Sampling temperature forwarded to the chat model.
        reasoning:
            Whether to enable model "thinking"/reasoning traces. Default ``False``
            so responses stay concise and parseable.
        **kwargs:
            Extra provider-specific kwargs (e.g. ``base_url``, ``num_ctx``).
        """
        provider = (provider or "ollama").lower().strip()
        if not name or not str(name).strip():
            raise ValueError("LLM model name is required")

        if provider == "ollama":
            return cls._create_ollama(
                str(name).strip(),
                temperature=temperature,
                reasoning=reasoning,
                **kwargs,
            )
        raise ValueError(
            f"Unknown LLM provider {provider!r}. Supported: 'ollama'."
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
    for key in ("provider", "model", "name", "temperature", "base_url", "reasoning"):
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
    return create_llm(str(name), **cfg), out
