"""PromptManager: load Jinja2 templates from ``resources/prompts/``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import BaseLoader, Environment, StrictUndefined, TemplateError


def default_prompts_dir() -> Path:
    """Resolve ``resources/prompts`` relative to the project root."""
    # src/alphaduel/prompts/manager.py -> parents[3] == project root
    return Path(__file__).resolve().parents[3] / "resources" / "prompts"


@dataclass(frozen=True)
class Prompt:
    """A single prompt template loaded from disk."""

    key: str
    content: str
    _env: Environment

    def render(self, **payload) -> str:
        """Render the Jinja2 template with ``payload`` and return the final string."""
        try:
            return self._env.from_string(self.content).render(**payload)
        except TemplateError as exc:
            raise ValueError(f"Failed to render prompt {self.key!r}: {exc}") from exc


class PromptManager:
    """Load and cache prompts from a directory of ``*.prompt`` files.

    Keys map 1:1 to filenames without the ``.prompt`` suffix, e.g.
    ``resources/prompts/base_agent_system.prompt`` → ``get("base_agent_system")``.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_prompts_dir()
        self._cache: dict[str, Prompt] = {}
        self._env = Environment(
            loader=BaseLoader(),
            undefined=StrictUndefined,
            autoescape=False,
            keep_trailing_newline=True,
        )

    def get(self, key: str) -> Prompt:
        if key in self._cache:
            return self._cache[key]
        path = self.root / f"{key}.prompt"
        if not path.is_file():
            raise KeyError(f"Prompt {key!r} not found at {path}")
        prompt = Prompt(key=key, content=path.read_text(encoding="utf-8"), _env=self._env)
        self._cache[key] = prompt
        return prompt

    def reload(self) -> None:
        """Drop the in-memory cache (useful after editing prompt files)."""
        self._cache.clear()

    def keys(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.stem for p in self.root.glob("*.prompt"))


#: Process-wide default manager pointing at ``resources/prompts/``.
prompt_manager = PromptManager()
