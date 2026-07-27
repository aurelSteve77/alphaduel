"""Load Jinja prompt templates from ``resources/prompts/<group>/<name>.prompt``.

Usage::

    from alphaduel.prompt_manager import PromptManager

    pm = PromptManager()
    raw = pm["llm_policy"]["system"].content
    text = pm["llm_policy"]["user"].render(ticker="AAPL", cash=10_000)
"""

from __future__ import annotations

from abc import ABCMeta
from collections.abc import Iterator, Mapping
from pathlib import Path

from jinja2 import Environment, StrictUndefined, Template

from alphaduel.utils.singleton import Singleton

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PROMPTS_ROOT = _PROJECT_ROOT / "resources" / "prompts"


class _SingletonABCMeta(Singleton, ABCMeta):
    """Combine :class:`Singleton` with :class:`abc.ABCMeta` for Mapping subclasses."""


class Prompt:
    """A single Jinja template loaded from a ``.prompt`` file."""

    def __init__(self, content: str, *, path: Path, env: Environment) -> None:
        self._content = content
        self._path = path
        self._template: Template = env.from_string(content)

    @property
    def content(self) -> str:
        """Raw template source (unrendered)."""
        return self._content

    @property
    def path(self) -> Path:
        """Absolute path of the source ``.prompt`` file."""
        return self._path

    def render(self, **payload: object) -> str:
        """Render the Jinja template with ``payload`` as template variables."""
        return self._template.render(**payload)

    def __repr__(self) -> str:
        return f"Prompt(path={self._path!s})"


class PromptGroup(Mapping[str, Prompt]):
    """Mapping of prompt name -> :class:`Prompt` within one folder."""

    def __init__(self, name: str, prompts: dict[str, Prompt]) -> None:
        self._name = name
        self._prompts = prompts

    def __getitem__(self, key: str) -> Prompt:
        try:
            return self._prompts[key]
        except KeyError as exc:
            known = ", ".join(sorted(self._prompts)) or "(none)"
            raise KeyError(
                f"Prompt {key!r} not found in group {self._name!r}. Known: {known}"
            ) from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self._prompts)

    def __len__(self) -> int:
        return len(self._prompts)

    def __repr__(self) -> str:
        return f"PromptGroup(name={self._name!r}, prompts={sorted(self._prompts)!r})"


class PromptManager(Mapping[str, PromptGroup], metaclass=_SingletonABCMeta):
    """Singleton index of all ``resources/prompts/<group>/<name>.prompt`` files.

    Access pattern::

        PromptManager()["group"]["name"].render(**payload)
        PromptManager()["group"]["name"].content
    """

    def __init__(self, root: Path | str | None = None) -> None:
        prompts_root = Path(root) if root is not None else _DEFAULT_PROMPTS_ROOT
        if not prompts_root.is_dir():
            raise FileNotFoundError(f"Prompts root not found: {prompts_root}")

        self._root = prompts_root.resolve()
        self._env = Environment(
            undefined=StrictUndefined,
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
        self._groups: dict[str, PromptGroup] = self._load(self._root)

    @property
    def root(self) -> Path:
        """Absolute path of the prompts directory."""
        return self._root

    def __getitem__(self, key: str) -> PromptGroup:
        try:
            return self._groups[key]
        except KeyError as exc:
            known = ", ".join(sorted(self._groups)) or "(none)"
            raise KeyError(
                f"Prompt group {key!r} not found under {self._root}. Known: {known}"
            ) from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self._groups)

    def __len__(self) -> int:
        return len(self._groups)

    def reload(self) -> None:
        """Re-scan ``root`` and replace the in-memory prompt index."""
        self._groups = self._load(self._root)

    def _load(self, root: Path) -> dict[str, PromptGroup]:
        groups: dict[str, PromptGroup] = {}
        for group_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            prompts: dict[str, Prompt] = {}
            for path in sorted(group_dir.glob("*.prompt")):
                content = path.read_text(encoding="utf-8")
                prompts[path.stem] = Prompt(content, path=path.resolve(), env=self._env)
            if prompts:
                groups[group_dir.name] = PromptGroup(group_dir.name, prompts)
        return groups
