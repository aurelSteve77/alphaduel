"""Tests for the PromptManager singleton and Jinja rendering."""

from __future__ import annotations

from pathlib import Path

import pytest
from jinja2 import UndefinedError

from alphaduel.prompt_manager import PromptManager
from alphaduel.utils.singleton import Singleton


@pytest.fixture
def prompts_root(tmp_path: Path) -> Path:
    group = tmp_path / "demo"
    group.mkdir()
    (group / "greet.prompt").write_text("Hello {{ name }}!\n", encoding="utf-8")
    (group / "raw.prompt").write_text("plain text", encoding="utf-8")
    return tmp_path


@pytest.fixture
def prompt_manager(prompts_root: Path) -> PromptManager:
    Singleton._instances.pop(PromptManager, None)
    manager = PromptManager(root=prompts_root)
    yield manager
    Singleton._instances.pop(PromptManager, None)


def test_is_singleton(prompt_manager: PromptManager, prompts_root: Path):
    assert PromptManager(root=prompts_root) is prompt_manager


def test_nested_access_and_content(prompt_manager: PromptManager):
    assert prompt_manager["demo"]["raw"].content == "plain text"
    assert "Hello {{ name }}!" in prompt_manager["demo"]["greet"].content


def test_render(prompt_manager: PromptManager):
    assert prompt_manager["demo"]["greet"].render(name="AAPL") == "Hello AAPL!\n"


def test_render_missing_variable_raises(prompt_manager: PromptManager):
    with pytest.raises(UndefinedError):
        prompt_manager["demo"]["greet"].render()


def test_unknown_group_and_prompt(prompt_manager: PromptManager):
    with pytest.raises(KeyError, match="group"):
        _ = prompt_manager["missing"]
    with pytest.raises(KeyError, match="Prompt"):
        _ = prompt_manager["demo"]["missing"]


def test_project_prompts_load():
    Singleton._instances.pop(PromptManager, None)
    pm = PromptManager()
    try:
        assert "llm_policy" in pm
        assert "system" in pm["llm_policy"]
        assert "user" in pm["llm_policy"]
        rendered = pm["llm_policy"]["system"].render(
            tickers=["AAPL", "MSFT"],
            max_shares=5,
            allow_short=False,
        )
        assert "AAPL, MSFT" in rendered
        assert "5" in rendered
    finally:
        Singleton._instances.pop(PromptManager, None)
