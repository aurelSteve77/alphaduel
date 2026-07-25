"""Tests for the Configuration singleton and dotted-key access."""

from __future__ import annotations

from alphaduel.configuration import Configuration


def test_is_singleton():
    assert Configuration() is Configuration()


def test_get_top_level_key():
    assert Configuration().get("seed") == 77


def test_get_dotted_key():
    assert Configuration().get("data.market.cache_dir") == "data/raw/market"


def test_missing_key_returns_default():
    assert Configuration().get("ddd.ddd.ddd") is None
    assert Configuration().get("ddd.ddd.ddd", default="x") == "x"


def test_as_dict_is_a_copy():
    d = Configuration().as_dict()
    d["seed"] = -1
    assert Configuration().get("seed") == 77


def test_path_points_to_project_yaml():
    assert Configuration().path.name == "project.yaml"
