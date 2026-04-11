"""Tests for graft.state."""

from graft.state import (
    FeatureState,
    _replace,
    _replace_bool,
    _replace_dict,
    _replace_int,
    _replace_list,
)


def test_replace_takes_latest():
    assert _replace("old", "new") == "new"


def test_replace_bool():
    assert _replace_bool(False, True) is True
    assert _replace_bool(True, False) is False


def test_replace_int():
    assert _replace_int(1, 42) == 42


def test_replace_list():
    assert _replace_list([1], [2, 3]) == [2, 3]


def test_replace_dict():
    assert _replace_dict({"a": 1}, {"b": 2}) == {"b": 2}


def test_feature_state_can_be_constructed():
    """FeatureState can be instantiated as a TypedDict."""
    state: FeatureState = {
        "repo_path": "/tmp/test",
        "project_id": "feat_test123",
        "feature_prompt": "Add dark mode",
    }
    assert state["repo_path"] == "/tmp/test"
    assert state["feature_prompt"] == "Add dark mode"
