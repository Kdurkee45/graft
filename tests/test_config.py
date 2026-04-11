"""Tests for graft.config."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from graft.config import Settings, _find_env_file


def test_settings_load_from_env(monkeypatch):
    """Settings.load() reads ANTHROPIC_API_KEY from environment."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")
    monkeypatch.delenv("GRAFT_MODEL", raising=False)
    monkeypatch.delenv("GRAFT_MAX_TURNS", raising=False)
    settings = Settings.load()
    assert settings.anthropic_api_key == "test-key-123"
    assert settings.model == "claude-opus-4-20250514"
    assert settings.max_agent_turns == 50


def test_settings_load_custom_model(monkeypatch):
    """Settings.load() respects GRAFT_MODEL override."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("GRAFT_MODEL", "claude-sonnet-4-20250514")
    settings = Settings.load()
    assert settings.model == "claude-sonnet-4-20250514"


def test_settings_load_missing_key(monkeypatch):
    """Settings.load() raises SystemExit without ANTHROPIC_API_KEY."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        Settings.load()


def test_find_env_file(tmp_path):
    """_find_env_file() finds .env in current directory."""
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=test")
    with patch("graft.config.Path.cwd", return_value=tmp_path):
        result = _find_env_file()
        assert result == env_file


def test_find_env_file_not_found(tmp_path):
    """_find_env_file() returns None when no .env exists."""
    with (
        patch("graft.config.Path.cwd", return_value=tmp_path),
        patch("graft.config.Path.home", return_value=tmp_path),
    ):
        result = _find_env_file()
        assert result is None
