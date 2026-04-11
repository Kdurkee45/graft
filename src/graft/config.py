"""Configuration — loads settings from environment and .env files."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _find_env_file() -> Path | None:
    candidates = [
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",
        Path.home() / ".graft" / ".env",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str
    github_token: str | None = None
    model: str = "claude-opus-4-20250514"
    max_agent_turns: int = 50
    projects_root: Path = field(
        default_factory=lambda: Path.home() / ".graft" / "projects"
    )

    @classmethod
    def load(cls) -> Settings:
        env_file = _find_env_file()
        if env_file:
            load_dotenv(env_file)

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise SystemExit(
                "ANTHROPIC_API_KEY not set. "
                "Add it to a .env file or export it in your shell."
            )

        max_turns_raw = os.environ.get("GRAFT_MAX_TURNS", "50")
        try:
            max_turns = int(max_turns_raw)
        except ValueError:
            raise SystemExit(
                f"GRAFT_MAX_TURNS must be a valid integer, got '{max_turns_raw}'."
            )
        if max_turns <= 0:
            raise SystemExit(
                f"GRAFT_MAX_TURNS must be a positive integer, got {max_turns}."
            )

        return cls(
            anthropic_api_key=api_key,
            github_token=os.environ.get("GITHUB_TOKEN"),
            model=os.environ.get("GRAFT_MODEL", "claude-opus-4-20250514"),
            max_agent_turns=max_turns,
        )
