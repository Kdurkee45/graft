"""Shared utility functions for stage modules.

Consolidates file-discovery, cwd-resolution, and cleanup patterns
that are common across discover, research, and other stages.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graft.ui import UI


async def async_read_text(path: Path) -> str:
    """Read a file's text content without blocking the event loop.

    Wraps :meth:`Path.read_text` via :func:`asyncio.to_thread` so that
    callers in async node functions avoid stalling the loop on disk I/O.
    """
    return await asyncio.to_thread(path.read_text)


def resolve_stage_cwd(repo_path: str, scope_path: str) -> str:
    """Return the working directory for a stage.

    If *scope_path* is non-empty and the corresponding subdirectory exists
    under *repo_path*, return that scoped directory.  Otherwise fall back to
    *repo_path* itself.
    """
    if scope_path:
        scoped_dir = Path(repo_path) / scope_path
        if scoped_dir.exists():
            return str(scoped_dir)
    return repo_path


def find_artifact(filename: str, stage_cwd: str, repo_path: str) -> Path:
    """Locate an artifact file, checking *stage_cwd* first then *repo_path*.

    Returns the first existing path, or falls back to ``stage_cwd / filename``
    (which may not exist — callers should check).
    """
    primary = Path(stage_cwd) / filename
    if primary.exists():
        return primary
    fallback = Path(repo_path) / filename
    if fallback.exists():
        return fallback
    return primary  # default even if missing — callers check .exists()


async def read_json_artifact(
    name: str,
    stage_cwd: str,
    repo_path: str,
    ui: UI | None = None,
) -> dict:
    """Find, read, and parse a JSON artifact.

    Returns the parsed dict, or ``{}`` if the file is missing or
    contains invalid JSON.  When *ui* is provided, parse failures
    are reported via :meth:`ui.error`.
    """
    path = find_artifact(name, stage_cwd, repo_path)
    if not path.exists():
        return {}
    try:
        return json.loads(await async_read_text(path))  # type: ignore[return-value]
    except json.JSONDecodeError:
        if ui is not None:
            ui.error(f"Failed to parse {name}.")
        return {}


async def read_text_artifact(
    name: str,
    stage_cwd: str,
    repo_path: str,
    fallback: str = "",
) -> str:
    """Find and read a text artifact, returning *fallback* if missing."""
    path = find_artifact(name, stage_cwd, repo_path)
    if path.exists():
        return await async_read_text(path)
    return fallback


def cleanup_artifacts(stage_cwd: str, repo_path: str, filenames: list[str]) -> None:
    """Remove temporary artifact files from both *stage_cwd* and *repo_path*.

    Silently skips files that don't exist.
    """
    for name in filenames:
        for base in (stage_cwd, repo_path):
            p = Path(base) / name
            if p.exists():
                p.unlink()
