"""Shared utility functions for stage modules.

Consolidates file-discovery, cwd-resolution, and cleanup patterns
that are common across discover, research, and other stages.
"""

from __future__ import annotations

import asyncio
import functools
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graft.state import FeatureState
    from graft.ui import UI

from graft.artifacts import mark_stage_complete


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


def cleanup_artifacts(stage_cwd: str, repo_path: str, filenames: list[str]) -> None:
    """Remove temporary artifact files from both *stage_cwd* and *repo_path*.

    Silently skips files that don't exist.
    """
    for name in filenames:
        for base in (stage_cwd, repo_path):
            p = Path(base) / name
            if p.exists():
                p.unlink()


def stage_node(name: str):
    """Decorator that wraps a stage node function with lifecycle boilerplate.

    Automatically calls ``ui.stage_start(name)`` before the wrapped function,
    ``mark_stage_complete(project_dir, name)`` and ``ui.stage_done(name)``
    after, and injects ``"current_stage"`` into the return dict.

    Usage::

        @stage_node("discover")
        async def discover_node(state: FeatureState, ui: UI) -> dict[str, Any]:
            # ... stage-specific logic only ...
            return {"codebase_profile": profile}
    """

    def decorator(fn):  # noqa: ANN001
        @functools.wraps(fn)
        async def wrapper(state: FeatureState, ui: UI) -> dict[str, Any]:
            ui.stage_start(name)
            result = await fn(state, ui)
            mark_stage_complete(state["project_dir"], name)
            ui.stage_done(name)
            result["current_stage"] = name
            return result

        return wrapper

    return decorator
