"""Shared utility functions for stage modules.

Consolidates file-discovery, cwd-resolution, and cleanup patterns
that are common across discover, research, and other stages.
"""

from __future__ import annotations

import asyncio
from pathlib import Path


async def async_read_text(path: Path) -> str:
    """Read a file's text content without blocking the event loop.

    Wraps :meth:`Path.read_text` via :func:`asyncio.to_thread` so that
    callers in async node functions avoid stalling the loop on disk I/O.
    """
    return await asyncio.to_thread(path.read_text)


async def async_subprocess_run(
    args: list[str],
    *,
    cwd: str | None = None,
) -> asyncio.subprocess.Process:
    """Run a subprocess without blocking the event loop.

    Mirrors a ``subprocess.run(args, cwd=..., capture_output=True, text=True)``
    call but uses :func:`asyncio.create_subprocess_exec` so the calling
    coroutine can ``await`` it instead of blocking.

    Returns the completed :class:`asyncio.subprocess.Process` after
    ``communicate()`` has been called (stdout/stderr populated, returncode set).
    """
    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    return proc


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
