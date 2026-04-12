"""Shared utility functions for stage modules.

Consolidates file-discovery, cwd-resolution, and cleanup patterns
that are common across discover, research, and other stages.
"""

from __future__ import annotations

from pathlib import Path


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
