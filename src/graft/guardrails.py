"""PreToolUse guardrails for Claude Agent SDK sessions.

Provides a Bash command hook that blocks dangerous or wasteful commands
before they execute. The hook returns a denial with a helpful message so
the agent can self-correct (e.g., searching within the workspace instead
of the entire filesystem).
"""

from __future__ import annotations

import re
from typing import Any

from claude_agent_sdk.types import (
    HookContext,
    HookInput,
    HookMatcher,
    PreToolUseHookSpecificOutput,
    SyncHookJSONOutput,
)

# ---------------------------------------------------------------------------
# Dangerous / wasteful command patterns
# ---------------------------------------------------------------------------

# `find` rooted at /
# NOTE: `find -name foo` (no path) defaults to cwd and is safe — don't block it.
_FIND_ROOT_RE = re.compile(
    r"""
    (?:^|[;&|]\s*)        # start of command or chained after ; & |
    (?:sudo\s+)?          # optional sudo prefix
    find\s+               # the find command
    (?:
        /\s               # explicit "/ " (root with next arg)
      | /\s*$             # just "/" at end of string
      | /\s+-             # "/" followed by a flag like -name
    )
    """,
    re.VERBOSE,
)

# Broader pattern: find starting from very high-level dirs that aren't the project.
# Only matches shallow system paths (at most one level deep, e.g. /usr/local).
# Deep paths like /Users/kevin/projects/my-app are allowed — they're project-specific.
_FIND_BROAD_RE = re.compile(
    r"""
    (?:^|[;&|]\s*)
    (?:sudo\s+)?
    find\s+
    /(?:usr|opt|var|etc|System|Library|Applications|home|Users)
    (?:/[^/\s]+)?          # allow at most one more path component (e.g., /usr/local)
    (?:\s|$)               # must end with whitespace or EOL (not /deeper/path)
    """,
    re.VERBOSE,
)

# rm -rf / or rm -rf /* (catastrophic)
_RM_ROOT_RE = re.compile(
    r"""
    (?:^|[;&|]\s*)
    (?:sudo\s+)?
    rm\s+
    (?:-[a-zA-Z]*r[a-zA-Z]*\s+)?   # flags containing -r
    (?:-[a-zA-Z]*f[a-zA-Z]*\s+)?   # flags containing -f
    /\s*(?:\*|\s|$)                 # targeting / or /*
    """,
    re.VERBOSE,
)

# dd writing to raw block devices
_DD_DEVICE_RE = re.compile(
    r"""
    (?:^|[;&|]\s*)
    (?:sudo\s+)?
    dd\s+.*of=/dev/
    """,
    re.VERBOSE,
)

# mkfs / format commands on devices
_MKFS_RE = re.compile(
    r"""
    (?:^|[;&|]\s*)
    (?:sudo\s+)?
    mkfs
    """,
    re.VERBOSE,
)

# Commands that should be denied with explanations
_DENY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        _FIND_ROOT_RE,
        "Searching from / traverses the entire filesystem"
        " and can hang for 30+ minutes. Use find within"
        " the project directory instead"
        " (e.g., find . -name 'file.py').",
    ),
    (
        _FIND_BROAD_RE,
        "Searching system directories (/usr, /opt, /var, etc.) is slow and unlikely to "
        "find project files. Search within the project directory instead.",
    ),
    (
        _RM_ROOT_RE,
        "Removing files from the root filesystem is not allowed.",
    ),
    (
        _DD_DEVICE_RE,
        "Writing directly to block devices is not allowed.",
    ),
    (
        _MKFS_RE,
        "Formatting filesystems is not allowed.",
    ),
]


def _extract_command(tool_input: dict[str, Any]) -> str | None:
    """Extract the shell command string from Bash tool input."""
    # Claude Code Bash tool uses {"command": "..."}
    cmd = tool_input.get("command")
    if isinstance(cmd, str):
        return cmd
    return None


def _check_command(command: str) -> str | None:
    """Check a command against deny patterns.

    Returns a denial reason string, or None if the command is allowed.
    """
    for pattern, reason in _DENY_PATTERNS:
        if pattern.search(command):
            return reason
    return None


async def bash_guardrail(
    hook_input: HookInput,
    tool_use_id: str | None,
    context: HookContext,
) -> SyncHookJSONOutput:
    """PreToolUse hook that inspects Bash commands for dangerous patterns.

    Wired into the Claude Agent SDK via ``HookMatcher(matcher="Bash", ...)``.
    Returns allow/deny decisions that the SDK enforces before command execution.
    """
    # Extract the command from hook input
    ti = hook_input.get("tool_input", {})
    command = _extract_command(ti if isinstance(ti, dict) else {})

    if command is None:
        # Can't inspect — allow by default
        output: PreToolUseHookSpecificOutput = {"hookEventName": "PreToolUse"}
        return SyncHookJSONOutput(hookSpecificOutput=output)

    denial_reason = _check_command(command)

    if denial_reason:
        output = PreToolUseHookSpecificOutput(
            hookEventName="PreToolUse",
            permissionDecision="deny",
            permissionDecisionReason=denial_reason,
        )
        return SyncHookJSONOutput(hookSpecificOutput=output)

    # Allow the command
    output = PreToolUseHookSpecificOutput(hookEventName="PreToolUse")
    return SyncHookJSONOutput(hookSpecificOutput=output)


def make_guardrail_hooks() -> dict[str, list[HookMatcher]]:
    """Build hook configuration for run_agent().

    Returns a dict suitable for ``ClaudeAgentOptions(hooks=...)``.
    """
    return {
        "PreToolUse": [
            HookMatcher(
                matcher="Bash",
                hooks=[bash_guardrail],
            ),
        ],
    }
