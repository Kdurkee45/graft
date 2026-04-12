"""Thin wrapper around the Claude Agent SDK for running persona-driven agents."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from claude_agent_sdk import ClaudeAgentOptions, query

from graft import artifacts
from graft.ui import UI

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0


@dataclass
class AgentResult:
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    raw_messages: list = field(default_factory=list)
    elapsed_seconds: float = 0.0
    turns_used: int = 0


async def run_agent(
    *,
    persona: str,
    system_prompt: str,
    user_prompt: str,
    cwd: str,
    project_dir: str,
    stage: str,
    ui: UI,
    model: str | None = None,
    max_turns: int = 50,
    allowed_tools: list[str] | None = None,
) -> AgentResult:
    """Spawn a Claude Agent SDK session with the given persona and capture output.

    Retries transient failures with exponential backoff.
    Every message is logged to ``<project_dir>/logs/<stage>.log``.
    """
    if allowed_tools is None:
        allowed_tools = [
            "Read",
            "Write",
            "Edit",
            "MultiEdit",
            "Bash",
            "Glob",
            "Grep",
        ]

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        cwd=cwd,
        max_turns=max_turns,
        allowed_tools=allowed_tools,
        permission_mode="bypassPermissions",
        model=model,
    )

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        raw_messages: list = []

        ui.stage_log(stage, f"[bold]Agent ({persona})[/bold] starting…")
        start = time.monotonic()

        try:
            async for message in query(prompt=user_prompt, options=options):
                raw_messages.append(message)
                _process_message(
                    message, text_parts, tool_calls, stage, ui, project_dir
                )

            elapsed = time.monotonic() - start
            full_text = "\n".join(text_parts)
            artifacts.save_log(project_dir, stage, full_text)

            ui.stage_log(
                stage,
                f"[bold]Agent ({persona})[/bold] finished "
                f"({elapsed:.1f}s, {len(raw_messages)} messages).",
            )
            return AgentResult(
                text=full_text,
                tool_calls=tool_calls,
                raw_messages=raw_messages,
                elapsed_seconds=elapsed,
                turns_used=len(raw_messages),
            )

        except (ConnectionError, TimeoutError, OSError) as exc:
            last_error = exc
            delay = RETRY_BACKOFF_BASE**attempt
            ui.stage_log(
                stage,
                f"[yellow]Agent ({persona}) failed (attempt {attempt}/{MAX_RETRIES}): "
                f"{exc}. Retrying in {delay:.0f}s…[/yellow]",
            )
            await asyncio.sleep(delay)

    raise RuntimeError(
        f"Agent ({persona}) failed after {MAX_RETRIES} attempts: {last_error}"
    )


def _process_message(
    message, text_parts: list, tool_calls: list, stage: str, ui: UI, project_dir: str
) -> None:
    """Extract text and tool-use info from a single SDK message."""
    if not hasattr(message, "content"):
        return

    for block in message.content:
        if hasattr(block, "text") and block.text:
            text_parts.append(block.text)
            preview = block.text[:200].replace("\n", " ")
            ui.stage_log(stage, preview)
        elif hasattr(block, "name"):
            tool_calls.append(
                {"tool": block.name, "input": getattr(block, "input", {})}
            )
            ui.stage_log(stage, f"  ↳ tool: {block.name}")
