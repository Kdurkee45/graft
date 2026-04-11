"""Tests for graft.agent."""

from __future__ import annotations

import asyncio
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graft.agent import (
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    AgentResult,
    _process_message,
    run_agent,
)
from graft.ui import UI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_text_block(text: str) -> SimpleNamespace:
    """Create a fake SDK text content block."""
    return SimpleNamespace(text=text)


def _make_tool_block(name: str, input_data: dict | None = None) -> SimpleNamespace:
    """Create a fake SDK tool-use content block."""
    return SimpleNamespace(name=name, input=input_data or {})


def _make_message(*blocks) -> SimpleNamespace:
    """Wrap content blocks in a message-like object with a .content attribute."""
    return SimpleNamespace(content=list(blocks))


def _make_ui() -> UI:
    """Return a non-verbose UI so stage_log is a no-op (no console output)."""
    return UI(verbose=False)


# ---------------------------------------------------------------------------
# AgentResult dataclass
# ---------------------------------------------------------------------------

class TestAgentResult:
    def test_construction_minimal(self):
        """AgentResult can be created with only the required 'text' field."""
        result = AgentResult(text="hello")
        assert result.text == "hello"
        assert result.tool_calls == []
        assert result.raw_messages == []
        assert result.elapsed_seconds == 0.0
        assert result.turns_used == 0

    def test_construction_full(self):
        """AgentResult stores all supplied fields."""
        calls = [{"tool": "Bash", "input": {"command": "ls"}}]
        msgs = [{"role": "assistant"}]
        result = AgentResult(
            text="done",
            tool_calls=calls,
            raw_messages=msgs,
            elapsed_seconds=1.5,
            turns_used=3,
        )
        assert result.text == "done"
        assert result.tool_calls is calls
        assert result.raw_messages is msgs
        assert result.elapsed_seconds == 1.5
        assert result.turns_used == 3

    def test_default_factory_isolation(self):
        """Each instance gets its own default list (no shared mutable default)."""
        a = AgentResult(text="a")
        b = AgentResult(text="b")
        a.tool_calls.append({"tool": "Read"})
        assert b.tool_calls == []

    def test_has_expected_fields(self):
        """AgentResult exposes exactly the documented fields."""
        names = {f.name for f in fields(AgentResult)}
        assert names == {"text", "tool_calls", "raw_messages", "elapsed_seconds", "turns_used"}


# ---------------------------------------------------------------------------
# _process_message
# ---------------------------------------------------------------------------

class TestProcessMessage:
    def test_text_block_appended(self):
        """Text content blocks are collected in text_parts."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        msg = _make_message(_make_text_block("Hello world"))
        _process_message(msg, text_parts, tool_calls, "discover", _make_ui(), "/tmp/proj")
        assert text_parts == ["Hello world"]
        assert tool_calls == []

    def test_tool_use_block_appended(self):
        """Tool-use content blocks are collected in tool_calls."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        msg = _make_message(_make_tool_block("Bash", {"command": "ls"}))
        _process_message(msg, text_parts, tool_calls, "discover", _make_ui(), "/tmp/proj")
        assert text_parts == []
        assert tool_calls == [{"tool": "Bash", "input": {"command": "ls"}}]

    def test_mixed_blocks(self):
        """A message with both text and tool-use blocks populates both lists."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        msg = _make_message(
            _make_text_block("I will list files."),
            _make_tool_block("Bash", {"command": "ls"}),
            _make_text_block("Done."),
        )
        _process_message(msg, text_parts, tool_calls, "execute", _make_ui(), "/tmp/proj")
        assert text_parts == ["I will list files.", "Done."]
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "Bash"

    def test_empty_text_block_ignored(self):
        """A text block with empty string is skipped (block.text is falsy)."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        msg = _make_message(SimpleNamespace(text=""))
        _process_message(msg, text_parts, tool_calls, "plan", _make_ui(), "/tmp/proj")
        assert text_parts == []

    def test_message_without_content_attribute(self):
        """Messages lacking a .content attribute are silently skipped."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        msg = SimpleNamespace(role="system")  # no .content
        _process_message(msg, text_parts, tool_calls, "discover", _make_ui(), "/tmp/proj")
        assert text_parts == []
        assert tool_calls == []

    def test_tool_block_without_input(self):
        """Tool block without .input attribute defaults to empty dict."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        block = SimpleNamespace(name="Read")  # no .input attribute
        msg = _make_message(block)
        _process_message(msg, text_parts, tool_calls, "verify", _make_ui(), "/tmp/proj")
        assert tool_calls == [{"tool": "Read", "input": {}}]

    def test_tool_result_block_ignored(self):
        """Blocks that have neither .text nor .name are silently ignored."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        block = SimpleNamespace(type="tool_result", content="output here")
        msg = _make_message(block)
        _process_message(msg, text_parts, tool_calls, "execute", _make_ui(), "/tmp/proj")
        assert text_parts == []
        assert tool_calls == []

    def test_text_preview_truncated_in_log(self):
        """stage_log is called with a preview truncated to 200 chars."""
        ui = MagicMock(spec=UI)
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        long_text = "A" * 500
        msg = _make_message(_make_text_block(long_text))
        _process_message(msg, text_parts, tool_calls, "research", ui, "/tmp/proj")
        # The full text is still stored
        assert text_parts == [long_text]
        # The preview logged to UI is at most 200 chars
        logged_preview = ui.stage_log.call_args_list[-1][0][1]
        assert len(logged_preview) <= 200

    def test_multiline_text_preview_collapsed(self):
        """Newlines in the logged preview are replaced with spaces."""
        ui = MagicMock(spec=UI)
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        msg = _make_message(_make_text_block("line1\nline2\nline3"))
        _process_message(msg, text_parts, tool_calls, "plan", ui, "/tmp/proj")
        logged_preview = ui.stage_log.call_args_list[-1][0][1]
        assert "\n" not in logged_preview

    def test_empty_content_list(self):
        """A message with an empty content list does nothing."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        msg = SimpleNamespace(content=[])
        _process_message(msg, text_parts, tool_calls, "discover", _make_ui(), "/tmp/proj")
        assert text_parts == []
        assert tool_calls == []


# ---------------------------------------------------------------------------
# run_agent – success path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRunAgentSuccess:
    @patch("graft.agent.artifacts")
    @patch("graft.agent.query")
    @patch("graft.agent.ClaudeAgentOptions")
    async def test_happy_path_returns_agent_result(
        self, mock_opts_cls, mock_query, mock_artifacts
    ):
        """run_agent returns an AgentResult with text, tool_calls and timing."""
        text_msg = _make_message(_make_text_block("All done."))
        tool_msg = _make_message(_make_tool_block("Bash", {"command": "echo hi"}))

        async def fake_query(**kwargs):
            for m in [text_msg, tool_msg]:
                yield m

        mock_query.side_effect = fake_query

        result = await run_agent(
            persona="coder",
            system_prompt="You are a coder.",
            user_prompt="Write code.",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="execute",
            ui=_make_ui(),
        )

        assert isinstance(result, AgentResult)
        assert "All done." in result.text
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["tool"] == "Bash"
        assert result.turns_used == 2
        assert result.elapsed_seconds > 0
        mock_artifacts.save_log.assert_called_once()

    @patch("graft.agent.artifacts")
    @patch("graft.agent.query")
    @patch("graft.agent.ClaudeAgentOptions")
    async def test_default_allowed_tools(self, mock_opts_cls, mock_query, mock_artifacts):
        """When allowed_tools is None, defaults are applied."""
        async def fake_query(**kwargs):
            return
            yield  # make it an async generator

        mock_query.side_effect = fake_query

        await run_agent(
            persona="auditor",
            system_prompt="Audit.",
            user_prompt="Check.",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="verify",
            ui=_make_ui(),
        )

        call_kwargs = mock_opts_cls.call_args[1]
        assert "Read" in call_kwargs["allowed_tools"]
        assert "Write" in call_kwargs["allowed_tools"]
        assert "Bash" in call_kwargs["allowed_tools"]

    @patch("graft.agent.artifacts")
    @patch("graft.agent.query")
    @patch("graft.agent.ClaudeAgentOptions")
    async def test_custom_allowed_tools(self, mock_opts_cls, mock_query, mock_artifacts):
        """Custom allowed_tools override the defaults."""
        async def fake_query(**kwargs):
            return
            yield

        mock_query.side_effect = fake_query

        await run_agent(
            persona="reader",
            system_prompt="Read only.",
            user_prompt="Read.",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="discover",
            ui=_make_ui(),
            allowed_tools=["Read", "Glob"],
        )

        call_kwargs = mock_opts_cls.call_args[1]
        assert call_kwargs["allowed_tools"] == ["Read", "Glob"]

    @patch("graft.agent.artifacts")
    @patch("graft.agent.query")
    @patch("graft.agent.ClaudeAgentOptions")
    async def test_model_passed_when_provided(self, mock_opts_cls, mock_query, mock_artifacts):
        """When model is specified, it appears in ClaudeAgentOptions kwargs."""
        async def fake_query(**kwargs):
            return
            yield

        mock_query.side_effect = fake_query

        await run_agent(
            persona="coder",
            system_prompt="Code.",
            user_prompt="Do it.",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="execute",
            ui=_make_ui(),
            model="claude-sonnet-4-20250514",
        )

        call_kwargs = mock_opts_cls.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"

    @patch("graft.agent.artifacts")
    @patch("graft.agent.query")
    @patch("graft.agent.ClaudeAgentOptions")
    async def test_model_omitted_when_none(self, mock_opts_cls, mock_query, mock_artifacts):
        """When model is None, it is not passed to ClaudeAgentOptions."""
        async def fake_query(**kwargs):
            return
            yield

        mock_query.side_effect = fake_query

        await run_agent(
            persona="coder",
            system_prompt="Code.",
            user_prompt="Do it.",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="execute",
            ui=_make_ui(),
        )

        call_kwargs = mock_opts_cls.call_args[1]
        assert "model" not in call_kwargs

    @patch("graft.agent.artifacts")
    @patch("graft.agent.query")
    @patch("graft.agent.ClaudeAgentOptions")
    async def test_empty_stream(self, mock_opts_cls, mock_query, mock_artifacts):
        """Agent that yields no messages returns an empty AgentResult."""
        async def fake_query(**kwargs):
            return
            yield

        mock_query.side_effect = fake_query

        result = await run_agent(
            persona="ghost",
            system_prompt="Say nothing.",
            user_prompt="…",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="discover",
            ui=_make_ui(),
        )

        assert result.text == ""
        assert result.tool_calls == []
        assert result.raw_messages == []
        assert result.turns_used == 0


# ---------------------------------------------------------------------------
# run_agent – retry / backoff logic
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRunAgentRetry:
    @patch("graft.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("graft.agent.artifacts")
    @patch("graft.agent.query")
    @patch("graft.agent.ClaudeAgentOptions")
    async def test_retry_on_connection_error(
        self, mock_opts_cls, mock_query, mock_artifacts, mock_sleep
    ):
        """run_agent retries on ConnectionError and succeeds on next attempt."""
        call_count = 0

        async def fake_query(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("connection reset")
            yield _make_message(_make_text_block("recovered"))

        mock_query.side_effect = fake_query

        result = await run_agent(
            persona="coder",
            system_prompt="Code.",
            user_prompt="Do it.",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="execute",
            ui=_make_ui(),
        )

        assert "recovered" in result.text
        assert call_count == 2
        mock_sleep.assert_called_once()
        # First retry delay = RETRY_BACKOFF_BASE ** 1
        assert mock_sleep.call_args[0][0] == RETRY_BACKOFF_BASE ** 1

    @patch("graft.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("graft.agent.artifacts")
    @patch("graft.agent.query")
    @patch("graft.agent.ClaudeAgentOptions")
    async def test_retry_on_timeout_error(
        self, mock_opts_cls, mock_query, mock_artifacts, mock_sleep
    ):
        """run_agent retries on TimeoutError."""
        call_count = 0

        async def fake_query(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("timed out")
            yield _make_message(_make_text_block("ok"))

        mock_query.side_effect = fake_query

        result = await run_agent(
            persona="coder",
            system_prompt="Code.",
            user_prompt="Do it.",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="execute",
            ui=_make_ui(),
        )

        assert result.text == "ok"

    @patch("graft.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("graft.agent.artifacts")
    @patch("graft.agent.query")
    @patch("graft.agent.ClaudeAgentOptions")
    async def test_retry_on_os_error(
        self, mock_opts_cls, mock_query, mock_artifacts, mock_sleep
    ):
        """run_agent retries on OSError."""
        call_count = 0

        async def fake_query(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("broken pipe")
            yield _make_message(_make_text_block("fixed"))

        mock_query.side_effect = fake_query

        result = await run_agent(
            persona="coder",
            system_prompt="Code.",
            user_prompt="Do it.",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="execute",
            ui=_make_ui(),
        )

        assert result.text == "fixed"

    @patch("graft.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("graft.agent.artifacts")
    @patch("graft.agent.query")
    @patch("graft.agent.ClaudeAgentOptions")
    async def test_max_retries_exceeded_raises_runtime_error(
        self, mock_opts_cls, mock_query, mock_artifacts, mock_sleep
    ):
        """After MAX_RETRIES failures, run_agent raises RuntimeError."""
        async def always_fail(**kwargs):
            raise ConnectionError("persistent failure")
            yield  # make it an async generator

        mock_query.side_effect = always_fail

        with pytest.raises(RuntimeError, match=r"failed after 3 attempts"):
            await run_agent(
                persona="coder",
                system_prompt="Code.",
                user_prompt="Do it.",
                cwd="/tmp",
                project_dir="/tmp/proj",
                stage="execute",
                ui=_make_ui(),
            )

        assert mock_sleep.call_count == MAX_RETRIES

    @patch("graft.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("graft.agent.artifacts")
    @patch("graft.agent.query")
    @patch("graft.agent.ClaudeAgentOptions")
    async def test_exponential_backoff_delays(
        self, mock_opts_cls, mock_query, mock_artifacts, mock_sleep
    ):
        """Backoff delays follow RETRY_BACKOFF_BASE ** attempt."""
        async def always_fail(**kwargs):
            raise TimeoutError("timeout")
            yield  # make it an async generator

        mock_query.side_effect = always_fail

        with pytest.raises(RuntimeError):
            await run_agent(
                persona="coder",
                system_prompt="Code.",
                user_prompt="Do it.",
                cwd="/tmp",
                project_dir="/tmp/proj",
                stage="execute",
                ui=_make_ui(),
            )

        delays = [call[0][0] for call in mock_sleep.call_args_list]
        expected = [RETRY_BACKOFF_BASE ** i for i in range(1, MAX_RETRIES + 1)]
        assert delays == expected

    @patch("graft.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("graft.agent.artifacts")
    @patch("graft.agent.query")
    @patch("graft.agent.ClaudeAgentOptions")
    async def test_non_retryable_error_propagates_immediately(
        self, mock_opts_cls, mock_query, mock_artifacts, mock_sleep
    ):
        """Errors not in the retry list (e.g., ValueError) propagate immediately."""
        async def raise_value_error(**kwargs):
            raise ValueError("bad input")
            yield  # make it an async generator

        mock_query.side_effect = raise_value_error

        with pytest.raises(ValueError, match="bad input"):
            await run_agent(
                persona="coder",
                system_prompt="Code.",
                user_prompt="Do it.",
                cwd="/tmp",
                project_dir="/tmp/proj",
                stage="execute",
                ui=_make_ui(),
            )

        mock_sleep.assert_not_called()

    @patch("graft.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("graft.agent.artifacts")
    @patch("graft.agent.query")
    @patch("graft.agent.ClaudeAgentOptions")
    async def test_error_during_iteration_triggers_retry(
        self, mock_opts_cls, mock_query, mock_artifacts, mock_sleep
    ):
        """An error raised mid-stream (during async iteration) triggers retry."""
        call_count = 0

        async def fake_query(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield _make_message(_make_text_block("partial"))
                raise ConnectionError("stream interrupted")
            yield _make_message(_make_text_block("complete"))

        mock_query.side_effect = fake_query

        result = await run_agent(
            persona="coder",
            system_prompt="Code.",
            user_prompt="Do it.",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="execute",
            ui=_make_ui(),
        )

        # The successful attempt should have clean state (not carry over partial)
        assert result.text == "complete"
        assert call_count == 2

    @patch("graft.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("graft.agent.artifacts")
    @patch("graft.agent.query")
    @patch("graft.agent.ClaudeAgentOptions")
    async def test_runtime_error_includes_last_exception(
        self, mock_opts_cls, mock_query, mock_artifacts, mock_sleep
    ):
        """The RuntimeError message includes the last underlying exception."""
        async def always_fail(**kwargs):
            raise OSError("disk full")
            yield  # make it an async generator

        mock_query.side_effect = always_fail

        with pytest.raises(RuntimeError, match="disk full"):
            await run_agent(
                persona="coder",
                system_prompt="Code.",
                user_prompt="Do it.",
                cwd="/tmp",
                project_dir="/tmp/proj",
                stage="execute",
                ui=_make_ui(),
            )


# ---------------------------------------------------------------------------
# run_agent – UI logging
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRunAgentLogging:
    @patch("graft.agent.artifacts")
    @patch("graft.agent.query")
    @patch("graft.agent.ClaudeAgentOptions")
    async def test_ui_start_and_finish_logged(
        self, mock_opts_cls, mock_query, mock_artifacts
    ):
        """UI receives starting and finished messages."""
        async def fake_query(**kwargs):
            return
            yield

        mock_query.side_effect = fake_query
        ui = MagicMock(spec=UI)

        await run_agent(
            persona="researcher",
            system_prompt="Research.",
            user_prompt="Find info.",
            cwd="/tmp",
            project_dir="/tmp/proj",
            stage="research",
            ui=ui,
        )

        log_calls = [call[0][1] for call in ui.stage_log.call_args_list]
        assert any("starting" in msg for msg in log_calls)
        assert any("finished" in msg for msg in log_calls)

    @patch("graft.agent.artifacts")
    @patch("graft.agent.query")
    @patch("graft.agent.ClaudeAgentOptions")
    async def test_save_log_called_with_correct_args(
        self, mock_opts_cls, mock_query, mock_artifacts
    ):
        """artifacts.save_log is called with project_dir, stage, and concatenated text."""
        async def fake_query(**kwargs):
            yield _make_message(_make_text_block("Line 1"))
            yield _make_message(_make_text_block("Line 2"))

        mock_query.side_effect = fake_query

        await run_agent(
            persona="coder",
            system_prompt="Code.",
            user_prompt="Write.",
            cwd="/tmp",
            project_dir="/tmp/myproj",
            stage="execute",
            ui=_make_ui(),
        )

        mock_artifacts.save_log.assert_called_once_with(
            "/tmp/myproj", "execute", "Line 1\nLine 2"
        )


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_max_retries_is_positive(self):
        assert MAX_RETRIES > 0

    def test_retry_backoff_base_greater_than_one(self):
        assert RETRY_BACKOFF_BASE > 1.0
