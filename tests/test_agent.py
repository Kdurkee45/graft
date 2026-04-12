"""Tests for graft.agent."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from graft.agent import (
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    AgentResult,
    _process_message,
    run_agent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text_block(text: str) -> SimpleNamespace:
    """Create a fake text content block."""
    return SimpleNamespace(text=text)


def _make_tool_block(name: str, tool_input: dict | None = None) -> SimpleNamespace:
    """Create a fake tool_use content block."""
    return SimpleNamespace(name=name, input=tool_input or {})


def _make_message(*blocks) -> SimpleNamespace:
    """Create a fake SDK message with a content list."""
    return SimpleNamespace(content=list(blocks))


def _make_ui() -> MagicMock:
    """Return a mock UI with a stage_log method."""
    ui = MagicMock()
    ui.stage_log = MagicMock()
    return ui


def _base_kwargs(ui: MagicMock | None = None) -> dict:
    """Return the minimal keyword arguments for run_agent."""
    return {
        "persona": "tester",
        "system_prompt": "You are a test agent.",
        "user_prompt": "Do the thing.",
        "cwd": "/tmp",
        "project_dir": "/tmp/project",
        "stage": "execute",
        "ui": ui or _make_ui(),
    }


# ---------------------------------------------------------------------------
# AgentResult dataclass
# ---------------------------------------------------------------------------


class TestAgentResult:
    def test_defaults(self):
        result = AgentResult(text="hello")
        assert result.text == "hello"
        assert result.tool_calls == []
        assert result.raw_messages == []
        assert result.elapsed_seconds == 0.0
        assert result.turns_used == 0

    def test_custom_values(self):
        result = AgentResult(
            text="output",
            tool_calls=[{"tool": "Bash", "input": {}}],
            raw_messages=["m1"],
            elapsed_seconds=1.5,
            turns_used=3,
        )
        assert result.tool_calls == [{"tool": "Bash", "input": {}}]
        assert result.raw_messages == ["m1"]
        assert result.elapsed_seconds == 1.5
        assert result.turns_used == 3


# ---------------------------------------------------------------------------
# _process_message
# ---------------------------------------------------------------------------


class TestProcessMessage:
    def test_extracts_text_blocks(self):
        """Text blocks are appended to text_parts and logged via ui."""
        msg = _make_message(_make_text_block("Hello world"))
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()

        _process_message(msg, text_parts, tool_calls, "execute", ui, "/tmp/proj")

        assert text_parts == ["Hello world"]
        assert tool_calls == []
        ui.stage_log.assert_called_once()
        assert "Hello world" in ui.stage_log.call_args[0][1]

    def test_extracts_multiple_text_blocks(self):
        msg = _make_message(_make_text_block("Part 1"), _make_text_block("Part 2"))
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()

        _process_message(msg, text_parts, tool_calls, "execute", ui, "/tmp/proj")

        assert text_parts == ["Part 1", "Part 2"]
        assert ui.stage_log.call_count == 2

    def test_extracts_tool_use_blocks(self):
        """Tool-use blocks are appended to tool_calls."""
        msg = _make_message(_make_tool_block("Bash", {"command": "ls"}))
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()

        _process_message(msg, text_parts, tool_calls, "execute", ui, "/tmp/proj")

        assert text_parts == []
        assert tool_calls == [{"tool": "Bash", "input": {"command": "ls"}}]
        ui.stage_log.assert_called_once()
        assert "Bash" in ui.stage_log.call_args[0][1]

    def test_extracts_tool_use_without_input(self):
        """Tool block without an input attribute defaults to empty dict."""
        block = SimpleNamespace(name="Read")  # no 'input' attribute
        msg = _make_message(block)
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()

        _process_message(msg, text_parts, tool_calls, "execute", ui, "/tmp/proj")

        assert tool_calls == [{"tool": "Read", "input": {}}]

    def test_skips_message_without_content(self):
        """Messages that lack a content attribute are silently skipped."""
        msg = SimpleNamespace(role="system")  # no 'content'
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()

        _process_message(msg, text_parts, tool_calls, "execute", ui, "/tmp/proj")

        assert text_parts == []
        assert tool_calls == []
        ui.stage_log.assert_not_called()

    def test_skips_empty_text_blocks(self):
        """Text blocks with empty string are not appended."""
        block = SimpleNamespace(text="")
        msg = _make_message(block)
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()

        _process_message(msg, text_parts, tool_calls, "execute", ui, "/tmp/proj")

        assert text_parts == []

    def test_mixed_text_and_tool_blocks(self):
        """Messages with both text and tool blocks extract both."""
        msg = _make_message(
            _make_text_block("Thinking…"),
            _make_tool_block("Grep", {"pattern": "TODO"}),
            _make_text_block("Found 3 results."),
        )
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()

        _process_message(msg, text_parts, tool_calls, "execute", ui, "/tmp/proj")

        assert text_parts == ["Thinking…", "Found 3 results."]
        assert tool_calls == [{"tool": "Grep", "input": {"pattern": "TODO"}}]
        assert ui.stage_log.call_count == 3

    def test_long_text_preview_truncated(self):
        """Text preview passed to ui.stage_log is truncated to 200 chars."""
        long_text = "A" * 500
        msg = _make_message(_make_text_block(long_text))
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()

        _process_message(msg, text_parts, tool_calls, "execute", ui, "/tmp/proj")

        # Full text kept in text_parts
        assert text_parts == [long_text]
        # Preview is truncated
        logged = ui.stage_log.call_args[0][1]
        assert len(logged) <= 200


# ---------------------------------------------------------------------------
# run_agent – success path
# ---------------------------------------------------------------------------


class TestRunAgentSuccess:
    @pytest.fixture
    def ui(self):
        return _make_ui()

    async def _run_with_messages(self, messages, ui=None, **extra_kwargs):
        """Helper: patch SDK query to yield *messages* and call run_agent."""
        if ui is None:
            ui = _make_ui()

        async def fake_query(prompt, options):
            for m in messages:
                yield m

        kwargs = _base_kwargs(ui)
        kwargs.update(extra_kwargs)

        with (
            patch("graft.agent.query", side_effect=fake_query),
            patch("graft.agent.artifacts") as mock_artifacts,
            patch("graft.agent.ClaudeAgentOptions"),
        ):
            result = await run_agent(**kwargs)
        return result, mock_artifacts

    async def test_returns_agent_result(self, ui):
        """Successful run returns AgentResult with assembled text."""
        msg = _make_message(_make_text_block("Line 1"), _make_text_block("Line 2"))
        result, _ = await self._run_with_messages([msg], ui)

        assert isinstance(result, AgentResult)
        assert result.text == "Line 1\nLine 2"
        assert result.turns_used == 1
        assert result.elapsed_seconds > 0.0

    async def test_tool_calls_collected(self, ui):
        """Tool use blocks are collected in the result."""
        msg = _make_message(_make_tool_block("Bash", {"command": "echo hi"}))
        result, _ = await self._run_with_messages([msg], ui)

        assert result.tool_calls == [{"tool": "Bash", "input": {"command": "echo hi"}}]

    async def test_raw_messages_preserved(self, ui):
        """All raw message objects are kept in raw_messages."""
        msg1 = _make_message(_make_text_block("a"))
        msg2 = _make_message(_make_text_block("b"))
        result, _ = await self._run_with_messages([msg1, msg2], ui)

        assert result.raw_messages == [msg1, msg2]
        assert result.turns_used == 2

    async def test_save_log_called(self, ui):
        """artifacts.save_log is called with the full text."""
        msg = _make_message(_make_text_block("log this"))
        _, mock_artifacts = await self._run_with_messages([msg], ui)

        mock_artifacts.save_log.assert_called_once_with(
            "/tmp/project", "execute", "log this"
        )

    async def test_empty_stream_returns_empty_text(self, ui):
        """An agent session with zero messages returns empty text."""
        result, _ = await self._run_with_messages([], ui)

        assert result.text == ""
        assert result.tool_calls == []
        assert result.turns_used == 0


# ---------------------------------------------------------------------------
# run_agent – allowed_tools defaults
# ---------------------------------------------------------------------------


class TestAllowedToolsDefault:
    async def test_defaults_when_none(self):
        """When allowed_tools is None, a default list is supplied."""
        captured_options = {}

        def capture_options(**kwargs):
            captured_options.update(kwargs)
            return MagicMock()

        async def fake_query(prompt, options):
            return
            yield  # make it an async generator

        ui = _make_ui()
        kwargs = _base_kwargs(ui)
        # allowed_tools not set → defaults to None

        with (
            patch("graft.agent.query", side_effect=fake_query),
            patch("graft.agent.artifacts"),
            patch("graft.agent.ClaudeAgentOptions", side_effect=capture_options),
        ):
            await run_agent(**kwargs)

        assert "allowed_tools" in captured_options
        default_tools = captured_options["allowed_tools"]
        assert "Read" in default_tools
        assert "Write" in default_tools
        assert "Bash" in default_tools
        assert "Glob" in default_tools
        assert "Grep" in default_tools
        assert "Edit" in default_tools
        assert "MultiEdit" in default_tools

    async def test_custom_allowed_tools(self):
        """Explicitly passed allowed_tools are forwarded to options."""
        captured_options = {}

        def capture_options(**kwargs):
            captured_options.update(kwargs)
            return MagicMock()

        async def fake_query(prompt, options):
            return
            yield

        ui = _make_ui()
        kwargs = _base_kwargs(ui)
        kwargs["allowed_tools"] = ["Read", "Grep"]

        with (
            patch("graft.agent.query", side_effect=fake_query),
            patch("graft.agent.artifacts"),
            patch("graft.agent.ClaudeAgentOptions", side_effect=capture_options),
        ):
            await run_agent(**kwargs)

        assert captured_options["allowed_tools"] == ["Read", "Grep"]


# ---------------------------------------------------------------------------
# run_agent – model parameter
# ---------------------------------------------------------------------------


class TestModelParameter:
    async def test_model_included_when_set(self):
        """When model is provided, it appears in opts."""
        captured_options = {}

        def capture_options(**kwargs):
            captured_options.update(kwargs)
            return MagicMock()

        async def fake_query(prompt, options):
            return
            yield

        ui = _make_ui()
        kwargs = _base_kwargs(ui)
        kwargs["model"] = "claude-sonnet-4-20250514"

        with (
            patch("graft.agent.query", side_effect=fake_query),
            patch("graft.agent.artifacts"),
            patch("graft.agent.ClaudeAgentOptions", side_effect=capture_options),
        ):
            await run_agent(**kwargs)

        assert captured_options["model"] == "claude-sonnet-4-20250514"

    async def test_model_excluded_when_none(self):
        """When model is None (default), 'model' key is not in opts."""
        captured_options = {}

        def capture_options(**kwargs):
            captured_options.update(kwargs)
            return MagicMock()

        async def fake_query(prompt, options):
            return
            yield

        ui = _make_ui()
        kwargs = _base_kwargs(ui)
        # model defaults to None — not set

        with (
            patch("graft.agent.query", side_effect=fake_query),
            patch("graft.agent.artifacts"),
            patch("graft.agent.ClaudeAgentOptions", side_effect=capture_options),
        ):
            await run_agent(**kwargs)

        assert "model" not in captured_options


# ---------------------------------------------------------------------------
# run_agent – retry behaviour
# ---------------------------------------------------------------------------


class TestRunAgentRetries:
    async def test_retries_on_connection_error(self):
        """ConnectionError triggers retries with exponential backoff."""
        call_count = 0

        async def failing_then_ok(prompt, options):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("server gone")
            msg = _make_message(_make_text_block("recovered"))
            yield msg

        ui = _make_ui()
        kwargs = _base_kwargs(ui)

        with (
            patch("graft.agent.query", side_effect=failing_then_ok),
            patch("graft.agent.artifacts"),
            patch("graft.agent.ClaudeAgentOptions"),
            patch("graft.agent.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            result = await run_agent(**kwargs)

        assert result.text == "recovered"
        assert call_count == 3
        # Backoff: 2^1 = 2, 2^2 = 4
        mock_sleep.assert_any_call(RETRY_BACKOFF_BASE**1)
        mock_sleep.assert_any_call(RETRY_BACKOFF_BASE**2)

    async def test_retries_on_timeout_error(self):
        """TimeoutError also triggers retry."""
        call_count = 0

        async def fail_once(prompt, options):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("timed out")
            yield _make_message(_make_text_block("ok"))

        ui = _make_ui()
        kwargs = _base_kwargs(ui)

        with (
            patch("graft.agent.query", side_effect=fail_once),
            patch("graft.agent.artifacts"),
            patch("graft.agent.ClaudeAgentOptions"),
            patch("graft.agent.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await run_agent(**kwargs)

        assert result.text == "ok"
        assert call_count == 2

    async def test_retries_on_os_error(self):
        """OSError also triggers retry."""
        call_count = 0

        async def fail_once(prompt, options):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("network unreachable")
            yield _make_message(_make_text_block("ok"))

        ui = _make_ui()
        kwargs = _base_kwargs(ui)

        with (
            patch("graft.agent.query", side_effect=fail_once),
            patch("graft.agent.artifacts"),
            patch("graft.agent.ClaudeAgentOptions"),
            patch("graft.agent.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await run_agent(**kwargs)

        assert result.text == "ok"
        assert call_count == 2

    async def test_raises_after_max_retries(self):
        """After MAX_RETRIES failures, RuntimeError is raised."""

        async def always_fail(prompt, options):
            raise ConnectionError("persistent failure")
            yield  # noqa: unreachable — make it an async generator

        ui = _make_ui()
        kwargs = _base_kwargs(ui)

        with (
            patch("graft.agent.query", side_effect=always_fail),
            patch("graft.agent.artifacts"),
            patch("graft.agent.ClaudeAgentOptions"),
            patch("graft.agent.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            with pytest.raises(RuntimeError, match=f"after {MAX_RETRIES} attempts"):
                await run_agent(**kwargs)

        assert mock_sleep.call_count == MAX_RETRIES

    async def test_non_retryable_error_propagates(self):
        """A ValueError (not in retry list) propagates immediately."""

        async def raise_value_error(prompt, options):
            raise ValueError("bad input")
            yield

        ui = _make_ui()
        kwargs = _base_kwargs(ui)

        with (
            patch("graft.agent.query", side_effect=raise_value_error),
            patch("graft.agent.artifacts"),
            patch("graft.agent.ClaudeAgentOptions"),
            patch("graft.agent.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            with pytest.raises(ValueError, match="bad input"):
                await run_agent(**kwargs)

        mock_sleep.assert_not_called()

    async def test_retry_resets_accumulators(self):
        """Each retry attempt starts with fresh text_parts / tool_calls."""
        call_count = 0

        async def fail_then_ok(prompt, options):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Yield a message, then raise — text should be discarded
                yield _make_message(_make_text_block("stale data"))
                raise ConnectionError("mid-stream failure")
            yield _make_message(_make_text_block("fresh"))

        ui = _make_ui()
        kwargs = _base_kwargs(ui)

        with (
            patch("graft.agent.query", side_effect=fail_then_ok),
            patch("graft.agent.artifacts"),
            patch("graft.agent.ClaudeAgentOptions"),
            patch("graft.agent.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await run_agent(**kwargs)

        # Only the successful attempt's text should be present
        assert "stale data" not in result.text
        assert result.text == "fresh"


# ---------------------------------------------------------------------------
# run_agent – UI logging
# ---------------------------------------------------------------------------


class TestRunAgentUILogging:
    async def test_start_and_finish_logged(self):
        """UI receives 'starting' and 'finished' stage_log calls."""

        async def fake_query(prompt, options):
            yield _make_message(_make_text_block("done"))

        ui = _make_ui()
        kwargs = _base_kwargs(ui)

        with (
            patch("graft.agent.query", side_effect=fake_query),
            patch("graft.agent.artifacts"),
            patch("graft.agent.ClaudeAgentOptions"),
        ):
            await run_agent(**kwargs)

        logged_messages = [c[0][1] for c in ui.stage_log.call_args_list]
        assert any("starting" in m for m in logged_messages)
        assert any("finished" in m for m in logged_messages)

    async def test_retry_warning_logged(self):
        """On retry, the warning with attempt number is logged to UI."""
        call_count = 0

        async def fail_once(prompt, options):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("boom")
            yield _make_message(_make_text_block("ok"))

        ui = _make_ui()
        kwargs = _base_kwargs(ui)

        with (
            patch("graft.agent.query", side_effect=fail_once),
            patch("graft.agent.artifacts"),
            patch("graft.agent.ClaudeAgentOptions"),
            patch("graft.agent.asyncio.sleep", new_callable=AsyncMock),
        ):
            await run_agent(**kwargs)

        logged_messages = [c[0][1] for c in ui.stage_log.call_args_list]
        assert any("attempt 1" in m for m in logged_messages)
