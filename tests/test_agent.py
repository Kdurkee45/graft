"""Tests for graft.agent – run_agent wrapper and message processing."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
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
from graft.ui import UI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text_block(text: str) -> SimpleNamespace:
    """Return a mock content block with a `text` attribute."""
    return SimpleNamespace(text=text)


def _make_tool_block(name: str, input_data: dict | None = None) -> SimpleNamespace:
    """Return a mock content block with a `name` and optional `input`."""
    return SimpleNamespace(name=name, input=input_data or {})


def _make_message(*blocks) -> SimpleNamespace:
    """Return a mock SDK message with a `content` list."""
    return SimpleNamespace(content=list(blocks))


def _make_ui() -> UI:
    """Create a UI instance that won't try to write to a real terminal."""
    return UI(auto_approve=True, verbose=True)


# ---------------------------------------------------------------------------
# Async iterator helper – simulates `query()` yielding messages
# ---------------------------------------------------------------------------


class _AsyncIter:
    """Turn a regular iterable into an async iterator (``async for``)."""

    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


# ---------------------------------------------------------------------------
# AgentResult dataclass
# ---------------------------------------------------------------------------


class TestAgentResultDataclass:
    def test_defaults(self):
        r = AgentResult(text="hello")
        assert r.text == "hello"
        assert r.tool_calls == []
        assert r.raw_messages == []
        assert r.elapsed_seconds == 0.0
        assert r.turns_used == 0

    def test_with_values(self):
        r = AgentResult(
            text="done",
            tool_calls=[{"tool": "Read"}],
            raw_messages=["m1"],
            elapsed_seconds=1.5,
            turns_used=3,
        )
        assert r.tool_calls == [{"tool": "Read"}]
        assert r.turns_used == 3


# ---------------------------------------------------------------------------
# _process_message
# ---------------------------------------------------------------------------


class TestProcessMessage:
    def test_text_block_appended(self):
        ui = _make_ui()
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        msg = _make_message(_make_text_block("Hello world"))

        _process_message(msg, text_parts, tool_calls, "discover", ui, "/tmp/proj")

        assert text_parts == ["Hello world"]
        assert tool_calls == []

    def test_tool_use_block_appended(self):
        ui = _make_ui()
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        msg = _make_message(_make_tool_block("Read", {"file_path": "/a.py"}))

        _process_message(msg, text_parts, tool_calls, "discover", ui, "/tmp/proj")

        assert text_parts == []
        assert tool_calls == [{"tool": "Read", "input": {"file_path": "/a.py"}}]

    def test_mixed_blocks(self):
        ui = _make_ui()
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        msg = _make_message(
            _make_text_block("Analyzing…"),
            _make_tool_block("Bash", {"command": "ls"}),
            _make_text_block("Done."),
        )

        _process_message(msg, text_parts, tool_calls, "build", ui, "/tmp/proj")

        assert text_parts == ["Analyzing…", "Done."]
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "Bash"

    def test_message_without_content_attribute(self):
        """Messages lacking a `content` attribute should be silently skipped."""
        ui = _make_ui()
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        msg = SimpleNamespace(role="system")  # no content attr

        _process_message(msg, text_parts, tool_calls, "plan", ui, "/tmp/proj")

        assert text_parts == []
        assert tool_calls == []

    def test_empty_text_block_skipped(self):
        """A text block with empty string should not be appended."""
        ui = _make_ui()
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        msg = _make_message(SimpleNamespace(text=""))

        _process_message(msg, text_parts, tool_calls, "discover", ui, "/tmp/proj")

        assert text_parts == []

    def test_tool_block_without_input(self):
        """Tool blocks may lack an `input` attr; should default to {}."""
        ui = _make_ui()
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        block = SimpleNamespace(name="Glob")  # no `input` attr
        msg = _make_message(block)

        _process_message(msg, text_parts, tool_calls, "discover", ui, "/tmp/proj")

        assert tool_calls == [{"tool": "Glob", "input": {}}]


# ---------------------------------------------------------------------------
# run_agent – happy path
# ---------------------------------------------------------------------------


class TestRunAgentHappyPath:
    @patch("graft.agent.artifacts.save_log")
    @patch("graft.agent.query")
    async def test_returns_agent_result(self, mock_query, mock_save_log):
        messages = [
            _make_message(_make_text_block("Step 1")),
            _make_message(_make_tool_block("Read", {"file_path": "x.py"})),
            _make_message(_make_text_block("Step 2")),
        ]
        mock_query.return_value = _AsyncIter(messages)

        result = await run_agent(
            persona="researcher",
            system_prompt="You are a researcher.",
            user_prompt="Analyze the repo.",
            cwd="/repo",
            project_dir="/tmp/proj",
            stage="discover",
            ui=_make_ui(),
        )

        assert isinstance(result, AgentResult)
        assert "Step 1" in result.text
        assert "Step 2" in result.text
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["tool"] == "Read"
        assert result.turns_used == 3
        assert result.elapsed_seconds > 0

    @patch("graft.agent.artifacts.save_log")
    @patch("graft.agent.query")
    async def test_save_log_called(self, mock_query, mock_save_log):
        mock_query.return_value = _AsyncIter([_make_message(_make_text_block("done"))])

        await run_agent(
            persona="builder",
            system_prompt="sys",
            user_prompt="build it",
            cwd="/repo",
            project_dir="/tmp/proj",
            stage="build",
            ui=_make_ui(),
        )

        mock_save_log.assert_called_once()
        args = mock_save_log.call_args
        assert args[0][0] == "/tmp/proj"
        assert args[0][1] == "build"
        assert "done" in args[0][2]

    @patch("graft.agent.artifacts.save_log")
    @patch("graft.agent.query")
    async def test_empty_messages_produces_empty_text(self, mock_query, mock_save_log):
        mock_query.return_value = _AsyncIter([])

        result = await run_agent(
            persona="auditor",
            system_prompt="sys",
            user_prompt="check",
            cwd="/repo",
            project_dir="/tmp/proj",
            stage="grill",
            ui=_make_ui(),
        )

        assert result.text == ""
        assert result.tool_calls == []
        assert result.turns_used == 0


# ---------------------------------------------------------------------------
# run_agent – allowed_tools passthrough
# ---------------------------------------------------------------------------


class TestRunAgentAllowedTools:
    @patch("graft.agent.artifacts.save_log")
    @patch("graft.agent.ClaudeAgentOptions")
    @patch("graft.agent.query")
    async def test_default_allowed_tools(
        self, mock_query, mock_opts_cls, mock_save_log
    ):
        mock_query.return_value = _AsyncIter([])

        await run_agent(
            persona="p",
            system_prompt="s",
            user_prompt="u",
            cwd="/repo",
            project_dir="/tmp/proj",
            stage="discover",
            ui=_make_ui(),
        )

        call_kwargs = mock_opts_cls.call_args[1]
        assert "Read" in call_kwargs["allowed_tools"]
        assert "Write" in call_kwargs["allowed_tools"]
        assert "Bash" in call_kwargs["allowed_tools"]

    @patch("graft.agent.artifacts.save_log")
    @patch("graft.agent.ClaudeAgentOptions")
    @patch("graft.agent.query")
    async def test_custom_allowed_tools(self, mock_query, mock_opts_cls, mock_save_log):
        mock_query.return_value = _AsyncIter([])

        await run_agent(
            persona="p",
            system_prompt="s",
            user_prompt="u",
            cwd="/repo",
            project_dir="/tmp/proj",
            stage="discover",
            ui=_make_ui(),
            allowed_tools=["Read", "Grep"],
        )

        call_kwargs = mock_opts_cls.call_args[1]
        assert call_kwargs["allowed_tools"] == ["Read", "Grep"]


# ---------------------------------------------------------------------------
# run_agent – model forwarding
# ---------------------------------------------------------------------------


class TestRunAgentModelForwarding:
    @patch("graft.agent.artifacts.save_log")
    @patch("graft.agent.ClaudeAgentOptions")
    @patch("graft.agent.query")
    async def test_model_set_when_provided(
        self, mock_query, mock_opts_cls, mock_save_log
    ):
        mock_query.return_value = _AsyncIter([])

        await run_agent(
            persona="p",
            system_prompt="s",
            user_prompt="u",
            cwd="/repo",
            project_dir="/tmp/proj",
            stage="discover",
            ui=_make_ui(),
            model="claude-sonnet-4-20250514",
        )

        call_kwargs = mock_opts_cls.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"

    @patch("graft.agent.artifacts.save_log")
    @patch("graft.agent.ClaudeAgentOptions")
    @patch("graft.agent.query")
    async def test_model_omitted_when_none(
        self, mock_query, mock_opts_cls, mock_save_log
    ):
        mock_query.return_value = _AsyncIter([])

        await run_agent(
            persona="p",
            system_prompt="s",
            user_prompt="u",
            cwd="/repo",
            project_dir="/tmp/proj",
            stage="discover",
            ui=_make_ui(),
            model=None,
        )

        call_kwargs = mock_opts_cls.call_args[1]
        assert "model" not in call_kwargs


# ---------------------------------------------------------------------------
# run_agent – system_prompt & persona forwarding
# ---------------------------------------------------------------------------


class TestRunAgentPromptForwarding:
    @patch("graft.agent.artifacts.save_log")
    @patch("graft.agent.ClaudeAgentOptions")
    @patch("graft.agent.query")
    async def test_system_prompt_forwarded(
        self, mock_query, mock_opts_cls, mock_save_log
    ):
        mock_query.return_value = _AsyncIter([])

        await run_agent(
            persona="architect",
            system_prompt="You are a software architect.",
            user_prompt="Plan the feature.",
            cwd="/repo",
            project_dir="/tmp/proj",
            stage="plan",
            ui=_make_ui(),
        )

        call_kwargs = mock_opts_cls.call_args[1]
        assert call_kwargs["system_prompt"] == "You are a software architect."

    @patch("graft.agent.artifacts.save_log")
    @patch("graft.agent.query")
    async def test_user_prompt_forwarded(self, mock_query, mock_save_log):
        mock_query.return_value = _AsyncIter([])

        await run_agent(
            persona="p",
            system_prompt="s",
            user_prompt="Do the thing.",
            cwd="/repo",
            project_dir="/tmp/proj",
            stage="build",
            ui=_make_ui(),
        )

        call_kwargs = mock_query.call_args[1]
        assert call_kwargs["prompt"] == "Do the thing."


# ---------------------------------------------------------------------------
# run_agent – retry logic
# ---------------------------------------------------------------------------


class TestRunAgentRetry:
    @patch("graft.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("graft.agent.artifacts.save_log")
    @patch("graft.agent.query")
    async def test_retries_on_connection_error(
        self, mock_query, mock_save_log, mock_sleep
    ):
        """ConnectionError on first attempt, success on second."""

        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("connection reset")
            return _AsyncIter([_make_message(_make_text_block("ok"))])

        mock_query.side_effect = side_effect

        result = await run_agent(
            persona="p",
            system_prompt="s",
            user_prompt="u",
            cwd="/repo",
            project_dir="/tmp/proj",
            stage="discover",
            ui=_make_ui(),
        )

        assert result.text == "ok"
        assert call_count == 2
        mock_sleep.assert_called_once_with(RETRY_BACKOFF_BASE**1)

    @patch("graft.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("graft.agent.artifacts.save_log")
    @patch("graft.agent.query")
    async def test_retries_on_timeout_error(
        self, mock_query, mock_save_log, mock_sleep
    ):
        """TimeoutError on first two attempts, success on third."""

        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise TimeoutError("timed out")
            return _AsyncIter([_make_message(_make_text_block("recovered"))])

        mock_query.side_effect = side_effect

        result = await run_agent(
            persona="p",
            system_prompt="s",
            user_prompt="u",
            cwd="/repo",
            project_dir="/tmp/proj",
            stage="build",
            ui=_make_ui(),
        )

        assert result.text == "recovered"
        assert call_count == 3
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(RETRY_BACKOFF_BASE**1)
        mock_sleep.assert_any_call(RETRY_BACKOFF_BASE**2)

    @patch("graft.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("graft.agent.artifacts.save_log")
    @patch("graft.agent.query")
    async def test_retries_on_oserror(self, mock_query, mock_save_log, mock_sleep):
        """OSError triggers retry too."""

        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("network unreachable")
            return _AsyncIter([_make_message(_make_text_block("fine"))])

        mock_query.side_effect = side_effect

        result = await run_agent(
            persona="p",
            system_prompt="s",
            user_prompt="u",
            cwd="/repo",
            project_dir="/tmp/proj",
            stage="discover",
            ui=_make_ui(),
        )

        assert result.text == "fine"

    @patch("graft.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("graft.agent.query")
    async def test_max_retries_exceeded_raises_runtime_error(
        self, mock_query, mock_sleep
    ):
        """After MAX_RETRIES consecutive failures, RuntimeError is raised."""

        mock_query.side_effect = ConnectionError("down")

        with pytest.raises(RuntimeError, match=f"failed after {MAX_RETRIES} attempts"):
            await run_agent(
                persona="explorer",
                system_prompt="s",
                user_prompt="u",
                cwd="/repo",
                project_dir="/tmp/proj",
                stage="discover",
                ui=_make_ui(),
            )

        assert mock_query.call_count == MAX_RETRIES
        assert mock_sleep.call_count == MAX_RETRIES

    @patch("graft.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("graft.agent.query")
    async def test_exponential_backoff_delays(self, mock_query, mock_sleep):
        """Verify backoff delays follow RETRY_BACKOFF_BASE ** attempt."""

        mock_query.side_effect = TimeoutError("timeout")

        with pytest.raises(RuntimeError):
            await run_agent(
                persona="p",
                system_prompt="s",
                user_prompt="u",
                cwd="/repo",
                project_dir="/tmp/proj",
                stage="discover",
                ui=_make_ui(),
            )

        expected_delays = [RETRY_BACKOFF_BASE**i for i in range(1, MAX_RETRIES + 1)]
        actual_delays = [c.args[0] for c in mock_sleep.call_args_list]
        assert actual_delays == expected_delays

    @patch("graft.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("graft.agent.query")
    async def test_non_retryable_error_propagates_immediately(
        self, mock_query, mock_sleep
    ):
        """ValueError (not in retry set) should bubble up without retries."""

        mock_query.side_effect = ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            await run_agent(
                persona="p",
                system_prompt="s",
                user_prompt="u",
                cwd="/repo",
                project_dir="/tmp/proj",
                stage="discover",
                ui=_make_ui(),
            )

        assert mock_query.call_count == 1
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# run_agent – error during async iteration (mid-stream failure)
# ---------------------------------------------------------------------------


class TestRunAgentMidStreamRetry:
    @patch("graft.agent.asyncio.sleep", new_callable=AsyncMock)
    @patch("graft.agent.artifacts.save_log")
    @patch("graft.agent.query")
    async def test_retry_on_midstream_connection_error(
        self, mock_query, mock_save_log, mock_sleep
    ):
        """ConnectionError raised while iterating messages triggers retry."""

        class _FailingIter:
            def __init__(self):
                self._yielded = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._yielded:
                    self._yielded = True
                    return _make_message(_make_text_block("partial"))
                raise ConnectionError("stream dropped")

        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _FailingIter()
            return _AsyncIter([_make_message(_make_text_block("full answer"))])

        mock_query.side_effect = side_effect

        result = await run_agent(
            persona="p",
            system_prompt="s",
            user_prompt="u",
            cwd="/repo",
            project_dir="/tmp/proj",
            stage="discover",
            ui=_make_ui(),
        )

        assert result.text == "full answer"
        assert call_count == 2


# ---------------------------------------------------------------------------
# run_agent – max_turns forwarding
# ---------------------------------------------------------------------------


class TestRunAgentMaxTurns:
    @patch("graft.agent.artifacts.save_log")
    @patch("graft.agent.ClaudeAgentOptions")
    @patch("graft.agent.query")
    async def test_max_turns_forwarded(self, mock_query, mock_opts_cls, mock_save_log):
        mock_query.return_value = _AsyncIter([])

        await run_agent(
            persona="p",
            system_prompt="s",
            user_prompt="u",
            cwd="/repo",
            project_dir="/tmp/proj",
            stage="discover",
            ui=_make_ui(),
            max_turns=25,
        )

        call_kwargs = mock_opts_cls.call_args[1]
        assert call_kwargs["max_turns"] == 25
