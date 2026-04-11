"""Tests for graft.agent — AgentResult dataclass and run_agent orchestration."""

from __future__ import annotations

import asyncio
from dataclasses import fields
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

def _make_ui() -> MagicMock:
    """Return a mock UI with the methods used by run_agent / _process_message."""
    ui = MagicMock()
    ui.stage_log = MagicMock()
    return ui


def _text_block(text: str):
    """Create a SimpleNamespace that looks like a text content block."""
    return SimpleNamespace(text=text)


def _tool_block(name: str, input_data: dict | None = None):
    """Create a SimpleNamespace that looks like a tool-use content block."""
    return SimpleNamespace(name=name, input=input_data or {})


def _message(blocks):
    """Create a SimpleNamespace that looks like an SDK message with .content."""
    return SimpleNamespace(content=blocks)


async def _async_gen_from(items):
    """Turn a list of items into an async generator (simulating claude_agent_sdk.query)."""
    for item in items:
        yield item


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
        """AgentResult with all fields set."""
        calls = [{"tool": "Read", "input": {}}]
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

    def test_default_factory_independence(self):
        """Default mutable fields are independent across instances."""
        r1 = AgentResult(text="a")
        r2 = AgentResult(text="b")
        r1.tool_calls.append({"tool": "X"})
        assert r2.tool_calls == []

    def test_field_names(self):
        """Verify the set of field names matches the public API."""
        names = {f.name for f in fields(AgentResult)}
        assert names == {"text", "tool_calls", "raw_messages", "elapsed_seconds", "turns_used"}


# ---------------------------------------------------------------------------
# _process_message
# ---------------------------------------------------------------------------

class TestProcessMessage:
    def test_text_block_appended(self):
        """Text blocks are appended to text_parts and previewed in stage_log."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()
        msg = _message([_text_block("Hello world")])

        _process_message(msg, text_parts, tool_calls, "discover", ui, "/tmp/proj")

        assert text_parts == ["Hello world"]
        assert tool_calls == []
        ui.stage_log.assert_called_once()
        # The preview should contain the text (truncated to 200 chars, newlines replaced)
        logged = ui.stage_log.call_args[0][1]
        assert "Hello world" in logged

    def test_tool_use_block_appended(self):
        """Tool-use blocks are appended to tool_calls list."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()
        msg = _message([_tool_block("Bash", {"command": "ls"})])

        _process_message(msg, text_parts, tool_calls, "execute", ui, "/tmp/proj")

        assert text_parts == []
        assert tool_calls == [{"tool": "Bash", "input": {"command": "ls"}}]
        ui.stage_log.assert_called_once_with("execute", "  ↳ tool: Bash")

    def test_mixed_blocks(self):
        """A message with both text and tool blocks populates both lists."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()
        msg = _message([
            _text_block("Analysing files…"),
            _tool_block("Grep", {"pattern": "TODO"}),
            _text_block("Found 3 matches."),
        ])

        _process_message(msg, text_parts, tool_calls, "research", ui, "/tmp/proj")

        assert text_parts == ["Analysing files…", "Found 3 matches."]
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "Grep"
        assert ui.stage_log.call_count == 3

    def test_message_without_content_attr(self):
        """Messages lacking a .content attribute are silently skipped."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()
        msg = SimpleNamespace(role="assistant")  # no .content

        _process_message(msg, text_parts, tool_calls, "plan", ui, "/tmp/proj")

        assert text_parts == []
        assert tool_calls == []
        ui.stage_log.assert_not_called()

    def test_empty_content_list(self):
        """Message with an empty content list produces no output."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()
        msg = _message([])

        _process_message(msg, text_parts, tool_calls, "verify", ui, "/tmp/proj")

        assert text_parts == []
        assert tool_calls == []

    def test_text_block_empty_string_ignored(self):
        """A text block with empty string is skipped (block.text is falsy)."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()
        msg = _message([_text_block("")])

        _process_message(msg, text_parts, tool_calls, "discover", ui, "/tmp/proj")

        assert text_parts == []
        ui.stage_log.assert_not_called()

    def test_tool_block_without_input(self):
        """A tool block missing the .input attribute defaults to empty dict."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()
        block = SimpleNamespace(name="Read")  # no .input attribute
        msg = _message([block])

        _process_message(msg, text_parts, tool_calls, "execute", ui, "/tmp/proj")

        assert tool_calls == [{"tool": "Read", "input": {}}]

    def test_long_text_preview_truncated(self):
        """Preview logged to UI is truncated to 200 chars with newlines stripped."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()
        long_text = "A" * 300
        msg = _message([_text_block(long_text)])

        _process_message(msg, text_parts, tool_calls, "plan", ui, "/tmp/proj")

        logged = ui.stage_log.call_args[0][1]
        assert len(logged) == 200

    def test_text_preview_newlines_replaced(self):
        """Newlines in the preview are replaced with spaces."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        ui = _make_ui()
        msg = _message([_text_block("line1\nline2\nline3")])

        _process_message(msg, text_parts, tool_calls, "plan", ui, "/tmp/proj")

        logged = ui.stage_log.call_args[0][1]
        assert "\n" not in logged
        assert "line1 line2 line3" == logged


# ---------------------------------------------------------------------------
# run_agent — success path
# ---------------------------------------------------------------------------

class TestRunAgentSuccess:
    @pytest.fixture
    def mock_query(self):
        """Patch claude_agent_sdk.query as used in graft.agent."""
        with patch("graft.agent.query") as mocked:
            yield mocked

    @pytest.fixture
    def mock_save_log(self):
        with patch("graft.agent.artifacts.save_log") as mocked:
            yield mocked

    async def test_basic_success_returns_agent_result(self, mock_query, mock_save_log):
        """Successful run returns AgentResult with text and tool_calls."""
        msg1 = _message([_text_block("Step 1 done.")])
        msg2 = _message([_tool_block("Bash", {"command": "echo ok"})])
        msg3 = _message([_text_block("All done.")])

        mock_query.return_value = _async_gen_from([msg1, msg2, msg3])
        ui = _make_ui()

        result = await run_agent(
            persona="coder",
            system_prompt="You are a coder.",
            user_prompt="Write code.",
            cwd="/repo",
            project_dir="/tmp/proj",
            stage="execute",
            ui=ui,
        )

        assert isinstance(result, AgentResult)
        assert "Step 1 done." in result.text
        assert "All done." in result.text
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["tool"] == "Bash"
        assert len(result.raw_messages) == 3
        assert result.turns_used == 3
        assert result.elapsed_seconds > 0

    async def test_save_log_called(self, mock_query, mock_save_log):
        """artifacts.save_log is called with the joined text."""
        msg = _message([_text_block("output text")])
        mock_query.return_value = _async_gen_from([msg])
        ui = _make_ui()

        await run_agent(
            persona="auditor",
            system_prompt="sys",
            user_prompt="user",
            cwd="/repo",
            project_dir="/tmp/proj",
            stage="verify",
            ui=ui,
        )

        mock_save_log.assert_called_once_with("/tmp/proj", "verify", "output text")

    async def test_model_override(self, mock_query, mock_save_log):
        """When model is passed, it is included in ClaudeAgentOptions."""
        mock_query.return_value = _async_gen_from([])
        ui = _make_ui()

        with patch("graft.agent.ClaudeAgentOptions") as MockOpts:
            MockOpts.return_value = MagicMock()
            mock_query.return_value = _async_gen_from([])

            await run_agent(
                persona="researcher",
                system_prompt="sys",
                user_prompt="user",
                cwd="/repo",
                project_dir="/tmp/proj",
                stage="research",
                ui=ui,
                model="claude-sonnet-4-20250514",
            )

            # Verify ClaudeAgentOptions was called with model in kwargs
            opts_kwargs = MockOpts.call_args[1]
            assert opts_kwargs["model"] == "claude-sonnet-4-20250514"

    async def test_model_not_set_when_none(self, mock_query, mock_save_log):
        """When model is None, it is NOT passed to ClaudeAgentOptions."""
        mock_query.return_value = _async_gen_from([])
        ui = _make_ui()

        with patch("graft.agent.ClaudeAgentOptions") as MockOpts:
            MockOpts.return_value = MagicMock()
            mock_query.return_value = _async_gen_from([])

            await run_agent(
                persona="planner",
                system_prompt="sys",
                user_prompt="user",
                cwd="/repo",
                project_dir="/tmp/proj",
                stage="plan",
                ui=ui,
                model=None,
            )

            opts_kwargs = MockOpts.call_args[1]
            assert "model" not in opts_kwargs

    async def test_allowed_tools_default(self, mock_query, mock_save_log):
        """Default allowed_tools includes the standard set."""
        mock_query.return_value = _async_gen_from([])
        ui = _make_ui()

        with patch("graft.agent.ClaudeAgentOptions") as MockOpts:
            MockOpts.return_value = MagicMock()
            mock_query.return_value = _async_gen_from([])

            await run_agent(
                persona="coder",
                system_prompt="sys",
                user_prompt="user",
                cwd="/repo",
                project_dir="/tmp/proj",
                stage="execute",
                ui=ui,
            )

            opts_kwargs = MockOpts.call_args[1]
            expected_tools = ["Read", "Write", "Edit", "MultiEdit", "Bash", "Glob", "Grep"]
            assert opts_kwargs["allowed_tools"] == expected_tools

    async def test_allowed_tools_custom(self, mock_query, mock_save_log):
        """Custom allowed_tools overrides the defaults."""
        mock_query.return_value = _async_gen_from([])
        ui = _make_ui()

        with patch("graft.agent.ClaudeAgentOptions") as MockOpts:
            MockOpts.return_value = MagicMock()
            mock_query.return_value = _async_gen_from([])

            await run_agent(
                persona="researcher",
                system_prompt="sys",
                user_prompt="user",
                cwd="/repo",
                project_dir="/tmp/proj",
                stage="research",
                ui=ui,
                allowed_tools=["Read", "Grep"],
            )

            opts_kwargs = MockOpts.call_args[1]
            assert opts_kwargs["allowed_tools"] == ["Read", "Grep"]

    async def test_ui_stage_log_called_on_start_and_finish(self, mock_query, mock_save_log):
        """UI receives start and finish log messages."""
        mock_query.return_value = _async_gen_from([])
        ui = _make_ui()

        await run_agent(
            persona="planner",
            system_prompt="sys",
            user_prompt="user",
            cwd="/repo",
            project_dir="/tmp/proj",
            stage="plan",
            ui=ui,
        )

        log_messages = [c[0][1] for c in ui.stage_log.call_args_list]
        assert any("starting" in m for m in log_messages)
        assert any("finished" in m for m in log_messages)

    async def test_empty_message_stream(self, mock_query, mock_save_log):
        """Zero messages still returns a valid AgentResult."""
        mock_query.return_value = _async_gen_from([])
        ui = _make_ui()

        result = await run_agent(
            persona="coder",
            system_prompt="sys",
            user_prompt="user",
            cwd="/repo",
            project_dir="/tmp/proj",
            stage="execute",
            ui=ui,
        )

        assert result.text == ""
        assert result.tool_calls == []
        assert result.raw_messages == []
        assert result.turns_used == 0


# ---------------------------------------------------------------------------
# run_agent — retry / backoff logic
# ---------------------------------------------------------------------------

class TestRunAgentRetry:
    @pytest.fixture
    def mock_save_log(self):
        with patch("graft.agent.artifacts.save_log") as mocked:
            yield mocked

    @pytest.mark.parametrize("exc_class", [ConnectionError, TimeoutError, OSError])
    async def test_retries_transient_errors(self, exc_class, mock_save_log):
        """Transient errors are retried, and success on last attempt returns result."""
        call_count = 0

        async def _query_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # Simulate failure part-way through async iteration
                async def _failing_gen():
                    raise exc_class(f"transient {call_count}")
                    yield  # noqa: unreachable — makes this an async generator
                return _failing_gen()
            else:
                return _async_gen_from([_message([_text_block("success")])])

        with (
            patch("graft.agent.query", side_effect=_query_side_effect),
            patch("graft.agent.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            ui = _make_ui()
            result = await run_agent(
                persona="coder",
                system_prompt="sys",
                user_prompt="do it",
                cwd="/repo",
                project_dir="/tmp/proj",
                stage="execute",
                ui=ui,
            )

            assert result.text == "success"
            # Verify exponential backoff: delay = RETRY_BACKOFF_BASE ** attempt
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(RETRY_BACKOFF_BASE**1)  # 2.0
            mock_sleep.assert_any_call(RETRY_BACKOFF_BASE**2)  # 4.0

    async def test_exhausts_retries_raises_runtime_error(self, mock_save_log):
        """After MAX_RETRIES failures, raises RuntimeError wrapping the last error."""
        async def _always_fail(**kwargs):
            async def _gen():
                raise ConnectionError("permanent failure")
                yield  # noqa: unreachable
            return _gen()

        with (
            patch("graft.agent.query", side_effect=_always_fail),
            patch("graft.agent.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            ui = _make_ui()
            with pytest.raises(RuntimeError, match=r"failed after 3 attempts"):
                await run_agent(
                    persona="coder",
                    system_prompt="sys",
                    user_prompt="do it",
                    cwd="/repo",
                    project_dir="/tmp/proj",
                    stage="execute",
                    ui=ui,
                )

            # All retry sleeps should have happened
            assert mock_sleep.call_count == MAX_RETRIES

    async def test_retry_backoff_delays(self, mock_save_log):
        """Verify the exact sequence of backoff delays across all retry attempts."""
        async def _always_fail(**kwargs):
            async def _gen():
                raise TimeoutError("timeout")
                yield  # noqa: unreachable
            return _gen()

        with (
            patch("graft.agent.query", side_effect=_always_fail),
            patch("graft.agent.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            ui = _make_ui()
            with pytest.raises(RuntimeError):
                await run_agent(
                    persona="coder",
                    system_prompt="sys",
                    user_prompt="do it",
                    cwd="/repo",
                    project_dir="/tmp/proj",
                    stage="execute",
                    ui=ui,
                )

            expected_delays = [RETRY_BACKOFF_BASE**i for i in range(1, MAX_RETRIES + 1)]
            actual_delays = [c[0][0] for c in mock_sleep.call_args_list]
            assert actual_delays == expected_delays  # [2.0, 4.0, 8.0]

    async def test_non_transient_error_not_retried(self, mock_save_log):
        """Non-transient exceptions (e.g. ValueError) propagate immediately."""
        async def _value_error(**kwargs):
            async def _gen():
                raise ValueError("bad input")
                yield  # noqa: unreachable
            return _gen()

        with (
            patch("graft.agent.query", side_effect=_value_error),
            patch("graft.agent.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            ui = _make_ui()
            with pytest.raises(ValueError, match="bad input"):
                await run_agent(
                    persona="coder",
                    system_prompt="sys",
                    user_prompt="do it",
                    cwd="/repo",
                    project_dir="/tmp/proj",
                    stage="execute",
                    ui=ui,
                )

            # No retries for non-transient errors
            mock_sleep.assert_not_called()

    async def test_retry_resets_accumulated_state(self, mock_save_log):
        """On retry, text_parts / tool_calls / raw_messages are reset."""
        call_count = 0

        async def _query_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                async def _partial_then_fail():
                    yield _message([_text_block("partial")])
                    raise ConnectionError("oops")
                return _partial_then_fail()
            else:
                return _async_gen_from([_message([_text_block("final")])])

        with (
            patch("graft.agent.query", side_effect=_query_side_effect),
            patch("graft.agent.asyncio.sleep", new_callable=AsyncMock),
        ):
            ui = _make_ui()
            result = await run_agent(
                persona="coder",
                system_prompt="sys",
                user_prompt="do it",
                cwd="/repo",
                project_dir="/tmp/proj",
                stage="execute",
                ui=ui,
            )

            # "partial" should NOT appear — the retry resets state
            assert result.text == "final"
            assert result.turns_used == 1

    async def test_retry_logs_warning_to_ui(self, mock_save_log):
        """Each transient failure logs a yellow warning to the UI."""
        call_count = 0

        async def _query_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                async def _gen():
                    raise OSError("disk full")
                    yield  # noqa: unreachable
                return _gen()
            else:
                return _async_gen_from([])

        with (
            patch("graft.agent.query", side_effect=_query_side_effect),
            patch("graft.agent.asyncio.sleep", new_callable=AsyncMock),
        ):
            ui = _make_ui()
            await run_agent(
                persona="coder",
                system_prompt="sys",
                user_prompt="do it",
                cwd="/repo",
                project_dir="/tmp/proj",
                stage="execute",
                ui=ui,
            )

            log_messages = [c[0][1] for c in ui.stage_log.call_args_list]
            retry_msgs = [m for m in log_messages if "failed" in m and "attempt" in m]
            assert len(retry_msgs) == 1
            assert "1/3" in retry_msgs[0]
            assert "yellow" in retry_msgs[0]


# ---------------------------------------------------------------------------
# run_agent — ClaudeAgentOptions construction
# ---------------------------------------------------------------------------

class TestRunAgentOptions:
    async def test_options_include_permission_mode(self):
        """Options always include permission_mode=bypassPermissions."""
        with (
            patch("graft.agent.ClaudeAgentOptions") as MockOpts,
            patch("graft.agent.query") as mock_query,
            patch("graft.agent.artifacts.save_log"),
        ):
            MockOpts.return_value = MagicMock()
            mock_query.return_value = _async_gen_from([])
            ui = _make_ui()

            await run_agent(
                persona="coder",
                system_prompt="sys",
                user_prompt="user",
                cwd="/repo",
                project_dir="/tmp/proj",
                stage="execute",
                ui=ui,
            )

            opts_kwargs = MockOpts.call_args[1]
            assert opts_kwargs["permission_mode"] == "bypassPermissions"

    async def test_options_max_turns(self):
        """max_turns parameter is forwarded to ClaudeAgentOptions."""
        with (
            patch("graft.agent.ClaudeAgentOptions") as MockOpts,
            patch("graft.agent.query") as mock_query,
            patch("graft.agent.artifacts.save_log"),
        ):
            MockOpts.return_value = MagicMock()
            mock_query.return_value = _async_gen_from([])
            ui = _make_ui()

            await run_agent(
                persona="coder",
                system_prompt="sys",
                user_prompt="user",
                cwd="/repo",
                project_dir="/tmp/proj",
                stage="execute",
                ui=ui,
                max_turns=10,
            )

            opts_kwargs = MockOpts.call_args[1]
            assert opts_kwargs["max_turns"] == 10

    async def test_query_receives_user_prompt_and_options(self):
        """query() is called with the correct prompt and options object."""
        sentinel_opts = MagicMock()
        with (
            patch("graft.agent.ClaudeAgentOptions", return_value=sentinel_opts),
            patch("graft.agent.query") as mock_query,
            patch("graft.agent.artifacts.save_log"),
        ):
            mock_query.return_value = _async_gen_from([])
            ui = _make_ui()

            await run_agent(
                persona="coder",
                system_prompt="sys",
                user_prompt="Please fix the bug",
                cwd="/repo",
                project_dir="/tmp/proj",
                stage="execute",
                ui=ui,
            )

            mock_query.assert_called_once_with(
                prompt="Please fix the bug",
                options=sentinel_opts,
            )
