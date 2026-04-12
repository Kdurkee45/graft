"""Tests for graft.agent — run_agent, _process_message, AgentResult."""

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text_block(text: str) -> SimpleNamespace:
    """Create a fake SDK message block with a ``text`` attribute."""
    return SimpleNamespace(text=text)


def _make_tool_block(name: str, input_data: dict | None = None) -> SimpleNamespace:
    """Create a fake SDK message block with ``name`` and ``input`` attributes."""
    return SimpleNamespace(name=name, input=input_data or {})


def _make_message(*blocks) -> SimpleNamespace:
    """Wrap blocks in a message-like object with a ``content`` list."""
    return SimpleNamespace(content=list(blocks))


async def _async_iter(items):
    """Turn an iterable into an async generator, mimicking ``query()``."""
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project_dir(tmp_path):
    """Temporary project directory with required sub-structure."""
    d = tmp_path / "project"
    d.mkdir()
    (d / "logs").mkdir()
    return d


@pytest.fixture
def ui():
    """Mock UI exposing the methods agent.py calls."""
    m = MagicMock()
    m.stage_log = MagicMock()
    return m


def _common_kwargs(project_dir, ui, **overrides):
    """Build the keyword arguments shared across most run_agent calls."""
    defaults = dict(
        persona="tester",
        system_prompt="You are a test agent.",
        user_prompt="Do something.",
        cwd=str(project_dir),
        project_dir=str(project_dir),
        stage="test_stage",
        ui=ui,
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# AgentResult dataclass
# ---------------------------------------------------------------------------


class TestAgentResult:
    """Basic sanity checks for the AgentResult dataclass."""

    def test_defaults(self):
        r = AgentResult(text="hello")
        assert r.text == "hello"
        assert r.tool_calls == []
        assert r.raw_messages == []
        assert r.elapsed_seconds == 0.0
        assert r.turns_used == 0

    def test_custom_fields(self):
        r = AgentResult(
            text="output",
            tool_calls=[{"tool": "Read", "input": {}}],
            raw_messages=["m1"],
            elapsed_seconds=3.5,
            turns_used=2,
        )
        assert r.turns_used == 2
        assert len(r.tool_calls) == 1
        assert r.elapsed_seconds == 3.5

    def test_tool_calls_default_is_independent(self):
        """Each instance should get its own mutable list."""
        r1 = AgentResult(text="a")
        r2 = AgentResult(text="b")
        r1.tool_calls.append({"tool": "Bash", "input": {}})
        assert r2.tool_calls == []


# ---------------------------------------------------------------------------
# _process_message
# ---------------------------------------------------------------------------


class TestProcessMessage:
    """Unit tests for the internal _process_message helper."""

    def test_text_block_appended(self, ui, project_dir):
        msg = _make_message(_make_text_block("hello world"))
        text_parts: list[str] = []
        tool_calls: list[dict] = []

        _process_message(msg, text_parts, tool_calls, "stage", ui, str(project_dir))

        assert text_parts == ["hello world"]
        assert tool_calls == []
        ui.stage_log.assert_called_once()

    def test_tool_block_appended(self, ui, project_dir):
        msg = _make_message(_make_tool_block("Bash", {"command": "ls"}))
        text_parts: list[str] = []
        tool_calls: list[dict] = []

        _process_message(msg, text_parts, tool_calls, "stage", ui, str(project_dir))

        assert text_parts == []
        assert tool_calls == [{"tool": "Bash", "input": {"command": "ls"}}]

    def test_mixed_blocks(self, ui, project_dir):
        msg = _make_message(
            _make_text_block("step 1"),
            _make_tool_block("Write", {"path": "/tmp/f"}),
            _make_text_block("step 2"),
        )
        text_parts: list[str] = []
        tool_calls: list[dict] = []

        _process_message(msg, text_parts, tool_calls, "stage", ui, str(project_dir))

        assert text_parts == ["step 1", "step 2"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "Write"

    def test_message_without_content_attr(self, ui, project_dir):
        """Messages lacking a ``content`` attribute are silently skipped."""
        msg = SimpleNamespace(role="system")  # no .content
        text_parts: list[str] = []
        tool_calls: list[dict] = []

        _process_message(msg, text_parts, tool_calls, "stage", ui, str(project_dir))

        assert text_parts == []
        assert tool_calls == []

    def test_empty_text_block_skipped(self, ui, project_dir):
        """A block with text="" should not be appended."""
        msg = _make_message(SimpleNamespace(text=""))
        text_parts: list[str] = []
        tool_calls: list[dict] = []

        _process_message(msg, text_parts, tool_calls, "stage", ui, str(project_dir))

        assert text_parts == []

    def test_tool_block_without_input(self, ui, project_dir):
        """Tool blocks missing the ``input`` attr default to empty dict."""
        block = SimpleNamespace(name="Glob")  # no .input attribute
        msg = _make_message(block)
        tool_calls: list[dict] = []

        _process_message(msg, [], tool_calls, "stage", ui, str(project_dir))

        assert tool_calls == [{"tool": "Glob", "input": {}}]

    def test_long_text_preview_truncated(self, ui, project_dir):
        """UI preview is capped at 200 chars with newlines collapsed."""
        long_text = "A" * 300
        msg = _make_message(_make_text_block(long_text))

        _process_message(msg, [], [], "stage", ui, str(project_dir))

        # The preview passed to stage_log should be truncated
        logged_preview = ui.stage_log.call_args[0][1]
        assert len(logged_preview) <= 200


# ---------------------------------------------------------------------------
# run_agent — happy path
# ---------------------------------------------------------------------------


class TestRunAgentSuccess:
    """Tests where the agent runs to completion without errors."""

    async def test_returns_agent_result_with_text(self, project_dir, ui):
        msg = _make_message(_make_text_block("result text"))

        with (
            patch("graft.agent.query", return_value=_async_iter([msg])),
            patch("graft.agent.ClaudeAgentOptions"),
            patch("graft.agent.artifacts"),
        ):
            result = await run_agent(**_common_kwargs(project_dir, ui))

        assert isinstance(result, AgentResult)
        assert result.text == "result text"

    async def test_returns_tool_calls(self, project_dir, ui):
        msg = _make_message(
            _make_text_block("doing work"),
            _make_tool_block("Edit", {"file": "a.py"}),
        )

        with (
            patch("graft.agent.query", return_value=_async_iter([msg])),
            patch("graft.agent.ClaudeAgentOptions"),
            patch("graft.agent.artifacts"),
        ):
            result = await run_agent(**_common_kwargs(project_dir, ui))

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["tool"] == "Edit"
        assert result.tool_calls[0]["input"] == {"file": "a.py"}

    async def test_elapsed_seconds_positive(self, project_dir, ui):
        msg = _make_message(_make_text_block("done"))

        with (
            patch("graft.agent.query", return_value=_async_iter([msg])),
            patch("graft.agent.ClaudeAgentOptions"),
            patch("graft.agent.artifacts"),
        ):
            result = await run_agent(**_common_kwargs(project_dir, ui))

        assert result.elapsed_seconds > 0

    async def test_turns_used_equals_message_count(self, project_dir, ui):
        msgs = [
            _make_message(_make_text_block("msg 1")),
            _make_message(_make_text_block("msg 2")),
            _make_message(_make_text_block("msg 3")),
        ]

        with (
            patch("graft.agent.query", return_value=_async_iter(msgs)),
            patch("graft.agent.ClaudeAgentOptions"),
            patch("graft.agent.artifacts"),
        ):
            result = await run_agent(**_common_kwargs(project_dir, ui))

        assert result.turns_used == 3
        assert len(result.raw_messages) == 3

    async def test_text_joined_with_newlines(self, project_dir, ui):
        msgs = [
            _make_message(_make_text_block("line 1")),
            _make_message(_make_text_block("line 2")),
        ]

        with (
            patch("graft.agent.query", return_value=_async_iter(msgs)),
            patch("graft.agent.ClaudeAgentOptions"),
            patch("graft.agent.artifacts"),
        ):
            result = await run_agent(**_common_kwargs(project_dir, ui))

        assert result.text == "line 1\nline 2"

    async def test_save_log_called(self, project_dir, ui):
        msg = _make_message(_make_text_block("logged output"))

        with (
            patch("graft.agent.query", return_value=_async_iter([msg])),
            patch("graft.agent.ClaudeAgentOptions"),
            patch("graft.agent.artifacts") as mock_artifacts,
        ):
            await run_agent(**_common_kwargs(project_dir, ui))

        mock_artifacts.save_log.assert_called_once_with(
            str(project_dir), "test_stage", "logged output"
        )

    async def test_empty_stream_returns_empty_text(self, project_dir, ui):
        """An agent that produces no messages returns an empty result."""
        with (
            patch("graft.agent.query", return_value=_async_iter([])),
            patch("graft.agent.ClaudeAgentOptions"),
            patch("graft.agent.artifacts"),
        ):
            result = await run_agent(**_common_kwargs(project_dir, ui))

        assert result.text == ""
        assert result.tool_calls == []
        assert result.turns_used == 0


# ---------------------------------------------------------------------------
# run_agent — ClaudeAgentOptions construction
# ---------------------------------------------------------------------------


class TestAgentOptionsConstruction:
    """Verify ClaudeAgentOptions is built with the right arguments."""

    async def test_default_allowed_tools(self, project_dir, ui):
        with (
            patch("graft.agent.query", return_value=_async_iter([])),
            patch("graft.agent.ClaudeAgentOptions") as mock_cls,
            patch("graft.agent.artifacts"),
        ):
            await run_agent(**_common_kwargs(project_dir, ui))

        _kwargs = mock_cls.call_args[1]
        assert _kwargs["allowed_tools"] == [
            "Read",
            "Write",
            "Edit",
            "MultiEdit",
            "Bash",
            "Glob",
            "Grep",
        ]

    async def test_custom_allowed_tools(self, project_dir, ui):
        with (
            patch("graft.agent.query", return_value=_async_iter([])),
            patch("graft.agent.ClaudeAgentOptions") as mock_cls,
            patch("graft.agent.artifacts"),
        ):
            await run_agent(
                **_common_kwargs(project_dir, ui, allowed_tools=["Read", "Bash"])
            )

        _kwargs = mock_cls.call_args[1]
        assert _kwargs["allowed_tools"] == ["Read", "Bash"]

    async def test_permission_mode_set(self, project_dir, ui):
        with (
            patch("graft.agent.query", return_value=_async_iter([])),
            patch("graft.agent.ClaudeAgentOptions") as mock_cls,
            patch("graft.agent.artifacts"),
        ):
            await run_agent(**_common_kwargs(project_dir, ui))

        _kwargs = mock_cls.call_args[1]
        assert _kwargs["permission_mode"] == "bypassPermissions"

    async def test_model_included_when_set(self, project_dir, ui):
        with (
            patch("graft.agent.query", return_value=_async_iter([])),
            patch("graft.agent.ClaudeAgentOptions") as mock_cls,
            patch("graft.agent.artifacts"),
        ):
            await run_agent(
                **_common_kwargs(project_dir, ui, model="claude-sonnet-4-20250514")
            )

        _kwargs = mock_cls.call_args[1]
        assert _kwargs["model"] == "claude-sonnet-4-20250514"

    async def test_model_omitted_when_none(self, project_dir, ui):
        with (
            patch("graft.agent.query", return_value=_async_iter([])),
            patch("graft.agent.ClaudeAgentOptions") as mock_cls,
            patch("graft.agent.artifacts"),
        ):
            await run_agent(**_common_kwargs(project_dir, ui, model=None))

        _kwargs = mock_cls.call_args[1]
        assert "model" not in _kwargs

    async def test_max_turns_forwarded(self, project_dir, ui):
        with (
            patch("graft.agent.query", return_value=_async_iter([])),
            patch("graft.agent.ClaudeAgentOptions") as mock_cls,
            patch("graft.agent.artifacts"),
        ):
            await run_agent(**_common_kwargs(project_dir, ui, max_turns=25))

        _kwargs = mock_cls.call_args[1]
        assert _kwargs["max_turns"] == 25

    async def test_system_prompt_forwarded(self, project_dir, ui):
        with (
            patch("graft.agent.query", return_value=_async_iter([])),
            patch("graft.agent.ClaudeAgentOptions") as mock_cls,
            patch("graft.agent.artifacts"),
        ):
            await run_agent(
                **_common_kwargs(project_dir, ui, system_prompt="custom prompt")
            )

        _kwargs = mock_cls.call_args[1]
        assert _kwargs["system_prompt"] == "custom prompt"

    async def test_cwd_forwarded(self, project_dir, ui):
        with (
            patch("graft.agent.query", return_value=_async_iter([])),
            patch("graft.agent.ClaudeAgentOptions") as mock_cls,
            patch("graft.agent.artifacts"),
        ):
            await run_agent(**_common_kwargs(project_dir, ui))

        _kwargs = mock_cls.call_args[1]
        assert _kwargs["cwd"] == str(project_dir)


# ---------------------------------------------------------------------------
# run_agent — retry logic
# ---------------------------------------------------------------------------


class TestRunAgentRetry:
    """Retry with exponential backoff on transient exceptions."""

    async def test_retries_connection_error(self, project_dir, ui):
        """ConnectionError triggers retry then succeeds on second attempt."""
        call_count = 0

        def _query_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("network down")
            return _async_iter([_make_message(_make_text_block("ok"))])

        with (
            patch("graft.agent.query", side_effect=_query_side_effect),
            patch("graft.agent.ClaudeAgentOptions"),
            patch("graft.agent.artifacts"),
            patch("graft.agent.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            result = await run_agent(**_common_kwargs(project_dir, ui))

        assert result.text == "ok"
        # First retry delay: RETRY_BACKOFF_BASE ** 1 = 2.0
        mock_sleep.assert_called_once_with(RETRY_BACKOFF_BASE**1)

    async def test_retries_timeout_error(self, project_dir, ui):
        call_count = 0

        def _query_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("timed out")
            return _async_iter([_make_message(_make_text_block("recovered"))])

        with (
            patch("graft.agent.query", side_effect=_query_side_effect),
            patch("graft.agent.ClaudeAgentOptions"),
            patch("graft.agent.artifacts"),
            patch("graft.agent.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await run_agent(**_common_kwargs(project_dir, ui))

        assert result.text == "recovered"

    async def test_retries_os_error(self, project_dir, ui):
        call_count = 0

        def _query_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("disk error")
            return _async_iter([_make_message(_make_text_block("ok"))])

        with (
            patch("graft.agent.query", side_effect=_query_side_effect),
            patch("graft.agent.ClaudeAgentOptions"),
            patch("graft.agent.artifacts"),
            patch("graft.agent.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await run_agent(**_common_kwargs(project_dir, ui))

        assert result.text == "ok"

    async def test_exponential_backoff_delays(self, project_dir, ui):
        """All MAX_RETRIES attempts fail — verify exponential sleep delays."""

        def _query_fail(**kwargs):
            raise ConnectionError("persistent failure")

        with (
            patch("graft.agent.query", side_effect=_query_fail),
            patch("graft.agent.ClaudeAgentOptions"),
            patch("graft.agent.artifacts"),
            patch("graft.agent.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            with pytest.raises(RuntimeError, match="failed after 3 attempts"):
                await run_agent(**_common_kwargs(project_dir, ui))

        # Expect sleep calls with delays: 2^1=2.0, 2^2=4.0, 2^3=8.0
        expected_delays = [RETRY_BACKOFF_BASE**i for i in range(1, MAX_RETRIES + 1)]
        actual_delays = [c.args[0] for c in mock_sleep.call_args_list]
        assert actual_delays == expected_delays

    async def test_max_retries_raises_runtime_error(self, project_dir, ui):
        """After MAX_RETRIES failed attempts, RuntimeError is raised."""

        def _query_fail(**kwargs):
            raise ConnectionError("always fails")

        with (
            patch("graft.agent.query", side_effect=_query_fail),
            patch("graft.agent.ClaudeAgentOptions"),
            patch("graft.agent.artifacts"),
            patch("graft.agent.asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                await run_agent(**_common_kwargs(project_dir, ui))

        assert "failed after 3 attempts" in str(exc_info.value)
        assert "always fails" in str(exc_info.value)

    async def test_non_transient_error_not_retried(self, project_dir, ui):
        """ValueError is not a transient error — it should propagate immediately."""

        def _query_fail(**kwargs):
            raise ValueError("bad input")

        with (
            patch("graft.agent.query", side_effect=_query_fail),
            patch("graft.agent.ClaudeAgentOptions"),
            patch("graft.agent.artifacts"),
            patch("graft.agent.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            with pytest.raises(ValueError, match="bad input"):
                await run_agent(**_common_kwargs(project_dir, ui))

        mock_sleep.assert_not_called()

    async def test_retry_resets_accumulators(self, project_dir, ui):
        """On retry, text_parts/tool_calls/raw_messages start fresh."""
        call_count = 0

        def _query_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Yield one message then raise — partial data should be discarded
                async def _partial():
                    yield _make_message(_make_text_block("partial"))
                    raise ConnectionError("mid-stream failure")

                return _partial()
            return _async_iter([_make_message(_make_text_block("clean result"))])

        with (
            patch("graft.agent.query", side_effect=_query_side_effect),
            patch("graft.agent.ClaudeAgentOptions"),
            patch("graft.agent.artifacts"),
            patch("graft.agent.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await run_agent(**_common_kwargs(project_dir, ui))

        # Only the second attempt's data should be in the result
        assert result.text == "clean result"
        assert result.turns_used == 1

    async def test_success_on_third_attempt(self, project_dir, ui):
        """Agent fails twice then succeeds on the third attempt."""
        call_count = 0

        def _query_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise TimeoutError(f"attempt {call_count}")
            return _async_iter([_make_message(_make_text_block("finally"))])

        with (
            patch("graft.agent.query", side_effect=_query_side_effect),
            patch("graft.agent.ClaudeAgentOptions"),
            patch("graft.agent.artifacts"),
            patch("graft.agent.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            result = await run_agent(**_common_kwargs(project_dir, ui))

        assert result.text == "finally"
        assert mock_sleep.call_count == 2


# ---------------------------------------------------------------------------
# run_agent — log file creation
# ---------------------------------------------------------------------------


class TestRunAgentLogFile:
    """Verify log artifacts are written through the artifacts module."""

    async def test_log_saved_with_correct_stage(self, project_dir, ui):
        msg = _make_message(_make_text_block("log this"))

        with (
            patch("graft.agent.query", return_value=_async_iter([msg])),
            patch("graft.agent.ClaudeAgentOptions"),
            patch("graft.agent.artifacts") as mock_art,
        ):
            await run_agent(**_common_kwargs(project_dir, ui, stage="discover"))

        mock_art.save_log.assert_called_once()
        args = mock_art.save_log.call_args[0]
        assert args[0] == str(project_dir)
        assert args[1] == "discover"
        assert args[2] == "log this"

    async def test_log_contains_all_text_parts(self, project_dir, ui):
        msgs = [
            _make_message(_make_text_block("part A")),
            _make_message(_make_text_block("part B")),
        ]

        with (
            patch("graft.agent.query", return_value=_async_iter(msgs)),
            patch("graft.agent.ClaudeAgentOptions"),
            patch("graft.agent.artifacts") as mock_art,
        ):
            await run_agent(**_common_kwargs(project_dir, ui))

        saved_content = mock_art.save_log.call_args[0][2]
        assert "part A" in saved_content
        assert "part B" in saved_content

    async def test_real_log_file_creation(self, project_dir, ui):
        """Integration-style: don't mock artifacts, verify actual file on disk."""
        msg = _make_message(_make_text_block("written to disk"))

        with (
            patch("graft.agent.query", return_value=_async_iter([msg])),
            patch("graft.agent.ClaudeAgentOptions"),
        ):
            await run_agent(**_common_kwargs(project_dir, ui, stage="execute"))

        log_path = project_dir / "logs" / "execute.log"
        assert log_path.exists()
        content = log_path.read_text()
        assert "written to disk" in content


# ---------------------------------------------------------------------------
# run_agent — UI interaction
# ---------------------------------------------------------------------------


class TestRunAgentUI:
    """Verify UI feedback messages during agent execution."""

    async def test_starting_message_logged(self, project_dir, ui):
        with (
            patch("graft.agent.query", return_value=_async_iter([])),
            patch("graft.agent.ClaudeAgentOptions"),
            patch("graft.agent.artifacts"),
        ):
            await run_agent(**_common_kwargs(project_dir, ui, persona="researcher"))

        # First call should be the "starting" message
        first_call = ui.stage_log.call_args_list[0]
        assert "researcher" in first_call.args[1]
        assert "starting" in first_call.args[1]

    async def test_finished_message_logged(self, project_dir, ui):
        msg = _make_message(_make_text_block("done"))

        with (
            patch("graft.agent.query", return_value=_async_iter([msg])),
            patch("graft.agent.ClaudeAgentOptions"),
            patch("graft.agent.artifacts"),
        ):
            await run_agent(**_common_kwargs(project_dir, ui, persona="planner"))

        # Last call should be the "finished" message
        last_call = ui.stage_log.call_args_list[-1]
        assert "planner" in last_call.args[1]
        assert "finished" in last_call.args[1]

    async def test_retry_warning_logged(self, project_dir, ui):
        call_count = 0

        def _query_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("oops")
            return _async_iter([_make_message(_make_text_block("ok"))])

        with (
            patch("graft.agent.query", side_effect=_query_side_effect),
            patch("graft.agent.ClaudeAgentOptions"),
            patch("graft.agent.artifacts"),
            patch("graft.agent.asyncio.sleep", new_callable=AsyncMock),
        ):
            await run_agent(**_common_kwargs(project_dir, ui))

        # One of the stage_log calls should mention the failure / retry
        all_logs = [c.args[1] for c in ui.stage_log.call_args_list]
        assert any(
            "failed" in log.lower() or "attempt" in log.lower() for log in all_logs
        )


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    """Verify the retry configuration constants are correct."""

    def test_max_retries(self):
        assert MAX_RETRIES == 3

    def test_retry_backoff_base(self):
        assert RETRY_BACKOFF_BASE == 2.0
