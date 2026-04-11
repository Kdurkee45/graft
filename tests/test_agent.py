"""Tests for graft.agent — Claude Agent SDK wrapper."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

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
    """Return a mock UI with verbose enabled so stage_log actually records."""
    ui = MagicMock()
    ui.verbose = True
    return ui


async def _async_gen(items):
    """Turn a plain list into an async generator (mimics ``query()``)."""
    for item in items:
        yield item


async def _async_gen_raise(items, exc, *, after: int = 0):
    """Yield *after* items then raise *exc*."""
    for i, item in enumerate(items):
        if i >= after:
            raise exc
        yield item


def _text_block(text: str):
    """Create a message with a single text content block."""
    block = SimpleNamespace(text=text)
    return SimpleNamespace(content=[block])


def _tool_block(name: str, tool_input: dict | None = None):
    """Create a message with a single tool-use content block."""
    block = SimpleNamespace(name=name, input=tool_input or {})
    return SimpleNamespace(content=[block])


def _run(coro):
    """Synchronously run an async coroutine."""
    return asyncio.run(coro)


def _default_kwargs(**overrides):
    """Build the mandatory keyword arguments for ``run_agent``."""
    defaults = dict(
        persona="tester",
        system_prompt="You are a test agent.",
        user_prompt="Do the thing.",
        cwd="/tmp",
        project_dir="/tmp/proj",
        stage="execute",
        ui=_make_ui(),
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# AgentResult dataclass
# ---------------------------------------------------------------------------

def test_agent_result_defaults():
    """AgentResult fields have sensible zero-value defaults."""
    result = AgentResult(text="hello")
    assert result.text == "hello"
    assert result.tool_calls == []
    assert result.raw_messages == []
    assert result.elapsed_seconds == 0.0
    assert result.turns_used == 0


def test_agent_result_custom_values():
    """AgentResult stores all custom values passed at construction."""
    calls = [{"tool": "Bash", "input": {"command": "ls"}}]
    msgs = [{"role": "assistant"}]
    result = AgentResult(
        text="output",
        tool_calls=calls,
        raw_messages=msgs,
        elapsed_seconds=1.5,
        turns_used=3,
    )
    assert result.text == "output"
    assert result.tool_calls == calls
    assert result.raw_messages == msgs
    assert result.elapsed_seconds == 1.5
    assert result.turns_used == 3


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_constants():
    """Verify the retry constants are set to expected values."""
    assert MAX_RETRIES == 3
    assert RETRY_BACKOFF_BASE == 2.0


# ---------------------------------------------------------------------------
# _process_message
# ---------------------------------------------------------------------------

def test_process_message_extracts_text():
    """Text blocks are appended to text_parts and previewed via ui.stage_log."""
    ui = _make_ui()
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    message = _text_block("Hello world")

    _process_message(message, text_parts, tool_calls, "execute", ui, "/tmp/proj")

    assert text_parts == ["Hello world"]
    assert tool_calls == []
    ui.stage_log.assert_called()


def test_process_message_extracts_tool_calls():
    """Tool-use blocks are appended to tool_calls with name and input."""
    ui = _make_ui()
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    message = _tool_block("Bash", {"command": "ls -la"})

    _process_message(message, text_parts, tool_calls, "execute", ui, "/tmp/proj")

    assert text_parts == []
    assert len(tool_calls) == 1
    assert tool_calls[0] == {"tool": "Bash", "input": {"command": "ls -la"}}


def test_process_message_tool_call_missing_input():
    """Tool blocks without an 'input' attribute default to empty dict."""
    ui = _make_ui()
    tool_calls: list[dict] = []
    # block has name but no input attribute
    block = SimpleNamespace(name="Grep")
    message = SimpleNamespace(content=[block])

    _process_message(message, [], tool_calls, "execute", ui, "/tmp/proj")

    assert tool_calls[0] == {"tool": "Grep", "input": {}}


def test_process_message_no_content_attribute():
    """Messages without a 'content' attribute are silently skipped."""
    ui = _make_ui()
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    message = SimpleNamespace(role="system")  # no .content

    _process_message(message, text_parts, tool_calls, "execute", ui, "/tmp/proj")

    assert text_parts == []
    assert tool_calls == []


def test_process_message_empty_text_skipped():
    """A text block with empty string is not appended."""
    ui = _make_ui()
    text_parts: list[str] = []
    block = SimpleNamespace(text="")
    message = SimpleNamespace(content=[block])

    _process_message(message, text_parts, [], "execute", ui, "/tmp/proj")

    assert text_parts == []


def test_process_message_mixed_content():
    """A message with both text and tool blocks extracts both."""
    ui = _make_ui()
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    text_block = SimpleNamespace(text="Analysis complete")
    tool_use_block = SimpleNamespace(name="Write", input={"path": "/a.py", "content": "x"})
    message = SimpleNamespace(content=[text_block, tool_use_block])

    _process_message(message, text_parts, tool_calls, "execute", ui, "/tmp/proj")

    assert text_parts == ["Analysis complete"]
    assert tool_calls == [{"tool": "Write", "input": {"path": "/a.py", "content": "x"}}]


def test_process_message_long_text_preview_truncated():
    """Text previews sent to UI are capped at 200 chars."""
    ui = _make_ui()
    long_text = "A" * 500
    message = _text_block(long_text)

    _process_message(message, [], [], "execute", ui, "/tmp/proj")

    # The preview passed to stage_log should be at most 200 chars
    logged_preview = ui.stage_log.call_args_list[-1][0][1]
    assert len(logged_preview) <= 200


# ---------------------------------------------------------------------------
# run_agent — success path
# ---------------------------------------------------------------------------

@patch("graft.agent.artifacts")
@patch("graft.agent.ClaudeAgentOptions")
@patch("graft.agent.query")
def test_run_agent_success(mock_query, mock_opts_cls, mock_artifacts):
    """Successful run returns AgentResult with text joined from messages."""
    msg1 = _text_block("Part one.")
    msg2 = _text_block("Part two.")
    mock_query.return_value = _async_gen([msg1, msg2])

    kwargs = _default_kwargs()
    result = _run(run_agent(**kwargs))

    assert isinstance(result, AgentResult)
    assert result.text == "Part one.\nPart two."
    assert result.turns_used == 2
    assert result.elapsed_seconds > 0


@patch("graft.agent.artifacts")
@patch("graft.agent.ClaudeAgentOptions")
@patch("graft.agent.query")
def test_run_agent_calls_save_log(mock_query, mock_opts_cls, mock_artifacts):
    """artifacts.save_log is called with the project_dir, stage, and full text."""
    msg = _text_block("logged output")
    mock_query.return_value = _async_gen([msg])

    kwargs = _default_kwargs(project_dir="/my/proj", stage="discover")
    _run(run_agent(**kwargs))

    mock_artifacts.save_log.assert_called_once_with(
        "/my/proj", "discover", "logged output"
    )


@patch("graft.agent.artifacts")
@patch("graft.agent.ClaudeAgentOptions")
@patch("graft.agent.query")
def test_run_agent_default_tools(mock_query, mock_opts_cls, mock_artifacts):
    """When allowed_tools is None the default tool list is used."""
    mock_query.return_value = _async_gen([])

    kwargs = _default_kwargs(allowed_tools=None)
    _run(run_agent(**kwargs))

    # Inspect the kwargs passed to ClaudeAgentOptions
    opts_kwargs = mock_opts_cls.call_args[1]
    assert opts_kwargs["allowed_tools"] == [
        "Read", "Write", "Edit", "MultiEdit",
        "Bash", "Glob", "Grep",
    ]


@patch("graft.agent.artifacts")
@patch("graft.agent.ClaudeAgentOptions")
@patch("graft.agent.query")
def test_run_agent_custom_tools(mock_query, mock_opts_cls, mock_artifacts):
    """Custom allowed_tools list is forwarded to options."""
    mock_query.return_value = _async_gen([])

    kwargs = _default_kwargs(allowed_tools=["Read", "Bash"])
    _run(run_agent(**kwargs))

    opts_kwargs = mock_opts_cls.call_args[1]
    assert opts_kwargs["allowed_tools"] == ["Read", "Bash"]


@patch("graft.agent.artifacts")
@patch("graft.agent.ClaudeAgentOptions")
@patch("graft.agent.query")
def test_run_agent_passes_model(mock_query, mock_opts_cls, mock_artifacts):
    """When model is provided it appears in ClaudeAgentOptions kwargs."""
    mock_query.return_value = _async_gen([])

    kwargs = _default_kwargs(model="claude-sonnet-4-20250514")
    _run(run_agent(**kwargs))

    opts_kwargs = mock_opts_cls.call_args[1]
    assert opts_kwargs["model"] == "claude-sonnet-4-20250514"


@patch("graft.agent.artifacts")
@patch("graft.agent.ClaudeAgentOptions")
@patch("graft.agent.query")
def test_run_agent_no_model_by_default(mock_query, mock_opts_cls, mock_artifacts):
    """When model is None it is not included in the options dict."""
    mock_query.return_value = _async_gen([])

    kwargs = _default_kwargs(model=None)
    _run(run_agent(**kwargs))

    opts_kwargs = mock_opts_cls.call_args[1]
    assert "model" not in opts_kwargs


# ---------------------------------------------------------------------------
# run_agent — retry / failure paths
# ---------------------------------------------------------------------------

@patch("graft.agent.asyncio.sleep", return_value=asyncio.coroutine(lambda: None)())
@patch("graft.agent.artifacts")
@patch("graft.agent.ClaudeAgentOptions")
@patch("graft.agent.query")
def test_run_agent_retries_on_connection_error(
    mock_query, mock_opts_cls, mock_artifacts, mock_sleep
):
    """ConnectionError triggers a retry; success on second attempt returns result."""
    call_count = 0

    async def _side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("server reset")
        async for item in _async_gen([_text_block("ok")]):
            yield item

    mock_query.side_effect = _side_effect

    kwargs = _default_kwargs()
    result = _run(run_agent(**kwargs))

    assert result.text == "ok"
    assert call_count == 2


@patch("graft.agent.asyncio.sleep", return_value=asyncio.coroutine(lambda: None)())
@patch("graft.agent.artifacts")
@patch("graft.agent.ClaudeAgentOptions")
@patch("graft.agent.query")
def test_run_agent_retries_on_timeout_error(
    mock_query, mock_opts_cls, mock_artifacts, mock_sleep
):
    """TimeoutError also triggers retry logic."""
    call_count = 0

    async def _side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise TimeoutError("timed out")
        async for item in _async_gen([_text_block("recovered")]):
            yield item

    mock_query.side_effect = _side_effect

    kwargs = _default_kwargs()
    result = _run(run_agent(**kwargs))

    assert result.text == "recovered"


@patch("graft.agent.asyncio.sleep", return_value=asyncio.coroutine(lambda: None)())
@patch("graft.agent.artifacts")
@patch("graft.agent.ClaudeAgentOptions")
@patch("graft.agent.query")
def test_run_agent_retries_on_os_error(
    mock_query, mock_opts_cls, mock_artifacts, mock_sleep
):
    """OSError also triggers retry logic."""
    call_count = 0

    async def _side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OSError("network unreachable")
        async for item in _async_gen([_text_block("back")]):
            yield item

    mock_query.side_effect = _side_effect

    kwargs = _default_kwargs()
    result = _run(run_agent(**kwargs))

    assert result.text == "back"


@patch("graft.agent.asyncio.sleep", return_value=asyncio.coroutine(lambda: None)())
@patch("graft.agent.artifacts")
@patch("graft.agent.ClaudeAgentOptions")
@patch("graft.agent.query")
def test_run_agent_raises_after_max_retries(
    mock_query, mock_opts_cls, mock_artifacts, mock_sleep
):
    """After MAX_RETRIES consecutive failures a RuntimeError is raised."""

    async def _always_fail(**kwargs):
        raise ConnectionError("persistent failure")
        yield  # noqa: unreachable — makes this an async generator

    mock_query.side_effect = _always_fail

    kwargs = _default_kwargs()
    try:
        _run(run_agent(**kwargs))
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "failed after 3 attempts" in str(exc)
        assert "persistent failure" in str(exc)


@patch("graft.agent.asyncio.sleep")
@patch("graft.agent.artifacts")
@patch("graft.agent.ClaudeAgentOptions")
@patch("graft.agent.query")
def test_run_agent_exponential_backoff(
    mock_query, mock_opts_cls, mock_artifacts, mock_sleep
):
    """Backoff delays follow RETRY_BACKOFF_BASE ** attempt."""
    # Make sleep a coroutine that records calls
    sleep_calls = []

    async def _fake_sleep(seconds):
        sleep_calls.append(seconds)

    mock_sleep.side_effect = _fake_sleep

    async def _always_fail(**kwargs):
        raise ConnectionError("fail")
        yield  # noqa: unreachable

    mock_query.side_effect = _always_fail

    kwargs = _default_kwargs()
    try:
        _run(run_agent(**kwargs))
    except RuntimeError:
        pass

    # attempt 1 → 2^1=2, attempt 2 → 2^2=4, attempt 3 → 2^3=8
    assert sleep_calls == [2.0, 4.0, 8.0]


@patch("graft.agent.asyncio.sleep")
@patch("graft.agent.artifacts")
@patch("graft.agent.ClaudeAgentOptions")
@patch("graft.agent.query")
def test_run_agent_mid_stream_error_retries(
    mock_query, mock_opts_cls, mock_artifacts, mock_sleep
):
    """Error raised mid-iteration (after yielding some messages) triggers retry."""
    call_count = 0

    async def _fake_sleep(seconds):
        pass

    mock_sleep.side_effect = _fake_sleep

    async def _side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield _text_block("partial")
            raise ConnectionError("mid-stream failure")
        async for item in _async_gen([_text_block("complete")]):
            yield item

    mock_query.side_effect = _side_effect

    kwargs = _default_kwargs()
    result = _run(run_agent(**kwargs))

    # The second attempt should produce a clean result (first attempt's data discarded)
    assert result.text == "complete"
    assert call_count == 2


# ---------------------------------------------------------------------------
# run_agent — collects tool calls
# ---------------------------------------------------------------------------

@patch("graft.agent.artifacts")
@patch("graft.agent.ClaudeAgentOptions")
@patch("graft.agent.query")
def test_run_agent_collects_tool_calls(mock_query, mock_opts_cls, mock_artifacts):
    """Tool-use messages are collected in result.tool_calls."""
    msg = _tool_block("Bash", {"command": "echo hi"})
    mock_query.return_value = _async_gen([msg])

    kwargs = _default_kwargs()
    result = _run(run_agent(**kwargs))

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["tool"] == "Bash"
    assert result.tool_calls[0]["input"] == {"command": "echo hi"}


@patch("graft.agent.artifacts")
@patch("graft.agent.ClaudeAgentOptions")
@patch("graft.agent.query")
def test_run_agent_raw_messages_captured(mock_query, mock_opts_cls, mock_artifacts):
    """All raw messages from the SDK stream are preserved."""
    msg1 = _text_block("a")
    msg2 = _tool_block("Read", {"path": "/x"})
    msg3 = SimpleNamespace(role="system")  # no content
    mock_query.return_value = _async_gen([msg1, msg2, msg3])

    kwargs = _default_kwargs()
    result = _run(run_agent(**kwargs))

    assert len(result.raw_messages) == 3
    assert result.raw_messages[0] is msg1
    assert result.raw_messages[2] is msg3
