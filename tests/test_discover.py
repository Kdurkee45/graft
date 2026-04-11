"""Tests for graft.stages.discover."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from graft.agent import AgentResult
from graft.stages.discover import SYSTEM_PROMPT, discover_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    *,
    repo_path: str = "/repo",
    project_dir: str = "/projects/feat_abc",
    scope_path: str = "",
    feature_prompt: str = "",
    model: str | None = None,
) -> dict:
    """Build a minimal FeatureState dict for testing."""
    state: dict = {
        "repo_path": repo_path,
        "project_dir": project_dir,
    }
    if scope_path:
        state["scope_path"] = scope_path
    if feature_prompt:
        state["feature_prompt"] = feature_prompt
    if model is not None:
        state["model"] = model
    return state


def _make_ui() -> MagicMock:
    """Return a mock UI with the methods discover_node calls."""
    ui = MagicMock()
    ui.stage_start = MagicMock()
    ui.stage_done = MagicMock()
    ui.error = MagicMock()
    ui.coverage_warning = MagicMock()
    return ui


SAMPLE_PROFILE = {
    "timestamp": "2025-01-01T00:00:00Z",
    "project": {"name": "acme", "languages": ["python"]},
    "coverage_warnings": [
        {
            "module": "src/billing.py",
            "coverage_pct": 10,
            "recommendation": "add unit tests",
        }
    ],
}

SAMPLE_REPORT = "# Discovery Report\n\nSample report content."


# ---------------------------------------------------------------------------
# Tests: basic happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_discover_happy_path_reads_files_from_cwd():
    """When agent writes both output files to discover_cwd, they are read correctly."""
    ui = _make_ui()
    state = _make_state()

    agent_result = AgentResult(text="fallback text", elapsed_seconds=42.0, turns_used=5)

    # Track which paths are probed for existence and what they return
    # The files exist in discover_cwd (== repo_path since no scope_path)
    profile_json = json.dumps(SAMPLE_PROFILE)

    existing_files: dict[str, str] = {
        "/repo/discovery_report.md": SAMPLE_REPORT,
        "/repo/codebase_profile.json": profile_json,
    }

    original_path_init = Path.__new__

    def fake_exists(self: Path) -> bool:
        return str(self) in existing_files

    def fake_read_text(self: Path, *a, **kw) -> str:
        return existing_files[str(self)]

    def fake_unlink(self: Path, *a, **kw) -> None:
        existing_files.pop(str(self), None)

    with (
        patch("graft.stages.discover.run_agent", new_callable=AsyncMock, return_value=agent_result) as mock_run,
        patch("graft.stages.discover.save_artifact") as mock_save,
        patch("graft.stages.discover.mark_stage_complete") as mock_mark,
        patch.object(Path, "exists", fake_exists),
        patch.object(Path, "read_text", fake_read_text),
        patch.object(Path, "unlink", fake_unlink),
    ):
        result = await discover_node(state, ui)

    # -- Verify run_agent was called with correct kwargs
    mock_run.assert_awaited_once()
    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["persona"] == "Principal Codebase Archaeologist"
    assert call_kwargs["cwd"] == "/repo"
    assert call_kwargs["stage"] == "discover"
    assert call_kwargs["max_turns"] == 40

    # -- Verify artifacts saved
    assert mock_save.call_count == 2
    # First call: discovery_report.md
    mock_save.assert_any_call("/projects/feat_abc", "discovery_report.md", SAMPLE_REPORT)
    # Second call: codebase_profile.json
    saved_profile_call = [
        c for c in mock_save.call_args_list
        if c.args[1] == "codebase_profile.json"
    ][0]
    assert json.loads(saved_profile_call.args[2]) == SAMPLE_PROFILE

    # -- Verify mark_stage_complete
    mock_mark.assert_called_once_with("/projects/feat_abc", "discover")

    # -- Verify UI lifecycle
    ui.stage_start.assert_called_once_with("discover")
    ui.stage_done.assert_called_once_with("discover")

    # -- Verify coverage_warning displayed
    ui.coverage_warning.assert_called_once_with(SAMPLE_PROFILE["coverage_warnings"])

    # -- Verify returned state dict
    assert result["codebase_profile"] == SAMPLE_PROFILE
    assert result["discovery_report"] == SAMPLE_REPORT
    assert result["current_stage"] == "discover"


# ---------------------------------------------------------------------------
# Tests: scope_path resolution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_discover_scope_path_resolves_to_subdirectory():
    """When scope_path is set and the scoped dir exists, cwd is the scoped dir."""
    ui = _make_ui()
    state = _make_state(scope_path="packages/frontend")

    agent_result = AgentResult(text="fallback")

    profile_json = json.dumps({"coverage_warnings": []})
    existing_files: dict[str, str] = {
        "/repo/packages/frontend": "__dir__",  # scoped_dir.exists() → True
        "/repo/packages/frontend/discovery_report.md": "scoped report",
        "/repo/packages/frontend/codebase_profile.json": profile_json,
    }

    def fake_exists(self: Path) -> bool:
        return str(self) in existing_files

    def fake_read_text(self: Path, *a, **kw) -> str:
        return existing_files[str(self)]

    def fake_unlink(self: Path, *a, **kw) -> None:
        existing_files.pop(str(self), None)

    with (
        patch("graft.stages.discover.run_agent", new_callable=AsyncMock, return_value=agent_result) as mock_run,
        patch("graft.stages.discover.save_artifact"),
        patch("graft.stages.discover.mark_stage_complete"),
        patch.object(Path, "exists", fake_exists),
        patch.object(Path, "read_text", fake_read_text),
        patch.object(Path, "unlink", fake_unlink),
    ):
        result = await discover_node(state, ui)

    # The agent should run in the scoped directory
    assert mock_run.call_args.kwargs["cwd"] == "/repo/packages/frontend"

    # The prompt should mention the scope
    prompt = mock_run.call_args.kwargs["user_prompt"]
    assert "packages/frontend" in prompt
    assert "SCOPE" in prompt


@pytest.mark.asyncio
async def test_discover_scope_path_nonexistent_falls_back_to_repo():
    """When scope_path doesn't exist on disk, discover_cwd stays as repo_path."""
    ui = _make_ui()
    state = _make_state(scope_path="nonexistent/path")

    agent_result = AgentResult(text="fallback")

    profile_json = json.dumps({"coverage_warnings": []})
    existing_files: dict[str, str] = {
        # /repo/nonexistent/path does NOT exist
        "/repo/discovery_report.md": "report",
        "/repo/codebase_profile.json": profile_json,
    }

    def fake_exists(self: Path) -> bool:
        return str(self) in existing_files

    def fake_read_text(self: Path, *a, **kw) -> str:
        return existing_files[str(self)]

    def fake_unlink(self: Path, *a, **kw) -> None:
        existing_files.pop(str(self), None)

    with (
        patch("graft.stages.discover.run_agent", new_callable=AsyncMock, return_value=agent_result) as mock_run,
        patch("graft.stages.discover.save_artifact"),
        patch("graft.stages.discover.mark_stage_complete"),
        patch.object(Path, "exists", fake_exists),
        patch.object(Path, "read_text", fake_read_text),
        patch.object(Path, "unlink", fake_unlink),
    ):
        await discover_node(state, ui)

    # Falls back to repo_path
    assert mock_run.call_args.kwargs["cwd"] == "/repo"


# ---------------------------------------------------------------------------
# Tests: fallback when output files are missing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_discover_report_falls_back_to_result_text():
    """When neither discover_cwd nor repo_path has discovery_report.md, use result.text."""
    ui = _make_ui()
    state = _make_state()

    fallback_text = "Agent text output as fallback"
    agent_result = AgentResult(text=fallback_text)

    # No files exist at all
    existing_files: dict[str, str] = {}

    def fake_exists(self: Path) -> bool:
        return str(self) in existing_files

    def fake_read_text(self: Path, *a, **kw) -> str:
        return existing_files[str(self)]

    def fake_unlink(self: Path, *a, **kw) -> None:
        pass

    with (
        patch("graft.stages.discover.run_agent", new_callable=AsyncMock, return_value=agent_result),
        patch("graft.stages.discover.save_artifact") as mock_save,
        patch("graft.stages.discover.mark_stage_complete"),
        patch.object(Path, "exists", fake_exists),
        patch.object(Path, "read_text", fake_read_text),
        patch.object(Path, "unlink", fake_unlink),
    ):
        result = await discover_node(state, ui)

    # Report should fall back to result.text
    assert result["discovery_report"] == fallback_text
    mock_save.assert_any_call("/projects/feat_abc", "discovery_report.md", fallback_text)

    # Profile should be empty dict
    assert result["codebase_profile"] == {}


@pytest.mark.asyncio
async def test_discover_profile_falls_back_to_repo_path():
    """When profile is missing in discover_cwd but exists in repo_path, it's found."""
    ui = _make_ui()
    # Use scope_path so discover_cwd != repo_path
    state = _make_state(scope_path="sub")

    profile = {"coverage_warnings": [], "project": {"name": "test"}}
    profile_json = json.dumps(profile)
    agent_result = AgentResult(text="fallback")

    existing_files: dict[str, str] = {
        "/repo/sub": "__dir__",  # scope exists
        # Profile not in /repo/sub, but in /repo
        "/repo/codebase_profile.json": profile_json,
        # Report not in /repo/sub, but in /repo
        "/repo/discovery_report.md": "report from repo",
    }

    def fake_exists(self: Path) -> bool:
        return str(self) in existing_files

    def fake_read_text(self: Path, *a, **kw) -> str:
        return existing_files[str(self)]

    def fake_unlink(self: Path, *a, **kw) -> None:
        existing_files.pop(str(self), None)

    with (
        patch("graft.stages.discover.run_agent", new_callable=AsyncMock, return_value=agent_result),
        patch("graft.stages.discover.save_artifact"),
        patch("graft.stages.discover.mark_stage_complete"),
        patch.object(Path, "exists", fake_exists),
        patch.object(Path, "read_text", fake_read_text),
        patch.object(Path, "unlink", fake_unlink),
    ):
        result = await discover_node(state, ui)

    assert result["codebase_profile"] == profile
    assert result["discovery_report"] == "report from repo"


# ---------------------------------------------------------------------------
# Tests: JSON decode error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_discover_invalid_json_profile_shows_error():
    """When codebase_profile.json is invalid JSON, ui.error is called and profile is empty."""
    ui = _make_ui()
    state = _make_state()

    agent_result = AgentResult(text="fallback")

    existing_files: dict[str, str] = {
        "/repo/discovery_report.md": "report",
        "/repo/codebase_profile.json": "NOT VALID JSON {{{",
    }

    def fake_exists(self: Path) -> bool:
        return str(self) in existing_files

    def fake_read_text(self: Path, *a, **kw) -> str:
        return existing_files[str(self)]

    def fake_unlink(self: Path, *a, **kw) -> None:
        existing_files.pop(str(self), None)

    with (
        patch("graft.stages.discover.run_agent", new_callable=AsyncMock, return_value=agent_result),
        patch("graft.stages.discover.save_artifact"),
        patch("graft.stages.discover.mark_stage_complete"),
        patch.object(Path, "exists", fake_exists),
        patch.object(Path, "read_text", fake_read_text),
        patch.object(Path, "unlink", fake_unlink),
    ):
        result = await discover_node(state, ui)

    ui.error.assert_called_once_with(
        "Failed to parse codebase_profile.json from agent output."
    )
    assert result["codebase_profile"] == {}


# ---------------------------------------------------------------------------
# Tests: coverage_warnings
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_discover_no_coverage_warnings_does_not_call_ui():
    """When coverage_warnings is empty, ui.coverage_warning is NOT called."""
    ui = _make_ui()
    state = _make_state()

    agent_result = AgentResult(text="fallback")

    profile = {"coverage_warnings": []}
    existing_files: dict[str, str] = {
        "/repo/discovery_report.md": "report",
        "/repo/codebase_profile.json": json.dumps(profile),
    }

    def fake_exists(self: Path) -> bool:
        return str(self) in existing_files

    def fake_read_text(self: Path, *a, **kw) -> str:
        return existing_files[str(self)]

    def fake_unlink(self: Path, *a, **kw) -> None:
        existing_files.pop(str(self), None)

    with (
        patch("graft.stages.discover.run_agent", new_callable=AsyncMock, return_value=agent_result),
        patch("graft.stages.discover.save_artifact"),
        patch("graft.stages.discover.mark_stage_complete"),
        patch.object(Path, "exists", fake_exists),
        patch.object(Path, "read_text", fake_read_text),
        patch.object(Path, "unlink", fake_unlink),
    ):
        await discover_node(state, ui)

    ui.coverage_warning.assert_not_called()


@pytest.mark.asyncio
async def test_discover_missing_profile_no_coverage_warnings():
    """When profile doesn't exist at all, coverage_warnings defaults to empty list."""
    ui = _make_ui()
    state = _make_state()

    agent_result = AgentResult(text="fallback")

    existing_files: dict[str, str] = {}

    def fake_exists(self: Path) -> bool:
        return str(self) in existing_files

    def fake_read_text(self: Path, *a, **kw) -> str:
        return existing_files[str(self)]

    def fake_unlink(self: Path, *a, **kw) -> None:
        pass

    with (
        patch("graft.stages.discover.run_agent", new_callable=AsyncMock, return_value=agent_result),
        patch("graft.stages.discover.save_artifact"),
        patch("graft.stages.discover.mark_stage_complete"),
        patch.object(Path, "exists", fake_exists),
        patch.object(Path, "read_text", fake_read_text),
        patch.object(Path, "unlink", fake_unlink),
    ):
        result = await discover_node(state, ui)

    ui.coverage_warning.assert_not_called()
    # Empty dict has no coverage_warnings key — .get returns []
    assert result["codebase_profile"] == {}


# ---------------------------------------------------------------------------
# Tests: file cleanup
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_discover_cleans_up_temporary_files():
    """All 4 potential file locations are checked and unlinked if they exist."""
    ui = _make_ui()
    state = _make_state(scope_path="sub")

    agent_result = AgentResult(text="fallback")

    profile_json = json.dumps({"coverage_warnings": []})
    existing_files: dict[str, str] = {
        "/repo/sub": "__dir__",
        "/repo/sub/discovery_report.md": "report",
        "/repo/sub/codebase_profile.json": profile_json,
        "/repo/discovery_report.md": "report2",
        "/repo/codebase_profile.json": profile_json,
    }

    unlinked: list[str] = []

    def fake_exists(self: Path) -> bool:
        return str(self) in existing_files

    def fake_read_text(self: Path, *a, **kw) -> str:
        return existing_files[str(self)]

    def fake_unlink(self: Path, *a, **kw) -> None:
        unlinked.append(str(self))

    with (
        patch("graft.stages.discover.run_agent", new_callable=AsyncMock, return_value=agent_result),
        patch("graft.stages.discover.save_artifact"),
        patch("graft.stages.discover.mark_stage_complete"),
        patch.object(Path, "exists", fake_exists),
        patch.object(Path, "read_text", fake_read_text),
        patch.object(Path, "unlink", fake_unlink),
    ):
        await discover_node(state, ui)

    # All 4 paths should be cleaned up
    assert "/repo/sub/discovery_report.md" in unlinked
    assert "/repo/sub/codebase_profile.json" in unlinked
    assert "/repo/discovery_report.md" in unlinked
    assert "/repo/codebase_profile.json" in unlinked


@pytest.mark.asyncio
async def test_discover_cleanup_skips_nonexistent_files():
    """Cleanup loop only unlinks files that exist — no errors for missing ones."""
    ui = _make_ui()
    state = _make_state()

    agent_result = AgentResult(text="fallback")

    # No output files exist
    existing_files: dict[str, str] = {}
    unlinked: list[str] = []

    def fake_exists(self: Path) -> bool:
        return str(self) in existing_files

    def fake_read_text(self: Path, *a, **kw) -> str:
        return existing_files.get(str(self), "")

    def fake_unlink(self: Path, *a, **kw) -> None:
        unlinked.append(str(self))

    with (
        patch("graft.stages.discover.run_agent", new_callable=AsyncMock, return_value=agent_result),
        patch("graft.stages.discover.save_artifact"),
        patch("graft.stages.discover.mark_stage_complete"),
        patch.object(Path, "exists", fake_exists),
        patch.object(Path, "read_text", fake_read_text),
        patch.object(Path, "unlink", fake_unlink),
    ):
        await discover_node(state, ui)

    # Nothing to unlink — no errors raised
    assert unlinked == []


# ---------------------------------------------------------------------------
# Tests: prompt construction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_discover_prompt_includes_feature_prompt():
    """When feature_prompt is set, it appears in the user prompt."""
    ui = _make_ui()
    state = _make_state(feature_prompt="Add Stripe billing integration")

    agent_result = AgentResult(text="fallback")
    existing_files: dict[str, str] = {}

    def fake_exists(self: Path) -> bool:
        return str(self) in existing_files

    def fake_read_text(self: Path, *a, **kw) -> str:
        return existing_files.get(str(self), "")

    def fake_unlink(self: Path, *a, **kw) -> None:
        pass

    with (
        patch("graft.stages.discover.run_agent", new_callable=AsyncMock, return_value=agent_result) as mock_run,
        patch("graft.stages.discover.save_artifact"),
        patch("graft.stages.discover.mark_stage_complete"),
        patch.object(Path, "exists", fake_exists),
        patch.object(Path, "read_text", fake_read_text),
        patch.object(Path, "unlink", fake_unlink),
    ):
        await discover_node(state, ui)

    prompt = mock_run.call_args.kwargs["user_prompt"]
    assert "Add Stripe billing integration" in prompt
    assert "UPCOMING FEATURE" in prompt
    assert "integration-critical modules" in prompt


@pytest.mark.asyncio
async def test_discover_prompt_minimal_without_optional_fields():
    """Without scope_path or feature_prompt, prompt is minimal."""
    ui = _make_ui()
    state = _make_state()

    agent_result = AgentResult(text="fallback")
    existing_files: dict[str, str] = {}

    def fake_exists(self: Path) -> bool:
        return str(self) in existing_files

    def fake_read_text(self: Path, *a, **kw) -> str:
        return existing_files.get(str(self), "")

    def fake_unlink(self: Path, *a, **kw) -> None:
        pass

    with (
        patch("graft.stages.discover.run_agent", new_callable=AsyncMock, return_value=agent_result) as mock_run,
        patch("graft.stages.discover.save_artifact"),
        patch("graft.stages.discover.mark_stage_complete"),
        patch.object(Path, "exists", fake_exists),
        patch.object(Path, "read_text", fake_read_text),
        patch.object(Path, "unlink", fake_unlink),
    ):
        await discover_node(state, ui)

    prompt = mock_run.call_args.kwargs["user_prompt"]
    assert "SCOPE" not in prompt
    assert "UPCOMING FEATURE" not in prompt
    assert "Discover and map the codebase at: /repo" in prompt
    assert "comprehensive discovery report" in prompt


# ---------------------------------------------------------------------------
# Tests: run_agent kwargs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_discover_passes_model_and_allowed_tools():
    """Model from state and the fixed allowed_tools list are forwarded to run_agent."""
    ui = _make_ui()
    state = _make_state(model="claude-sonnet-4-20250514")

    agent_result = AgentResult(text="fallback")
    existing_files: dict[str, str] = {}

    def fake_exists(self: Path) -> bool:
        return str(self) in existing_files

    def fake_read_text(self: Path, *a, **kw) -> str:
        return existing_files.get(str(self), "")

    def fake_unlink(self: Path, *a, **kw) -> None:
        pass

    with (
        patch("graft.stages.discover.run_agent", new_callable=AsyncMock, return_value=agent_result) as mock_run,
        patch("graft.stages.discover.save_artifact"),
        patch("graft.stages.discover.mark_stage_complete"),
        patch.object(Path, "exists", fake_exists),
        patch.object(Path, "read_text", fake_read_text),
        patch.object(Path, "unlink", fake_unlink),
    ):
        await discover_node(state, ui)

    kw = mock_run.call_args.kwargs
    assert kw["model"] == "claude-sonnet-4-20250514"
    assert kw["allowed_tools"] == ["Bash", "Read", "Glob", "Grep"]
    assert kw["system_prompt"] == SYSTEM_PROMPT
    assert kw["project_dir"] == "/projects/feat_abc"


# ---------------------------------------------------------------------------
# Tests: return dict shape
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_discover_return_dict_has_required_keys():
    """Returned dict always has codebase_profile, discovery_report, current_stage."""
    ui = _make_ui()
    state = _make_state()

    agent_result = AgentResult(text="fallback")
    existing_files: dict[str, str] = {}

    def fake_exists(self: Path) -> bool:
        return str(self) in existing_files

    def fake_read_text(self: Path, *a, **kw) -> str:
        return existing_files.get(str(self), "")

    def fake_unlink(self: Path, *a, **kw) -> None:
        pass

    with (
        patch("graft.stages.discover.run_agent", new_callable=AsyncMock, return_value=agent_result),
        patch("graft.stages.discover.save_artifact"),
        patch("graft.stages.discover.mark_stage_complete"),
        patch.object(Path, "exists", fake_exists),
        patch.object(Path, "read_text", fake_read_text),
        patch.object(Path, "unlink", fake_unlink),
    ):
        result = await discover_node(state, ui)

    assert set(result.keys()) == {"codebase_profile", "discovery_report", "current_stage"}
    assert isinstance(result["codebase_profile"], dict)
    assert isinstance(result["discovery_report"], str)
    assert result["current_stage"] == "discover"


# ---------------------------------------------------------------------------
# Tests: SYSTEM_PROMPT constant
# ---------------------------------------------------------------------------

def test_system_prompt_contains_key_instructions():
    """SYSTEM_PROMPT includes the critical deliverables the agent must produce."""
    assert "codebase_profile.json" in SYSTEM_PROMPT
    assert "discovery_report.md" in SYSTEM_PROMPT
    assert "coverage_warnings" in SYSTEM_PROMPT
    assert "Principal Codebase Archaeologist" in SYSTEM_PROMPT
