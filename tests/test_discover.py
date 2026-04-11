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

def _make_state(tmp_path: Path, **overrides) -> dict:
    """Build a minimal FeatureState dict backed by real temp directories."""
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    (project / "artifacts").mkdir(exist_ok=True)
    (project / "logs").mkdir(exist_ok=True)
    # metadata.json required by mark_stage_complete
    meta = {
        "project_id": "feat_test",
        "repo_path": str(repo),
        "feature_prompt": "Add dark mode",
        "status": "in_progress",
        "stages_completed": [],
        "last_updated": "2026-01-01T00:00:00+00:00",
    }
    (project / "metadata.json").write_text(json.dumps(meta))

    state: dict = {
        "repo_path": str(repo),
        "project_dir": str(project),
        "scope_path": "",
        "feature_prompt": "",
    }
    state.update(overrides)
    return state


def _make_ui() -> MagicMock:
    """Return a mock UI object with the methods discover_node calls."""
    ui = MagicMock()
    ui.stage_start = MagicMock()
    ui.stage_done = MagicMock()
    ui.error = MagicMock()
    ui.coverage_warning = MagicMock()
    return ui


def _agent_result(text: str = "Discovery complete.") -> AgentResult:
    return AgentResult(
        text=text,
        tool_calls=[],
        raw_messages=[],
        elapsed_seconds=1.0,
        turns_used=5,
    )


SAMPLE_PROFILE = {
    "timestamp": "2026-01-01T00:00:00Z",
    "project": {
        "name": "my-app",
        "languages": ["python"],
        "frameworks": ["fastapi"],
        "package_manager": "pip",
        "monorepo": False,
    },
    "services": [],
    "data_model": {},
    "patterns": {},
    "test_infrastructure": {},
    "conventions": {},
    "coverage_warnings": [],
}

SAMPLE_REPORT = "# Discovery Report\n\nThe codebase is well-organized."


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_discover_node_happy_path(mock_run_agent, tmp_path):
    """Full happy path: agent produces both files in cwd, artifacts are saved,
    cleanup runs, state dict has the right shape."""
    state = _make_state(tmp_path)
    ui = _make_ui()
    repo = Path(state["repo_path"])

    # Place agent output files in cwd (== repo_path when no scope)
    (repo / "discovery_report.md").write_text(SAMPLE_REPORT)
    (repo / "codebase_profile.json").write_text(json.dumps(SAMPLE_PROFILE))

    mock_run_agent.return_value = _agent_result()

    result = await discover_node(state, ui)

    # -- Verify run_agent called with correct kwargs
    mock_run_agent.assert_awaited_once()
    kwargs = mock_run_agent.call_args.kwargs
    assert kwargs["persona"] == "Principal Codebase Archaeologist"
    assert kwargs["system_prompt"] == SYSTEM_PROMPT
    assert kwargs["cwd"] == str(repo)
    assert kwargs["stage"] == "discover"
    assert kwargs["max_turns"] == 40
    assert "Bash" in kwargs["allowed_tools"]

    # -- Returned state has required keys
    assert result["codebase_profile"] == SAMPLE_PROFILE
    assert result["discovery_report"] == SAMPLE_REPORT
    assert result["current_stage"] == "discover"

    # -- Artifacts saved to project dir
    artifacts_dir = Path(state["project_dir"]) / "artifacts"
    assert (artifacts_dir / "discovery_report.md").read_text() == SAMPLE_REPORT
    saved_profile = json.loads((artifacts_dir / "codebase_profile.json").read_text())
    assert saved_profile == SAMPLE_PROFILE

    # -- Agent output files cleaned up from repo
    assert not (repo / "discovery_report.md").exists()
    assert not (repo / "codebase_profile.json").exists()

    # -- mark_stage_complete was called
    meta = json.loads((Path(state["project_dir"]) / "metadata.json").read_text())
    assert "discover" in meta["stages_completed"]

    # -- UI lifecycle
    ui.stage_start.assert_called_once_with("discover")
    ui.stage_done.assert_called_once_with("discover")


@pytest.mark.asyncio
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_discover_node_scope_path(mock_run_agent, tmp_path):
    """When scope_path is set and the scoped directory exists, the agent cwd
    should be the scoped subdirectory, and the prompt should mention scope."""
    state = _make_state(tmp_path, scope_path="services/api")
    ui = _make_ui()
    repo = Path(state["repo_path"])
    scoped = repo / "services" / "api"
    scoped.mkdir(parents=True)

    # Place files in the scoped dir (agent's cwd)
    (scoped / "discovery_report.md").write_text(SAMPLE_REPORT)
    (scoped / "codebase_profile.json").write_text(json.dumps(SAMPLE_PROFILE))

    mock_run_agent.return_value = _agent_result()

    result = await discover_node(state, ui)

    kwargs = mock_run_agent.call_args.kwargs
    assert kwargs["cwd"] == str(scoped)
    assert "SCOPE" in kwargs["user_prompt"]
    assert "services/api" in kwargs["user_prompt"]

    # Scoped files cleaned up
    assert not (scoped / "discovery_report.md").exists()
    assert not (scoped / "codebase_profile.json").exists()


@pytest.mark.asyncio
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_discover_node_scope_path_nonexistent(mock_run_agent, tmp_path):
    """When scope_path is set but directory doesn't exist, cwd falls back to
    repo_path."""
    state = _make_state(tmp_path, scope_path="nonexistent/path")
    ui = _make_ui()
    repo = Path(state["repo_path"])

    (repo / "discovery_report.md").write_text(SAMPLE_REPORT)
    (repo / "codebase_profile.json").write_text(json.dumps(SAMPLE_PROFILE))

    mock_run_agent.return_value = _agent_result()

    await discover_node(state, ui)

    kwargs = mock_run_agent.call_args.kwargs
    assert kwargs["cwd"] == str(repo)


@pytest.mark.asyncio
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_discover_node_feature_prompt_in_prompt(mock_run_agent, tmp_path):
    """When feature_prompt is provided, the user prompt includes it."""
    state = _make_state(tmp_path, feature_prompt="Add dark mode toggle")
    ui = _make_ui()
    repo = Path(state["repo_path"])

    (repo / "discovery_report.md").write_text(SAMPLE_REPORT)
    (repo / "codebase_profile.json").write_text(json.dumps(SAMPLE_PROFILE))

    mock_run_agent.return_value = _agent_result()

    await discover_node(state, ui)

    prompt = mock_run_agent.call_args.kwargs["user_prompt"]
    assert "UPCOMING FEATURE" in prompt
    assert "Add dark mode toggle" in prompt
    assert "integration-critical" in prompt


@pytest.mark.asyncio
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_discover_node_files_in_repo_fallback(mock_run_agent, tmp_path):
    """When output files are NOT in cwd (scoped dir) but ARE in repo_path,
    the fallback path is used."""
    state = _make_state(tmp_path, scope_path="services/api")
    ui = _make_ui()
    repo = Path(state["repo_path"])
    scoped = repo / "services" / "api"
    scoped.mkdir(parents=True)

    # Files placed at repo root, not scoped dir
    (repo / "discovery_report.md").write_text(SAMPLE_REPORT)
    (repo / "codebase_profile.json").write_text(json.dumps(SAMPLE_PROFILE))

    mock_run_agent.return_value = _agent_result()

    result = await discover_node(state, ui)

    assert result["discovery_report"] == SAMPLE_REPORT
    assert result["codebase_profile"] == SAMPLE_PROFILE

    # Repo-level files cleaned up
    assert not (repo / "discovery_report.md").exists()
    assert not (repo / "codebase_profile.json").exists()


@pytest.mark.asyncio
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_discover_node_missing_output_files(mock_run_agent, tmp_path):
    """When neither cwd nor repo_path contains agent output files, discovery
    report falls back to result.text, profile is empty dict."""
    state = _make_state(tmp_path)
    ui = _make_ui()

    agent_text = "Agent produced text but no files."
    mock_run_agent.return_value = _agent_result(text=agent_text)

    result = await discover_node(state, ui)

    # Report falls back to agent text
    assert result["discovery_report"] == agent_text
    # Profile is empty dict
    assert result["codebase_profile"] == {}

    # Artifact still saved (empty profile)
    artifacts_dir = Path(state["project_dir"]) / "artifacts"
    assert (artifacts_dir / "discovery_report.md").read_text() == agent_text
    saved_profile = json.loads((artifacts_dir / "codebase_profile.json").read_text())
    assert saved_profile == {}


@pytest.mark.asyncio
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_discover_node_malformed_json(mock_run_agent, tmp_path):
    """When codebase_profile.json contains invalid JSON, the profile is empty
    and a UI error is shown."""
    state = _make_state(tmp_path)
    ui = _make_ui()
    repo = Path(state["repo_path"])

    (repo / "discovery_report.md").write_text(SAMPLE_REPORT)
    (repo / "codebase_profile.json").write_text("{invalid json content!!!")

    mock_run_agent.return_value = _agent_result()

    result = await discover_node(state, ui)

    assert result["codebase_profile"] == {}
    ui.error.assert_called_once_with(
        "Failed to parse codebase_profile.json from agent output."
    )

    # Despite the error, the rest of the pipeline continues
    assert result["discovery_report"] == SAMPLE_REPORT
    assert result["current_stage"] == "discover"

    # Cleanup still happens
    assert not (repo / "codebase_profile.json").exists()
    assert not (repo / "discovery_report.md").exists()


@pytest.mark.asyncio
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_discover_node_coverage_warnings_displayed(mock_run_agent, tmp_path):
    """When codebase_profile has coverage_warnings, ui.coverage_warning is called."""
    state = _make_state(tmp_path)
    ui = _make_ui()
    repo = Path(state["repo_path"])

    warnings = [
        {"module": "src/auth.py", "coverage_pct": 12, "recommendation": "Add tests"},
        {"module": "src/billing.py", "coverage_pct": 5, "recommendation": "Add tests"},
    ]
    profile_with_warnings = {**SAMPLE_PROFILE, "coverage_warnings": warnings}

    (repo / "discovery_report.md").write_text(SAMPLE_REPORT)
    (repo / "codebase_profile.json").write_text(json.dumps(profile_with_warnings))

    mock_run_agent.return_value = _agent_result()

    result = await discover_node(state, ui)

    ui.coverage_warning.assert_called_once_with(warnings)
    assert result["codebase_profile"]["coverage_warnings"] == warnings


@pytest.mark.asyncio
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_discover_node_no_coverage_warnings(mock_run_agent, tmp_path):
    """When coverage_warnings is empty, ui.coverage_warning is NOT called."""
    state = _make_state(tmp_path)
    ui = _make_ui()
    repo = Path(state["repo_path"])

    (repo / "discovery_report.md").write_text(SAMPLE_REPORT)
    (repo / "codebase_profile.json").write_text(json.dumps(SAMPLE_PROFILE))  # empty warnings

    mock_run_agent.return_value = _agent_result()

    await discover_node(state, ui)

    ui.coverage_warning.assert_not_called()


@pytest.mark.asyncio
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_discover_node_cleanup_all_locations(mock_run_agent, tmp_path):
    """Cleanup removes files from BOTH cwd and repo_path when both exist
    (e.g. agent wrote to both locations)."""
    state = _make_state(tmp_path, scope_path="sub")
    ui = _make_ui()
    repo = Path(state["repo_path"])
    scoped = repo / "sub"
    scoped.mkdir()

    # Files in BOTH locations
    for d in [scoped, repo]:
        (d / "discovery_report.md").write_text("report")
        (d / "codebase_profile.json").write_text("{}")

    mock_run_agent.return_value = _agent_result()

    await discover_node(state, ui)

    # All four files removed
    assert not (scoped / "discovery_report.md").exists()
    assert not (scoped / "codebase_profile.json").exists()
    assert not (repo / "discovery_report.md").exists()
    assert not (repo / "codebase_profile.json").exists()


@pytest.mark.asyncio
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_discover_node_model_passed_through(mock_run_agent, tmp_path):
    """The model key from state is forwarded to run_agent."""
    state = _make_state(tmp_path, model="claude-sonnet-4-20250514")
    ui = _make_ui()
    repo = Path(state["repo_path"])

    (repo / "discovery_report.md").write_text(SAMPLE_REPORT)
    (repo / "codebase_profile.json").write_text(json.dumps(SAMPLE_PROFILE))

    mock_run_agent.return_value = _agent_result()

    await discover_node(state, ui)

    assert mock_run_agent.call_args.kwargs["model"] == "claude-sonnet-4-20250514"


@pytest.mark.asyncio
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_discover_node_no_optional_state_keys(mock_run_agent, tmp_path):
    """When scope_path and feature_prompt are absent from state, prompt
    contains only the base discovery instruction."""
    repo = tmp_path / "repo"
    repo.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    (project / "artifacts").mkdir()
    (project / "logs").mkdir()
    meta = {
        "project_id": "feat_test",
        "repo_path": str(repo),
        "feature_prompt": "",
        "status": "in_progress",
        "stages_completed": [],
        "last_updated": "2026-01-01T00:00:00+00:00",
    }
    (project / "metadata.json").write_text(json.dumps(meta))

    state: dict = {
        "repo_path": str(repo),
        "project_dir": str(project),
        # No scope_path or feature_prompt at all
    }
    ui = _make_ui()

    (repo / "discovery_report.md").write_text(SAMPLE_REPORT)
    (repo / "codebase_profile.json").write_text(json.dumps(SAMPLE_PROFILE))

    mock_run_agent.return_value = _agent_result()

    result = await discover_node(state, ui)

    prompt = mock_run_agent.call_args.kwargs["user_prompt"]
    assert "SCOPE" not in prompt
    assert "UPCOMING FEATURE" not in prompt
    assert "Discover and map the codebase" in prompt
    assert "comprehensive discovery report" in prompt


@pytest.mark.asyncio
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_discover_node_agent_exception_propagates(mock_run_agent, tmp_path):
    """If run_agent raises an exception, it propagates out of discover_node."""
    state = _make_state(tmp_path)
    ui = _make_ui()

    mock_run_agent.side_effect = RuntimeError("Agent SDK connection failed")

    with pytest.raises(RuntimeError, match="Agent SDK connection failed"):
        await discover_node(state, ui)

    # stage_start was called before the error
    ui.stage_start.assert_called_once_with("discover")
    # stage_done was NOT called since we errored out
    ui.stage_done.assert_not_called()
