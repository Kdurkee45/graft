"""Tests for graft.stages.discover."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from graft.agent import AgentResult
from graft.stages.discover import discover_node


@pytest.fixture
def ui():
    """Mock UI with all methods used by discover_node."""
    mock = MagicMock()
    mock.stage_start = MagicMock()
    mock.stage_done = MagicMock()
    mock.coverage_warning = MagicMock()
    mock.error = MagicMock()
    return mock


@pytest.fixture
def project_dir(tmp_path):
    """Create a project directory with the artifacts subdirectory."""
    p = tmp_path / "project"
    (p / "artifacts").mkdir(parents=True)
    return p


@pytest.fixture
def repo_path(tmp_path):
    """Create a repo directory to simulate the target codebase."""
    r = tmp_path / "repo"
    r.mkdir()
    return r


def _make_state(repo_path, project_dir, **overrides):
    """Build a minimal FeatureState dict for discover_node."""
    state = {
        "repo_path": str(repo_path),
        "project_dir": str(project_dir),
    }
    state.update(overrides)
    return state


def _agent_result(text="discovery done"):
    return AgentResult(
        text=text, tool_calls=[], raw_messages=[], elapsed_seconds=1.0, turns_used=5
    )


# ── Test 1: happy-path — reads files, saves artifacts, returns correct state ──


@pytest.mark.asyncio
@patch("graft.stages.discover.mark_stage_complete")
@patch("graft.stages.discover.save_artifact")
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_discover_happy_path(
    mock_run_agent, mock_save, mock_mark, repo_path, project_dir, ui
):
    """discover_node reads codebase_profile.json + discovery_report.md from repo dir,
    saves both as artifacts, and returns state with codebase_profile, discovery_report,
    and current_stage."""
    profile = {"project": {"name": "acme"}, "coverage_warnings": []}
    report_text = "# Discovery Report\nAll is well."

    (repo_path / "codebase_profile.json").write_text(json.dumps(profile))
    (repo_path / "discovery_report.md").write_text(report_text)

    mock_run_agent.return_value = _agent_result()
    state = _make_state(repo_path, project_dir)

    result = await discover_node(state, ui)

    assert result["codebase_profile"] == profile
    assert result["discovery_report"] == report_text
    assert result["current_stage"] == "discover"

    # Artifacts saved
    assert mock_save.call_count == 2
    mock_save.assert_any_call(str(project_dir), "discovery_report.md", report_text)
    mock_save.assert_any_call(
        str(project_dir), "codebase_profile.json", json.dumps(profile, indent=2)
    )


# ── Test 2: fallback to result.text when report file not written ─────────────


@pytest.mark.asyncio
@patch("graft.stages.discover.mark_stage_complete")
@patch("graft.stages.discover.save_artifact")
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_fallback_to_result_text(
    mock_run_agent, mock_save, mock_mark, repo_path, project_dir, ui
):
    """When discovery_report.md is not written by the agent, fall back to result.text."""
    mock_run_agent.return_value = _agent_result(text="Agent raw text output")
    state = _make_state(repo_path, project_dir)

    result = await discover_node(state, ui)

    assert result["discovery_report"] == "Agent raw text output"
    assert result["codebase_profile"] == {}


# ── Test 3: invalid JSON in codebase_profile.json → empty dict ───────────────


@pytest.mark.asyncio
@patch("graft.stages.discover.mark_stage_complete")
@patch("graft.stages.discover.save_artifact")
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_invalid_json_profile_fallback(
    mock_run_agent, mock_save, mock_mark, repo_path, project_dir, ui
):
    """Invalid JSON in codebase_profile.json yields an empty dict and calls ui.error."""
    (repo_path / "codebase_profile.json").write_text("{bad json!!!")
    mock_run_agent.return_value = _agent_result()
    state = _make_state(repo_path, project_dir)

    result = await discover_node(state, ui)

    assert result["codebase_profile"] == {}
    ui.error.assert_called_once_with(
        "Failed to parse codebase_profile.json from agent output."
    )


# ── Test 4: scope_path is applied correctly to discover_cwd ──────────────────


@pytest.mark.asyncio
@patch("graft.stages.discover.mark_stage_complete")
@patch("graft.stages.discover.save_artifact")
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_scope_path_applied(
    mock_run_agent, mock_save, mock_mark, repo_path, project_dir, ui
):
    """When scope_path is provided and exists, run_agent uses the scoped directory as cwd."""
    scoped = repo_path / "packages" / "web"
    scoped.mkdir(parents=True)

    profile = {"project": {"name": "web"}, "coverage_warnings": []}
    (scoped / "codebase_profile.json").write_text(json.dumps(profile))
    (scoped / "discovery_report.md").write_text("Scoped report.")

    mock_run_agent.return_value = _agent_result()
    state = _make_state(repo_path, project_dir, scope_path="packages/web")

    result = await discover_node(state, ui)

    # Verify run_agent was called with the scoped cwd
    _, kwargs = mock_run_agent.call_args
    assert kwargs["cwd"] == str(scoped)

    assert result["codebase_profile"] == profile
    assert result["discovery_report"] == "Scoped report."


@pytest.mark.asyncio
@patch("graft.stages.discover.mark_stage_complete")
@patch("graft.stages.discover.save_artifact")
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_scope_path_nonexistent_falls_back(
    mock_run_agent, mock_save, mock_mark, repo_path, project_dir, ui
):
    """When scope_path does not exist on disk, discover_cwd falls back to repo_path."""
    mock_run_agent.return_value = _agent_result()
    state = _make_state(repo_path, project_dir, scope_path="nonexistent/path")

    await discover_node(state, ui)

    _, kwargs = mock_run_agent.call_args
    assert kwargs["cwd"] == str(repo_path)


# ── Test 5: coverage_warnings are displayed via ui.coverage_warning ──────────


@pytest.mark.asyncio
@patch("graft.stages.discover.mark_stage_complete")
@patch("graft.stages.discover.save_artifact")
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_coverage_warnings_displayed(
    mock_run_agent, mock_save, mock_mark, repo_path, project_dir, ui
):
    """coverage_warnings from the profile are forwarded to ui.coverage_warning."""
    warnings = [
        {
            "module": "src/services/auth.ts",
            "coverage_pct": 5,
            "recommendation": "add tests",
        }
    ]
    profile = {"project": {"name": "acme"}, "coverage_warnings": warnings}
    (repo_path / "codebase_profile.json").write_text(json.dumps(profile))

    mock_run_agent.return_value = _agent_result()
    state = _make_state(repo_path, project_dir)

    await discover_node(state, ui)

    ui.coverage_warning.assert_called_once_with(warnings)


@pytest.mark.asyncio
@patch("graft.stages.discover.mark_stage_complete")
@patch("graft.stages.discover.save_artifact")
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_no_coverage_warnings_no_call(
    mock_run_agent, mock_save, mock_mark, repo_path, project_dir, ui
):
    """When coverage_warnings is empty, ui.coverage_warning is NOT called."""
    profile = {"project": {"name": "acme"}, "coverage_warnings": []}
    (repo_path / "codebase_profile.json").write_text(json.dumps(profile))

    mock_run_agent.return_value = _agent_result()
    state = _make_state(repo_path, project_dir)

    await discover_node(state, ui)

    ui.coverage_warning.assert_not_called()


# ── Test 6: cleanup removes temp files from repo directory ───────────────────


@pytest.mark.asyncio
@patch("graft.stages.discover.mark_stage_complete")
@patch("graft.stages.discover.save_artifact")
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_cleanup_removes_temp_files(
    mock_run_agent, mock_save, mock_mark, repo_path, project_dir, ui
):
    """After loading, discover_node removes discovery_report.md and codebase_profile.json
    from both repo_path and discover_cwd."""
    (repo_path / "codebase_profile.json").write_text("{}")
    (repo_path / "discovery_report.md").write_text("report")

    mock_run_agent.return_value = _agent_result()
    state = _make_state(repo_path, project_dir)

    await discover_node(state, ui)

    assert not (repo_path / "codebase_profile.json").exists()
    assert not (repo_path / "discovery_report.md").exists()


@pytest.mark.asyncio
@patch("graft.stages.discover.mark_stage_complete")
@patch("graft.stages.discover.save_artifact")
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_cleanup_scoped_and_repo_files(
    mock_run_agent, mock_save, mock_mark, repo_path, project_dir, ui
):
    """Cleanup removes files from both the scoped dir and the repo root."""
    scoped = repo_path / "packages" / "api"
    scoped.mkdir(parents=True)

    # Files in both locations
    (scoped / "codebase_profile.json").write_text("{}")
    (scoped / "discovery_report.md").write_text("scoped")
    (repo_path / "codebase_profile.json").write_text("{}")
    (repo_path / "discovery_report.md").write_text("root")

    mock_run_agent.return_value = _agent_result()
    state = _make_state(repo_path, project_dir, scope_path="packages/api")

    await discover_node(state, ui)

    assert not (scoped / "codebase_profile.json").exists()
    assert not (scoped / "discovery_report.md").exists()
    assert not (repo_path / "codebase_profile.json").exists()
    assert not (repo_path / "discovery_report.md").exists()


# ── Test 7: mark_stage_complete and ui.stage_done are called ─────────────────


@pytest.mark.asyncio
@patch("graft.stages.discover.mark_stage_complete")
@patch("graft.stages.discover.save_artifact")
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_stage_lifecycle_calls(
    mock_run_agent, mock_save, mock_mark, repo_path, project_dir, ui
):
    """mark_stage_complete and ui.stage_start/stage_done are called with 'discover'."""
    mock_run_agent.return_value = _agent_result()
    state = _make_state(repo_path, project_dir)

    await discover_node(state, ui)

    ui.stage_start.assert_called_once_with("discover")
    ui.stage_done.assert_called_once_with("discover")
    mock_mark.assert_called_once_with(str(project_dir), "discover")


# ── Test 8: run_agent receives correct kwargs ────────────────────────────────


@pytest.mark.asyncio
@patch("graft.stages.discover.mark_stage_complete")
@patch("graft.stages.discover.save_artifact")
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_run_agent_called_with_correct_kwargs(
    mock_run_agent, mock_save, mock_mark, repo_path, project_dir, ui
):
    """run_agent is called with the expected persona, stage, and allowed_tools."""
    mock_run_agent.return_value = _agent_result()
    state = _make_state(repo_path, project_dir, model="opus")

    await discover_node(state, ui)

    mock_run_agent.assert_called_once()
    _, kwargs = mock_run_agent.call_args
    assert kwargs["persona"] == "Principal Codebase Archaeologist"
    assert kwargs["stage"] == "discover"
    assert kwargs["cwd"] == str(repo_path)
    assert kwargs["project_dir"] == str(project_dir)
    assert kwargs["model"] == "opus"
    assert kwargs["max_turns"] == 40
    assert kwargs["allowed_tools"] == ["Bash", "Read", "Glob", "Grep"]
    assert kwargs["ui"] is ui


# ── Test 9: feature_prompt is included in user_prompt ─────────────────────────


@pytest.mark.asyncio
@patch("graft.stages.discover.mark_stage_complete")
@patch("graft.stages.discover.save_artifact")
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_feature_prompt_included(
    mock_run_agent, mock_save, mock_mark, repo_path, project_dir, ui
):
    """When feature_prompt is set, it appears in the user_prompt sent to run_agent."""
    mock_run_agent.return_value = _agent_result()
    state = _make_state(repo_path, project_dir, feature_prompt="Add dark mode toggle")

    await discover_node(state, ui)

    _, kwargs = mock_run_agent.call_args
    assert "Add dark mode toggle" in kwargs["user_prompt"]
    assert "UPCOMING FEATURE" in kwargs["user_prompt"]


@pytest.mark.asyncio
@patch("graft.stages.discover.mark_stage_complete")
@patch("graft.stages.discover.save_artifact")
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_scope_path_included_in_prompt(
    mock_run_agent, mock_save, mock_mark, repo_path, project_dir, ui
):
    """When scope_path is set, SCOPE instruction appears in the user_prompt."""
    scoped = repo_path / "packages" / "web"
    scoped.mkdir(parents=True)
    mock_run_agent.return_value = _agent_result()
    state = _make_state(repo_path, project_dir, scope_path="packages/web")

    await discover_node(state, ui)

    _, kwargs = mock_run_agent.call_args
    assert "SCOPE" in kwargs["user_prompt"]
    assert "packages/web" in kwargs["user_prompt"]


# ── Test 10: profile in scoped dir with fallback to repo_path ────────────────


@pytest.mark.asyncio
@patch("graft.stages.discover.mark_stage_complete")
@patch("graft.stages.discover.save_artifact")
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_profile_fallback_to_repo_path(
    mock_run_agent, mock_save, mock_mark, repo_path, project_dir, ui
):
    """When profile/report are not in discover_cwd but are in repo_path, they are found."""
    scoped = repo_path / "packages" / "web"
    scoped.mkdir(parents=True)

    profile = {"project": {"name": "fallback"}, "coverage_warnings": []}
    (repo_path / "codebase_profile.json").write_text(json.dumps(profile))
    (repo_path / "discovery_report.md").write_text("Fallback report")

    mock_run_agent.return_value = _agent_result()
    state = _make_state(repo_path, project_dir, scope_path="packages/web")

    result = await discover_node(state, ui)

    assert result["codebase_profile"] == profile
    assert result["discovery_report"] == "Fallback report"


# ── Test 11: no coverage_warnings key in profile ─────────────────────────────


@pytest.mark.asyncio
@patch("graft.stages.discover.mark_stage_complete")
@patch("graft.stages.discover.save_artifact")
@patch("graft.stages.discover.run_agent", new_callable=AsyncMock)
async def test_profile_without_coverage_warnings_key(
    mock_run_agent, mock_save, mock_mark, repo_path, project_dir, ui
):
    """Profile missing coverage_warnings key does not crash — defaults to empty list."""
    profile = {"project": {"name": "minimal"}}
    (repo_path / "codebase_profile.json").write_text(json.dumps(profile))

    mock_run_agent.return_value = _agent_result()
    state = _make_state(repo_path, project_dir)

    result = await discover_node(state, ui)

    assert result["codebase_profile"] == profile
    ui.coverage_warning.assert_not_called()
