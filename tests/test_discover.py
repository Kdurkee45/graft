"""Tests for graft.stages.discover."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from graft.agent import AgentResult
from graft.stages.discover import SYSTEM_PROMPT, discover_node


@pytest.fixture
def project_dir(tmp_path):
    """Create a minimal project directory with artifacts and logs dirs."""
    pdir = tmp_path / "project"
    (pdir / "artifacts").mkdir(parents=True)
    (pdir / "logs").mkdir(parents=True)
    # metadata.json required by mark_stage_complete
    meta = {
        "project_id": "feat_test",
        "repo_path": str(tmp_path / "repo"),
        "feature_prompt": "test feature",
        "status": "in_progress",
        "stages_completed": [],
        "created_at": "2025-01-01T00:00:00",
    }
    (pdir / "metadata.json").write_text(json.dumps(meta))
    return pdir


@pytest.fixture
def repo_dir(tmp_path):
    """Create a temporary repo directory."""
    rdir = tmp_path / "repo"
    rdir.mkdir()
    return rdir


@pytest.fixture
def ui():
    """Return a mock UI object with all expected methods."""
    mock_ui = MagicMock()
    mock_ui.stage_start = MagicMock()
    mock_ui.stage_done = MagicMock()
    mock_ui.stage_log = MagicMock()
    mock_ui.error = MagicMock()
    mock_ui.coverage_warning = MagicMock()
    return mock_ui


@pytest.fixture
def sample_profile():
    """A realistic codebase_profile.json payload."""
    return {
        "timestamp": "2025-01-01T00:00:00Z",
        "project": {
            "name": "test-project",
            "languages": ["python"],
            "frameworks": ["fastapi"],
            "package_manager": "pip",
            "monorepo": False,
        },
        "services": [],
        "data_model": {"orm": "sqlalchemy", "tables": [], "key_relationships": []},
        "patterns": {
            "state_management": "n/a",
            "api": "REST",
            "auth": "JWT",
            "components": "n/a",
            "routing": "fastapi",
        },
        "test_infrastructure": {
            "framework": "pytest",
            "runner": "pytest",
            "coverage": "pytest-cov",
            "e2e": None,
            "conventions": "tests/ directory",
        },
        "conventions": {
            "git_workflow": "feature branches",
            "commit_style": "conventional",
            "code_style": "ruff",
        },
        "coverage_warnings": [],
    }


def _make_state(repo_dir, project_dir, **overrides):
    """Build a minimal FeatureState dict for testing."""
    state = {
        "repo_path": str(repo_dir),
        "project_dir": str(project_dir),
    }
    state.update(overrides)
    return state


def _make_agent_result(text="Agent completed discovery."):
    """Build a minimal AgentResult."""
    return AgentResult(
        text=text,
        tool_calls=[],
        raw_messages=[],
        elapsed_seconds=5.0,
        turns_used=10,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discover_node_calls_run_agent_with_correct_args(
    repo_dir, project_dir, ui
):
    """run_agent is called with the correct persona and read-only tools."""
    mock_result = _make_agent_result()

    with patch("graft.stages.discover.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_result
        await discover_node(_make_state(repo_dir, project_dir), ui)

    mock_run.assert_called_once()
    kwargs = mock_run.call_args.kwargs
    assert kwargs["persona"] == "Principal Codebase Archaeologist"
    assert kwargs["system_prompt"] == SYSTEM_PROMPT
    assert kwargs["stage"] == "discover"
    assert kwargs["allowed_tools"] == ["Bash", "Read", "Glob", "Grep"]
    assert kwargs["cwd"] == str(repo_dir)
    assert kwargs["project_dir"] == str(project_dir)
    assert kwargs["max_turns"] == 40


@pytest.mark.asyncio
async def test_discover_node_reads_profile_from_file(
    repo_dir, project_dir, ui, sample_profile
):
    """When the agent writes codebase_profile.json, it is parsed and returned."""
    # Write agent output files in the repo dir
    (repo_dir / "codebase_profile.json").write_text(json.dumps(sample_profile))
    (repo_dir / "discovery_report.md").write_text("# Discovery Report\nAll good.")

    mock_result = _make_agent_result()
    with patch("graft.stages.discover.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_result
        result = await discover_node(_make_state(repo_dir, project_dir), ui)

    assert result["codebase_profile"] == sample_profile
    assert result["discovery_report"] == "# Discovery Report\nAll good."
    assert result["current_stage"] == "discover"


@pytest.mark.asyncio
async def test_discover_node_falls_back_to_result_text_when_no_report_file(
    repo_dir, project_dir, ui
):
    """When discovery_report.md is not created by the agent, fall back to result.text."""
    mock_result = _make_agent_result(text="Fallback discovery text from agent.")
    with patch("graft.stages.discover.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_result
        result = await discover_node(_make_state(repo_dir, project_dir), ui)

    assert result["discovery_report"] == "Fallback discovery text from agent."
    # Profile should be empty dict since no file was written
    assert result["codebase_profile"] == {}


@pytest.mark.asyncio
async def test_discover_node_saves_artifacts(repo_dir, project_dir, ui, sample_profile):
    """save_artifact is called for both the report and the profile."""
    (repo_dir / "codebase_profile.json").write_text(json.dumps(sample_profile))
    (repo_dir / "discovery_report.md").write_text("# Report")

    mock_result = _make_agent_result()
    with (
        patch("graft.stages.discover.run_agent", new_callable=AsyncMock) as mock_run,
        patch("graft.stages.discover.save_artifact") as mock_save,
        patch("graft.stages.discover.mark_stage_complete") as mock_mark,
    ):
        mock_run.return_value = mock_result
        await discover_node(_make_state(repo_dir, project_dir), ui)

    # Should save both artifacts
    save_calls = mock_save.call_args_list
    assert len(save_calls) == 2

    # First call: discovery_report.md
    assert save_calls[0] == call(str(project_dir), "discovery_report.md", "# Report")

    # Second call: codebase_profile.json (as pretty-printed JSON)
    profile_call_content = save_calls[1][0][2]
    assert json.loads(profile_call_content) == sample_profile

    # mark_stage_complete called with "discover"
    mock_mark.assert_called_once_with(str(project_dir), "discover")


@pytest.mark.asyncio
async def test_discover_node_shows_coverage_warnings(
    repo_dir, project_dir, ui
):
    """When the profile has coverage_warnings, ui.coverage_warning is called."""
    warnings = [
        {
            "module": "src/services/billing.py",
            "coverage_pct": 5,
            "recommendation": "Add unit tests for billing service",
        }
    ]
    profile = {"coverage_warnings": warnings}
    (repo_dir / "codebase_profile.json").write_text(json.dumps(profile))

    mock_result = _make_agent_result()
    with patch("graft.stages.discover.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_result
        await discover_node(_make_state(repo_dir, project_dir), ui)

    ui.coverage_warning.assert_called_once_with(warnings)


@pytest.mark.asyncio
async def test_discover_node_no_coverage_warning_when_empty(
    repo_dir, project_dir, ui
):
    """When coverage_warnings is empty, ui.coverage_warning is NOT called."""
    profile = {"coverage_warnings": []}
    (repo_dir / "codebase_profile.json").write_text(json.dumps(profile))

    mock_result = _make_agent_result()
    with patch("graft.stages.discover.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_result
        await discover_node(_make_state(repo_dir, project_dir), ui)

    ui.coverage_warning.assert_not_called()


@pytest.mark.asyncio
async def test_discover_node_cleans_up_generated_files(
    repo_dir, project_dir, ui, sample_profile
):
    """Agent-generated files in the repo dir are removed after processing."""
    report_file = repo_dir / "discovery_report.md"
    profile_file = repo_dir / "codebase_profile.json"
    report_file.write_text("# Report")
    profile_file.write_text(json.dumps(sample_profile))

    assert report_file.exists()
    assert profile_file.exists()

    mock_result = _make_agent_result()
    with patch("graft.stages.discover.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_result
        await discover_node(_make_state(repo_dir, project_dir), ui)

    # Files should be cleaned up
    assert not report_file.exists()
    assert not profile_file.exists()


@pytest.mark.asyncio
async def test_discover_node_respects_scope_path(repo_dir, project_dir, ui):
    """When scope_path is set and exists, the agent's cwd is the scoped directory."""
    scoped = repo_dir / "packages" / "backend"
    scoped.mkdir(parents=True)

    mock_result = _make_agent_result()
    with patch("graft.stages.discover.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_result
        await discover_node(
            _make_state(
                repo_dir, project_dir,
                scope_path="packages/backend",
            ),
            ui,
        )

    kwargs = mock_run.call_args.kwargs
    assert kwargs["cwd"] == str(scoped)
    # user_prompt should mention SCOPE
    assert "SCOPE" in kwargs["user_prompt"]
    assert "packages/backend" in kwargs["user_prompt"]


@pytest.mark.asyncio
async def test_discover_node_scope_path_nonexistent_falls_back(
    repo_dir, project_dir, ui
):
    """When scope_path doesn't exist as a directory, cwd falls back to repo_path."""
    mock_result = _make_agent_result()
    with patch("graft.stages.discover.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_result
        await discover_node(
            _make_state(
                repo_dir, project_dir,
                scope_path="nonexistent/path",
            ),
            ui,
        )

    kwargs = mock_run.call_args.kwargs
    # Falls back to repo_path since scoped dir doesn't exist
    assert kwargs["cwd"] == str(repo_dir)


@pytest.mark.asyncio
async def test_discover_node_invalid_json_profile(repo_dir, project_dir, ui):
    """When codebase_profile.json contains invalid JSON, an error is logged."""
    (repo_dir / "codebase_profile.json").write_text("not valid json {{{")
    (repo_dir / "discovery_report.md").write_text("# Report")

    mock_result = _make_agent_result()
    with patch("graft.stages.discover.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_result
        result = await discover_node(_make_state(repo_dir, project_dir), ui)

    ui.error.assert_called_once_with(
        "Failed to parse codebase_profile.json from agent output."
    )
    # Profile should be empty dict on parse failure
    assert result["codebase_profile"] == {}


@pytest.mark.asyncio
async def test_discover_node_feature_prompt_included_in_user_prompt(
    repo_dir, project_dir, ui
):
    """When feature_prompt is provided, it appears in the user_prompt sent to the agent."""
    mock_result = _make_agent_result()
    with patch("graft.stages.discover.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_result
        await discover_node(
            _make_state(
                repo_dir, project_dir,
                feature_prompt="Add dark mode support",
            ),
            ui,
        )

    kwargs = mock_run.call_args.kwargs
    assert "Add dark mode support" in kwargs["user_prompt"]
    assert "UPCOMING FEATURE" in kwargs["user_prompt"]
    assert "integration-critical" in kwargs["user_prompt"]


@pytest.mark.asyncio
async def test_discover_node_ui_lifecycle(repo_dir, project_dir, ui):
    """stage_start and stage_done are called with 'discover'."""
    mock_result = _make_agent_result()
    with patch("graft.stages.discover.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_result
        await discover_node(_make_state(repo_dir, project_dir), ui)

    ui.stage_start.assert_called_once_with("discover")
    ui.stage_done.assert_called_once_with("discover")


@pytest.mark.asyncio
async def test_discover_node_passes_model_from_state(repo_dir, project_dir, ui):
    """The model parameter from state is forwarded to run_agent."""
    mock_result = _make_agent_result()
    with patch("graft.stages.discover.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_result
        await discover_node(
            _make_state(repo_dir, project_dir, model="claude-sonnet-4-20250514"),
            ui,
        )

    kwargs = mock_run.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4-20250514"


@pytest.mark.asyncio
async def test_discover_node_model_none_when_not_in_state(
    repo_dir, project_dir, ui
):
    """When model is not in state, None is passed to run_agent."""
    mock_result = _make_agent_result()
    with patch("graft.stages.discover.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_result
        await discover_node(_make_state(repo_dir, project_dir), ui)

    kwargs = mock_run.call_args.kwargs
    assert kwargs["model"] is None


@pytest.mark.asyncio
async def test_discover_node_cleans_files_in_scoped_and_repo_dirs(
    repo_dir, project_dir, ui, sample_profile
):
    """Cleanup removes files from both the scoped cwd and the repo root."""
    scoped = repo_dir / "packages" / "backend"
    scoped.mkdir(parents=True)

    # Place files in both locations
    (scoped / "discovery_report.md").write_text("# Scoped Report")
    (scoped / "codebase_profile.json").write_text(json.dumps(sample_profile))
    (repo_dir / "discovery_report.md").write_text("# Root Report")
    (repo_dir / "codebase_profile.json").write_text(json.dumps(sample_profile))

    mock_result = _make_agent_result()
    with patch("graft.stages.discover.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_result
        await discover_node(
            _make_state(
                repo_dir, project_dir,
                scope_path="packages/backend",
            ),
            ui,
        )

    # All four files should be cleaned up
    assert not (scoped / "discovery_report.md").exists()
    assert not (scoped / "codebase_profile.json").exists()
    assert not (repo_dir / "discovery_report.md").exists()
    assert not (repo_dir / "codebase_profile.json").exists()


@pytest.mark.asyncio
async def test_discover_node_reads_from_scoped_dir_first(
    repo_dir, project_dir, ui, sample_profile
):
    """When files exist in the scoped dir, they are read from there (not repo root)."""
    scoped = repo_dir / "packages" / "frontend"
    scoped.mkdir(parents=True)

    scoped_profile = {**sample_profile, "project": {**sample_profile["project"], "name": "scoped-project"}}
    (scoped / "codebase_profile.json").write_text(json.dumps(scoped_profile))
    (scoped / "discovery_report.md").write_text("# Scoped Report")

    # Also place different files in root (should NOT be read)
    root_profile = {**sample_profile, "project": {**sample_profile["project"], "name": "root-project"}}
    (repo_dir / "codebase_profile.json").write_text(json.dumps(root_profile))
    (repo_dir / "discovery_report.md").write_text("# Root Report")

    mock_result = _make_agent_result()
    with patch("graft.stages.discover.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_result
        result = await discover_node(
            _make_state(
                repo_dir, project_dir,
                scope_path="packages/frontend",
            ),
            ui,
        )

    # Should read the scoped version, not the root version
    assert result["codebase_profile"]["project"]["name"] == "scoped-project"
    assert result["discovery_report"] == "# Scoped Report"
