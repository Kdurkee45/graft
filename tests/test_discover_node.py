"""Tests for graft.stages.discover — discover_node function."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, call, patch

import pytest

from graft.stages.discover import discover_node
from graft.ui import UI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeAgentResult:
    """Minimal stand-in for AgentResult."""

    text: str = "fallback report text"
    tool_calls: list[dict] = field(default_factory=list)
    raw_messages: list = field(default_factory=list)
    elapsed_seconds: float = 1.0
    turns_used: int = 5


@pytest.fixture
def repo_path(tmp_path):
    """Simulated repository root."""
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


@pytest.fixture
def project_dir(tmp_path):
    """Simulated project directory with artifacts/ and metadata.json."""
    proj = tmp_path / "project"
    proj.mkdir()
    (proj / "artifacts").mkdir()
    # Minimal metadata so mark_stage_complete won't choke
    meta = {"stages_completed": []}
    (proj / "metadata.json").write_text(json.dumps(meta))
    return proj


@pytest.fixture
def base_state(repo_path, project_dir):
    """Minimal FeatureState dict accepted by discover_node."""
    return {
        "repo_path": str(repo_path),
        "project_dir": str(project_dir),
        "feature_prompt": "Add dark mode",
        "scope_path": "",
        "model": None,
    }


@pytest.fixture
def ui():
    return UI(verbose=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

PROFILE_PAYLOAD = {
    "project": {"name": "my-app", "languages": ["python"]},
    "coverage_warnings": [],
}


@pytest.mark.asyncio
async def test_discover_node_happy_path(repo_path, project_dir, base_state, ui):
    """Agent writes both output files; node reads, saves artifacts, cleans up."""
    report_content = "# Discovery Report\nAll looks good."
    profile_content = json.dumps(PROFILE_PAYLOAD)

    # Simulate agent writing files into repo_path
    (repo_path / "discovery_report.md").write_text(report_content)
    (repo_path / "codebase_profile.json").write_text(profile_content)

    fake_result = FakeAgentResult(text="ignored when files exist")

    with patch(
        "graft.stages.discover.run_agent",
        new_callable=AsyncMock,
        return_value=fake_result,
    ):
        result = await discover_node(base_state, ui)

    # Returned state dict has correct keys and values
    assert result["current_stage"] == "discover"
    assert result["discovery_report"] == report_content
    assert result["codebase_profile"] == PROFILE_PAYLOAD

    # Artifacts persisted to project_dir
    saved_report = (project_dir / "artifacts" / "discovery_report.md").read_text()
    assert saved_report == report_content

    saved_profile = json.loads(
        (project_dir / "artifacts" / "codebase_profile.json").read_text()
    )
    assert saved_profile == PROFILE_PAYLOAD

    # Temp files cleaned up from repo_path
    assert not (repo_path / "discovery_report.md").exists()
    assert not (repo_path / "codebase_profile.json").exists()

    # Stage marked complete in metadata
    meta = json.loads((project_dir / "metadata.json").read_text())
    assert "discover" in meta["stages_completed"]


@pytest.mark.asyncio
async def test_discover_node_missing_files_falls_back_to_result_text(
    repo_path, project_dir, base_state, ui
):
    """When agent fails to produce files, node uses result.text for the report."""
    fallback_text = "Agent produced this text instead of a file."
    fake_result = FakeAgentResult(text=fallback_text)

    with patch(
        "graft.stages.discover.run_agent",
        new_callable=AsyncMock,
        return_value=fake_result,
    ):
        result = await discover_node(base_state, ui)

    # Report falls back to result.text
    assert result["discovery_report"] == fallback_text
    # Profile defaults to empty dict when file missing
    assert result["codebase_profile"] == {}

    # Artifacts still saved (even if fallback content)
    saved_report = (project_dir / "artifacts" / "discovery_report.md").read_text()
    assert saved_report == fallback_text


@pytest.mark.asyncio
async def test_discover_node_invalid_json_profile(
    repo_path, project_dir, base_state, ui
):
    """Malformed codebase_profile.json results in empty dict, no crash."""
    (repo_path / "discovery_report.md").write_text("# Report")
    (repo_path / "codebase_profile.json").write_text("{invalid json!!")

    fake_result = FakeAgentResult()

    with patch(
        "graft.stages.discover.run_agent",
        new_callable=AsyncMock,
        return_value=fake_result,
    ):
        result = await discover_node(base_state, ui)

    assert result["codebase_profile"] == {}
    assert result["current_stage"] == "discover"


@pytest.mark.asyncio
async def test_artifacts_saved_via_save_artifact(
    repo_path, project_dir, base_state, ui
):
    """Verify save_artifact is called for both report and profile."""
    (repo_path / "discovery_report.md").write_text("# Report")
    (repo_path / "codebase_profile.json").write_text(json.dumps(PROFILE_PAYLOAD))
    fake_result = FakeAgentResult()

    with (
        patch(
            "graft.stages.discover.run_agent",
            new_callable=AsyncMock,
            return_value=fake_result,
        ),
        patch("graft.stages.discover.save_artifact") as mock_save,
    ):
        await discover_node(base_state, ui)

    assert mock_save.call_count == 2
    # First call: discovery_report.md
    first_call = mock_save.call_args_list[0]
    assert first_call == call(str(project_dir), "discovery_report.md", "# Report")
    # Second call: codebase_profile.json
    second_call = mock_save.call_args_list[1]
    assert second_call[0][0] == str(project_dir)
    assert second_call[0][1] == "codebase_profile.json"
    parsed = json.loads(second_call[0][2])
    assert parsed == PROFILE_PAYLOAD


@pytest.mark.asyncio
async def test_temp_files_cleaned_up(repo_path, project_dir, base_state, ui):
    """Temporary agent output files are unlinked from repo_path after reading."""
    (repo_path / "discovery_report.md").write_text("report")
    (repo_path / "codebase_profile.json").write_text("{}")
    fake_result = FakeAgentResult()

    with patch(
        "graft.stages.discover.run_agent",
        new_callable=AsyncMock,
        return_value=fake_result,
    ):
        await discover_node(base_state, ui)

    assert not (repo_path / "discovery_report.md").exists()
    assert not (repo_path / "codebase_profile.json").exists()


@pytest.mark.asyncio
async def test_returned_state_keys(repo_path, project_dir, base_state, ui):
    """Return dict contains exactly codebase_profile, discovery_report, current_stage."""
    fake_result = FakeAgentResult(text="report body")

    with patch(
        "graft.stages.discover.run_agent",
        new_callable=AsyncMock,
        return_value=fake_result,
    ):
        result = await discover_node(base_state, ui)

    assert set(result.keys()) == {
        "codebase_profile",
        "discovery_report",
        "current_stage",
    }
    assert isinstance(result["codebase_profile"], dict)
    assert isinstance(result["discovery_report"], str)
    assert result["current_stage"] == "discover"


@pytest.mark.asyncio
async def test_prompt_includes_repo_path_and_feature_prompt(
    repo_path, project_dir, base_state, ui
):
    """User prompt forwarded to run_agent contains repo_path and feature_prompt."""
    fake_result = FakeAgentResult()

    with patch(
        "graft.stages.discover.run_agent",
        new_callable=AsyncMock,
        return_value=fake_result,
    ) as mock_agent:
        await discover_node(base_state, ui)

    _, kwargs = mock_agent.call_args
    user_prompt = kwargs["user_prompt"]
    assert str(repo_path) in user_prompt
    assert "Add dark mode" in user_prompt


@pytest.mark.asyncio
async def test_prompt_includes_scope_path(repo_path, project_dir, base_state, ui):
    """When scope_path is set and the directory exists, prompt includes SCOPE note."""
    scoped = repo_path / "backend"
    scoped.mkdir()
    base_state["scope_path"] = "backend"

    fake_result = FakeAgentResult()

    with patch(
        "graft.stages.discover.run_agent",
        new_callable=AsyncMock,
        return_value=fake_result,
    ) as mock_agent:
        await discover_node(base_state, ui)

    _, kwargs = mock_agent.call_args
    user_prompt = kwargs["user_prompt"]
    assert "backend" in user_prompt
    assert "SCOPE" in user_prompt
    # cwd should be the scoped directory
    assert kwargs["cwd"] == str(scoped)


@pytest.mark.asyncio
async def test_scope_path_dir_missing_uses_repo_root(
    repo_path, project_dir, base_state, ui
):
    """When scope_path doesn't exist on disk, cwd remains repo_path."""
    base_state["scope_path"] = "nonexistent_subdir"
    fake_result = FakeAgentResult()

    with patch(
        "graft.stages.discover.run_agent",
        new_callable=AsyncMock,
        return_value=fake_result,
    ) as mock_agent:
        await discover_node(base_state, ui)

    _, kwargs = mock_agent.call_args
    assert kwargs["cwd"] == str(repo_path)


@pytest.mark.asyncio
async def test_scoped_dir_output_files(repo_path, project_dir, base_state, ui):
    """When scope_path is used, agent output is read from scoped directory."""
    scoped = repo_path / "frontend"
    scoped.mkdir()
    base_state["scope_path"] = "frontend"

    report_text = "# Scoped Report"
    profile_data = {"project": {"name": "frontend-app"}, "coverage_warnings": []}

    (scoped / "discovery_report.md").write_text(report_text)
    (scoped / "codebase_profile.json").write_text(json.dumps(profile_data))

    fake_result = FakeAgentResult()

    with patch(
        "graft.stages.discover.run_agent",
        new_callable=AsyncMock,
        return_value=fake_result,
    ):
        result = await discover_node(base_state, ui)

    assert result["discovery_report"] == report_text
    assert result["codebase_profile"] == profile_data
    # Cleaned up from scoped dir
    assert not (scoped / "discovery_report.md").exists()
    assert not (scoped / "codebase_profile.json").exists()


@pytest.mark.asyncio
async def test_coverage_warnings_displayed(repo_path, project_dir, base_state, ui):
    """When codebase_profile contains coverage_warnings, ui.coverage_warning is called."""
    warnings = [
        {"module": "src/foo.py", "coverage_pct": 10, "recommendation": "test it"}
    ]
    profile = {"coverage_warnings": warnings}
    (repo_path / "discovery_report.md").write_text("report")
    (repo_path / "codebase_profile.json").write_text(json.dumps(profile))

    fake_result = FakeAgentResult()

    with patch(
        "graft.stages.discover.run_agent",
        new_callable=AsyncMock,
        return_value=fake_result,
    ):
        with patch.object(ui, "coverage_warning") as mock_cov:
            await discover_node(base_state, ui)

    mock_cov.assert_called_once_with(warnings)


@pytest.mark.asyncio
async def test_no_coverage_warnings_not_called(repo_path, project_dir, base_state, ui):
    """When coverage_warnings is empty, ui.coverage_warning is not called."""
    profile = {"coverage_warnings": []}
    (repo_path / "discovery_report.md").write_text("report")
    (repo_path / "codebase_profile.json").write_text(json.dumps(profile))

    fake_result = FakeAgentResult()

    with patch(
        "graft.stages.discover.run_agent",
        new_callable=AsyncMock,
        return_value=fake_result,
    ):
        with patch.object(ui, "coverage_warning") as mock_cov:
            await discover_node(base_state, ui)

    mock_cov.assert_not_called()


@pytest.mark.asyncio
async def test_run_agent_called_with_correct_kwargs(
    repo_path, project_dir, base_state, ui
):
    """run_agent receives the expected keyword arguments."""
    fake_result = FakeAgentResult()

    with patch(
        "graft.stages.discover.run_agent",
        new_callable=AsyncMock,
        return_value=fake_result,
    ) as mock_agent:
        await discover_node(base_state, ui)

    mock_agent.assert_called_once()
    _, kwargs = mock_agent.call_args
    assert kwargs["persona"] == "Principal Codebase Archaeologist"
    assert kwargs["cwd"] == str(repo_path)
    assert kwargs["project_dir"] == str(project_dir)
    assert kwargs["stage"] == "discover"
    assert kwargs["ui"] is ui
    assert kwargs["model"] is None
    assert kwargs["max_turns"] == 40
    assert kwargs["allowed_tools"] == ["Bash", "Read", "Glob", "Grep"]
