"""Tests for graft.stages.research."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest

from graft.stages.research import research_node


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ui():
    ui = MagicMock()
    ui.stage_start = Mock()
    ui.stage_done = Mock()
    ui.error = Mock()
    ui.info = Mock()
    return ui


@pytest.fixture
def repo_dir(tmp_path):
    """Simulated repo directory."""
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


@pytest.fixture
def project_dir(tmp_path):
    """Simulated project directory with artifacts subdir."""
    proj = tmp_path / "project"
    proj.mkdir()
    (proj / "artifacts").mkdir()
    (proj / "metadata.json").write_text(json.dumps({
        "project_id": "feat_test",
        "stages_completed": ["discover"],
        "status": "in_progress",
    }))
    return proj


@pytest.fixture
def base_state(repo_dir, project_dir):
    """Minimal valid state for research_node."""
    return {
        "repo_path": str(repo_dir),
        "project_dir": str(project_dir),
        "feature_prompt": "Add dark mode toggle",
        "codebase_profile": {
            "project": {"name": "myapp", "languages": ["python"]},
            "patterns": {"api": "REST"},
        },
    }


def _agent_result(text: str = "fallback output") -> Mock:
    """Create a mock AgentResult."""
    result = Mock()
    result.text = text
    return result


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_research_node_happy_path(base_state, repo_dir, project_dir, mock_ui):
    """Full happy path: agent writes both files, node parses and returns them."""
    assessment = {
        "feature_prompt": "Add dark mode toggle",
        "reusable_components": [],
        "new_artifacts_needed": [],
        "open_questions": [
            {"question": "Should dark mode persist?", "category": "intent",
             "recommended_answer": "Yes, store in localStorage."}
        ],
    }
    report_content = "# Research Report\n\nDark mode analysis..."

    # Simulate agent writing files into the repo
    (repo_dir / "research_report.md").write_text(report_content)
    (repo_dir / "technical_assessment.json").write_text(json.dumps(assessment))

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent, \
         patch("graft.stages.research.save_artifact") as mock_save, \
         patch("graft.stages.research.mark_stage_complete") as mock_mark:
        mock_agent.return_value = _agent_result()
        result = await research_node(base_state, mock_ui)

    # Returned state keys
    assert result["technical_assessment"] == assessment
    assert result["research_report"] == report_content
    assert result["current_stage"] == "research"

    # save_artifact called for both outputs
    assert mock_save.call_count == 2
    mock_save.assert_any_call(str(project_dir), "research_report.md", report_content)
    mock_save.assert_any_call(
        str(project_dir), "technical_assessment.json",
        json.dumps(assessment, indent=2),
    )

    # Stage lifecycle
    mock_mark.assert_called_once_with(str(project_dir), "research")
    mock_ui.stage_start.assert_called_once_with("research")
    mock_ui.stage_done.assert_called_once_with("research")

    # Open questions surfaced
    mock_ui.info.assert_called_once()
    assert "1 open question" in mock_ui.info.call_args[0][0]

    # Cleanup: files removed from repo dir
    assert not (repo_dir / "research_report.md").exists()
    assert not (repo_dir / "technical_assessment.json").exists()


@pytest.mark.asyncio
async def test_prompt_includes_codebase_profile_and_feature(base_state, mock_ui):
    """Verify the user prompt sent to run_agent contains the profile and feature."""
    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent, \
         patch("graft.stages.research.save_artifact"), \
         patch("graft.stages.research.mark_stage_complete"):
        mock_agent.return_value = _agent_result()
        await research_node(base_state, mock_ui)

    _args, kwargs = mock_agent.call_args
    prompt = kwargs["user_prompt"]
    assert "Add dark mode toggle" in prompt
    assert '"name": "myapp"' in prompt
    assert "CODEBASE PROFILE:" in prompt


@pytest.mark.asyncio
async def test_prompt_includes_constraints_when_present(base_state, mock_ui):
    """Constraints list is appended to the prompt when provided."""
    base_state["constraints"] = ["No new dependencies", "Must support Python 3.11"]

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent, \
         patch("graft.stages.research.save_artifact"), \
         patch("graft.stages.research.mark_stage_complete"):
        mock_agent.return_value = _agent_result()
        await research_node(base_state, mock_ui)

    prompt = mock_agent.call_args.kwargs["user_prompt"]
    assert "CONSTRAINTS:" in prompt
    assert "No new dependencies" in prompt
    assert "Must support Python 3.11" in prompt


# ---------------------------------------------------------------------------
# Fallback paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_report_falls_back_to_agent_text_when_no_file(
    base_state, repo_dir, project_dir, mock_ui,
):
    """When agent doesn't write research_report.md, use result.text as fallback."""
    # No files created in repo_dir — agent only returns text
    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent, \
         patch("graft.stages.research.save_artifact") as mock_save, \
         patch("graft.stages.research.mark_stage_complete"):
        mock_agent.return_value = _agent_result("Agent textual output")
        result = await research_node(base_state, mock_ui)

    assert result["research_report"] == "Agent textual output"
    # First save_artifact call is for the report
    mock_save.assert_any_call(
        str(project_dir), "research_report.md", "Agent textual output",
    )


@pytest.mark.asyncio
async def test_assessment_empty_dict_when_no_json_file(
    base_state, repo_dir, mock_ui,
):
    """When agent doesn't write technical_assessment.json, return empty dict."""
    # Write only the report, not the JSON
    (repo_dir / "research_report.md").write_text("report")

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent, \
         patch("graft.stages.research.save_artifact"), \
         patch("graft.stages.research.mark_stage_complete"):
        mock_agent.return_value = _agent_result()
        result = await research_node(base_state, mock_ui)

    assert result["technical_assessment"] == {}
    mock_ui.error.assert_not_called()


@pytest.mark.asyncio
async def test_malformed_json_triggers_error_and_empty_dict(
    base_state, repo_dir, mock_ui,
):
    """Malformed technical_assessment.json → ui.error + empty assessment."""
    (repo_dir / "technical_assessment.json").write_text("{bad json!!!")
    (repo_dir / "research_report.md").write_text("report")

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent, \
         patch("graft.stages.research.save_artifact"), \
         patch("graft.stages.research.mark_stage_complete"):
        mock_agent.return_value = _agent_result()
        result = await research_node(base_state, mock_ui)

    assert result["technical_assessment"] == {}
    mock_ui.error.assert_called_once()
    assert "Failed to parse" in mock_ui.error.call_args[0][0]


# ---------------------------------------------------------------------------
# Edge cases: state variations
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_codebase_profile_defaults_to_empty_dict(
    repo_dir, project_dir, mock_ui,
):
    """When codebase_profile is absent from state, prompt uses '{}'."""
    state = {
        "repo_path": str(repo_dir),
        "project_dir": str(project_dir),
        "feature_prompt": "Add export feature",
    }

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent, \
         patch("graft.stages.research.save_artifact"), \
         patch("graft.stages.research.mark_stage_complete"):
        mock_agent.return_value = _agent_result()
        result = await research_node(state, mock_ui)

    # Should not crash — proceeds with empty profile
    prompt = mock_agent.call_args.kwargs["user_prompt"]
    assert "{}" in prompt
    assert result["current_stage"] == "research"


@pytest.mark.asyncio
async def test_missing_feature_prompt_defaults_to_empty_string(
    repo_dir, project_dir, mock_ui,
):
    """When feature_prompt is absent from state, prompt uses empty string."""
    state = {
        "repo_path": str(repo_dir),
        "project_dir": str(project_dir),
        "codebase_profile": {"project": {"name": "app"}},
    }

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent, \
         patch("graft.stages.research.save_artifact"), \
         patch("graft.stages.research.mark_stage_complete"):
        mock_agent.return_value = _agent_result()
        result = await research_node(state, mock_ui)

    prompt = mock_agent.call_args.kwargs["user_prompt"]
    assert "FEATURE: \n" in prompt or "FEATURE: " in prompt
    assert result["current_stage"] == "research"


@pytest.mark.asyncio
async def test_agent_empty_text_and_no_files(
    base_state, mock_ui,
):
    """Agent returns empty text and writes no files — still completes."""
    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent, \
         patch("graft.stages.research.save_artifact") as mock_save, \
         patch("graft.stages.research.mark_stage_complete") as mock_mark:
        mock_agent.return_value = _agent_result("")
        result = await research_node(base_state, mock_ui)

    assert result["research_report"] == ""
    assert result["technical_assessment"] == {}
    mock_mark.assert_called_once_with(base_state["project_dir"], "research")


# ---------------------------------------------------------------------------
# Scope path handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scope_path_sets_cwd_when_dir_exists(
    base_state, repo_dir, mock_ui,
):
    """When scope_path points to an existing subdir, agent cwd is scoped."""
    scoped = repo_dir / "packages" / "core"
    scoped.mkdir(parents=True)
    base_state["scope_path"] = "packages/core"

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent, \
         patch("graft.stages.research.save_artifact"), \
         patch("graft.stages.research.mark_stage_complete"):
        mock_agent.return_value = _agent_result()
        await research_node(base_state, mock_ui)

    assert mock_agent.call_args.kwargs["cwd"] == str(scoped)


@pytest.mark.asyncio
async def test_scope_path_ignored_when_dir_missing(
    base_state, repo_dir, mock_ui,
):
    """When scope_path doesn't exist on disk, cwd falls back to repo_path."""
    base_state["scope_path"] = "nonexistent/subdir"

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent, \
         patch("graft.stages.research.save_artifact"), \
         patch("graft.stages.research.mark_stage_complete"):
        mock_agent.return_value = _agent_result()
        await research_node(base_state, mock_ui)

    assert mock_agent.call_args.kwargs["cwd"] == str(repo_dir)


# ---------------------------------------------------------------------------
# File cleanup
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cleanup_removes_files_from_both_scoped_and_repo_dirs(
    base_state, repo_dir, mock_ui,
):
    """Cleanup removes artifacts from both scoped cwd and repo root."""
    scoped = repo_dir / "packages" / "web"
    scoped.mkdir(parents=True)
    base_state["scope_path"] = "packages/web"

    # Files in both locations
    (scoped / "research_report.md").write_text("scoped report")
    (scoped / "technical_assessment.json").write_text("{}")
    (repo_dir / "research_report.md").write_text("root report")
    (repo_dir / "technical_assessment.json").write_text("{}")

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent, \
         patch("graft.stages.research.save_artifact"), \
         patch("graft.stages.research.mark_stage_complete"):
        mock_agent.return_value = _agent_result()
        await research_node(base_state, mock_ui)

    assert not (scoped / "research_report.md").exists()
    assert not (scoped / "technical_assessment.json").exists()
    assert not (repo_dir / "research_report.md").exists()
    assert not (repo_dir / "technical_assessment.json").exists()


# ---------------------------------------------------------------------------
# Scoped file fallback to repo root
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_files_read_from_repo_root_when_not_in_scoped_dir(
    base_state, repo_dir, mock_ui,
):
    """When scoped dir has no files but repo root does, read from root."""
    scoped = repo_dir / "packages" / "api"
    scoped.mkdir(parents=True)
    base_state["scope_path"] = "packages/api"

    assessment = {"feature_prompt": "test", "open_questions": []}
    (repo_dir / "research_report.md").write_text("root report")
    (repo_dir / "technical_assessment.json").write_text(json.dumps(assessment))

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent, \
         patch("graft.stages.research.save_artifact"), \
         patch("graft.stages.research.mark_stage_complete"):
        mock_agent.return_value = _agent_result()
        result = await research_node(base_state, mock_ui)

    assert result["research_report"] == "root report"
    assert result["technical_assessment"] == assessment


# ---------------------------------------------------------------------------
# Open questions info message
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_open_questions_means_no_info_message(
    base_state, repo_dir, mock_ui,
):
    """When assessment has no open_questions, ui.info is not called."""
    assessment = {"feature_prompt": "test", "open_questions": []}
    (repo_dir / "technical_assessment.json").write_text(json.dumps(assessment))

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent, \
         patch("graft.stages.research.save_artifact"), \
         patch("graft.stages.research.mark_stage_complete"):
        mock_agent.return_value = _agent_result()
        await research_node(base_state, mock_ui)

    mock_ui.info.assert_not_called()


@pytest.mark.asyncio
async def test_multiple_open_questions_count_in_info(
    base_state, repo_dir, mock_ui,
):
    """Info message reports correct count of open questions."""
    assessment = {
        "open_questions": [
            {"question": "Q1", "category": "intent", "recommended_answer": "A1"},
            {"question": "Q2", "category": "edge_case", "recommended_answer": "A2"},
            {"question": "Q3", "category": "preference", "recommended_answer": "A3"},
        ],
    }
    (repo_dir / "technical_assessment.json").write_text(json.dumps(assessment))

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent, \
         patch("graft.stages.research.save_artifact"), \
         patch("graft.stages.research.mark_stage_complete"):
        mock_agent.return_value = _agent_result()
        await research_node(base_state, mock_ui)

    mock_ui.info.assert_called_once()
    assert "3 open question" in mock_ui.info.call_args[0][0]


# ---------------------------------------------------------------------------
# run_agent call arguments
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_agent_called_with_correct_kwargs(base_state, mock_ui):
    """Verify persona, stage, allowed_tools, and max_turns forwarded correctly."""
    base_state["model"] = "claude-sonnet-4-20250514"

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent, \
         patch("graft.stages.research.save_artifact"), \
         patch("graft.stages.research.mark_stage_complete"):
        mock_agent.return_value = _agent_result()
        await research_node(base_state, mock_ui)

    kwargs = mock_agent.call_args.kwargs
    assert kwargs["persona"] == "Staff Software Architect (Feature Specialist)"
    assert kwargs["stage"] == "research"
    assert kwargs["max_turns"] == 30
    assert kwargs["allowed_tools"] == ["Bash", "Read", "Glob", "Grep"]
    assert kwargs["model"] == "claude-sonnet-4-20250514"
    assert kwargs["project_dir"] == base_state["project_dir"]
