"""Tests for graft.stages.research."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graft.agent import AgentResult
from graft.stages.research import research_node


@pytest.fixture
def tmp_repo(tmp_path):
    """Create a temporary repo directory."""
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


@pytest.fixture
def tmp_project(tmp_path):
    """Create a temporary project directory with required structure."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "artifacts").mkdir()
    (project / "logs").mkdir()
    meta = {
        "project_id": "feat_test1234",
        "repo_path": str(tmp_path / "repo"),
        "feature_prompt": "Add dark mode",
        "status": "in_progress",
        "stages_completed": [],
    }
    (project / "metadata.json").write_text(json.dumps(meta))
    return project


@pytest.fixture
def ui():
    """Return a mock UI with all required methods."""
    mock = MagicMock()
    mock.stage_start = MagicMock()
    mock.stage_done = MagicMock()
    mock.info = MagicMock()
    mock.error = MagicMock()
    return mock


@pytest.fixture
def base_state(tmp_repo, tmp_project):
    """Minimal valid FeatureState dict for research_node."""
    return {
        "repo_path": str(tmp_repo),
        "project_dir": str(tmp_project),
        "feature_prompt": "Add dark mode",
        "codebase_profile": {"language": "python", "framework": "flask"},
    }


def _make_agent_result(text: str = "fallback text") -> AgentResult:
    return AgentResult(
        text=text, tool_calls=[], raw_messages=[], elapsed_seconds=1.0, turns_used=5
    )


# ── 1. Happy path: reads technical_assessment.json, saves artifacts, returns correct state ──


@pytest.mark.asyncio
@patch("graft.stages.research.run_agent", new_callable=AsyncMock)
async def test_research_node_happy_path(
    mock_run_agent, base_state, tmp_repo, tmp_project, ui
):
    """research_node reads agent-written files and returns expected state dict."""
    assessment = {
        "feature_prompt": "Add dark mode",
        "reusable_components": [],
        "new_artifacts_needed": [],
        "pattern_to_follow": "src/features/settings",
        "edge_cases": ["Theme flicker on load"],
        "integration_points": ["navbar"],
        "open_questions": [
            {
                "question": "Default theme?",
                "category": "preference",
                "recommended_answer": "system",
            }
        ],
    }
    report_content = "# Research Report\n\nDark mode feasibility analysis."

    # Simulate agent writing files to repo
    (tmp_repo / "technical_assessment.json").write_text(json.dumps(assessment))
    (tmp_repo / "research_report.md").write_text(report_content)

    mock_run_agent.return_value = _make_agent_result()

    result = await research_node(base_state, ui)

    assert result["technical_assessment"] == assessment
    assert result["research_report"] == report_content
    assert result["current_stage"] == "research"

    # Artifacts saved to project dir
    saved_report = (Path(tmp_project) / "artifacts" / "research_report.md").read_text()
    assert saved_report == report_content

    saved_assessment = json.loads(
        (Path(tmp_project) / "artifacts" / "technical_assessment.json").read_text()
    )
    assert saved_assessment == assessment


# ── 2. Fallback to result.text when research_report.md not found ──


@pytest.mark.asyncio
@patch("graft.stages.research.run_agent", new_callable=AsyncMock)
async def test_fallback_to_result_text(
    mock_run_agent, base_state, tmp_repo, tmp_project, ui
):
    """When research_report.md does not exist, research_report falls back to result.text."""
    mock_run_agent.return_value = _make_agent_result(text="Agent raw output for report")

    # No research_report.md on disk — only assessment
    assessment = {"feature_prompt": "Add dark mode", "open_questions": []}
    (tmp_repo / "technical_assessment.json").write_text(json.dumps(assessment))

    result = await research_node(base_state, ui)

    assert result["research_report"] == "Agent raw output for report"
    saved = (Path(tmp_project) / "artifacts" / "research_report.md").read_text()
    assert saved == "Agent raw output for report"


# ── 3. Fallback to empty dict on malformed JSON ──


@pytest.mark.asyncio
@patch("graft.stages.research.run_agent", new_callable=AsyncMock)
async def test_malformed_json_fallback(mock_run_agent, base_state, tmp_repo, ui):
    """Malformed technical_assessment.json results in empty dict and ui.error call."""
    mock_run_agent.return_value = _make_agent_result()

    (tmp_repo / "technical_assessment.json").write_text("{invalid json!!!}")
    (tmp_repo / "research_report.md").write_text("report")

    result = await research_node(base_state, ui)

    assert result["technical_assessment"] == {}
    ui.error.assert_called_once_with(
        "Failed to parse technical_assessment.json from agent output."
    )


# ── 4. scope_path is respected for working directory ──


@pytest.mark.asyncio
@patch("graft.stages.research.run_agent", new_callable=AsyncMock)
async def test_scope_path_sets_cwd(
    mock_run_agent, base_state, tmp_repo, tmp_project, ui
):
    """When scope_path is set, the agent cwd is repo_path / scope_path."""
    scoped_dir = tmp_repo / "packages" / "frontend"
    scoped_dir.mkdir(parents=True)

    base_state["scope_path"] = "packages/frontend"

    assessment = {"feature_prompt": "scoped", "open_questions": []}
    (scoped_dir / "technical_assessment.json").write_text(json.dumps(assessment))
    (scoped_dir / "research_report.md").write_text("scoped report")

    mock_run_agent.return_value = _make_agent_result()

    result = await research_node(base_state, ui)

    # Verify run_agent was called with the scoped directory as cwd
    call_kwargs = mock_run_agent.call_args[1]
    assert call_kwargs["cwd"] == str(scoped_dir)

    assert result["technical_assessment"] == assessment
    assert result["research_report"] == "scoped report"


@pytest.mark.asyncio
@patch("graft.stages.research.run_agent", new_callable=AsyncMock)
async def test_scope_path_nonexistent_falls_back_to_repo(
    mock_run_agent, base_state, tmp_repo, ui
):
    """When scope_path dir doesn't exist, cwd falls back to repo_path."""
    base_state["scope_path"] = "nonexistent/path"
    mock_run_agent.return_value = _make_agent_result()

    # Files at repo root
    (tmp_repo / "technical_assessment.json").write_text(json.dumps({}))
    (tmp_repo / "research_report.md").write_text("root report")

    await research_node(base_state, ui)

    call_kwargs = mock_run_agent.call_args[1]
    assert call_kwargs["cwd"] == str(tmp_repo)


# ── 5. Constraints are appended to prompt ──


@pytest.mark.asyncio
@patch("graft.stages.research.run_agent", new_callable=AsyncMock)
async def test_constraints_appended_to_prompt(mock_run_agent, base_state, tmp_repo, ui):
    """Constraints list is joined with semicolons and appended to the agent prompt."""
    base_state["constraints"] = ["No new dependencies", "Must support IE11"]
    mock_run_agent.return_value = _make_agent_result()

    (tmp_repo / "technical_assessment.json").write_text(json.dumps({}))
    (tmp_repo / "research_report.md").write_text("report")

    await research_node(base_state, ui)

    call_kwargs = mock_run_agent.call_args[1]
    user_prompt = call_kwargs["user_prompt"]
    assert "CONSTRAINTS: No new dependencies; Must support IE11" in user_prompt


@pytest.mark.asyncio
@patch("graft.stages.research.run_agent", new_callable=AsyncMock)
async def test_no_constraints_omits_section(mock_run_agent, base_state, tmp_repo, ui):
    """When constraints is empty, no CONSTRAINTS section appears in the prompt."""
    base_state["constraints"] = []
    mock_run_agent.return_value = _make_agent_result()

    (tmp_repo / "technical_assessment.json").write_text(json.dumps({}))
    (tmp_repo / "research_report.md").write_text("report")

    await research_node(base_state, ui)

    call_kwargs = mock_run_agent.call_args[1]
    assert "CONSTRAINTS" not in call_kwargs["user_prompt"]


# ── 6. open_questions count is logged via ui.info ──


@pytest.mark.asyncio
@patch("graft.stages.research.run_agent", new_callable=AsyncMock)
async def test_open_questions_logged(mock_run_agent, base_state, tmp_repo, ui):
    """When open_questions exist, ui.info is called with the count."""
    assessment = {
        "open_questions": [
            {"question": "q1", "category": "intent", "recommended_answer": "a1"},
            {"question": "q2", "category": "preference", "recommended_answer": "a2"},
            {"question": "q3", "category": "edge_case", "recommended_answer": "a3"},
        ]
    }
    (tmp_repo / "technical_assessment.json").write_text(json.dumps(assessment))
    (tmp_repo / "research_report.md").write_text("report")
    mock_run_agent.return_value = _make_agent_result()

    await research_node(base_state, ui)

    ui.info.assert_called_once_with(
        "Research identified 3 open question(s) for the Grill phase."
    )


@pytest.mark.asyncio
@patch("graft.stages.research.run_agent", new_callable=AsyncMock)
async def test_no_open_questions_no_info_call(mock_run_agent, base_state, tmp_repo, ui):
    """When no open_questions, ui.info is not called."""
    assessment = {"open_questions": []}
    (tmp_repo / "technical_assessment.json").write_text(json.dumps(assessment))
    (tmp_repo / "research_report.md").write_text("report")
    mock_run_agent.return_value = _make_agent_result()

    await research_node(base_state, ui)

    ui.info.assert_not_called()


# ── 7. Temp file cleanup removes both scoped and root-level files ──


@pytest.mark.asyncio
@patch("graft.stages.research.run_agent", new_callable=AsyncMock)
async def test_cleanup_removes_repo_level_files(
    mock_run_agent, base_state, tmp_repo, ui
):
    """Agent-generated files in the repo root are cleaned up after processing."""
    (tmp_repo / "technical_assessment.json").write_text(json.dumps({}))
    (tmp_repo / "research_report.md").write_text("report")
    mock_run_agent.return_value = _make_agent_result()

    await research_node(base_state, ui)

    assert not (tmp_repo / "technical_assessment.json").exists()
    assert not (tmp_repo / "research_report.md").exists()


@pytest.mark.asyncio
@patch("graft.stages.research.run_agent", new_callable=AsyncMock)
async def test_cleanup_removes_scoped_and_root_files(
    mock_run_agent, base_state, tmp_repo, ui
):
    """With scope_path, files in both the scoped dir and repo root are cleaned up."""
    scoped_dir = tmp_repo / "packages" / "frontend"
    scoped_dir.mkdir(parents=True)
    base_state["scope_path"] = "packages/frontend"

    # Files in both locations
    (scoped_dir / "technical_assessment.json").write_text(json.dumps({}))
    (scoped_dir / "research_report.md").write_text("scoped report")
    (tmp_repo / "technical_assessment.json").write_text(json.dumps({}))
    (tmp_repo / "research_report.md").write_text("root report")

    mock_run_agent.return_value = _make_agent_result()

    await research_node(base_state, ui)

    # All four files cleaned up
    assert not (scoped_dir / "technical_assessment.json").exists()
    assert not (scoped_dir / "research_report.md").exists()
    assert not (tmp_repo / "technical_assessment.json").exists()
    assert not (tmp_repo / "research_report.md").exists()


# ── 8. mark_stage_complete and ui.stage_done are called ──


@pytest.mark.asyncio
@patch("graft.stages.research.run_agent", new_callable=AsyncMock)
async def test_mark_stage_complete_called(
    mock_run_agent, base_state, tmp_repo, tmp_project, ui
):
    """mark_stage_complete records 'research' in metadata.json."""
    (tmp_repo / "technical_assessment.json").write_text(json.dumps({}))
    (tmp_repo / "research_report.md").write_text("report")
    mock_run_agent.return_value = _make_agent_result()

    await research_node(base_state, ui)

    meta = json.loads((Path(tmp_project) / "metadata.json").read_text())
    assert "research" in meta["stages_completed"]


@pytest.mark.asyncio
@patch("graft.stages.research.run_agent", new_callable=AsyncMock)
async def test_ui_stage_lifecycle(mock_run_agent, base_state, tmp_repo, ui):
    """ui.stage_start and ui.stage_done are called with 'research'."""
    (tmp_repo / "technical_assessment.json").write_text(json.dumps({}))
    (tmp_repo / "research_report.md").write_text("report")
    mock_run_agent.return_value = _make_agent_result()

    await research_node(base_state, ui)

    ui.stage_start.assert_called_once_with("research")
    ui.stage_done.assert_called_once_with("research")


# ── 9. run_agent called with expected parameters ──


@pytest.mark.asyncio
@patch("graft.stages.research.run_agent", new_callable=AsyncMock)
async def test_run_agent_call_parameters(mock_run_agent, base_state, tmp_repo, ui):
    """run_agent is invoked with the correct persona, stage, and tools."""
    base_state["model"] = "sonnet"
    (tmp_repo / "technical_assessment.json").write_text(json.dumps({}))
    (tmp_repo / "research_report.md").write_text("report")
    mock_run_agent.return_value = _make_agent_result()

    await research_node(base_state, ui)

    mock_run_agent.assert_called_once()
    kwargs = mock_run_agent.call_args[1]
    assert kwargs["persona"] == "Staff Software Architect (Feature Specialist)"
    assert kwargs["stage"] == "research"
    assert kwargs["model"] == "sonnet"
    assert kwargs["max_turns"] == 30
    assert kwargs["allowed_tools"] == ["Bash", "Read", "Glob", "Grep"]
    assert kwargs["project_dir"] == str(base_state["project_dir"])
    assert "FEATURE: Add dark mode" in kwargs["user_prompt"]
    assert "CODEBASE PROFILE" in kwargs["user_prompt"]


# ── 10. No assessment file produces empty dict ──


@pytest.mark.asyncio
@patch("graft.stages.research.run_agent", new_callable=AsyncMock)
async def test_no_assessment_file_returns_empty_dict(
    mock_run_agent, base_state, tmp_repo, ui
):
    """When no technical_assessment.json exists anywhere, result is empty dict."""
    mock_run_agent.return_value = _make_agent_result(text="just text")

    # No files on disk at all
    result = await research_node(base_state, ui)

    assert result["technical_assessment"] == {}
    assert result["research_report"] == "just text"


# ── 11. Scoped dir file takes precedence over repo root file ──


@pytest.mark.asyncio
@patch("graft.stages.research.run_agent", new_callable=AsyncMock)
async def test_scoped_files_take_precedence(mock_run_agent, base_state, tmp_repo, ui):
    """Files in the scoped directory are preferred over repo root ones."""
    scoped_dir = tmp_repo / "src" / "app"
    scoped_dir.mkdir(parents=True)
    base_state["scope_path"] = "src/app"

    scoped_assessment = {"source": "scoped", "open_questions": []}
    root_assessment = {"source": "root", "open_questions": []}

    (scoped_dir / "technical_assessment.json").write_text(json.dumps(scoped_assessment))
    (scoped_dir / "research_report.md").write_text("scoped report")
    (tmp_repo / "technical_assessment.json").write_text(json.dumps(root_assessment))
    (tmp_repo / "research_report.md").write_text("root report")

    mock_run_agent.return_value = _make_agent_result()

    result = await research_node(base_state, ui)

    # Scoped versions should win
    assert result["technical_assessment"]["source"] == "scoped"
    assert result["research_report"] == "scoped report"


# ── 12. Missing optional state keys use defaults ──


@pytest.mark.asyncio
@patch("graft.stages.research.run_agent", new_callable=AsyncMock)
async def test_missing_optional_state_keys(mock_run_agent, tmp_repo, tmp_project, ui):
    """State without feature_prompt, codebase_profile, scope_path, constraints works."""
    state = {
        "repo_path": str(tmp_repo),
        "project_dir": str(tmp_project),
    }
    (tmp_repo / "technical_assessment.json").write_text(json.dumps({}))
    (tmp_repo / "research_report.md").write_text("report")
    mock_run_agent.return_value = _make_agent_result()

    result = await research_node(state, ui)

    assert result["current_stage"] == "research"
    kwargs = mock_run_agent.call_args[1]
    # feature_prompt defaults to empty string
    assert (
        "FEATURE: \n" in kwargs["user_prompt"] or "FEATURE: " in kwargs["user_prompt"]
    )
    # No CONSTRAINTS section
    assert "CONSTRAINTS" not in kwargs["user_prompt"]
