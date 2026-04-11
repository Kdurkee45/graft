"""Tests for graft.stages.research."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from graft.agent import AgentResult
from graft.stages.research import research_node


def _make_state(tmp_path: Path, **overrides) -> dict:
    """Build a minimal FeatureState dict rooted in *tmp_path*."""
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    (project / "artifacts").mkdir(parents=True, exist_ok=True)
    (project / "metadata.json").write_text(json.dumps({
        "stages_completed": [],
    }))

    base: dict = {
        "repo_path": str(repo),
        "project_dir": str(project),
        "feature_prompt": "Add a CSV export endpoint",
        "codebase_profile": {"framework": "fastapi"},
    }
    base.update(overrides)
    return base


def _make_ui() -> MagicMock:
    """Return a mock UI with the methods research_node calls."""
    ui = MagicMock()
    ui.stage_start = MagicMock()
    ui.stage_done = MagicMock()
    ui.info = MagicMock()
    ui.error = MagicMock()
    return ui


# ── run_agent call correctness ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_agent_called_with_correct_persona_and_tools(tmp_path):
    """research_node passes the right persona and read-only tool list."""
    state = _make_state(tmp_path)
    ui = _make_ui()
    mock_result = AgentResult(text="fallback text")

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock, return_value=mock_result) as mock_run:
        await research_node(state, ui)

    mock_run.assert_awaited_once()
    kwargs = mock_run.call_args.kwargs
    assert kwargs["persona"] == "Staff Software Architect (Feature Specialist)"
    assert kwargs["allowed_tools"] == ["Bash", "Read", "Glob", "Grep"]
    assert kwargs["stage"] == "research"
    assert kwargs["max_turns"] == 30
    assert kwargs["cwd"] == state["repo_path"]
    assert kwargs["project_dir"] == state["project_dir"]


# ── technical_assessment.json parsing ────────────────────────────────────


@pytest.mark.asyncio
async def test_reads_and_parses_technical_assessment(tmp_path):
    """research_node reads technical_assessment.json written by the agent."""
    state = _make_state(tmp_path)
    ui = _make_ui()

    assessment = {
        "feature_prompt": "CSV export",
        "reusable_components": [],
        "new_artifacts_needed": [],
        "open_questions": [
            {"question": "Use streaming?", "category": "intent", "recommended_answer": "Yes"},
        ],
    }
    repo = Path(state["repo_path"])
    (repo / "technical_assessment.json").write_text(json.dumps(assessment))
    (repo / "research_report.md").write_text("# Research Report\nDetails here.")

    mock_result = AgentResult(text="fallback")

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock, return_value=mock_result):
        result = await research_node(state, ui)

    assert result["technical_assessment"] == assessment
    assert result["research_report"] == "# Research Report\nDetails here."
    assert result["current_stage"] == "research"


# ── fallback to result.text ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_falls_back_to_result_text_when_report_missing(tmp_path):
    """When research_report.md is not written, research_node uses result.text."""
    state = _make_state(tmp_path)
    ui = _make_ui()

    mock_result = AgentResult(text="Agent produced this text instead.")

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock, return_value=mock_result):
        result = await research_node(state, ui)

    assert result["research_report"] == "Agent produced this text instead."
    # With no assessment file, should get empty dict
    assert result["technical_assessment"] == {}


# ── open_questions ui.info logging ───────────────────────────────────────


@pytest.mark.asyncio
async def test_logs_open_questions_count(tmp_path):
    """research_node logs the count of open questions via ui.info."""
    state = _make_state(tmp_path)
    ui = _make_ui()

    assessment = {
        "open_questions": [
            {"question": "Q1", "category": "intent", "recommended_answer": "A1"},
            {"question": "Q2", "category": "edge_case", "recommended_answer": "A2"},
            {"question": "Q3", "category": "preference", "recommended_answer": "A3"},
        ],
    }
    repo = Path(state["repo_path"])
    (repo / "technical_assessment.json").write_text(json.dumps(assessment))

    mock_result = AgentResult(text="")

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock, return_value=mock_result):
        await research_node(state, ui)

    ui.info.assert_called_once_with(
        "Research identified 3 open question(s) for the Grill phase."
    )


@pytest.mark.asyncio
async def test_no_info_logged_when_no_open_questions(tmp_path):
    """ui.info is NOT called when there are zero open questions."""
    state = _make_state(tmp_path)
    ui = _make_ui()

    assessment = {"open_questions": []}
    repo = Path(state["repo_path"])
    (repo / "technical_assessment.json").write_text(json.dumps(assessment))

    mock_result = AgentResult(text="")

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock, return_value=mock_result):
        await research_node(state, ui)

    ui.info.assert_not_called()


# ── file cleanup ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cleans_up_generated_files(tmp_path):
    """research_node removes research_report.md and technical_assessment.json
    from both repo_path and research_cwd after processing."""
    state = _make_state(tmp_path)
    ui = _make_ui()

    repo = Path(state["repo_path"])
    (repo / "research_report.md").write_text("report")
    (repo / "technical_assessment.json").write_text("{}")

    mock_result = AgentResult(text="")

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock, return_value=mock_result):
        await research_node(state, ui)

    assert not (repo / "research_report.md").exists()
    assert not (repo / "technical_assessment.json").exists()


# ── scope_path handling ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_respects_scope_path(tmp_path):
    """When scope_path is set and exists, run_agent cwd is the scoped dir."""
    state = _make_state(tmp_path, scope_path="backend/api")
    ui = _make_ui()

    scoped = Path(state["repo_path"]) / "backend" / "api"
    scoped.mkdir(parents=True)

    mock_result = AgentResult(text="scoped result")

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock, return_value=mock_result) as mock_run:
        await research_node(state, ui)

    kwargs = mock_run.call_args.kwargs
    assert kwargs["cwd"] == str(scoped)


@pytest.mark.asyncio
async def test_scope_path_nonexistent_falls_back_to_repo(tmp_path):
    """When scope_path directory doesn't exist, cwd stays at repo_path."""
    state = _make_state(tmp_path, scope_path="nonexistent/path")
    ui = _make_ui()

    mock_result = AgentResult(text="fallback")

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock, return_value=mock_result) as mock_run:
        await research_node(state, ui)

    kwargs = mock_run.call_args.kwargs
    assert kwargs["cwd"] == state["repo_path"]


# ── JSONDecodeError handling ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_handles_json_decode_error_in_assessment(tmp_path):
    """Malformed JSON in technical_assessment.json triggers ui.error and
    returns an empty dict for technical_assessment."""
    state = _make_state(tmp_path)
    ui = _make_ui()

    repo = Path(state["repo_path"])
    (repo / "technical_assessment.json").write_text("NOT VALID JSON {{{")

    mock_result = AgentResult(text="report text")

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock, return_value=mock_result):
        result = await research_node(state, ui)

    ui.error.assert_called_once_with(
        "Failed to parse technical_assessment.json from agent output."
    )
    assert result["technical_assessment"] == {}


# ── stage lifecycle ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stage_start_and_done_called(tmp_path):
    """research_node calls ui.stage_start and ui.stage_done with 'research'."""
    state = _make_state(tmp_path)
    ui = _make_ui()
    mock_result = AgentResult(text="")

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock, return_value=mock_result):
        await research_node(state, ui)

    ui.stage_start.assert_called_once_with("research")
    ui.stage_done.assert_called_once_with("research")


# ── artifacts saved correctly ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_artifacts_saved_to_project_dir(tmp_path):
    """research_node writes both artifacts into project_dir/artifacts/."""
    state = _make_state(tmp_path)
    ui = _make_ui()

    assessment = {"open_questions": []}
    repo = Path(state["repo_path"])
    (repo / "technical_assessment.json").write_text(json.dumps(assessment))
    (repo / "research_report.md").write_text("# Full Report")

    mock_result = AgentResult(text="")

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock, return_value=mock_result):
        await research_node(state, ui)

    project_dir = Path(state["project_dir"])
    report_artifact = project_dir / "artifacts" / "research_report.md"
    assessment_artifact = project_dir / "artifacts" / "technical_assessment.json"

    assert report_artifact.exists()
    assert report_artifact.read_text() == "# Full Report"
    assert assessment_artifact.exists()
    assert json.loads(assessment_artifact.read_text()) == assessment


# ── constraints included in prompt ───────────────────────────────────────


@pytest.mark.asyncio
async def test_constraints_included_in_prompt(tmp_path):
    """When constraints are provided, they appear in the user_prompt."""
    state = _make_state(tmp_path, constraints=["No new dependencies", "Must be async"])
    ui = _make_ui()
    mock_result = AgentResult(text="")

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock, return_value=mock_result) as mock_run:
        await research_node(state, ui)

    user_prompt = mock_run.call_args.kwargs["user_prompt"]
    assert "No new dependencies" in user_prompt
    assert "Must be async" in user_prompt


# ── scoped dir file lookup ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reads_files_from_scoped_dir_first(tmp_path):
    """When scope_path is set, files in the scoped dir take priority over repo root."""
    state = _make_state(tmp_path, scope_path="services")
    ui = _make_ui()

    repo = Path(state["repo_path"])
    scoped = repo / "services"
    scoped.mkdir()

    # Put files in scoped dir (should be preferred)
    scoped_assessment = {"open_questions": [], "source": "scoped"}
    (scoped / "technical_assessment.json").write_text(json.dumps(scoped_assessment))
    (scoped / "research_report.md").write_text("Scoped report")

    # Also put files in repo root (should NOT be used)
    root_assessment = {"open_questions": [], "source": "root"}
    (repo / "technical_assessment.json").write_text(json.dumps(root_assessment))
    (repo / "research_report.md").write_text("Root report")

    mock_result = AgentResult(text="")

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock, return_value=mock_result):
        result = await research_node(state, ui)

    assert result["technical_assessment"]["source"] == "scoped"
    assert result["research_report"] == "Scoped report"


# ── model passthrough ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_model_passed_through_to_run_agent(tmp_path):
    """The model from state is forwarded to run_agent."""
    state = _make_state(tmp_path, model="claude-sonnet-4-20250514")
    ui = _make_ui()
    mock_result = AgentResult(text="")

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock, return_value=mock_result) as mock_run:
        await research_node(state, ui)

    assert mock_run.call_args.kwargs["model"] == "claude-sonnet-4-20250514"
