"""Tests for graft.stages.research — research_node function."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graft.agent import AgentResult
from graft.stages.research import research_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(tmp_path: Path, **overrides) -> dict:
    """Build a minimal FeatureState dict backed by tmp_path."""
    project_dir = tmp_path / "project"
    (project_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (project_dir / "logs").mkdir(parents=True, exist_ok=True)
    # metadata.json needed by mark_stage_complete
    meta = {
        "project_id": "feat_test1234",
        "repo_path": str(tmp_path / "repo"),
        "feature_prompt": "Add a widget feature",
        "created_at": "2025-01-01T00:00:00Z",
        "status": "in_progress",
        "stages_completed": [],
    }
    (project_dir / "metadata.json").write_text(json.dumps(meta))

    repo_path = tmp_path / "repo"
    repo_path.mkdir(parents=True, exist_ok=True)

    state: dict = {
        "repo_path": str(repo_path),
        "project_dir": str(project_dir),
        "feature_prompt": "Add a widget feature",
        "codebase_profile": {"language": "python", "framework": "flask"},
    }
    state.update(overrides)
    return state


def _make_ui() -> MagicMock:
    """Return a mock UI object that records calls."""
    ui = MagicMock()
    ui.stage_start = MagicMock()
    ui.stage_done = MagicMock()
    ui.stage_log = MagicMock()
    ui.info = MagicMock()
    ui.error = MagicMock()
    return ui


def _fake_agent_result(text: str = "Agent research output") -> AgentResult:
    return AgentResult(
        text=text,
        tool_calls=[],
        raw_messages=[],
        elapsed_seconds=5.0,
        turns_used=3,
    )


SAMPLE_ASSESSMENT = {
    "feature_prompt": "Add a widget feature",
    "reusable_components": [
        {"path": "src/components/button.py", "reason": "existing UI component"},
    ],
    "new_artifacts_needed": [
        {"type": "component", "name": "Widget", "description": "New widget"},
    ],
    "pattern_to_follow": "src/features/dashboard/",
    "edge_cases": ["concurrent access"],
    "integration_points": ["notification service"],
    "open_questions": [
        {
            "question": "Should the widget auto-refresh?",
            "category": "intent",
            "recommended_answer": "Yes, every 30s based on similar patterns.",
        },
        {
            "question": "Who can create widgets?",
            "category": "preference",
            "recommended_answer": "Admin users only, matching existing RBAC.",
        },
    ],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_reads_both_files(tmp_path: Path):
    """Agent writes research_report.md and technical_assessment.json;
    node reads them, saves artifacts, cleans up, and returns correct state."""
    state = _make_state(tmp_path)
    ui = _make_ui()

    repo_path = Path(state["repo_path"])
    report_content = "# Research Report\nThe codebase uses Flask with SQLAlchemy."
    (repo_path / "research_report.md").write_text(report_content)
    (repo_path / "technical_assessment.json").write_text(
        json.dumps(SAMPLE_ASSESSMENT, indent=2)
    )

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = _fake_agent_result()
        result = await research_node(state, ui)

    # Return value
    assert result["research_report"] == report_content
    assert result["technical_assessment"] == SAMPLE_ASSESSMENT
    assert result["current_stage"] == "research"

    # Artifacts saved to project_dir
    project_dir = Path(state["project_dir"])
    saved_report = (project_dir / "artifacts" / "research_report.md").read_text()
    assert saved_report == report_content

    saved_assessment = json.loads(
        (project_dir / "artifacts" / "technical_assessment.json").read_text()
    )
    assert saved_assessment == SAMPLE_ASSESSMENT

    # Temp files cleaned up from repo_path
    assert not (repo_path / "research_report.md").exists()
    assert not (repo_path / "technical_assessment.json").exists()

    # UI lifecycle
    ui.stage_start.assert_called_once_with("research")
    ui.stage_done.assert_called_once_with("research")


@pytest.mark.asyncio
async def test_fallback_when_output_files_missing(tmp_path: Path):
    """When agent produces no output files, research_report falls back to
    result.text and technical_assessment is empty."""
    state = _make_state(tmp_path)
    ui = _make_ui()

    agent_text = "Fallback: agent could not write files."

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = _fake_agent_result(text=agent_text)
        result = await research_node(state, ui)

    # Falls back to agent text for report
    assert result["research_report"] == agent_text
    # Empty dict for assessment
    assert result["technical_assessment"] == {}

    # Artifacts still saved (empty assessment, fallback report)
    project_dir = Path(state["project_dir"])
    assert (project_dir / "artifacts" / "research_report.md").read_text() == agent_text
    saved_assessment = json.loads(
        (project_dir / "artifacts" / "technical_assessment.json").read_text()
    )
    assert saved_assessment == {}


@pytest.mark.asyncio
async def test_prompt_includes_codebase_profile_and_feature(tmp_path: Path):
    """The user_prompt passed to run_agent includes codebase_profile and
    feature_prompt from state."""
    state = _make_state(
        tmp_path,
        feature_prompt="Build an admin dashboard",
        codebase_profile={"language": "typescript", "framework": "next.js"},
    )
    ui = _make_ui()

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = _fake_agent_result()
        await research_node(state, ui)

    call_kwargs = mock_agent.call_args.kwargs
    prompt = call_kwargs["user_prompt"]

    assert "Build an admin dashboard" in prompt
    assert "typescript" in prompt
    assert "next.js" in prompt
    # repo_path should be in the prompt
    assert state["repo_path"] in prompt


@pytest.mark.asyncio
async def test_open_questions_extracted_and_ui_informed(tmp_path: Path):
    """open_questions are extracted from technical_assessment and the UI
    is told how many were found."""
    state = _make_state(tmp_path)
    ui = _make_ui()

    repo_path = Path(state["repo_path"])
    (repo_path / "research_report.md").write_text("Report")
    (repo_path / "technical_assessment.json").write_text(json.dumps(SAMPLE_ASSESSMENT))

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = _fake_agent_result()
        result = await research_node(state, ui)

    questions = result["technical_assessment"]["open_questions"]
    assert len(questions) == 2
    assert questions[0]["question"] == "Should the widget auto-refresh?"

    # UI informed about the count
    ui.info.assert_called_once()
    info_msg = ui.info.call_args[0][0]
    assert "2" in info_msg
    assert "open question" in info_msg.lower()


@pytest.mark.asyncio
async def test_stage_completion_marked(tmp_path: Path):
    """mark_stage_complete records 'research' in metadata.json."""
    state = _make_state(tmp_path)
    ui = _make_ui()

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = _fake_agent_result()
        await research_node(state, ui)

    meta = json.loads((Path(state["project_dir"]) / "metadata.json").read_text())
    assert "research" in meta["stages_completed"]


@pytest.mark.asyncio
async def test_cleanup_removes_files_from_repo_path(tmp_path: Path):
    """After research_node finishes, temp files in repo_path are deleted."""
    state = _make_state(tmp_path)
    ui = _make_ui()

    repo_path = Path(state["repo_path"])
    (repo_path / "research_report.md").write_text("temp report")
    (repo_path / "technical_assessment.json").write_text("{}")

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = _fake_agent_result()
        await research_node(state, ui)

    assert not (repo_path / "research_report.md").exists()
    assert not (repo_path / "technical_assessment.json").exists()


@pytest.mark.asyncio
async def test_invalid_json_in_assessment(tmp_path: Path):
    """Malformed technical_assessment.json is handled gracefully with
    an empty dict and a ui.error call."""
    state = _make_state(tmp_path)
    ui = _make_ui()

    repo_path = Path(state["repo_path"])
    (repo_path / "research_report.md").write_text("Report")
    (repo_path / "technical_assessment.json").write_text("NOT VALID JSON {{{")

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = _fake_agent_result()
        result = await research_node(state, ui)

    assert result["technical_assessment"] == {}
    ui.error.assert_called_once()
    assert (
        "parse" in ui.error.call_args[0][0].lower()
        or "json" in ui.error.call_args[0][0].lower()
    )


@pytest.mark.asyncio
async def test_scope_path_sets_cwd(tmp_path: Path):
    """When scope_path is set and exists, run_agent receives the scoped
    directory as its cwd."""
    state = _make_state(tmp_path, scope_path="src/features")
    ui = _make_ui()

    scoped_dir = Path(state["repo_path"]) / "src" / "features"
    scoped_dir.mkdir(parents=True, exist_ok=True)

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = _fake_agent_result()
        await research_node(state, ui)

    call_kwargs = mock_agent.call_args.kwargs
    assert call_kwargs["cwd"] == str(scoped_dir)


@pytest.mark.asyncio
async def test_scope_path_nonexistent_falls_back(tmp_path: Path):
    """When scope_path doesn't exist on disk, cwd falls back to repo_path."""
    state = _make_state(tmp_path, scope_path="nonexistent/path")
    ui = _make_ui()

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = _fake_agent_result()
        await research_node(state, ui)

    call_kwargs = mock_agent.call_args.kwargs
    assert call_kwargs["cwd"] == state["repo_path"]


@pytest.mark.asyncio
async def test_constraints_included_in_prompt(tmp_path: Path):
    """Constraints from state are joined and included in the user_prompt."""
    state = _make_state(
        tmp_path,
        constraints=["must use PostgreSQL", "no new npm packages"],
    )
    ui = _make_ui()

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = _fake_agent_result()
        await research_node(state, ui)

    prompt = mock_agent.call_args.kwargs["user_prompt"]
    assert "must use PostgreSQL" in prompt
    assert "no new npm packages" in prompt
    assert "CONSTRAINTS" in prompt


@pytest.mark.asyncio
async def test_scoped_dir_files_preferred_over_repo_root(tmp_path: Path):
    """When scope_path is active and files exist there, they take priority
    over files in the repo root."""
    state = _make_state(tmp_path, scope_path="src")
    ui = _make_ui()

    repo_path = Path(state["repo_path"])
    scoped_dir = repo_path / "src"
    scoped_dir.mkdir(parents=True, exist_ok=True)

    scoped_report = "# Scoped research report"
    scoped_assessment = {"feature_prompt": "scoped", "open_questions": []}

    # Files in scoped dir (should be preferred)
    (scoped_dir / "research_report.md").write_text(scoped_report)
    (scoped_dir / "technical_assessment.json").write_text(json.dumps(scoped_assessment))
    # Files also in repo root (should NOT be used)
    (repo_path / "research_report.md").write_text("root report")
    (repo_path / "technical_assessment.json").write_text(
        json.dumps({"feature_prompt": "root"})
    )

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = _fake_agent_result()
        result = await research_node(state, ui)

    assert result["research_report"] == scoped_report
    assert result["technical_assessment"]["feature_prompt"] == "scoped"

    # Cleanup removes from both locations
    assert not (scoped_dir / "research_report.md").exists()
    assert not (scoped_dir / "technical_assessment.json").exists()
    assert not (repo_path / "research_report.md").exists()
    assert not (repo_path / "technical_assessment.json").exists()


@pytest.mark.asyncio
async def test_no_open_questions_skips_ui_info(tmp_path: Path):
    """When technical_assessment has no open_questions, ui.info is not called."""
    state = _make_state(tmp_path)
    ui = _make_ui()

    assessment_no_questions = {
        "feature_prompt": "Simple feature",
        "open_questions": [],
    }
    repo_path = Path(state["repo_path"])
    (repo_path / "research_report.md").write_text("Report")
    (repo_path / "technical_assessment.json").write_text(
        json.dumps(assessment_no_questions)
    )

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = _fake_agent_result()
        await research_node(state, ui)

    ui.info.assert_not_called()


@pytest.mark.asyncio
async def test_model_passed_to_agent(tmp_path: Path):
    """The model parameter from state is forwarded to run_agent."""
    state = _make_state(tmp_path, model="claude-sonnet-4-20250514")
    ui = _make_ui()

    with patch("graft.stages.research.run_agent", new_callable=AsyncMock) as mock_agent:
        mock_agent.return_value = _fake_agent_result()
        await research_node(state, ui)

    call_kwargs = mock_agent.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-20250514"
    assert call_kwargs["max_turns"] == 30
    assert call_kwargs["stage"] == "research"
