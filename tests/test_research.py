"""Tests for graft.stages.research — the Research stage node."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from graft.agent import AgentResult
from graft.stages.research import SYSTEM_PROMPT, research_node
from graft.state import FeatureState
from graft.ui import UI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ui() -> UI:
    """Return a real UI instance with verbose off (no console output)."""
    return UI(auto_approve=True, verbose=False)


def _base_state(tmp_path: Path, **overrides) -> FeatureState:
    """Return a minimal FeatureState suitable for research_node."""
    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    # metadata.json required by mark_stage_complete
    meta = {"stages_completed": []}
    (project_dir / "metadata.json").write_text(json.dumps(meta))

    repo_path = tmp_path / "repo"
    repo_path.mkdir(parents=True, exist_ok=True)

    state: FeatureState = {
        "repo_path": str(repo_path),
        "project_dir": str(project_dir),
        "feature_prompt": "Add dark mode toggle",
        "codebase_profile": {"framework": "next.js", "language": "typescript"},
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


SAMPLE_ASSESSMENT = {
    "feature_prompt": "Add dark mode toggle",
    "reusable_components": [{"path": "src/ui/Button.tsx", "reason": "shared UI"}],
    "new_artifacts_needed": [
        {"type": "component", "name": "ThemeToggle", "description": "toggle component"}
    ],
    "pattern_to_follow": "src/features/settings/",
    "edge_cases": ["concurrent theme changes"],
    "integration_points": ["navigation bar"],
    "open_questions": [
        {
            "question": "Should dark mode persist?",
            "category": "intent",
            "recommended_answer": "Yes, store in localStorage.",
        }
    ],
}

SAMPLE_REPORT = "# Research Report\n\nDark mode is feasible.\n"


def _write_agent_output_files(directory: str | Path) -> None:
    """Simulate agent writing output files into *directory*."""
    d = Path(directory)
    (d / "technical_assessment.json").write_text(json.dumps(SAMPLE_ASSESSMENT))
    (d / "research_report.md").write_text(SAMPLE_REPORT)


# ---------------------------------------------------------------------------
# Core happy-path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_research_node_happy_path(tmp_path):
    """Full successful run: agent writes both output files, node returns correct state."""
    state = _base_state(tmp_path)
    ui = _make_ui()

    async def _fake_run_agent(**kwargs):
        # Simulate agent creating the output files in cwd
        _write_agent_output_files(kwargs["cwd"])
        return AgentResult(text="agent fallback text")

    with patch("graft.stages.research.run_agent", new_callable=lambda: lambda: AsyncMock(side_effect=_fake_run_agent)) as mock_run, \
         patch("graft.stages.research.mark_stage_complete") as mock_mark:
        result = await research_node(state, ui)

    # Return value has the right keys and values
    assert result["current_stage"] == "research"
    assert result["technical_assessment"] == SAMPLE_ASSESSMENT
    assert result["research_report"] == SAMPLE_REPORT

    # mark_stage_complete called with correct args
    mock_mark.assert_called_once_with(state["project_dir"], "research")


@pytest.mark.asyncio
async def test_research_node_saves_artifacts(tmp_path):
    """Artifacts are saved via save_artifact for both report and assessment."""
    state = _base_state(tmp_path)
    ui = _make_ui()

    async def _fake_run_agent(**kwargs):
        _write_agent_output_files(kwargs["cwd"])
        return AgentResult(text="fallback")

    with patch("graft.stages.research.run_agent", new_callable=lambda: lambda: AsyncMock(side_effect=_fake_run_agent)), \
         patch("graft.stages.research.mark_stage_complete"), \
         patch("graft.stages.research.save_artifact") as mock_save:
        await research_node(state, ui)

    # Two save_artifact calls: research_report.md then technical_assessment.json
    assert mock_save.call_count == 2
    names_saved = [c.args[1] for c in mock_save.call_args_list]
    assert "research_report.md" in names_saved
    assert "technical_assessment.json" in names_saved


# ---------------------------------------------------------------------------
# File cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_research_node_cleans_up_output_files(tmp_path):
    """Agent output files in repo_path are deleted after being read."""
    state = _base_state(tmp_path)
    repo_path = Path(state["repo_path"])
    ui = _make_ui()

    async def _fake_run_agent(**kwargs):
        _write_agent_output_files(kwargs["cwd"])
        return AgentResult(text="fallback")

    with patch("graft.stages.research.run_agent", new_callable=lambda: lambda: AsyncMock(side_effect=_fake_run_agent)), \
         patch("graft.stages.research.mark_stage_complete"), \
         patch("graft.stages.research.save_artifact"):
        await research_node(state, ui)

    assert not (repo_path / "research_report.md").exists()
    assert not (repo_path / "technical_assessment.json").exists()


# ---------------------------------------------------------------------------
# Fallback when output files are missing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_research_report_falls_back_to_result_text(tmp_path):
    """When research_report.md is not found, result.text is used instead."""
    state = _base_state(tmp_path)
    ui = _make_ui()
    fallback_text = "Agent produced this text but no file."

    async def _fake_run_agent(**kwargs):
        # Do NOT write research_report.md — only assessment
        cwd = Path(kwargs["cwd"])
        (cwd / "technical_assessment.json").write_text(json.dumps(SAMPLE_ASSESSMENT))
        return AgentResult(text=fallback_text)

    with patch("graft.stages.research.run_agent", new_callable=lambda: lambda: AsyncMock(side_effect=_fake_run_agent)), \
         patch("graft.stages.research.mark_stage_complete"), \
         patch("graft.stages.research.save_artifact"):
        result = await research_node(state, ui)

    assert result["research_report"] == fallback_text


@pytest.mark.asyncio
async def test_technical_assessment_empty_when_file_missing(tmp_path):
    """When technical_assessment.json is not found, an empty dict is returned."""
    state = _base_state(tmp_path)
    ui = _make_ui()

    async def _fake_run_agent(**kwargs):
        # Write only the report, not the assessment
        cwd = Path(kwargs["cwd"])
        (cwd / "research_report.md").write_text(SAMPLE_REPORT)
        return AgentResult(text="fallback")

    with patch("graft.stages.research.run_agent", new_callable=lambda: lambda: AsyncMock(side_effect=_fake_run_agent)), \
         patch("graft.stages.research.mark_stage_complete"), \
         patch("graft.stages.research.save_artifact"):
        result = await research_node(state, ui)

    assert result["technical_assessment"] == {}


@pytest.mark.asyncio
async def test_no_output_files_at_all(tmp_path):
    """When agent writes neither file, report falls back and assessment is empty."""
    state = _base_state(tmp_path)
    ui = _make_ui()
    fallback = "only agent text"

    async def _fake_run_agent(**kwargs):
        return AgentResult(text=fallback)

    with patch("graft.stages.research.run_agent", new_callable=lambda: lambda: AsyncMock(side_effect=_fake_run_agent)), \
         patch("graft.stages.research.mark_stage_complete"), \
         patch("graft.stages.research.save_artifact"):
        result = await research_node(state, ui)

    assert result["research_report"] == fallback
    assert result["technical_assessment"] == {}


# ---------------------------------------------------------------------------
# scope_path handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scope_path_changes_cwd(tmp_path):
    """When scope_path is set and the directory exists, agent runs in that subdir."""
    state = _base_state(tmp_path)
    repo_path = Path(state["repo_path"])
    scoped = repo_path / "packages" / "frontend"
    scoped.mkdir(parents=True)
    state["scope_path"] = "packages/frontend"
    ui = _make_ui()

    captured_cwd = {}

    async def _fake_run_agent(**kwargs):
        captured_cwd["value"] = kwargs["cwd"]
        _write_agent_output_files(kwargs["cwd"])
        return AgentResult(text="text")

    with patch("graft.stages.research.run_agent", new_callable=lambda: lambda: AsyncMock(side_effect=_fake_run_agent)), \
         patch("graft.stages.research.mark_stage_complete"), \
         patch("graft.stages.research.save_artifact"):
        await research_node(state, ui)

    assert captured_cwd["value"] == str(scoped)


@pytest.mark.asyncio
async def test_scope_path_nonexistent_falls_back_to_repo(tmp_path):
    """When scope_path doesn't exist on disk, cwd falls back to repo_path."""
    state = _base_state(tmp_path)
    state["scope_path"] = "does/not/exist"
    ui = _make_ui()

    captured_cwd = {}

    async def _fake_run_agent(**kwargs):
        captured_cwd["value"] = kwargs["cwd"]
        return AgentResult(text="text")

    with patch("graft.stages.research.run_agent", new_callable=lambda: lambda: AsyncMock(side_effect=_fake_run_agent)), \
         patch("graft.stages.research.mark_stage_complete"), \
         patch("graft.stages.research.save_artifact"):
        await research_node(state, ui)

    assert captured_cwd["value"] == state["repo_path"]


@pytest.mark.asyncio
async def test_scope_path_empty_string_uses_repo_path(tmp_path):
    """An empty scope_path means no scoping — cwd = repo_path."""
    state = _base_state(tmp_path)
    state["scope_path"] = ""
    ui = _make_ui()

    captured_cwd = {}

    async def _fake_run_agent(**kwargs):
        captured_cwd["value"] = kwargs["cwd"]
        return AgentResult(text="text")

    with patch("graft.stages.research.run_agent", new_callable=lambda: lambda: AsyncMock(side_effect=_fake_run_agent)), \
         patch("graft.stages.research.mark_stage_complete"), \
         patch("graft.stages.research.save_artifact"):
        await research_node(state, ui)

    assert captured_cwd["value"] == state["repo_path"]


@pytest.mark.asyncio
async def test_scope_path_files_read_from_scoped_dir(tmp_path):
    """Output files written in scope_path dir are found and read correctly."""
    state = _base_state(tmp_path)
    repo_path = Path(state["repo_path"])
    scoped = repo_path / "apps" / "web"
    scoped.mkdir(parents=True)
    state["scope_path"] = "apps/web"
    ui = _make_ui()

    async def _fake_run_agent(**kwargs):
        _write_agent_output_files(kwargs["cwd"])
        return AgentResult(text="fallback")

    with patch("graft.stages.research.run_agent", new_callable=lambda: lambda: AsyncMock(side_effect=_fake_run_agent)), \
         patch("graft.stages.research.mark_stage_complete"), \
         patch("graft.stages.research.save_artifact"):
        result = await research_node(state, ui)

    assert result["research_report"] == SAMPLE_REPORT
    assert result["technical_assessment"] == SAMPLE_ASSESSMENT
    # Files cleaned up in both scoped dir and repo root
    assert not (scoped / "research_report.md").exists()
    assert not (scoped / "technical_assessment.json").exists()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_includes_codebase_profile_json(tmp_path):
    """The user_prompt sent to run_agent contains serialised codebase_profile."""
    state = _base_state(tmp_path)
    profile = {"framework": "django", "language": "python", "orm": "django-orm"}
    state["codebase_profile"] = profile
    ui = _make_ui()

    captured_prompt = {}

    async def _fake_run_agent(**kwargs):
        captured_prompt["value"] = kwargs["user_prompt"]
        return AgentResult(text="text")

    with patch("graft.stages.research.run_agent", new_callable=lambda: lambda: AsyncMock(side_effect=_fake_run_agent)), \
         patch("graft.stages.research.mark_stage_complete"), \
         patch("graft.stages.research.save_artifact"):
        await research_node(state, ui)

    prompt = captured_prompt["value"]
    assert "CODEBASE PROFILE:" in prompt
    assert '"framework": "django"' in prompt
    assert '"orm": "django-orm"' in prompt


@pytest.mark.asyncio
async def test_prompt_includes_feature_prompt(tmp_path):
    """The user_prompt sent to run_agent includes the feature description."""
    state = _base_state(tmp_path)
    state["feature_prompt"] = "Implement SSO login via SAML"
    ui = _make_ui()

    captured_prompt = {}

    async def _fake_run_agent(**kwargs):
        captured_prompt["value"] = kwargs["user_prompt"]
        return AgentResult(text="text")

    with patch("graft.stages.research.run_agent", new_callable=lambda: lambda: AsyncMock(side_effect=_fake_run_agent)), \
         patch("graft.stages.research.mark_stage_complete"), \
         patch("graft.stages.research.save_artifact"):
        await research_node(state, ui)

    assert "FEATURE: Implement SSO login via SAML" in captured_prompt["value"]


@pytest.mark.asyncio
async def test_prompt_includes_constraints_when_present(tmp_path):
    """Constraints are joined with semicolons and included in the prompt."""
    state = _base_state(tmp_path)
    state["constraints"] = ["no new dependencies", "must be accessible"]
    ui = _make_ui()

    captured_prompt = {}

    async def _fake_run_agent(**kwargs):
        captured_prompt["value"] = kwargs["user_prompt"]
        return AgentResult(text="text")

    with patch("graft.stages.research.run_agent", new_callable=lambda: lambda: AsyncMock(side_effect=_fake_run_agent)), \
         patch("graft.stages.research.mark_stage_complete"), \
         patch("graft.stages.research.save_artifact"):
        await research_node(state, ui)

    assert "CONSTRAINTS:" in captured_prompt["value"]
    assert "no new dependencies" in captured_prompt["value"]
    assert "must be accessible" in captured_prompt["value"]


@pytest.mark.asyncio
async def test_prompt_omits_constraints_when_empty(tmp_path):
    """When constraints list is empty, CONSTRAINTS section is not in the prompt."""
    state = _base_state(tmp_path)
    state["constraints"] = []
    ui = _make_ui()

    captured_prompt = {}

    async def _fake_run_agent(**kwargs):
        captured_prompt["value"] = kwargs["user_prompt"]
        return AgentResult(text="text")

    with patch("graft.stages.research.run_agent", new_callable=lambda: lambda: AsyncMock(side_effect=_fake_run_agent)), \
         patch("graft.stages.research.mark_stage_complete"), \
         patch("graft.stages.research.save_artifact"):
        await research_node(state, ui)

    assert "CONSTRAINTS:" not in captured_prompt["value"]


@pytest.mark.asyncio
async def test_prompt_includes_repo_path(tmp_path):
    """The prompt tells the agent where the codebase lives."""
    state = _base_state(tmp_path)
    ui = _make_ui()

    captured_prompt = {}

    async def _fake_run_agent(**kwargs):
        captured_prompt["value"] = kwargs["user_prompt"]
        return AgentResult(text="text")

    with patch("graft.stages.research.run_agent", new_callable=lambda: lambda: AsyncMock(side_effect=_fake_run_agent)), \
         patch("graft.stages.research.mark_stage_complete"), \
         patch("graft.stages.research.save_artifact"):
        await research_node(state, ui)

    assert state["repo_path"] in captured_prompt["value"]


# ---------------------------------------------------------------------------
# run_agent call parameters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_agent_called_with_correct_params(tmp_path):
    """Verify run_agent receives the expected keyword arguments."""
    state = _base_state(tmp_path)
    state["model"] = "claude-sonnet-4-20250514"
    ui = _make_ui()

    captured_kwargs = {}

    async def _fake_run_agent(**kwargs):
        captured_kwargs.update(kwargs)
        return AgentResult(text="text")

    with patch("graft.stages.research.run_agent", new_callable=lambda: lambda: AsyncMock(side_effect=_fake_run_agent)), \
         patch("graft.stages.research.mark_stage_complete"), \
         patch("graft.stages.research.save_artifact"):
        await research_node(state, ui)

    assert captured_kwargs["persona"] == "Staff Software Architect (Feature Specialist)"
    assert captured_kwargs["system_prompt"] == SYSTEM_PROMPT
    assert captured_kwargs["stage"] == "research"
    assert captured_kwargs["project_dir"] == state["project_dir"]
    assert captured_kwargs["model"] == "claude-sonnet-4-20250514"
    assert captured_kwargs["max_turns"] == 30
    assert captured_kwargs["allowed_tools"] == ["Bash", "Read", "Glob", "Grep"]
    assert captured_kwargs["ui"] is ui


@pytest.mark.asyncio
async def test_run_agent_model_none_when_not_set(tmp_path):
    """When state has no model key, run_agent receives model=None."""
    state = _base_state(tmp_path)
    # Ensure no model key
    state.pop("model", None)  # type: ignore[misc]
    ui = _make_ui()

    captured_kwargs = {}

    async def _fake_run_agent(**kwargs):
        captured_kwargs.update(kwargs)
        return AgentResult(text="text")

    with patch("graft.stages.research.run_agent", new_callable=lambda: lambda: AsyncMock(side_effect=_fake_run_agent)), \
         patch("graft.stages.research.mark_stage_complete"), \
         patch("graft.stages.research.save_artifact"):
        await research_node(state, ui)

    assert captured_kwargs["model"] is None


# ---------------------------------------------------------------------------
# JSON parse error in technical_assessment.json
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_malformed_assessment_json_returns_empty_dict(tmp_path):
    """If technical_assessment.json contains invalid JSON, result is empty dict."""
    state = _base_state(tmp_path)
    ui = _make_ui()

    async def _fake_run_agent(**kwargs):
        cwd = Path(kwargs["cwd"])
        (cwd / "research_report.md").write_text(SAMPLE_REPORT)
        (cwd / "technical_assessment.json").write_text("NOT VALID JSON {{{{")
        return AgentResult(text="fallback")

    with patch("graft.stages.research.run_agent", new_callable=lambda: lambda: AsyncMock(side_effect=_fake_run_agent)), \
         patch("graft.stages.research.mark_stage_complete"), \
         patch("graft.stages.research.save_artifact"):
        result = await research_node(state, ui)

    assert result["technical_assessment"] == {}


# ---------------------------------------------------------------------------
# UI interactions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ui_stage_start_and_done_called(tmp_path):
    """UI.stage_start and UI.stage_done are called with 'research'."""
    state = _base_state(tmp_path)
    ui = MagicMock(spec=UI)

    async def _fake_run_agent(**kwargs):
        return AgentResult(text="text")

    with patch("graft.stages.research.run_agent", new_callable=lambda: lambda: AsyncMock(side_effect=_fake_run_agent)), \
         patch("graft.stages.research.mark_stage_complete"), \
         patch("graft.stages.research.save_artifact"):
        await research_node(state, ui)

    ui.stage_start.assert_called_once_with("research")
    ui.stage_done.assert_called_once_with("research")


@pytest.mark.asyncio
async def test_ui_info_called_with_open_questions_count(tmp_path):
    """When assessment has open questions, ui.info reports their count."""
    state = _base_state(tmp_path)
    ui = MagicMock(spec=UI)

    async def _fake_run_agent(**kwargs):
        _write_agent_output_files(kwargs["cwd"])
        return AgentResult(text="text")

    with patch("graft.stages.research.run_agent", new_callable=lambda: lambda: AsyncMock(side_effect=_fake_run_agent)), \
         patch("graft.stages.research.mark_stage_complete"), \
         patch("graft.stages.research.save_artifact"):
        await research_node(state, ui)

    # SAMPLE_ASSESSMENT has 1 open question
    ui.info.assert_called_once()
    msg = ui.info.call_args[0][0]
    assert "1 open question" in msg


@pytest.mark.asyncio
async def test_ui_info_not_called_when_no_open_questions(tmp_path):
    """When assessment has no open questions, ui.info is not called."""
    state = _base_state(tmp_path)
    ui = MagicMock(spec=UI)

    assessment_no_questions = {**SAMPLE_ASSESSMENT, "open_questions": []}

    async def _fake_run_agent(**kwargs):
        cwd = Path(kwargs["cwd"])
        (cwd / "technical_assessment.json").write_text(json.dumps(assessment_no_questions))
        (cwd / "research_report.md").write_text(SAMPLE_REPORT)
        return AgentResult(text="text")

    with patch("graft.stages.research.run_agent", new_callable=lambda: lambda: AsyncMock(side_effect=_fake_run_agent)), \
         patch("graft.stages.research.mark_stage_complete"), \
         patch("graft.stages.research.save_artifact"):
        await research_node(state, ui)

    ui.info.assert_not_called()


# ---------------------------------------------------------------------------
# Fallback path for files in repo_path (not scoped cwd)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_files_found_in_repo_root_when_not_in_scoped_cwd(tmp_path):
    """If files aren't in scoped cwd, falls back to repo_path to find them."""
    state = _base_state(tmp_path)
    repo_path = Path(state["repo_path"])
    scoped = repo_path / "pkg"
    scoped.mkdir()
    state["scope_path"] = "pkg"
    ui = _make_ui()

    async def _fake_run_agent(**kwargs):
        # Write files to repo_path (not scoped cwd) — simulates agent writing in wrong dir
        _write_agent_output_files(repo_path)
        return AgentResult(text="fallback")

    with patch("graft.stages.research.run_agent", new_callable=lambda: lambda: AsyncMock(side_effect=_fake_run_agent)), \
         patch("graft.stages.research.mark_stage_complete"), \
         patch("graft.stages.research.save_artifact"):
        result = await research_node(state, ui)

    # Should find files via the fallback path
    assert result["research_report"] == SAMPLE_REPORT
    assert result["technical_assessment"] == SAMPLE_ASSESSMENT


# ---------------------------------------------------------------------------
# Cleanup removes files from both scoped and repo dirs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_removes_files_from_both_locations(tmp_path):
    """Files are cleaned up from both scoped cwd and repo root."""
    state = _base_state(tmp_path)
    repo_path = Path(state["repo_path"])
    scoped = repo_path / "sub"
    scoped.mkdir()
    state["scope_path"] = "sub"
    ui = _make_ui()

    async def _fake_run_agent(**kwargs):
        # Write files in both locations
        _write_agent_output_files(kwargs["cwd"])
        _write_agent_output_files(repo_path)
        return AgentResult(text="text")

    with patch("graft.stages.research.run_agent", new_callable=lambda: lambda: AsyncMock(side_effect=_fake_run_agent)), \
         patch("graft.stages.research.mark_stage_complete"), \
         patch("graft.stages.research.save_artifact"):
        await research_node(state, ui)

    # Both locations cleaned
    assert not (scoped / "research_report.md").exists()
    assert not (scoped / "technical_assessment.json").exists()
    assert not (repo_path / "research_report.md").exists()
    assert not (repo_path / "technical_assessment.json").exists()


# ---------------------------------------------------------------------------
# Edge case: missing optional state fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_feature_prompt_defaults_to_empty(tmp_path):
    """When feature_prompt is absent from state, it defaults to empty string."""
    state = _base_state(tmp_path)
    del state["feature_prompt"]  # type: ignore[misc]
    ui = _make_ui()

    captured_prompt = {}

    async def _fake_run_agent(**kwargs):
        captured_prompt["value"] = kwargs["user_prompt"]
        return AgentResult(text="text")

    with patch("graft.stages.research.run_agent", new_callable=lambda: lambda: AsyncMock(side_effect=_fake_run_agent)), \
         patch("graft.stages.research.mark_stage_complete"), \
         patch("graft.stages.research.save_artifact"):
        await research_node(state, ui)

    assert "FEATURE: " in captured_prompt["value"]


@pytest.mark.asyncio
async def test_missing_codebase_profile_defaults_to_empty_dict(tmp_path):
    """When codebase_profile is absent, it defaults to {} and appears as such in prompt."""
    state = _base_state(tmp_path)
    del state["codebase_profile"]  # type: ignore[misc]
    ui = _make_ui()

    captured_prompt = {}

    async def _fake_run_agent(**kwargs):
        captured_prompt["value"] = kwargs["user_prompt"]
        return AgentResult(text="text")

    with patch("graft.stages.research.run_agent", new_callable=lambda: lambda: AsyncMock(side_effect=_fake_run_agent)), \
         patch("graft.stages.research.mark_stage_complete"), \
         patch("graft.stages.research.save_artifact"):
        await research_node(state, ui)

    assert "{}" in captured_prompt["value"]


# ---------------------------------------------------------------------------
# SYSTEM_PROMPT constant validation
# ---------------------------------------------------------------------------


def test_system_prompt_mentions_required_outputs():
    """System prompt instructs the agent to produce both required files."""
    assert "research_report.md" in SYSTEM_PROMPT
    assert "technical_assessment.json" in SYSTEM_PROMPT


def test_system_prompt_mentions_open_questions():
    """System prompt requires open questions with recommended answers."""
    assert "open_questions" in SYSTEM_PROMPT
    assert "recommended_answer" in SYSTEM_PROMPT
