"""Tests for graft.stages.plan."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graft.agent import AgentResult
from graft.stages.plan import (
    _COST_PER_UNIT,
    _PIPELINE_OVERHEAD,
    estimate_cost,
    plan_node,
    plan_review_node,
    plan_review_router,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PLAN_RAW = {
    "plan_id": "feat_abc123",
    "feature_name": "Add trade ledger",
    "total_units": 3,
    "estimated_cost": "$5-12",
    "units": [
        {
            "unit_id": "feat_01",
            "title": "Create trades table migration",
            "description": "Add migration for trades table",
            "category": "database",
            "service": "packages/db",
            "risk": "low",
            "blast_radius": "1 file (new migration)",
            "depends_on": [],
            "acceptance_criteria": ["migration runs cleanly"],
            "pattern_reference": "migrations/001_initial.sql",
            "tests_included": False,
        },
        {
            "unit_id": "feat_02",
            "title": "Add trade API endpoint",
            "description": "REST endpoint for creating trades",
            "category": "api",
            "service": "packages/api",
            "risk": "medium",
            "blast_radius": "2 files",
            "depends_on": ["feat_01"],
            "acceptance_criteria": ["POST /trades returns 201"],
            "pattern_reference": "routes/orders.ts",
            "tests_included": True,
        },
        {
            "unit_id": "feat_03",
            "title": "Add trade form component",
            "description": "React form for trade entry",
            "category": "component",
            "service": "packages/web",
            "risk": "high",
            "blast_radius": "3 files",
            "depends_on": ["feat_02"],
            "acceptance_criteria": ["form renders", "validation works"],
            "pattern_reference": "components/OrderForm.tsx",
            "tests_included": True,
        },
    ],
}


@pytest.fixture
def sample_units():
    return SAMPLE_PLAN_RAW["units"]


@pytest.fixture
def project_dir(tmp_path):
    """Create a minimal project directory with metadata.json for mark_stage_complete."""
    pdir = tmp_path / "feat_test"
    (pdir / "artifacts").mkdir(parents=True)
    (pdir / "logs").mkdir(parents=True)
    meta = {
        "project_id": "feat_test",
        "repo_path": "/tmp/repo",
        "feature_prompt": "test feature",
        "created_at": "2026-01-01T00:00:00Z",
        "status": "in_progress",
        "stages_completed": [],
    }
    (pdir / "metadata.json").write_text(json.dumps(meta))
    return pdir


@pytest.fixture
def repo_path(tmp_path):
    """Temporary repo directory for plan_node to write build_plan.json."""
    rdir = tmp_path / "repo"
    rdir.mkdir()
    return rdir


@pytest.fixture
def mock_ui():
    ui = MagicMock()
    ui.stage_start = MagicMock()
    ui.stage_done = MagicMock()
    ui.show_artifact = MagicMock()
    ui.info = MagicMock()
    ui.error = MagicMock()
    ui.prompt_plan_review = MagicMock()
    return ui


def _make_state(repo_path, project_dir, **overrides):
    """Build a FeatureState-like dict with sensible defaults."""
    state = {
        "repo_path": str(repo_path),
        "project_dir": str(project_dir),
        "feature_prompt": "Add a trade ledger feature",
        "codebase_profile": {"framework": "Next.js", "language": "TypeScript"},
        "technical_assessment": {"gaps": ["no trade model"], "reuse": ["order model"]},
        "feature_spec": {"decisions": [{"q": "DB?", "a": "Postgres"}]},
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# estimate_cost tests (existing + new edge cases)
# ---------------------------------------------------------------------------


def test_estimate_cost_empty_plan():
    """Empty plan still has pipeline overhead."""
    low, high = estimate_cost([])
    assert low > 0
    assert high > low


def test_estimate_cost_with_units():
    """Cost scales with number and risk of units."""
    plan = [
        {"risk": "low"},
        {"risk": "medium"},
        {"risk": "high"},
    ]
    low, high = estimate_cost(plan)
    assert low > 3.0  # More than just overhead
    assert high > low


def test_estimate_cost_exact_values():
    """Verify exact arithmetic against known constants."""
    plan = [{"risk": "low"}]
    low, high = estimate_cost(plan)
    expected_low = round(_PIPELINE_OVERHEAD[0] + _COST_PER_UNIT["low"][0], 2)
    expected_high = round(_PIPELINE_OVERHEAD[1] + _COST_PER_UNIT["low"][1], 2)
    assert low == expected_low
    assert high == expected_high


def test_estimate_cost_unknown_risk_defaults_to_medium():
    """Unknown risk level falls back to medium cost tier."""
    plan = [{"risk": "unknown_level"}]
    low_unknown, high_unknown = estimate_cost(plan)

    plan_medium = [{"risk": "medium"}]
    low_medium, high_medium = estimate_cost(plan_medium)

    assert low_unknown == low_medium
    assert high_unknown == high_medium


def test_estimate_cost_missing_risk_key_defaults_to_medium():
    """Unit with no 'risk' key falls back to medium."""
    plan = [{}]
    low, high = estimate_cost(plan)

    plan_medium = [{"risk": "medium"}]
    low_medium, high_medium = estimate_cost(plan_medium)

    assert low == low_medium
    assert high == high_medium


def test_estimate_cost_multiple_same_risk():
    """Multiple units of the same risk stack linearly."""
    plan = [{"risk": "high"}, {"risk": "high"}, {"risk": "high"}]
    low, high = estimate_cost(plan)
    expected_low = round(_PIPELINE_OVERHEAD[0] + 3 * _COST_PER_UNIT["high"][0], 2)
    expected_high = round(_PIPELINE_OVERHEAD[1] + 3 * _COST_PER_UNIT["high"][1], 2)
    assert low == expected_low
    assert high == expected_high


def test_estimate_cost_returns_rounded_floats():
    """Output is rounded to 2 decimal places."""
    plan = [{"risk": "low"}, {"risk": "medium"}, {"risk": "high"}]
    low, high = estimate_cost(plan)
    assert low == round(low, 2)
    assert high == round(high, 2)


# ---------------------------------------------------------------------------
# plan_review_router tests (existing)
# ---------------------------------------------------------------------------


def test_plan_review_router_approved():
    assert plan_review_router({"plan_approved": True}) == "execute"


def test_plan_review_router_not_approved():
    assert plan_review_router({"plan_approved": False}) == "plan"


def test_plan_review_router_missing_key():
    """Missing plan_approved is falsy → routes back to plan."""
    assert plan_review_router({}) == "plan"


# ---------------------------------------------------------------------------
# plan_node tests
# ---------------------------------------------------------------------------


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_parses_build_plan_json(
    mock_run_agent, mock_save_artifact, mock_mark_complete, repo_path, project_dir, mock_ui
):
    """plan_node reads build_plan.json written by the agent and returns parsed units."""
    # Agent writes build_plan.json to repo_path
    plan_file = repo_path / "build_plan.json"
    plan_file.write_text(json.dumps(SAMPLE_PLAN_RAW))
    mock_run_agent.return_value = AgentResult(text="Done")

    state = _make_state(repo_path, project_dir)
    result = await plan_node(state, mock_ui)

    assert result["build_plan"] == SAMPLE_PLAN_RAW["units"]
    assert result["current_stage"] == "plan"
    assert len(result["build_plan"]) == 3


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_cleans_up_build_plan_json(
    mock_run_agent, mock_save_artifact, mock_mark_complete, repo_path, project_dir, mock_ui
):
    """plan_node deletes build_plan.json from repo after reading it."""
    plan_file = repo_path / "build_plan.json"
    plan_file.write_text(json.dumps(SAMPLE_PLAN_RAW))
    mock_run_agent.return_value = AgentResult(text="Done")

    state = _make_state(repo_path, project_dir)
    await plan_node(state, mock_ui)

    assert not plan_file.exists(), "build_plan.json should be removed after parsing"


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_saves_artifact(
    mock_run_agent, mock_save_artifact, mock_mark_complete, repo_path, project_dir, mock_ui
):
    """plan_node calls save_artifact with the serialized plan."""
    plan_file = repo_path / "build_plan.json"
    plan_file.write_text(json.dumps(SAMPLE_PLAN_RAW))
    mock_run_agent.return_value = AgentResult(text="Done")

    state = _make_state(repo_path, project_dir)
    await plan_node(state, mock_ui)

    mock_save_artifact.assert_called_once()
    call_args = mock_save_artifact.call_args
    assert call_args[0][0] == str(project_dir)
    assert call_args[0][1] == "build_plan.json"
    saved_content = json.loads(call_args[0][2])
    assert saved_content == SAMPLE_PLAN_RAW


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_marks_stage_complete(
    mock_run_agent, mock_save_artifact, mock_mark_complete, repo_path, project_dir, mock_ui
):
    """plan_node calls mark_stage_complete with 'plan'."""
    plan_file = repo_path / "build_plan.json"
    plan_file.write_text(json.dumps(SAMPLE_PLAN_RAW))
    mock_run_agent.return_value = AgentResult(text="Done")

    state = _make_state(repo_path, project_dir)
    await plan_node(state, mock_ui)

    mock_mark_complete.assert_called_once_with(str(project_dir), "plan")


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_file_not_found_returns_empty_plan(
    mock_run_agent, mock_save_artifact, mock_mark_complete, repo_path, project_dir, mock_ui
):
    """When agent doesn't produce build_plan.json, return empty plan and log error."""
    mock_run_agent.return_value = AgentResult(text="I could not produce the plan")

    state = _make_state(repo_path, project_dir)
    result = await plan_node(state, mock_ui)

    assert result["build_plan"] == []
    mock_ui.error.assert_called_once_with("Agent did not produce build_plan.json.")


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_invalid_json_returns_empty_plan(
    mock_run_agent, mock_save_artifact, mock_mark_complete, repo_path, project_dir, mock_ui
):
    """Malformed JSON in build_plan.json is handled gracefully."""
    plan_file = repo_path / "build_plan.json"
    plan_file.write_text("{invalid json content!!!")
    mock_run_agent.return_value = AgentResult(text="Done")

    state = _make_state(repo_path, project_dir)
    result = await plan_node(state, mock_ui)

    assert result["build_plan"] == []
    mock_ui.error.assert_called_once_with(
        "Failed to parse build_plan.json — using empty plan."
    )


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_prompt_includes_codebase_profile(
    mock_run_agent, mock_save_artifact, mock_mark_complete, repo_path, project_dir, mock_ui
):
    """The prompt sent to run_agent includes codebase_profile JSON."""
    mock_run_agent.return_value = AgentResult(text="Done")

    profile = {"framework": "Next.js", "language": "TypeScript"}
    state = _make_state(repo_path, project_dir, codebase_profile=profile)
    await plan_node(state, mock_ui)

    call_kwargs = mock_run_agent.call_args[1]
    assert "CODEBASE PROFILE" in call_kwargs["user_prompt"]
    assert '"Next.js"' in call_kwargs["user_prompt"]
    assert '"TypeScript"' in call_kwargs["user_prompt"]


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_prompt_includes_technical_assessment(
    mock_run_agent, mock_save_artifact, mock_mark_complete, repo_path, project_dir, mock_ui
):
    """The prompt sent to run_agent includes technical_assessment JSON."""
    mock_run_agent.return_value = AgentResult(text="Done")

    assessment = {"gaps": ["no trade model"], "reuse": ["order model"]}
    state = _make_state(repo_path, project_dir, technical_assessment=assessment)
    await plan_node(state, mock_ui)

    call_kwargs = mock_run_agent.call_args[1]
    assert "TECHNICAL ASSESSMENT" in call_kwargs["user_prompt"]
    assert "no trade model" in call_kwargs["user_prompt"]


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_prompt_includes_feature_spec(
    mock_run_agent, mock_save_artifact, mock_mark_complete, repo_path, project_dir, mock_ui
):
    """The prompt sent to run_agent includes feature_spec JSON."""
    mock_run_agent.return_value = AgentResult(text="Done")

    spec = {"decisions": [{"q": "DB?", "a": "Postgres"}]}
    state = _make_state(repo_path, project_dir, feature_spec=spec)
    await plan_node(state, mock_ui)

    call_kwargs = mock_run_agent.call_args[1]
    assert "FEATURE SPEC" in call_kwargs["user_prompt"]
    assert "Postgres" in call_kwargs["user_prompt"]


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_prompt_includes_constraints(
    mock_run_agent, mock_save_artifact, mock_mark_complete, repo_path, project_dir, mock_ui
):
    """Constraints are appended to the prompt when present."""
    mock_run_agent.return_value = AgentResult(text="Done")

    state = _make_state(
        repo_path, project_dir, constraints=["no breaking changes", "must use REST"]
    )
    await plan_node(state, mock_ui)

    call_kwargs = mock_run_agent.call_args[1]
    assert "CONSTRAINTS" in call_kwargs["user_prompt"]
    assert "no breaking changes" in call_kwargs["user_prompt"]
    assert "must use REST" in call_kwargs["user_prompt"]


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_prompt_no_constraints_when_empty(
    mock_run_agent, mock_save_artifact, mock_mark_complete, repo_path, project_dir, mock_ui
):
    """No CONSTRAINTS section when constraints list is empty."""
    mock_run_agent.return_value = AgentResult(text="Done")

    state = _make_state(repo_path, project_dir, constraints=[])
    await plan_node(state, mock_ui)

    call_kwargs = mock_run_agent.call_args[1]
    assert "CONSTRAINTS" not in call_kwargs["user_prompt"]


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_max_units_in_prompt(
    mock_run_agent, mock_save_artifact, mock_mark_complete, repo_path, project_dir, mock_ui
):
    """max_units > 0 adds a maximum build units line to the prompt."""
    mock_run_agent.return_value = AgentResult(text="Done")

    state = _make_state(repo_path, project_dir, max_units=5)
    await plan_node(state, mock_ui)

    call_kwargs = mock_run_agent.call_args[1]
    assert "Maximum build units: 5" in call_kwargs["user_prompt"]


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_max_units_zero_not_in_prompt(
    mock_run_agent, mock_save_artifact, mock_mark_complete, repo_path, project_dir, mock_ui
):
    """max_units=0 (default) does not add maximum build units line."""
    mock_run_agent.return_value = AgentResult(text="Done")

    state = _make_state(repo_path, project_dir, max_units=0)
    await plan_node(state, mock_ui)

    call_kwargs = mock_run_agent.call_args[1]
    assert "Maximum build units" not in call_kwargs["user_prompt"]


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_calls_stage_start_and_done(
    mock_run_agent, mock_save_artifact, mock_mark_complete, repo_path, project_dir, mock_ui
):
    """plan_node calls ui.stage_start and ui.stage_done with 'plan'."""
    mock_run_agent.return_value = AgentResult(text="Done")

    state = _make_state(repo_path, project_dir)
    await plan_node(state, mock_ui)

    mock_ui.stage_start.assert_called_once_with("plan")
    mock_ui.stage_done.assert_called_once_with("plan")


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_passes_correct_agent_kwargs(
    mock_run_agent, mock_save_artifact, mock_mark_complete, repo_path, project_dir, mock_ui
):
    """Verify key agent kwargs: stage, cwd, max_turns, allowed_tools."""
    mock_run_agent.return_value = AgentResult(text="Done")

    state = _make_state(repo_path, project_dir, model="opus")
    await plan_node(state, mock_ui)

    call_kwargs = mock_run_agent.call_args[1]
    assert call_kwargs["stage"] == "plan"
    assert call_kwargs["cwd"] == str(repo_path)
    assert call_kwargs["project_dir"] == str(project_dir)
    assert call_kwargs["max_turns"] == 25
    assert call_kwargs["allowed_tools"] == ["Read", "Bash", "Glob", "Grep"]
    assert call_kwargs["model"] == "opus"
    assert call_kwargs["ui"] is mock_ui


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_plan_json_without_units_key(
    mock_run_agent, mock_save_artifact, mock_mark_complete, repo_path, project_dir, mock_ui
):
    """build_plan.json without 'units' key returns empty list."""
    plan_file = repo_path / "build_plan.json"
    plan_file.write_text(json.dumps({"plan_id": "feat_x", "feature_name": "test"}))
    mock_run_agent.return_value = AgentResult(text="Done")

    state = _make_state(repo_path, project_dir)
    result = await plan_node(state, mock_ui)

    assert result["build_plan"] == []


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_prompt_includes_feature_prompt(
    mock_run_agent, mock_save_artifact, mock_mark_complete, repo_path, project_dir, mock_ui
):
    """User's feature prompt is embedded in the agent prompt."""
    mock_run_agent.return_value = AgentResult(text="Done")

    state = _make_state(
        repo_path, project_dir, feature_prompt="Build a payments dashboard"
    )
    await plan_node(state, mock_ui)

    call_kwargs = mock_run_agent.call_args[1]
    assert "Build a payments dashboard" in call_kwargs["user_prompt"]
    assert "FEATURE:" in call_kwargs["user_prompt"]


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_prompt_includes_repo_path(
    mock_run_agent, mock_save_artifact, mock_mark_complete, repo_path, project_dir, mock_ui
):
    """The repo_path appears in the prompt for agent context."""
    mock_run_agent.return_value = AgentResult(text="Done")

    state = _make_state(repo_path, project_dir)
    await plan_node(state, mock_ui)

    call_kwargs = mock_run_agent.call_args[1]
    assert str(repo_path) in call_kwargs["user_prompt"]


# ---------------------------------------------------------------------------
# plan_review_node tests
# ---------------------------------------------------------------------------


async def test_plan_review_node_auto_approve(mock_ui):
    """auto_approve=True bypasses prompt and sets plan_approved=True."""
    state = {
        "build_plan": SAMPLE_PLAN_RAW["units"],
        "auto_approve": True,
    }

    result = await plan_review_node(state, mock_ui)

    assert result == {"plan_approved": True}
    mock_ui.show_artifact.assert_called_once()
    mock_ui.info.assert_called_once_with("Plan auto-approved.")
    # prompt_plan_review should NOT be called when auto-approving
    mock_ui.prompt_plan_review.assert_not_called()


async def test_plan_review_node_user_approves(mock_ui):
    """User approves the plan via prompt_plan_review."""
    state = {
        "build_plan": SAMPLE_PLAN_RAW["units"],
        "auto_approve": False,
    }
    mock_ui.prompt_plan_review.return_value = (True, "")

    result = await plan_review_node(state, mock_ui)

    assert result == {"plan_approved": True}
    mock_ui.prompt_plan_review.assert_called_once()
    mock_ui.info.assert_called_once_with("Plan approved — proceeding to execute.")


async def test_plan_review_node_user_rejects(mock_ui):
    """User rejects the plan — currently still sets plan_approved=True (re-plan not implemented)."""
    state = {
        "build_plan": SAMPLE_PLAN_RAW["units"],
        "auto_approve": False,
    }
    mock_ui.prompt_plan_review.return_value = (False, "Needs fewer units")

    result = await plan_review_node(state, mock_ui)

    # Re-planning not yet implemented — still proceeds
    assert result == {"plan_approved": True}
    # Feedback should be logged
    calls = [str(c) for c in mock_ui.info.call_args_list]
    assert any("Needs fewer units" in c for c in calls)
    assert any("Re-planning is not yet implemented" in c for c in calls)


async def test_plan_review_node_empty_plan(mock_ui):
    """Review node handles empty plan gracefully."""
    state = {
        "build_plan": [],
        "auto_approve": True,
    }

    result = await plan_review_node(state, mock_ui)

    assert result == {"plan_approved": True}
    # Summary should show 0 units
    summary_arg = mock_ui.show_artifact.call_args[0][1]
    assert "Total build units: 0" in summary_arg


async def test_plan_review_node_cost_in_summary(mock_ui):
    """Cost estimation is included in the summary displayed to user."""
    plan = [{"risk": "low"}, {"risk": "high"}]
    state = {
        "build_plan": plan,
        "auto_approve": True,
    }

    result = await plan_review_node(state, mock_ui)

    summary_arg = mock_ui.show_artifact.call_args[0][1]
    cost_low, cost_high = estimate_cost(plan)
    assert f"${cost_low:.2f}" in summary_arg
    assert f"${cost_high:.2f}" in summary_arg


async def test_plan_review_node_summary_shows_risk_colors(mock_ui):
    """Summary contains Rich-style color tags for risk levels."""
    state = {
        "build_plan": SAMPLE_PLAN_RAW["units"],
        "auto_approve": True,
    }

    await plan_review_node(state, mock_ui)

    summary_arg = mock_ui.show_artifact.call_args[0][1]
    assert "[green]" in summary_arg  # low risk
    assert "[yellow]" in summary_arg  # medium risk
    assert "[red]" in summary_arg  # high risk


async def test_plan_review_node_summary_shows_unit_details(mock_ui):
    """Summary includes unit IDs, titles, categories, and blast radius."""
    state = {
        "build_plan": SAMPLE_PLAN_RAW["units"],
        "auto_approve": True,
    }

    await plan_review_node(state, mock_ui)

    summary_arg = mock_ui.show_artifact.call_args[0][1]
    assert "feat_01" in summary_arg
    assert "Create trades table migration" in summary_arg
    assert "database" in summary_arg
    assert "1 file (new migration)" in summary_arg


async def test_plan_review_node_summary_shows_tests_tag(mock_ui):
    """Units with tests_included=True show the +tests tag."""
    state = {
        "build_plan": SAMPLE_PLAN_RAW["units"],
        "auto_approve": True,
    }

    await plan_review_node(state, mock_ui)

    summary_arg = mock_ui.show_artifact.call_args[0][1]
    assert "+tests" in summary_arg


async def test_plan_review_node_summary_shows_pattern_reference(mock_ui):
    """Summary includes pattern_reference for each unit."""
    state = {
        "build_plan": SAMPLE_PLAN_RAW["units"],
        "auto_approve": True,
    }

    await plan_review_node(state, mock_ui)

    summary_arg = mock_ui.show_artifact.call_args[0][1]
    assert "migrations/001_initial.sql" in summary_arg
    assert "routes/orders.ts" in summary_arg
    assert "components/OrderForm.tsx" in summary_arg


async def test_plan_review_node_unit_without_pattern_reference(mock_ui):
    """Units without pattern_reference don't add a pattern line."""
    plan = [
        {
            "unit_id": "feat_01",
            "title": "Something",
            "risk": "low",
            "category": "api",
            "blast_radius": "1 file",
            "tests_included": False,
        }
    ]
    state = {
        "build_plan": plan,
        "auto_approve": True,
    }

    await plan_review_node(state, mock_ui)

    summary_arg = mock_ui.show_artifact.call_args[0][1]
    assert "pattern:" not in summary_arg


async def test_plan_review_node_unit_with_unknown_risk(mock_ui):
    """Unit with unrecognized risk gets 'white' color and '?' display."""
    plan = [
        {
            "unit_id": "feat_01",
            "title": "Weird unit",
            "risk": "extreme",
            "category": "api",
            "blast_radius": "unknown",
            "tests_included": False,
        }
    ]
    state = {
        "build_plan": plan,
        "auto_approve": True,
    }

    await plan_review_node(state, mock_ui)

    summary_arg = mock_ui.show_artifact.call_args[0][1]
    assert "[white]" in summary_arg


async def test_plan_review_node_unit_missing_fields(mock_ui):
    """Units with missing optional fields use '?' and 'Untitled' defaults."""
    plan = [{}]  # completely empty unit dict
    state = {
        "build_plan": plan,
        "auto_approve": True,
    }

    await plan_review_node(state, mock_ui)

    summary_arg = mock_ui.show_artifact.call_args[0][1]
    assert "Untitled" in summary_arg
    assert "?" in summary_arg


async def test_plan_review_node_summary_unit_count(mock_ui):
    """Summary shows correct total unit count."""
    state = {
        "build_plan": SAMPLE_PLAN_RAW["units"],
        "auto_approve": True,
    }

    await plan_review_node(state, mock_ui)

    summary_arg = mock_ui.show_artifact.call_args[0][1]
    assert "Total build units: 3" in summary_arg


async def test_plan_review_node_missing_plan_key(mock_ui):
    """State without build_plan key defaults to empty list."""
    state = {"auto_approve": True}

    result = await plan_review_node(state, mock_ui)

    assert result == {"plan_approved": True}
    summary_arg = mock_ui.show_artifact.call_args[0][1]
    assert "Total build units: 0" in summary_arg
