"""Tests for graft.stages.plan."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from graft.stages.plan import (
    estimate_cost,
    plan_node,
    plan_review_node,
    plan_review_router,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

SAMPLE_PLAN_RAW = {
    "plan_id": "feat_00001",
    "feature_name": "Test Feature",
    "total_units": 3,
    "estimated_cost": "$5-12",
    "units": [
        {
            "unit_id": "feat_01",
            "title": "Create migration",
            "description": "Add DB migration",
            "category": "database",
            "service": "packages/db",
            "risk": "low",
            "blast_radius": "1 file",
            "depends_on": [],
            "acceptance_criteria": ["migration runs"],
            "pattern_reference": "migrations/001.sql",
            "tests_included": False,
        },
        {
            "unit_id": "feat_02",
            "title": "Create API endpoint",
            "description": "Add REST endpoint",
            "category": "api",
            "service": "packages/api",
            "risk": "medium",
            "blast_radius": "2 files",
            "depends_on": ["feat_01"],
            "acceptance_criteria": ["endpoint returns 200"],
            "pattern_reference": "api/routes/users.ts",
            "tests_included": True,
        },
        {
            "unit_id": "feat_03",
            "title": "Add UI component",
            "description": "Complex component",
            "category": "component",
            "service": "packages/ui",
            "risk": "high",
            "blast_radius": "5 files",
            "depends_on": ["feat_02"],
            "acceptance_criteria": ["renders correctly"],
            "pattern_reference": "components/Widget.tsx",
            "tests_included": True,
        },
    ],
}


def _make_state(tmp_path: Path, **overrides) -> dict:
    """Build a minimal FeatureState for plan_node tests."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir(exist_ok=True)
    project_dir = tmp_path / "project"
    project_dir.mkdir(exist_ok=True)
    (project_dir / "artifacts").mkdir(exist_ok=True)
    # metadata.json needed by mark_stage_complete
    (project_dir / "metadata.json").write_text(json.dumps({"stages_completed": []}))

    base: dict = {
        "repo_path": str(repo_path),
        "project_dir": str(project_dir),
        "feature_prompt": "Add a widget feature",
        "codebase_profile": {"lang": "typescript"},
        "technical_assessment": {"gaps": []},
        "feature_spec": {"title": "Widget"},
        "constraints": [],
        "max_units": 0,
        "model": None,
    }
    base.update(overrides)
    return base


def _mock_ui() -> MagicMock:
    """Return a mock UI with the methods plan.py uses."""
    ui = MagicMock()
    ui.stage_start = MagicMock()
    ui.stage_done = MagicMock()
    ui.error = MagicMock()
    ui.info = MagicMock()
    ui.show_artifact = MagicMock()
    ui.prompt_plan_review = MagicMock(return_value=(True, ""))
    return ui


# ---------------------------------------------------------------------------
# estimate_cost (existing tests kept, new edges added)
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


def test_estimate_cost_unknown_risk_defaults_to_medium():
    """Units with unrecognised risk fall back to medium cost."""
    plan_unknown = [{"risk": "unknown"}]
    plan_medium = [{"risk": "medium"}]
    assert estimate_cost(plan_unknown) == estimate_cost(plan_medium)


def test_estimate_cost_missing_risk_key_defaults_to_medium():
    """Units without a risk key fall back to medium cost."""
    plan_missing = [{}]
    plan_medium = [{"risk": "medium"}]
    assert estimate_cost(plan_missing) == estimate_cost(plan_medium)


# ---------------------------------------------------------------------------
# plan_review_router (existing tests kept)
# ---------------------------------------------------------------------------


def test_plan_review_router_approved():
    assert plan_review_router({"plan_approved": True}) == "execute"


def test_plan_review_router_not_approved():
    assert plan_review_router({"plan_approved": False}) == "plan"


def test_plan_review_router_missing_key():
    """Missing plan_approved is falsy → re-plan."""
    assert plan_review_router({}) == "plan"


# ---------------------------------------------------------------------------
# plan_node
# ---------------------------------------------------------------------------


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_happy_path(mock_agent, mock_save, mock_mark, tmp_path):
    """plan_node invokes agent, reads build_plan.json, extracts units, saves artifact, returns state."""
    state = _make_state(tmp_path)
    ui = _mock_ui()

    # Simulate agent writing build_plan.json into repo_path
    plan_path = Path(state["repo_path"]) / "build_plan.json"
    plan_path.write_text(json.dumps(SAMPLE_PLAN_RAW))

    mock_agent.return_value = MagicMock(text="Done")

    result = await plan_node(state, ui)

    # Agent was called once
    mock_agent.assert_awaited_once()
    kwargs = mock_agent.call_args.kwargs
    assert kwargs["stage"] == "plan"
    assert kwargs["cwd"] == state["repo_path"]
    assert "Read" in kwargs["allowed_tools"]

    # Returns extracted units
    assert result["build_plan"] == SAMPLE_PLAN_RAW["units"]
    assert result["current_stage"] == "plan"

    # Artifact saved with full plan JSON
    mock_save.assert_called_once()
    saved_content = json.loads(mock_save.call_args[0][2])
    assert saved_content["units"] == SAMPLE_PLAN_RAW["units"]

    # Stage marked complete
    mock_mark.assert_called_once_with(state["project_dir"], "plan")

    # UI lifecycle called
    ui.stage_start.assert_called_once_with("plan")
    ui.stage_done.assert_called_once_with("plan")


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_cleans_up_build_plan_json(
    mock_agent, mock_save, mock_mark, tmp_path
):
    """build_plan.json is deleted from the repo after reading."""
    state = _make_state(tmp_path)
    ui = _mock_ui()
    plan_path = Path(state["repo_path"]) / "build_plan.json"
    plan_path.write_text(json.dumps(SAMPLE_PLAN_RAW))

    mock_agent.return_value = MagicMock(text="Done")

    await plan_node(state, ui)

    assert not plan_path.exists(), "build_plan.json should be cleaned up from repo"


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_missing_build_plan_json(
    mock_agent, mock_save, mock_mark, tmp_path
):
    """When agent doesn't produce build_plan.json, error is logged and empty plan returned."""
    state = _make_state(tmp_path)
    ui = _mock_ui()

    mock_agent.return_value = MagicMock(text="Oops")

    result = await plan_node(state, ui)

    # Error reported via UI
    ui.error.assert_called_once_with("Agent did not produce build_plan.json.")

    # Returns empty plan
    assert result["build_plan"] == []
    assert result["current_stage"] == "plan"

    # Artifact still saved (empty raw dict)
    mock_save.assert_called_once()
    saved = json.loads(mock_save.call_args[0][2])
    assert saved == {}

    # Stage lifecycle still completes
    mock_mark.assert_called_once()
    ui.stage_done.assert_called_once_with("plan")


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_malformed_json(mock_agent, mock_save, mock_mark, tmp_path):
    """Malformed JSON in build_plan.json is handled gracefully."""
    state = _make_state(tmp_path)
    ui = _mock_ui()

    plan_path = Path(state["repo_path"]) / "build_plan.json"
    plan_path.write_text("{invalid json!!!")

    mock_agent.return_value = MagicMock(text="Done")

    result = await plan_node(state, ui)

    ui.error.assert_called_once_with(
        "Failed to parse build_plan.json — using empty plan."
    )
    assert result["build_plan"] == []
    # Malformed file is still cleaned up
    assert not plan_path.exists()


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_respects_max_units_in_prompt(
    mock_agent, mock_save, mock_mark, tmp_path
):
    """When max_units > 0, it is included in the prompt sent to the agent."""
    state = _make_state(tmp_path, max_units=5)
    ui = _mock_ui()

    mock_agent.return_value = MagicMock(text="Done")

    await plan_node(state, ui)

    user_prompt = mock_agent.call_args.kwargs["user_prompt"]
    assert "Maximum build units: 5" in user_prompt


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_max_units_zero_not_in_prompt(
    mock_agent, mock_save, mock_mark, tmp_path
):
    """When max_units is 0 (default), it is NOT included in the prompt."""
    state = _make_state(tmp_path, max_units=0)
    ui = _mock_ui()

    mock_agent.return_value = MagicMock(text="Done")

    await plan_node(state, ui)

    user_prompt = mock_agent.call_args.kwargs["user_prompt"]
    assert "Maximum build units" not in user_prompt


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_respects_constraints_in_prompt(
    mock_agent, mock_save, mock_mark, tmp_path
):
    """When constraints are provided, they appear in the agent prompt."""
    state = _make_state(tmp_path, constraints=["no breaking changes", "keep it simple"])
    ui = _mock_ui()

    mock_agent.return_value = MagicMock(text="Done")

    await plan_node(state, ui)

    user_prompt = mock_agent.call_args.kwargs["user_prompt"]
    assert "CONSTRAINTS:" in user_prompt
    assert "no breaking changes" in user_prompt
    assert "keep it simple" in user_prompt


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_empty_constraints_not_in_prompt(
    mock_agent, mock_save, mock_mark, tmp_path
):
    """When constraints list is empty, CONSTRAINTS section is not in prompt."""
    state = _make_state(tmp_path, constraints=[])
    ui = _mock_ui()

    mock_agent.return_value = MagicMock(text="Done")

    await plan_node(state, ui)

    user_prompt = mock_agent.call_args.kwargs["user_prompt"]
    assert "CONSTRAINTS:" not in user_prompt


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_passes_model_to_agent(
    mock_agent, mock_save, mock_mark, tmp_path
):
    """The model from state is forwarded to run_agent."""
    state = _make_state(tmp_path, model="claude-sonnet-4-20250514")
    ui = _mock_ui()

    mock_agent.return_value = MagicMock(text="Done")

    await plan_node(state, ui)

    assert mock_agent.call_args.kwargs["model"] == "claude-sonnet-4-20250514"


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_plan_json_missing_units_key(
    mock_agent, mock_save, mock_mark, tmp_path
):
    """Valid JSON but without 'units' key returns empty build_plan list."""
    state = _make_state(tmp_path)
    ui = _mock_ui()

    plan_path = Path(state["repo_path"]) / "build_plan.json"
    plan_path.write_text(json.dumps({"plan_id": "test", "no_units_here": True}))

    mock_agent.return_value = MagicMock(text="Done")

    result = await plan_node(state, ui)

    # No error — JSON was valid, just missing "units"
    ui.error.assert_not_called()
    assert result["build_plan"] == []


@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_includes_context_in_prompt(
    mock_agent, mock_save, mock_mark, tmp_path
):
    """Prompt includes feature_prompt, codebase_profile, technical_assessment, and feature_spec."""
    state = _make_state(
        tmp_path,
        feature_prompt="Build a dashboard",
        codebase_profile={"framework": "next"},
        technical_assessment={"risk": "low"},
        feature_spec={"pages": ["dashboard"]},
    )
    ui = _mock_ui()

    mock_agent.return_value = MagicMock(text="Done")

    await plan_node(state, ui)

    prompt = mock_agent.call_args.kwargs["user_prompt"]
    assert "Build a dashboard" in prompt
    assert "CODEBASE PROFILE:" in prompt
    assert "next" in prompt
    assert "TECHNICAL ASSESSMENT:" in prompt
    assert "FEATURE SPEC:" in prompt
    assert "dashboard" in prompt


# ---------------------------------------------------------------------------
# plan_review_node
# ---------------------------------------------------------------------------


async def test_plan_review_node_auto_approve():
    """When auto_approve=True, plan is approved without prompting."""
    ui = _mock_ui()
    state = {
        "build_plan": SAMPLE_PLAN_RAW["units"],
        "auto_approve": True,
    }

    result = await plan_review_node(state, ui)

    assert result == {"plan_approved": True}
    ui.show_artifact.assert_called_once()
    assert "Build Plan" in ui.show_artifact.call_args[0][0]
    ui.info.assert_called_once_with("Plan auto-approved.")
    # prompt_plan_review should NOT be called
    ui.prompt_plan_review.assert_not_called()


async def test_plan_review_node_manual_approve():
    """When auto_approve=False and user approves, returns plan_approved=True."""
    ui = _mock_ui()
    ui.prompt_plan_review.return_value = (True, "")
    state = {
        "build_plan": SAMPLE_PLAN_RAW["units"],
        "auto_approve": False,
    }

    result = await plan_review_node(state, ui)

    assert result == {"plan_approved": True}
    ui.prompt_plan_review.assert_called_once()
    ui.info.assert_called_once_with("Plan approved — proceeding to execute.")


async def test_plan_review_node_manual_reject_still_proceeds():
    """When user rejects the plan, feedback is logged but plan still proceeds (re-plan not implemented)."""
    ui = _mock_ui()
    ui.prompt_plan_review.return_value = (False, "Needs fewer units")
    state = {
        "build_plan": SAMPLE_PLAN_RAW["units"],
        "auto_approve": False,
    }

    result = await plan_review_node(state, ui)

    # Still returns approved because re-planning is not implemented
    assert result == {"plan_approved": True}
    # Feedback logged
    ui.info.assert_any_call("Plan feedback received: Needs fewer units")
    ui.info.assert_any_call(
        "Re-planning is not yet implemented — proceeding with current plan."
    )


async def test_plan_review_node_summary_contains_cost_estimate():
    """The plan summary passed to UI includes cost estimate."""
    ui = _mock_ui()
    state = {
        "build_plan": SAMPLE_PLAN_RAW["units"],
        "auto_approve": True,
    }

    await plan_review_node(state, ui)

    summary = ui.show_artifact.call_args[0][1]
    assert "Estimated cost:" in summary
    assert "$" in summary


async def test_plan_review_node_summary_contains_unit_count():
    """The plan summary includes total unit count."""
    ui = _mock_ui()
    state = {
        "build_plan": SAMPLE_PLAN_RAW["units"],
        "auto_approve": True,
    }

    await plan_review_node(state, ui)

    summary = ui.show_artifact.call_args[0][1]
    assert "Total build units: 3" in summary


async def test_plan_review_node_summary_risk_colors():
    """Each unit in the summary is tagged with the correct risk color markup."""
    ui = _mock_ui()
    state = {
        "build_plan": SAMPLE_PLAN_RAW["units"],
        "auto_approve": True,
    }

    await plan_review_node(state, ui)

    summary = ui.show_artifact.call_args[0][1]
    # low → green, medium → yellow, high → red
    assert "[green]" in summary
    assert "[yellow]" in summary
    assert "[red]" in summary


async def test_plan_review_node_summary_shows_tests_tag():
    """Units with tests_included=True get a +tests tag in summary."""
    ui = _mock_ui()
    state = {
        "build_plan": SAMPLE_PLAN_RAW["units"],
        "auto_approve": True,
    }

    await plan_review_node(state, ui)

    summary = ui.show_artifact.call_args[0][1]
    assert "+tests" in summary


async def test_plan_review_node_summary_shows_pattern_reference():
    """Units with pattern_reference show the reference in the summary."""
    ui = _mock_ui()
    state = {
        "build_plan": SAMPLE_PLAN_RAW["units"],
        "auto_approve": True,
    }

    await plan_review_node(state, ui)

    summary = ui.show_artifact.call_args[0][1]
    assert "pattern: migrations/001.sql" in summary
    assert "pattern: api/routes/users.ts" in summary


async def test_plan_review_node_summary_shows_category_and_blast_radius():
    """Each unit line includes category and blast_radius."""
    ui = _mock_ui()
    state = {
        "build_plan": SAMPLE_PLAN_RAW["units"],
        "auto_approve": True,
    }

    await plan_review_node(state, ui)

    summary = ui.show_artifact.call_args[0][1]
    assert "[database]" in summary
    assert "[api]" in summary
    assert "[component]" in summary
    assert "1 file" in summary
    assert "5 files" in summary


async def test_plan_review_node_empty_plan():
    """Empty plan is handled gracefully."""
    ui = _mock_ui()
    state = {
        "build_plan": [],
        "auto_approve": True,
    }

    result = await plan_review_node(state, ui)

    assert result == {"plan_approved": True}
    summary = ui.show_artifact.call_args[0][1]
    assert "Total build units: 0" in summary


async def test_plan_review_node_unit_missing_fields():
    """Units with missing optional fields use fallback values."""
    ui = _mock_ui()
    state = {
        "build_plan": [
            {
                "unit_id": "feat_01",
                # title, risk, category, blast_radius, tests_included, pattern_reference all missing
            }
        ],
        "auto_approve": True,
    }

    result = await plan_review_node(state, ui)

    assert result == {"plan_approved": True}
    summary = ui.show_artifact.call_args[0][1]
    # Fallback values
    assert "Untitled" in summary
    assert "[white]" in summary  # unknown risk → white
    assert "?" in summary  # missing fields default to '?'


async def test_plan_review_node_unit_no_pattern_reference():
    """Units without pattern_reference don't show pattern line."""
    ui = _mock_ui()
    state = {
        "build_plan": [
            {
                "unit_id": "feat_01",
                "title": "Simple task",
                "risk": "low",
                "category": "misc",
                "blast_radius": "1 file",
                "tests_included": False,
            }
        ],
        "auto_approve": True,
    }

    await plan_review_node(state, ui)

    summary = ui.show_artifact.call_args[0][1]
    assert "pattern:" not in summary


async def test_plan_review_node_default_auto_approve_false():
    """When auto_approve is not in state, it defaults to False and prompts for review."""
    ui = _mock_ui()
    ui.prompt_plan_review.return_value = (True, "")
    state = {
        "build_plan": [{"unit_id": "feat_01", "title": "Task", "risk": "low"}],
        # auto_approve not set
    }

    result = await plan_review_node(state, ui)

    assert result == {"plan_approved": True}
    ui.prompt_plan_review.assert_called_once()
    ui.show_artifact.assert_not_called()


async def test_plan_review_node_summary_passed_to_prompt():
    """The summary string generated is the same one passed to prompt_plan_review."""
    ui = _mock_ui()
    ui.prompt_plan_review.return_value = (True, "")
    plan = [{"unit_id": "u1", "title": "Task A", "risk": "high"}]
    state = {
        "build_plan": plan,
        "auto_approve": False,
    }

    await plan_review_node(state, ui)

    summary = ui.prompt_plan_review.call_args[0][0]
    assert "Total build units: 1" in summary
    assert "Task A" in summary
    assert "[red]" in summary
