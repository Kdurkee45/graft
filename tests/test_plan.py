"""Tests for graft.stages.plan."""

import json
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graft.stages.plan import (
    _COST_PER_UNIT,
    _PIPELINE_OVERHEAD,
    estimate_cost,
    plan_node,
    plan_review_node,
    plan_review_router,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeAgentResult:
    text: str = ""
    tool_calls: list = field(default_factory=list)
    raw_messages: list = field(default_factory=list)
    elapsed_seconds: float = 1.0
    turns_used: int = 1


def _make_ui() -> MagicMock:
    """Return a mock UI with the methods plan.py calls."""
    ui = MagicMock()
    ui.stage_start = MagicMock()
    ui.stage_done = MagicMock()
    ui.error = MagicMock()
    ui.info = MagicMock()
    ui.show_artifact = MagicMock()
    ui.prompt_plan_review = MagicMock(return_value=(True, ""))
    return ui


def _base_state(tmp_path, **overrides) -> dict:
    """Minimal FeatureState dict for plan_node / plan_review_node."""
    project_dir = tmp_path / "project"
    project_dir.mkdir(exist_ok=True)
    (project_dir / "artifacts").mkdir(exist_ok=True)
    (project_dir / "logs").mkdir(exist_ok=True)
    # metadata.json expected by mark_stage_complete
    meta = {"stages_completed": [], "last_updated": ""}
    (project_dir / "metadata.json").write_text(json.dumps(meta))

    state = {
        "repo_path": str(tmp_path / "repo"),
        "project_dir": str(project_dir),
        "feature_prompt": "Add a widget",
        "codebase_profile": {},
        "technical_assessment": {},
        "feature_spec": {},
        "constraints": [],
        "max_units": 0,
    }
    # Ensure repo_path directory exists so build_plan.json can be written
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)

    state.update(overrides)
    return state


# ===========================================================================
# Existing tests (unchanged)
# ===========================================================================


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


def test_plan_review_router_approved():
    assert plan_review_router({"plan_approved": True}) == "execute"


def test_plan_review_router_not_approved():
    assert plan_review_router({"plan_approved": False}) == "plan"


# ===========================================================================
# NEW — estimate_cost edge cases
# ===========================================================================


def test_estimate_cost_unknown_risk_defaults_to_medium():
    """A unit with an unrecognised risk level should use the medium bucket."""
    plan = [{"risk": "catastrophic"}]
    low, high = estimate_cost(plan)

    expected_low = round(_PIPELINE_OVERHEAD[0] + _COST_PER_UNIT["medium"][0], 2)
    expected_high = round(_PIPELINE_OVERHEAD[1] + _COST_PER_UNIT["medium"][1], 2)
    assert low == expected_low
    assert high == expected_high


def test_estimate_cost_missing_risk_key_defaults_to_medium():
    """A unit dict without a 'risk' key at all should default to medium."""
    plan = [{}]
    low, high = estimate_cost(plan)

    expected_low = round(_PIPELINE_OVERHEAD[0] + _COST_PER_UNIT["medium"][0], 2)
    expected_high = round(_PIPELINE_OVERHEAD[1] + _COST_PER_UNIT["medium"][1], 2)
    assert low == expected_low
    assert high == expected_high


def test_estimate_cost_all_low_risk():
    """Verify exact arithmetic for a homogeneous low-risk plan."""
    n = 5
    plan = [{"risk": "low"}] * n
    low, high = estimate_cost(plan)

    expected_low = round(_PIPELINE_OVERHEAD[0] + n * _COST_PER_UNIT["low"][0], 2)
    expected_high = round(_PIPELINE_OVERHEAD[1] + n * _COST_PER_UNIT["low"][1], 2)
    assert low == expected_low
    assert high == expected_high


# ===========================================================================
# NEW — plan_node
# ===========================================================================


@pytest.mark.asyncio
@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_reads_build_plan_json(
    mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path
):
    """plan_node should call run_agent, read build_plan.json, and return units."""
    state = _base_state(tmp_path)
    ui = _make_ui()

    # Simulate the agent writing build_plan.json into repo_path
    build_plan_data = {
        "plan_id": "feat_001",
        "feature_name": "Widget",
        "total_units": 2,
        "units": [
            {"unit_id": "feat_01", "title": "Migration", "risk": "low"},
            {"unit_id": "feat_02", "title": "API endpoint", "risk": "medium"},
        ],
    }
    plan_path = tmp_path / "repo" / "build_plan.json"

    async def _write_plan(**kwargs):
        plan_path.write_text(json.dumps(build_plan_data))
        return _FakeAgentResult(text="Done")

    mock_run_agent.side_effect = _write_plan

    result = await plan_node(state, ui)

    # run_agent was called exactly once
    mock_run_agent.assert_awaited_once()

    # Returned units match what was in the file
    assert len(result["build_plan"]) == 2
    assert result["build_plan"][0]["unit_id"] == "feat_01"
    assert result["current_stage"] == "plan"

    # Stage lifecycle
    ui.stage_start.assert_called_once_with("plan")
    ui.stage_done.assert_called_once_with("plan")

    # Artifact saved
    mock_save_artifact.assert_called_once()
    mock_mark_complete.assert_called_once_with(state["project_dir"], "plan")


@pytest.mark.asyncio
@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_missing_build_plan_json(
    mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path
):
    """When the agent does not produce build_plan.json, plan_node logs an error
    and returns an empty build_plan list."""
    state = _base_state(tmp_path)
    ui = _make_ui()
    mock_run_agent.return_value = _FakeAgentResult(text="oops")

    result = await plan_node(state, ui)

    assert result["build_plan"] == []
    ui.error.assert_called_once_with("Agent did not produce build_plan.json.")


@pytest.mark.asyncio
@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_cleans_up_build_plan_json(
    mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path
):
    """build_plan.json should be deleted from repo_path after being read."""
    state = _base_state(tmp_path)
    ui = _make_ui()

    plan_path = tmp_path / "repo" / "build_plan.json"

    async def _write_plan(**kwargs):
        plan_path.write_text(json.dumps({"units": [{"unit_id": "u1", "title": "t"}]}))
        return _FakeAgentResult()

    mock_run_agent.side_effect = _write_plan

    await plan_node(state, ui)

    # The file must no longer exist
    assert not plan_path.exists()


@pytest.mark.asyncio
@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_handles_invalid_json(
    mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path
):
    """If build_plan.json contains invalid JSON, plan_node logs an error
    and falls back to an empty plan."""
    state = _base_state(tmp_path)
    ui = _make_ui()

    plan_path = tmp_path / "repo" / "build_plan.json"

    async def _write_bad_json(**kwargs):
        plan_path.write_text("{invalid json!!")
        return _FakeAgentResult()

    mock_run_agent.side_effect = _write_bad_json

    result = await plan_node(state, ui)

    assert result["build_plan"] == []
    ui.error.assert_called_once_with(
        "Failed to parse build_plan.json — using empty plan."
    )
    # File is still cleaned up
    assert not plan_path.exists()


@pytest.mark.asyncio
@patch("graft.stages.plan.mark_stage_complete")
@patch("graft.stages.plan.save_artifact")
@patch("graft.stages.plan.run_agent", new_callable=AsyncMock)
async def test_plan_node_includes_constraints_and_max_units(
    mock_run_agent, mock_save_artifact, mock_mark_complete, tmp_path
):
    """When constraints and max_units are provided they appear in the prompt
    sent to run_agent."""
    state = _base_state(
        tmp_path, constraints=["No new deps"], max_units=5
    )
    ui = _make_ui()
    mock_run_agent.return_value = _FakeAgentResult()

    await plan_node(state, ui)

    call_kwargs = mock_run_agent.call_args.kwargs
    prompt = call_kwargs["user_prompt"]
    assert "No new deps" in prompt
    assert "Maximum build units: 5" in prompt


# ===========================================================================
# NEW — plan_review_node
# ===========================================================================


@pytest.mark.asyncio
async def test_plan_review_node_auto_approve():
    """With auto_approve=True the node should return plan_approved=True
    without prompting the user."""
    ui = _make_ui()
    state = {
        "build_plan": [
            {"unit_id": "u1", "title": "Stuff", "risk": "low"},
        ],
        "auto_approve": True,
    }

    result = await plan_review_node(state, ui)

    assert result == {"plan_approved": True}
    ui.show_artifact.assert_called_once()
    ui.info.assert_called_once_with("Plan auto-approved.")
    # prompt_plan_review must NOT be called when auto-approving
    ui.prompt_plan_review.assert_not_called()


@pytest.mark.asyncio
async def test_plan_review_node_manual_approval():
    """When the user approves, plan_approved=True is returned."""
    ui = _make_ui()
    ui.prompt_plan_review.return_value = (True, "")
    state = {
        "build_plan": [
            {"unit_id": "u1", "title": "Stuff", "risk": "medium"},
        ],
        "auto_approve": False,
    }

    result = await plan_review_node(state, ui)

    assert result == {"plan_approved": True}
    ui.prompt_plan_review.assert_called_once()
    ui.info.assert_called_once_with("Plan approved — proceeding to execute.")


@pytest.mark.asyncio
async def test_plan_review_node_feedback():
    """When the user rejects and provides feedback, we still proceed (for now)
    but the feedback is logged."""
    ui = _make_ui()
    ui.prompt_plan_review.return_value = (False, "Too many units")
    state = {
        "build_plan": [
            {"unit_id": "u1", "title": "Stuff", "risk": "high"},
        ],
        "auto_approve": False,
    }

    result = await plan_review_node(state, ui)

    # Current implementation proceeds even on rejection
    assert result == {"plan_approved": True}
    # Feedback is logged via ui.info
    ui.info.assert_any_call("Plan feedback received: Too many units")
    ui.info.assert_any_call(
        "Re-planning is not yet implemented — proceeding with current plan."
    )


@pytest.mark.asyncio
async def test_plan_review_node_empty_plan():
    """plan_review_node handles an empty plan without crashing."""
    ui = _make_ui()
    state = {"build_plan": [], "auto_approve": True}

    result = await plan_review_node(state, ui)

    assert result == {"plan_approved": True}
    # Summary should mention 0 units
    summary_text = ui.show_artifact.call_args[0][1]
    assert "Total build units: 0" in summary_text


@pytest.mark.asyncio
async def test_plan_review_node_summary_contains_unit_details():
    """The summary passed to the UI should contain unit IDs, titles, and risk."""
    ui = _make_ui()
    state = {
        "build_plan": [
            {
                "unit_id": "feat_01",
                "title": "Create migration",
                "risk": "low",
                "category": "database",
                "blast_radius": "1 file",
                "tests_included": True,
                "pattern_reference": "migrations/001.sql",
            },
            {
                "unit_id": "feat_02",
                "title": "Build API",
                "risk": "high",
                "category": "api",
                "blast_radius": "3 files",
                "tests_included": False,
            },
        ],
        "auto_approve": True,
    }

    await plan_review_node(state, ui)

    summary = ui.show_artifact.call_args[0][1]
    assert "feat_01" in summary
    assert "Create migration" in summary
    assert "feat_02" in summary
    assert "Build API" in summary
    assert "database" in summary
    assert "migrations/001.sql" in summary
    assert "+tests" in summary  # tests_included tag for feat_01
