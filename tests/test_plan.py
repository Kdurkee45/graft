"""Tests for graft.stages.plan."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
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
# Fixtures & helpers
# ---------------------------------------------------------------------------

def _make_state(tmp_path: Path, **overrides) -> dict:
    """Build a minimal FeatureState dict suitable for plan_node."""
    project_dir = tmp_path / "project"
    project_dir.mkdir(exist_ok=True)
    (project_dir / "artifacts").mkdir(exist_ok=True)
    (project_dir / "logs").mkdir(exist_ok=True)
    return {
        "repo_path": str(tmp_path / "repo"),
        "project_dir": str(project_dir),
        "feature_prompt": "Add a dashboard widget",
        "codebase_profile": {"language": "python"},
        "technical_assessment": {"gaps": []},
        "feature_spec": {"title": "Dashboard Widget"},
        "constraints": [],
        "max_units": 0,
        **overrides,
    }


def _sample_plan(n: int = 3, risk: str = "medium") -> list[dict]:
    """Return a list of n sample build-plan units."""
    return [
        {
            "unit_id": f"feat_{i:02d}",
            "title": f"Unit {i}",
            "description": f"Implement unit {i}",
            "category": "api",
            "service": "backend",
            "risk": risk,
            "blast_radius": "1 file",
            "depends_on": [],
            "acceptance_criteria": [f"unit {i} works"],
            "pattern_reference": f"src/example_{i}.py",
            "tests_included": i % 2 == 0,
        }
        for i in range(1, n + 1)
    ]


def _sample_plan_raw(n: int = 3, risk: str = "medium") -> dict:
    """Return a full build_plan.json structure."""
    units = _sample_plan(n, risk)
    return {
        "plan_id": "feat_ABCDE",
        "feature_name": "Dashboard Widget",
        "total_units": len(units),
        "estimated_cost": "$5-10",
        "units": units,
    }


@dataclass
class FakeAgentResult:
    text: str = "Plan generated."
    tool_calls: list = None
    raw_messages: list = None
    elapsed_seconds: float = 12.0
    turns_used: int = 5

    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []
        if self.raw_messages is None:
            self.raw_messages = []


def _make_ui() -> MagicMock:
    """Create a mock UI with all the methods plan.py uses."""
    ui = MagicMock()
    ui.stage_start = MagicMock()
    ui.stage_done = MagicMock()
    ui.error = MagicMock()
    ui.info = MagicMock()
    ui.show_artifact = MagicMock()
    ui.prompt_plan_review = MagicMock(return_value=(True, ""))
    return ui


# ===========================================================================
# Existing tests — estimate_cost & plan_review_router
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
    """Units with unrecognised risk level fall back to 'medium' pricing."""
    plan_unknown = [{"risk": "extreme"}]
    plan_medium = [{"risk": "medium"}]
    assert estimate_cost(plan_unknown) == estimate_cost(plan_medium)


def test_estimate_cost_missing_risk_key_defaults_to_medium():
    """Units that omit the risk key entirely fall back to 'medium'."""
    plan_missing = [{}]
    plan_medium = [{"risk": "medium"}]
    assert estimate_cost(plan_missing) == estimate_cost(plan_medium)


def test_estimate_cost_exact_values():
    """Verify arithmetic against known constants for a single low-risk unit."""
    low, high = estimate_cost([{"risk": "low"}])
    expected_low = round(_PIPELINE_OVERHEAD[0] + _COST_PER_UNIT["low"][0], 2)
    expected_high = round(_PIPELINE_OVERHEAD[1] + _COST_PER_UNIT["low"][1], 2)
    assert (low, high) == (expected_low, expected_high)


def test_estimate_cost_many_units_accumulate():
    """Cost grows linearly with the number of identical units."""
    one = estimate_cost([{"risk": "high"}])
    three = estimate_cost([{"risk": "high"}] * 3)
    # 3 units should add 2 extra unit-costs compared to 1 unit
    extra_low = round(2 * _COST_PER_UNIT["high"][0], 2)
    extra_high = round(2 * _COST_PER_UNIT["high"][1], 2)
    assert round(three[0] - one[0], 2) == extra_low
    assert round(three[1] - one[1], 2) == extra_high


# ===========================================================================
# NEW — plan_review_router edge cases
# ===========================================================================


def test_plan_review_router_missing_key_routes_to_plan():
    """When plan_approved is absent the router should route back to plan."""
    assert plan_review_router({}) == "plan"


# ===========================================================================
# NEW — plan_node async tests
# ===========================================================================


@pytest.mark.asyncio
async def test_plan_node_happy_path(tmp_path):
    """plan_node reads build_plan.json written by the agent and returns units."""
    state = _make_state(tmp_path)
    repo = Path(state["repo_path"])
    repo.mkdir(parents=True, exist_ok=True)
    ui = _make_ui()

    plan_raw = _sample_plan_raw(3)
    plan_file = repo / "build_plan.json"
    plan_file.write_text(json.dumps(plan_raw))

    with patch("graft.stages.plan.run_agent", new_callable=AsyncMock, return_value=FakeAgentResult()) as mock_agent, \
         patch("graft.stages.plan.save_artifact") as mock_save, \
         patch("graft.stages.plan.mark_stage_complete") as mock_mark:

        result = await plan_node(state, ui)

    # Verify returned state
    assert result["build_plan"] == plan_raw["units"]
    assert result["current_stage"] == "plan"

    # Agent was called once
    mock_agent.assert_awaited_once()

    # Artifact saved with full plan JSON
    mock_save.assert_called_once()
    saved_content = mock_save.call_args[0][2]
    assert json.loads(saved_content) == plan_raw

    # Stage completion recorded
    mock_mark.assert_called_once_with(state["project_dir"], "plan")

    # UI lifecycle
    ui.stage_start.assert_called_once_with("plan")
    ui.stage_done.assert_called_once_with("plan")

    # Temp file cleaned up
    assert not plan_file.exists()


@pytest.mark.asyncio
async def test_plan_node_no_plan_file(tmp_path):
    """When the agent fails to write build_plan.json, node returns empty plan."""
    state = _make_state(tmp_path)
    repo = Path(state["repo_path"])
    repo.mkdir(parents=True, exist_ok=True)
    ui = _make_ui()

    with patch("graft.stages.plan.run_agent", new_callable=AsyncMock, return_value=FakeAgentResult()), \
         patch("graft.stages.plan.save_artifact"), \
         patch("graft.stages.plan.mark_stage_complete"):

        result = await plan_node(state, ui)

    assert result["build_plan"] == []
    ui.error.assert_any_call("Agent did not produce build_plan.json.")


@pytest.mark.asyncio
async def test_plan_node_malformed_json(tmp_path):
    """Malformed JSON in build_plan.json results in empty plan + error."""
    state = _make_state(tmp_path)
    repo = Path(state["repo_path"])
    repo.mkdir(parents=True, exist_ok=True)
    ui = _make_ui()

    plan_file = repo / "build_plan.json"
    plan_file.write_text("{invalid json!!")

    with patch("graft.stages.plan.run_agent", new_callable=AsyncMock, return_value=FakeAgentResult()), \
         patch("graft.stages.plan.save_artifact"), \
         patch("graft.stages.plan.mark_stage_complete"):

        result = await plan_node(state, ui)

    assert result["build_plan"] == []
    ui.error.assert_any_call("Failed to parse build_plan.json — using empty plan.")


@pytest.mark.asyncio
async def test_plan_node_empty_units_in_plan(tmp_path):
    """A valid JSON plan with zero units returns an empty build_plan list."""
    state = _make_state(tmp_path)
    repo = Path(state["repo_path"])
    repo.mkdir(parents=True, exist_ok=True)
    ui = _make_ui()

    plan_raw = {"plan_id": "feat_EMPTY", "units": []}
    (repo / "build_plan.json").write_text(json.dumps(plan_raw))

    with patch("graft.stages.plan.run_agent", new_callable=AsyncMock, return_value=FakeAgentResult()), \
         patch("graft.stages.plan.save_artifact") as mock_save, \
         patch("graft.stages.plan.mark_stage_complete"):

        result = await plan_node(state, ui)

    assert result["build_plan"] == []
    # Error should NOT be called — the file exists and parses fine
    ui.error.assert_not_called()
    # Artifact still saved
    mock_save.assert_called_once()


@pytest.mark.asyncio
async def test_plan_node_passes_constraints_and_max_units(tmp_path):
    """Constraints and max_units are forwarded in the agent prompt."""
    state = _make_state(
        tmp_path,
        constraints=["no breaking changes", "keep bundle < 200kb"],
        max_units=5,
    )
    repo = Path(state["repo_path"])
    repo.mkdir(parents=True, exist_ok=True)
    ui = _make_ui()

    # Write a minimal plan so the rest of the function proceeds
    (repo / "build_plan.json").write_text(json.dumps({"units": []}))

    with patch("graft.stages.plan.run_agent", new_callable=AsyncMock, return_value=FakeAgentResult()) as mock_agent, \
         patch("graft.stages.plan.save_artifact"), \
         patch("graft.stages.plan.mark_stage_complete"):

        await plan_node(state, ui)

    # Inspect the user_prompt sent to run_agent
    call_kwargs = mock_agent.call_args
    user_prompt = call_kwargs.kwargs["user_prompt"]
    assert "no breaking changes" in user_prompt
    assert "keep bundle < 200kb" in user_prompt
    assert "Maximum build units: 5" in user_prompt


@pytest.mark.asyncio
async def test_plan_node_no_constraints_omitted_from_prompt(tmp_path):
    """When constraints is empty and max_units is 0, those sections are absent."""
    state = _make_state(tmp_path, constraints=[], max_units=0)
    repo = Path(state["repo_path"])
    repo.mkdir(parents=True, exist_ok=True)
    ui = _make_ui()

    (repo / "build_plan.json").write_text(json.dumps({"units": []}))

    with patch("graft.stages.plan.run_agent", new_callable=AsyncMock, return_value=FakeAgentResult()) as mock_agent, \
         patch("graft.stages.plan.save_artifact"), \
         patch("graft.stages.plan.mark_stage_complete"):

        await plan_node(state, ui)

    user_prompt = mock_agent.call_args.kwargs["user_prompt"]
    assert "CONSTRAINTS" not in user_prompt
    assert "Maximum build units" not in user_prompt


@pytest.mark.asyncio
async def test_plan_node_agent_receives_correct_kwargs(tmp_path):
    """Verify run_agent is called with the expected keyword arguments."""
    state = _make_state(tmp_path, model="claude-sonnet-4-20250514")
    repo = Path(state["repo_path"])
    repo.mkdir(parents=True, exist_ok=True)
    ui = _make_ui()

    (repo / "build_plan.json").write_text(json.dumps({"units": []}))

    with patch("graft.stages.plan.run_agent", new_callable=AsyncMock, return_value=FakeAgentResult()) as mock_agent, \
         patch("graft.stages.plan.save_artifact"), \
         patch("graft.stages.plan.mark_stage_complete"):

        await plan_node(state, ui)

    kwargs = mock_agent.call_args.kwargs
    assert kwargs["persona"] == "Staff Software Architect (Implementation Planner)"
    assert kwargs["cwd"] == str(repo)
    assert kwargs["project_dir"] == state["project_dir"]
    assert kwargs["stage"] == "plan"
    assert kwargs["ui"] is ui
    assert kwargs["model"] == "claude-sonnet-4-20250514"
    assert kwargs["max_turns"] == 25
    assert kwargs["allowed_tools"] == ["Read", "Bash", "Glob", "Grep"]


@pytest.mark.asyncio
async def test_plan_node_plan_file_cleaned_up_after_success(tmp_path):
    """build_plan.json is removed from repo_path after being read."""
    state = _make_state(tmp_path)
    repo = Path(state["repo_path"])
    repo.mkdir(parents=True, exist_ok=True)
    ui = _make_ui()

    plan_file = repo / "build_plan.json"
    plan_file.write_text(json.dumps(_sample_plan_raw(1)))

    with patch("graft.stages.plan.run_agent", new_callable=AsyncMock, return_value=FakeAgentResult()), \
         patch("graft.stages.plan.save_artifact"), \
         patch("graft.stages.plan.mark_stage_complete"):

        await plan_node(state, ui)

    assert not plan_file.exists()


# ===========================================================================
# NEW — plan_review_node async tests
# ===========================================================================


@pytest.mark.asyncio
async def test_plan_review_node_auto_approve(tmp_path):
    """When auto_approve is True, plan is approved without prompting."""
    state = _make_state(tmp_path, auto_approve=True, build_plan=_sample_plan(2))
    ui = _make_ui()

    result = await plan_review_node(state, ui)

    assert result == {"plan_approved": True}
    ui.show_artifact.assert_called_once()
    ui.info.assert_any_call("Plan auto-approved.")
    # prompt_plan_review should NOT be called
    ui.prompt_plan_review.assert_not_called()


@pytest.mark.asyncio
async def test_plan_review_node_manual_approve(tmp_path):
    """When user manually approves, return plan_approved True."""
    state = _make_state(tmp_path, auto_approve=False, build_plan=_sample_plan(2))
    ui = _make_ui()
    ui.prompt_plan_review.return_value = (True, "")

    result = await plan_review_node(state, ui)

    assert result == {"plan_approved": True}
    ui.prompt_plan_review.assert_called_once()
    ui.info.assert_any_call("Plan approved — proceeding to execute.")


@pytest.mark.asyncio
async def test_plan_review_node_manual_reject_with_feedback(tmp_path):
    """When user rejects, feedback is logged; still proceeds (re-plan not implemented)."""
    state = _make_state(tmp_path, auto_approve=False, build_plan=_sample_plan(2))
    ui = _make_ui()
    ui.prompt_plan_review.return_value = (False, "Need more tests")

    result = await plan_review_node(state, ui)

    # Currently proceeds anyway (re-planning not yet implemented)
    assert result == {"plan_approved": True}
    ui.info.assert_any_call("Plan feedback received: Need more tests")
    ui.info.assert_any_call(
        "Re-planning is not yet implemented — proceeding with current plan."
    )


@pytest.mark.asyncio
async def test_plan_review_node_empty_plan(tmp_path):
    """Review node handles an empty plan gracefully."""
    state = _make_state(tmp_path, auto_approve=True, build_plan=[])
    ui = _make_ui()

    result = await plan_review_node(state, ui)

    assert result == {"plan_approved": True}
    # The summary should mention 0 units
    summary_arg = ui.show_artifact.call_args[0][1]
    assert "Total build units: 0" in summary_arg


@pytest.mark.asyncio
async def test_plan_review_node_cost_display(tmp_path):
    """Cost estimate appears in the summary shown to the user."""
    plan = _sample_plan(2, risk="high")
    state = _make_state(tmp_path, auto_approve=True, build_plan=plan)
    ui = _make_ui()

    await plan_review_node(state, ui)

    summary = ui.show_artifact.call_args[0][1]
    low, high = estimate_cost(plan)
    assert f"${low:.2f}" in summary
    assert f"${high:.2f}" in summary


@pytest.mark.asyncio
async def test_plan_review_node_summary_contains_unit_details(tmp_path):
    """Summary lines include unit_id, title, risk, category and pattern_reference."""
    plan = [
        {
            "unit_id": "feat_01",
            "title": "Create migration",
            "risk": "low",
            "category": "database",
            "blast_radius": "1 file",
            "pattern_reference": "migrations/001.sql",
            "tests_included": True,
        },
        {
            "unit_id": "feat_02",
            "title": "Add API endpoint",
            "risk": "high",
            "category": "api",
            "blast_radius": "3 files",
            "pattern_reference": "",
            "tests_included": False,
        },
    ]
    state = _make_state(tmp_path, auto_approve=True, build_plan=plan)
    ui = _make_ui()

    await plan_review_node(state, ui)

    summary = ui.show_artifact.call_args[0][1]
    assert "feat_01" in summary
    assert "Create migration" in summary
    assert "feat_02" in summary
    assert "Add API endpoint" in summary
    assert "database" in summary
    assert "api" in summary
    assert "migrations/001.sql" in summary
    # tests_included True should render +tests tag
    assert "+tests" in summary
