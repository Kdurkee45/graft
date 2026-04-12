"""Tests for graft.stages.plan."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graft.stages.plan import (
    SYSTEM_PROMPT,
    estimate_cost,
    plan_node,
    plan_review_node,
    plan_review_router,
)

# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_UNIT_LOW = {
    "unit_id": "feat_01",
    "title": "Create migration",
    "description": "Add new table",
    "category": "database",
    "service": "packages/db",
    "risk": "low",
    "blast_radius": "1 file (new migration)",
    "depends_on": [],
    "acceptance_criteria": ["migration runs cleanly"],
    "pattern_reference": "migrations/001.sql",
    "tests_included": False,
}

_SAMPLE_UNIT_MEDIUM = {
    "unit_id": "feat_02",
    "title": "Add API endpoint",
    "description": "REST handler",
    "category": "api",
    "service": "packages/api",
    "risk": "medium",
    "blast_radius": "2 files",
    "depends_on": ["feat_01"],
    "acceptance_criteria": ["returns 200"],
    "pattern_reference": "routes/health.ts",
    "tests_included": True,
}

_SAMPLE_UNIT_HIGH = {
    "unit_id": "feat_03",
    "title": "Refactor auth middleware",
    "description": "Breaking change in auth flow",
    "category": "integration",
    "service": "packages/auth",
    "risk": "high",
    "blast_radius": "5 files",
    "depends_on": ["feat_01", "feat_02"],
    "acceptance_criteria": ["auth still works", "no regressions"],
    "pattern_reference": "",
    "tests_included": False,
}

_SAMPLE_BUILD_PLAN = {
    "plan_id": "feat_abc123",
    "feature_name": "Test Feature",
    "total_units": 3,
    "estimated_cost": "$5-12",
    "units": [_SAMPLE_UNIT_LOW, _SAMPLE_UNIT_MEDIUM, _SAMPLE_UNIT_HIGH],
}


@dataclass
class FakeAgentResult:
    """Minimal stand-in for graft.agent.AgentResult."""

    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    raw_messages: list = field(default_factory=list)
    elapsed_seconds: float = 1.0
    turns_used: int = 5


@pytest.fixture
def repo(tmp_path):
    """Temporary repo directory."""
    d = tmp_path / "repo"
    d.mkdir()
    return d


@pytest.fixture
def project(tmp_path):
    """Temporary project directory with required sub-structure."""
    d = tmp_path / "project"
    d.mkdir()
    (d / "artifacts").mkdir()
    (d / "logs").mkdir()
    meta = {"project_id": "feat_test01", "stages_completed": []}
    (d / "metadata.json").write_text(json.dumps(meta))
    return d


@pytest.fixture
def ui():
    """Mock UI object exposing the methods plan nodes call."""
    m = MagicMock()
    m.stage_start = MagicMock()
    m.stage_done = MagicMock()
    m.error = MagicMock()
    m.info = MagicMock()
    m.show_artifact = MagicMock()
    m.prompt_plan_review = MagicMock(return_value=(True, ""))
    return m


def _state(repo, project, **kw):
    """Build a minimal FeatureState dict."""
    base = {
        "repo_path": str(repo),
        "project_dir": str(project),
    }
    base.update(kw)
    return base


def _write_build_plan(repo, plan_dict):
    """Write a build_plan.json into the repo directory."""
    (Path(repo) / "build_plan.json").write_text(json.dumps(plan_dict))


# ---------------------------------------------------------------------------
# Existing tests (preserved)
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


def test_plan_review_router_approved():
    assert plan_review_router({"plan_approved": True}) == "execute"


def test_plan_review_router_not_approved():
    assert plan_review_router({"plan_approved": False}) == "plan"


# ---------------------------------------------------------------------------
# estimate_cost — additional coverage
# ---------------------------------------------------------------------------


class TestEstimateCost:
    """Additional cost estimation tests."""

    def test_unknown_risk_defaults_to_medium(self):
        """Units with unrecognised risk fall back to 'medium' cost."""
        plan = [{"risk": "critical"}]
        low, high = estimate_cost(plan)
        # Pipeline overhead + 1 medium unit
        assert low == round(3.00 + 0.60, 2)
        assert high == round(8.00 + 1.50, 2)

    def test_missing_risk_defaults_to_medium(self):
        """Units without a risk key default to 'medium'."""
        plan = [{}]
        low, high = estimate_cost(plan)
        assert low == round(3.00 + 0.60, 2)
        assert high == round(8.00 + 1.50, 2)

    def test_single_low_risk_unit(self):
        low, high = estimate_cost([{"risk": "low"}])
        assert low == round(3.00 + 0.30, 2)
        assert high == round(8.00 + 0.80, 2)

    def test_single_high_risk_unit(self):
        low, high = estimate_cost([{"risk": "high"}])
        assert low == round(3.00 + 1.00, 2)
        assert high == round(8.00 + 3.00, 2)

    def test_many_units_accumulate(self):
        """10 low-risk units should accumulate linearly."""
        plan = [{"risk": "low"}] * 10
        low, high = estimate_cost(plan)
        assert low == round(3.00 + 10 * 0.30, 2)
        assert high == round(8.00 + 10 * 0.80, 2)

    def test_returns_rounded_to_two_decimals(self):
        """Result must be rounded to 2 decimal places."""
        plan = [{"risk": "low"}, {"risk": "low"}, {"risk": "low"}]
        low, high = estimate_cost(plan)
        assert low == round(low, 2)
        assert high == round(high, 2)


# ---------------------------------------------------------------------------
# plan_node
# ---------------------------------------------------------------------------


class TestPlanNodeHappyPath:
    """plan_node with a well-formed build_plan.json from the agent."""

    async def test_returns_build_plan_units(self, repo, project, ui):
        """plan_node reads build_plan.json and returns its units list."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="done")
            _write_build_plan(repo, _SAMPLE_BUILD_PLAN)

            result = await plan_node(_state(repo, project), ui)

        assert result["build_plan"] == _SAMPLE_BUILD_PLAN["units"]
        assert result["current_stage"] == "plan"

    async def test_saves_artifact(self, repo, project, ui):
        """build_plan.json is persisted via save_artifact."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_build_plan(repo, _SAMPLE_BUILD_PLAN)

            await plan_node(_state(repo, project), ui)

        artifact_path = project / "artifacts" / "build_plan.json"
        assert artifact_path.exists()
        saved = json.loads(artifact_path.read_text())
        assert saved["units"] == _SAMPLE_BUILD_PLAN["units"]

    async def test_cleans_up_plan_file_from_repo(self, repo, project, ui):
        """build_plan.json in the repo is deleted after being read."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_build_plan(repo, _SAMPLE_BUILD_PLAN)

            await plan_node(_state(repo, project), ui)

        assert not (Path(repo) / "build_plan.json").exists()

    async def test_calls_mark_stage_complete(self, repo, project, ui):
        """plan stage is marked complete in metadata."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_build_plan(repo, _SAMPLE_BUILD_PLAN)

            await plan_node(_state(repo, project), ui)

        meta = json.loads((project / "metadata.json").read_text())
        assert "plan" in meta["stages_completed"]

    async def test_ui_lifecycle(self, repo, project, ui):
        """stage_start and stage_done are called in order."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_build_plan(repo, _SAMPLE_BUILD_PLAN)

            await plan_node(_state(repo, project), ui)

        ui.stage_start.assert_called_once_with("plan")
        ui.stage_done.assert_called_once_with("plan")


class TestPlanNodeAgentCall:
    """Verify run_agent is invoked with the right parameters."""

    async def test_agent_receives_correct_persona_and_tools(self, repo, project, ui):
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_build_plan(repo, _SAMPLE_BUILD_PLAN)

            await plan_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["persona"] == "Staff Software Architect (Implementation Planner)"
        assert kwargs["system_prompt"] == SYSTEM_PROMPT
        assert kwargs["stage"] == "plan"
        assert kwargs["max_turns"] == 25
        assert kwargs["allowed_tools"] == ["Read", "Bash", "Glob", "Grep"]
        assert kwargs["cwd"] == str(repo)
        assert kwargs["project_dir"] == str(project)

    async def test_prompt_contains_feature_prompt(self, repo, project, ui):
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_build_plan(repo, _SAMPLE_BUILD_PLAN)

            await plan_node(_state(repo, project, feature_prompt="Add dark mode"), ui)

        _, kwargs = mock_run.call_args
        assert "Add dark mode" in kwargs["user_prompt"]

    async def test_prompt_includes_constraints(self, repo, project, ui):
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_build_plan(repo, _SAMPLE_BUILD_PLAN)

            await plan_node(
                _state(repo, project, constraints=["no breaking changes", "use React"]),
                ui,
            )

        _, kwargs = mock_run.call_args
        assert "no breaking changes" in kwargs["user_prompt"]
        assert "use React" in kwargs["user_prompt"]
        assert "CONSTRAINTS:" in kwargs["user_prompt"]

    async def test_prompt_includes_max_units(self, repo, project, ui):
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_build_plan(repo, _SAMPLE_BUILD_PLAN)

            await plan_node(_state(repo, project, max_units=5), ui)

        _, kwargs = mock_run.call_args
        assert "Maximum build units: 5" in kwargs["user_prompt"]

    async def test_prompt_omits_constraints_when_empty(self, repo, project, ui):
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_build_plan(repo, _SAMPLE_BUILD_PLAN)

            await plan_node(_state(repo, project, constraints=[]), ui)

        _, kwargs = mock_run.call_args
        assert "CONSTRAINTS:" not in kwargs["user_prompt"]

    async def test_prompt_omits_max_units_when_zero(self, repo, project, ui):
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_build_plan(repo, _SAMPLE_BUILD_PLAN)

            await plan_node(_state(repo, project, max_units=0), ui)

        _, kwargs = mock_run.call_args
        assert "Maximum build units" not in kwargs["user_prompt"]

    async def test_prompt_includes_codebase_profile(self, repo, project, ui):
        profile = {"framework": "Next.js", "language": "TypeScript"}
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_build_plan(repo, _SAMPLE_BUILD_PLAN)

            await plan_node(_state(repo, project, codebase_profile=profile), ui)

        _, kwargs = mock_run.call_args
        assert "Next.js" in kwargs["user_prompt"]
        assert "CODEBASE PROFILE:" in kwargs["user_prompt"]

    async def test_model_forwarded(self, repo, project, ui):
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_build_plan(repo, _SAMPLE_BUILD_PLAN)

            await plan_node(_state(repo, project, model="claude-sonnet-4-20250514"), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["model"] == "claude-sonnet-4-20250514"


class TestPlanNodeErrorHandling:
    """plan_node when the agent produces no or malformed output."""

    async def test_missing_build_plan_json(self, repo, project, ui):
        """Agent didn't write build_plan.json — returns empty plan."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            # Deliberately don't write build_plan.json

            result = await plan_node(_state(repo, project), ui)

        assert result["build_plan"] == []
        ui.error.assert_called_once_with("Agent did not produce build_plan.json.")

    async def test_malformed_json(self, repo, project, ui):
        """build_plan.json with invalid JSON — returns empty plan."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            (Path(repo) / "build_plan.json").write_text("{not valid json!!!")

            result = await plan_node(_state(repo, project), ui)

        assert result["build_plan"] == []
        ui.error.assert_called_once_with(
            "Failed to parse build_plan.json — using empty plan."
        )

    async def test_malformed_json_still_saves_empty_artifact(self, repo, project, ui):
        """Even on parse failure, an (empty) artifact is saved."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            (Path(repo) / "build_plan.json").write_text("{bad json")

            await plan_node(_state(repo, project), ui)

        artifact = project / "artifacts" / "build_plan.json"
        assert artifact.exists()
        # plan_raw stays {} on parse error, so artifact is "{}"
        assert json.loads(artifact.read_text()) == {}

    async def test_malformed_json_cleans_up_file(self, repo, project, ui):
        """Malformed build_plan.json in repo is still deleted."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            (Path(repo) / "build_plan.json").write_text("{bad")

            await plan_node(_state(repo, project), ui)

        assert not (Path(repo) / "build_plan.json").exists()

    async def test_missing_file_still_marks_stage_complete(self, repo, project, ui):
        """Stage is always marked complete even without output."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()

            await plan_node(_state(repo, project), ui)

        meta = json.loads((project / "metadata.json").read_text())
        assert "plan" in meta["stages_completed"]

    async def test_plan_json_with_no_units_key(self, repo, project, ui):
        """build_plan.json that is valid JSON but has no 'units' key."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            plan_no_units = {"plan_id": "feat_abc", "feature_name": "test"}
            _write_build_plan(repo, plan_no_units)

            result = await plan_node(_state(repo, project), ui)

        assert result["build_plan"] == []
        # No error called — JSON was valid, just missing units
        ui.error.assert_not_called()

    async def test_plan_json_empty_units_list(self, repo, project, ui):
        """build_plan.json with an empty units array."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_build_plan(repo, {**_SAMPLE_BUILD_PLAN, "units": []})

            result = await plan_node(_state(repo, project), ui)

        assert result["build_plan"] == []
        ui.error.assert_not_called()


# ---------------------------------------------------------------------------
# plan_review_node
# ---------------------------------------------------------------------------


class TestPlanReviewNodeAutoApprove:
    """plan_review_node with auto_approve=True."""

    async def test_auto_approve_returns_approved(self, repo, project, ui):
        state = _state(
            repo,
            project,
            build_plan=[_SAMPLE_UNIT_LOW],
            auto_approve=True,
        )
        result = await plan_review_node(state, ui)
        assert result == {"plan_approved": True}

    async def test_auto_approve_shows_artifact(self, repo, project, ui):
        state = _state(
            repo,
            project,
            build_plan=[_SAMPLE_UNIT_LOW],
            auto_approve=True,
        )
        await plan_review_node(state, ui)
        ui.show_artifact.assert_called_once()
        title, summary = ui.show_artifact.call_args[0]
        assert title == "Build Plan"
        assert "Total build units: 1" in summary

    async def test_auto_approve_logs_info(self, repo, project, ui):
        state = _state(
            repo,
            project,
            build_plan=[],
            auto_approve=True,
        )
        await plan_review_node(state, ui)
        ui.info.assert_called_once_with("Plan auto-approved.")

    async def test_auto_approve_skips_prompt(self, repo, project, ui):
        """prompt_plan_review must not be called when auto-approving."""
        state = _state(
            repo,
            project,
            build_plan=[_SAMPLE_UNIT_LOW],
            auto_approve=True,
        )
        await plan_review_node(state, ui)
        ui.prompt_plan_review.assert_not_called()


class TestPlanReviewNodeInteractive:
    """plan_review_node with interactive approval (auto_approve=False)."""

    async def test_interactive_approval(self, repo, project, ui):
        ui.prompt_plan_review.return_value = (True, "")
        state = _state(
            repo,
            project,
            build_plan=[_SAMPLE_UNIT_LOW, _SAMPLE_UNIT_MEDIUM],
            auto_approve=False,
        )
        result = await plan_review_node(state, ui)

        assert result == {"plan_approved": True}
        ui.prompt_plan_review.assert_called_once()
        ui.info.assert_called_once_with("Plan approved — proceeding to execute.")

    async def test_interactive_passes_summary_to_prompt(self, repo, project, ui):
        """The summary string is forwarded to prompt_plan_review."""
        ui.prompt_plan_review.return_value = (True, "")
        state = _state(
            repo,
            project,
            build_plan=[_SAMPLE_UNIT_LOW],
            auto_approve=False,
        )
        await plan_review_node(state, ui)

        summary_arg = ui.prompt_plan_review.call_args[0][0]
        assert "Total build units: 1" in summary_arg
        assert "Estimated cost:" in summary_arg

    async def test_rejection_returns_not_approved(self, repo, project, ui):
        """Rejection sets plan_approved=False to route back to plan stage."""
        ui.prompt_plan_review.return_value = (False, "needs more tests")
        state = _state(
            repo,
            project,
            build_plan=[_SAMPLE_UNIT_LOW],
            auto_approve=False,
        )
        result = await plan_review_node(state, ui)

        assert result["plan_approved"] is False
        assert result["plan_feedback"] == "needs more tests"

    async def test_rejection_logs_feedback(self, repo, project, ui):
        """On rejection, feedback is logged with re-planning message."""
        ui.prompt_plan_review.return_value = (False, "needs more tests")
        state = _state(
            repo,
            project,
            build_plan=[_SAMPLE_UNIT_LOW],
            auto_approve=False,
        )
        await plan_review_node(state, ui)

        calls = ui.info.call_args_list
        assert any("needs more tests" in str(c) for c in calls)
        assert any("re-planning" in str(c).lower() for c in calls)

    async def test_default_auto_approve_is_false(self, repo, project, ui):
        """When auto_approve is not in state, defaults to False (interactive)."""
        ui.prompt_plan_review.return_value = (True, "")
        state = _state(
            repo,
            project,
            build_plan=[_SAMPLE_UNIT_LOW],
            # no auto_approve key
        )
        await plan_review_node(state, ui)

        # Should take interactive path
        ui.prompt_plan_review.assert_called_once()
        ui.show_artifact.assert_not_called()


class TestPlanReviewNodeSummary:
    """Verify the plan summary content and formatting."""

    async def test_summary_contains_unit_count_and_cost(self, repo, project, ui):
        state = _state(
            repo,
            project,
            build_plan=[_SAMPLE_UNIT_LOW, _SAMPLE_UNIT_MEDIUM, _SAMPLE_UNIT_HIGH],
            auto_approve=True,
        )
        await plan_review_node(state, ui)

        summary = ui.show_artifact.call_args[0][1]
        assert "Total build units: 3" in summary
        assert "Estimated cost:" in summary

    async def test_summary_risk_color_low(self, repo, project, ui):
        """Low-risk units use green color tag."""
        state = _state(
            repo,
            project,
            build_plan=[_SAMPLE_UNIT_LOW],
            auto_approve=True,
        )
        await plan_review_node(state, ui)

        summary = ui.show_artifact.call_args[0][1]
        assert "[green]" in summary
        assert "[/green]" in summary

    async def test_summary_risk_color_medium(self, repo, project, ui):
        """Medium-risk units use yellow color tag."""
        state = _state(
            repo,
            project,
            build_plan=[_SAMPLE_UNIT_MEDIUM],
            auto_approve=True,
        )
        await plan_review_node(state, ui)

        summary = ui.show_artifact.call_args[0][1]
        assert "[yellow]" in summary
        assert "[/yellow]" in summary

    async def test_summary_risk_color_high(self, repo, project, ui):
        """High-risk units use red color tag."""
        state = _state(
            repo,
            project,
            build_plan=[_SAMPLE_UNIT_HIGH],
            auto_approve=True,
        )
        await plan_review_node(state, ui)

        summary = ui.show_artifact.call_args[0][1]
        assert "[red]" in summary
        assert "[/red]" in summary

    async def test_summary_unknown_risk_uses_white(self, repo, project, ui):
        """Units with unrecognised risk get white color."""
        unit = {**_SAMPLE_UNIT_LOW, "risk": "critical"}
        state = _state(
            repo,
            project,
            build_plan=[unit],
            auto_approve=True,
        )
        await plan_review_node(state, ui)

        summary = ui.show_artifact.call_args[0][1]
        assert "[white]" in summary

    async def test_summary_missing_risk_uses_white(self, repo, project, ui):
        """Units without a risk key get white color (empty string not in map)."""
        unit = {k: v for k, v in _SAMPLE_UNIT_LOW.items() if k != "risk"}
        state = _state(
            repo,
            project,
            build_plan=[unit],
            auto_approve=True,
        )
        await plan_review_node(state, ui)

        summary = ui.show_artifact.call_args[0][1]
        assert "[white]" in summary

    async def test_summary_tests_included_tag(self, repo, project, ui):
        """Units with tests_included=True show +tests dim tag."""
        state = _state(
            repo,
            project,
            build_plan=[_SAMPLE_UNIT_MEDIUM],  # tests_included=True
            auto_approve=True,
        )
        await plan_review_node(state, ui)

        summary = ui.show_artifact.call_args[0][1]
        assert "[dim]+tests[/dim]" in summary

    async def test_summary_no_tests_tag_when_false(self, repo, project, ui):
        """Units with tests_included=False do not show +tests tag."""
        state = _state(
            repo,
            project,
            build_plan=[_SAMPLE_UNIT_LOW],  # tests_included=False
            auto_approve=True,
        )
        await plan_review_node(state, ui)

        summary = ui.show_artifact.call_args[0][1]
        assert "+tests" not in summary

    async def test_summary_pattern_reference_shown(self, repo, project, ui):
        """Units with a pattern_reference show the pattern line."""
        state = _state(
            repo,
            project,
            build_plan=[_SAMPLE_UNIT_LOW],
            auto_approve=True,
        )
        await plan_review_node(state, ui)

        summary = ui.show_artifact.call_args[0][1]
        assert "pattern: migrations/001.sql" in summary

    async def test_summary_no_pattern_line_when_empty(self, repo, project, ui):
        """Units with empty pattern_reference omit the pattern line."""
        state = _state(
            repo,
            project,
            build_plan=[_SAMPLE_UNIT_HIGH],  # pattern_reference=""
            auto_approve=True,
        )
        await plan_review_node(state, ui)

        summary = ui.show_artifact.call_args[0][1]
        assert "pattern:" not in summary

    async def test_summary_unit_id_and_title(self, repo, project, ui):
        state = _state(
            repo,
            project,
            build_plan=[_SAMPLE_UNIT_LOW],
            auto_approve=True,
        )
        await plan_review_node(state, ui)

        summary = ui.show_artifact.call_args[0][1]
        assert "feat_01" in summary
        assert "Create migration" in summary

    async def test_summary_category_and_blast_radius(self, repo, project, ui):
        state = _state(
            repo,
            project,
            build_plan=[_SAMPLE_UNIT_LOW],
            auto_approve=True,
        )
        await plan_review_node(state, ui)

        summary = ui.show_artifact.call_args[0][1]
        assert "[database]" in summary
        assert "1 file (new migration)" in summary

    async def test_empty_plan_summary(self, repo, project, ui):
        """Empty plan shows zero units with overhead cost."""
        state = _state(
            repo,
            project,
            build_plan=[],
            auto_approve=True,
        )
        await plan_review_node(state, ui)

        summary = ui.show_artifact.call_args[0][1]
        assert "Total build units: 0" in summary
        # Only pipeline overhead
        assert "$3.00" in summary
        assert "$8.00" in summary

    async def test_summary_multiple_units_numbered(self, repo, project, ui):
        """Units are numbered sequentially starting from 1."""
        state = _state(
            repo,
            project,
            build_plan=[_SAMPLE_UNIT_LOW, _SAMPLE_UNIT_MEDIUM, _SAMPLE_UNIT_HIGH],
            auto_approve=True,
        )
        await plan_review_node(state, ui)

        summary = ui.show_artifact.call_args[0][1]
        assert "  1." in summary
        assert "  2." in summary
        assert "  3." in summary

    async def test_summary_defaults_for_missing_fields(self, repo, project, ui):
        """Units missing optional fields show '?' fallbacks."""
        bare_unit = {}  # No keys at all
        state = _state(
            repo,
            project,
            build_plan=[bare_unit],
            auto_approve=True,
        )
        await plan_review_node(state, ui)

        summary = ui.show_artifact.call_args[0][1]
        assert "?: Untitled" in summary
        assert "[?]" in summary  # category fallback


# ---------------------------------------------------------------------------
# plan_review_router — additional coverage
# ---------------------------------------------------------------------------


class TestPlanReviewRouter:
    """Extra router edge cases."""

    def test_missing_plan_approved_key(self):
        """When plan_approved is absent, routes back to plan."""
        assert plan_review_router({}) == "plan"

    def test_truthy_non_bool_routes_to_execute(self):
        """Any truthy value for plan_approved routes to execute."""
        assert plan_review_router({"plan_approved": 1}) == "execute"

    def test_falsy_none_routes_to_plan(self):
        assert plan_review_router({"plan_approved": None}) == "plan"
