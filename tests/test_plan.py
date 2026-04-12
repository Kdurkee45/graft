"""Tests for graft.stages.plan."""

import json
from dataclasses import dataclass, field
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
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeAgentResult:
    """Minimal stand-in for graft.agent.AgentResult."""

    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    raw_messages: list = field(default_factory=list)
    elapsed_seconds: float = 1.0
    turns_used: int = 5


def _sample_plan_raw(units=None):
    """Return a valid build_plan.json dict."""
    if units is None:
        units = [
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
                "pattern_reference": "migrations/001_init.sql",
                "tests_included": False,
            },
            {
                "unit_id": "feat_02",
                "title": "Add trades API endpoint",
                "description": "REST endpoint for trades CRUD",
                "category": "api",
                "service": "packages/api",
                "risk": "medium",
                "blast_radius": "2 files",
                "depends_on": ["feat_01"],
                "acceptance_criteria": ["GET /trades returns 200"],
                "pattern_reference": "src/routes/orders.ts",
                "tests_included": True,
            },
            {
                "unit_id": "feat_03",
                "title": "Build trades dashboard",
                "description": "React component for trades view",
                "category": "component",
                "service": "packages/web",
                "risk": "high",
                "blast_radius": "5 files",
                "depends_on": ["feat_02"],
                "acceptance_criteria": ["renders trade list"],
                "pattern_reference": "src/components/OrderDash.tsx",
                "tests_included": True,
            },
        ]
    return {
        "plan_id": "feat_XXXXX",
        "feature_name": "Trades Feature",
        "total_units": len(units),
        "estimated_cost": "$8-15",
        "units": units,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    """Mock UI object exposing the methods plan stages call."""
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


# ---------------------------------------------------------------------------
# estimate_cost (existing tests, preserved)
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
    """Unknown risk level falls back to medium cost."""
    plan_unknown = [{"risk": "unknown_level"}]
    plan_medium = [{"risk": "medium"}]
    assert estimate_cost(plan_unknown) == estimate_cost(plan_medium)


def test_estimate_cost_missing_risk_key_defaults_to_medium():
    """Unit without 'risk' key falls back to medium cost."""
    plan_missing = [{}]
    plan_medium = [{"risk": "medium"}]
    assert estimate_cost(plan_missing) == estimate_cost(plan_medium)


def test_estimate_cost_returns_rounded_floats():
    """Values are rounded to 2 decimal places."""
    plan = [{"risk": "low"}, {"risk": "low"}, {"risk": "low"}]
    low, high = estimate_cost(plan)
    assert low == round(low, 2)
    assert high == round(high, 2)


def test_estimate_cost_exact_values():
    """Verify exact dollar amounts for known inputs."""
    # overhead = (3.00, 8.00), low = (0.30, 0.80)
    low, high = estimate_cost([{"risk": "low"}])
    assert low == 3.30
    assert high == 8.80


# ---------------------------------------------------------------------------
# plan_review_router (existing tests, preserved)
# ---------------------------------------------------------------------------


def test_plan_review_router_approved():
    assert plan_review_router({"plan_approved": True}) == "execute"


def test_plan_review_router_not_approved():
    assert plan_review_router({"plan_approved": False}) == "plan"


def test_plan_review_router_missing_key():
    """Missing plan_approved key should route back to plan."""
    assert plan_review_router({}) == "plan"


# ---------------------------------------------------------------------------
# plan_node — happy path
# ---------------------------------------------------------------------------


class TestPlanNodeHappyPath:
    """Core happy-path tests where agent produces valid build_plan.json."""

    async def test_returns_expected_keys(self, repo, project, ui):
        """plan_node returns build_plan and current_stage."""
        plan_raw = _sample_plan_raw()
        (repo / "build_plan.json").write_text(json.dumps(plan_raw))

        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="done")
            result = await plan_node(_state(repo, project), ui)

        assert set(result.keys()) == {"build_plan", "current_stage"}
        assert result["current_stage"] == "plan"

    async def test_returns_units_from_plan(self, repo, project, ui):
        """build_plan in result contains the units list."""
        plan_raw = _sample_plan_raw()
        (repo / "build_plan.json").write_text(json.dumps(plan_raw))

        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="done")
            result = await plan_node(_state(repo, project), ui)

        assert len(result["build_plan"]) == 3
        assert result["build_plan"][0]["unit_id"] == "feat_01"
        assert result["build_plan"][1]["unit_id"] == "feat_02"
        assert result["build_plan"][2]["unit_id"] == "feat_03"

    async def test_saves_artifact(self, repo, project, ui):
        """build_plan.json is persisted as an artifact."""
        plan_raw = _sample_plan_raw()
        (repo / "build_plan.json").write_text(json.dumps(plan_raw))

        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="done")
            await plan_node(_state(repo, project), ui)

        art_path = project / "artifacts" / "build_plan.json"
        assert art_path.exists()
        saved = json.loads(art_path.read_text())
        assert saved["plan_id"] == "feat_XXXXX"
        assert len(saved["units"]) == 3

    async def test_marks_stage_complete(self, repo, project, ui):
        """Stage 'plan' is recorded in metadata after success."""
        plan_raw = _sample_plan_raw()
        (repo / "build_plan.json").write_text(json.dumps(plan_raw))

        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="done")
            await plan_node(_state(repo, project), ui)

        meta = json.loads((project / "metadata.json").read_text())
        assert "plan" in meta["stages_completed"]

    async def test_cleans_up_temp_file(self, repo, project, ui):
        """build_plan.json is removed from repo after reading."""
        plan_raw = _sample_plan_raw()
        (repo / "build_plan.json").write_text(json.dumps(plan_raw))

        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="done")
            await plan_node(_state(repo, project), ui)

        assert not (repo / "build_plan.json").exists()

    async def test_ui_lifecycle(self, repo, project, ui):
        """stage_start and stage_done are called in order."""
        (repo / "build_plan.json").write_text(json.dumps(_sample_plan_raw()))

        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="done")
            await plan_node(_state(repo, project), ui)

        ui.stage_start.assert_called_once_with("plan")
        ui.stage_done.assert_called_once_with("plan")


# ---------------------------------------------------------------------------
# plan_node — agent invocation arguments
# ---------------------------------------------------------------------------


class TestPlanNodeAgentArgs:
    """Verify run_agent is invoked with the correct arguments."""

    async def test_system_prompt_is_constant(self, repo, project, ui):
        """run_agent receives the module-level SYSTEM_PROMPT."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await plan_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["system_prompt"] is SYSTEM_PROMPT

    async def test_allowed_tools(self, repo, project, ui):
        """Only Read, Bash, Glob, Grep tools are allowed."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await plan_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["allowed_tools"] == ["Read", "Bash", "Glob", "Grep"]

    async def test_stage_is_plan(self, repo, project, ui):
        """run_agent is called with stage='plan'."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await plan_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["stage"] == "plan"

    async def test_max_turns_is_25(self, repo, project, ui):
        """run_agent is called with max_turns=25."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await plan_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["max_turns"] == 25

    async def test_cwd_is_repo_path(self, repo, project, ui):
        """cwd passed to run_agent is the repo_path."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await plan_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["cwd"] == str(repo)

    async def test_model_forwarded(self, repo, project, ui):
        """model from state is passed through to run_agent."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await plan_node(_state(repo, project, model="claude-sonnet-4-20250514"), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["model"] == "claude-sonnet-4-20250514"

    async def test_persona(self, repo, project, ui):
        """Persona includes 'Architect'."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await plan_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert "Architect" in kwargs["persona"]


# ---------------------------------------------------------------------------
# plan_node — prompt construction
# ---------------------------------------------------------------------------


class TestPlanNodePrompt:
    """Verify the user prompt composition logic."""

    async def test_prompt_contains_repo_path(self, repo, project, ui):
        """User prompt references the repo path."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await plan_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert str(repo) in kwargs["user_prompt"]

    async def test_prompt_contains_feature_prompt(self, repo, project, ui):
        """User prompt includes the feature description."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await plan_node(_state(repo, project, feature_prompt="Add dark mode"), ui)

        _, kwargs = mock_run.call_args
        assert "Add dark mode" in kwargs["user_prompt"]

    async def test_prompt_contains_feature_spec(self, repo, project, ui):
        """User prompt includes serialized feature_spec."""
        spec = {"decisions": ["use CSS vars"], "qa": []}
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await plan_node(_state(repo, project, feature_spec=spec), ui)

        _, kwargs = mock_run.call_args
        assert "FEATURE SPEC" in kwargs["user_prompt"]
        assert "use CSS vars" in kwargs["user_prompt"]

    async def test_prompt_contains_technical_assessment(self, repo, project, ui):
        """User prompt includes serialized technical_assessment."""
        assessment = {"reuse": ["shared utils"], "gaps": ["no auth module"]}
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await plan_node(_state(repo, project, technical_assessment=assessment), ui)

        _, kwargs = mock_run.call_args
        assert "TECHNICAL ASSESSMENT" in kwargs["user_prompt"]
        assert "shared utils" in kwargs["user_prompt"]
        assert "no auth module" in kwargs["user_prompt"]

    async def test_prompt_contains_codebase_profile(self, repo, project, ui):
        """User prompt includes serialized codebase_profile."""
        profile = {"project": {"name": "acme", "lang": "typescript"}}
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await plan_node(_state(repo, project, codebase_profile=profile), ui)

        _, kwargs = mock_run.call_args
        assert "CODEBASE PROFILE" in kwargs["user_prompt"]
        assert "acme" in kwargs["user_prompt"]

    async def test_prompt_includes_constraints(self, repo, project, ui):
        """When constraints are set, they appear in the prompt."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await plan_node(
                _state(
                    repo,
                    project,
                    constraints=["no breaking changes", "keep < 10 units"],
                ),
                ui,
            )

        _, kwargs = mock_run.call_args
        assert "CONSTRAINTS" in kwargs["user_prompt"]
        assert "no breaking changes" in kwargs["user_prompt"]
        assert "keep < 10 units" in kwargs["user_prompt"]

    async def test_prompt_omits_constraints_when_empty(self, repo, project, ui):
        """When constraints is empty list, CONSTRAINTS section is omitted."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await plan_node(_state(repo, project, constraints=[]), ui)

        _, kwargs = mock_run.call_args
        assert "CONSTRAINTS" not in kwargs["user_prompt"]

    async def test_prompt_includes_max_units(self, repo, project, ui):
        """When max_units > 0, the limit appears in the prompt."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await plan_node(_state(repo, project, max_units=8), ui)

        _, kwargs = mock_run.call_args
        assert "Maximum build units: 8" in kwargs["user_prompt"]

    async def test_prompt_omits_max_units_when_zero(self, repo, project, ui):
        """When max_units is 0 (default), limit line is omitted."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await plan_node(_state(repo, project, max_units=0), ui)

        _, kwargs = mock_run.call_args
        assert "Maximum build units" not in kwargs["user_prompt"]

    async def test_prompt_ends_with_explore_instruction(self, repo, project, ui):
        """Prompt asks agent to explore codebase for pattern references."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await plan_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert "pattern_reference" in kwargs["user_prompt"]
        assert "Explore the codebase" in kwargs["user_prompt"]


# ---------------------------------------------------------------------------
# plan_node — error handling
# ---------------------------------------------------------------------------


class TestPlanNodeErrors:
    """Error paths: missing file, malformed JSON."""

    async def test_missing_build_plan_json(self, repo, project, ui):
        """When agent doesn't produce build_plan.json, error is shown and empty plan returned."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="oops")
            result = await plan_node(_state(repo, project), ui)

        ui.error.assert_called_once()
        assert "build_plan.json" in ui.error.call_args[0][0]
        assert result["build_plan"] == []

    async def test_malformed_json_shows_error(self, repo, project, ui):
        """Malformed JSON in build_plan.json triggers ui.error."""
        (repo / "build_plan.json").write_text("NOT VALID JSON {{{")

        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            result = await plan_node(_state(repo, project), ui)

        ui.error.assert_called_once()
        assert (
            "parse" in ui.error.call_args[0][0].lower()
            or "Failed" in ui.error.call_args[0][0]
        )
        assert result["build_plan"] == []

    async def test_malformed_json_still_saves_artifact(self, repo, project, ui):
        """Even with bad JSON, an empty artifact is saved."""
        (repo / "build_plan.json").write_text("{broken json")

        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await plan_node(_state(repo, project), ui)

        art_path = project / "artifacts" / "build_plan.json"
        assert art_path.exists()
        # plan_raw stays {} on parse failure, so artifact is serialized {}
        saved = json.loads(art_path.read_text())
        assert saved == {}

    async def test_malformed_json_still_marks_stage_complete(self, repo, project, ui):
        """Stage completes even when build_plan.json is malformed."""
        (repo / "build_plan.json").write_text(">>>invalid<<<")

        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await plan_node(_state(repo, project), ui)

        meta = json.loads((project / "metadata.json").read_text())
        assert "plan" in meta["stages_completed"]

    async def test_malformed_json_cleans_up_file(self, repo, project, ui):
        """build_plan.json is removed even when its content is broken."""
        (repo / "build_plan.json").write_text("oops")

        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await plan_node(_state(repo, project), ui)

        assert not (repo / "build_plan.json").exists()

    async def test_missing_file_still_marks_stage_complete(self, repo, project, ui):
        """Stage completes even when build_plan.json is absent."""
        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await plan_node(_state(repo, project), ui)

        meta = json.loads((project / "metadata.json").read_text())
        assert "plan" in meta["stages_completed"]

    async def test_valid_json_without_units_key(self, repo, project, ui):
        """JSON that parses but has no 'units' key yields empty plan."""
        (repo / "build_plan.json").write_text(json.dumps({"plan_id": "x"}))

        with patch("graft.stages.plan.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            result = await plan_node(_state(repo, project), ui)

        assert result["build_plan"] == []
        # No error call — the JSON parsed fine, just no units
        ui.error.assert_not_called()


# ---------------------------------------------------------------------------
# plan_review_node — auto-approve
# ---------------------------------------------------------------------------


class TestPlanReviewNodeAutoApprove:
    """plan_review_node with auto_approve=True."""

    async def test_auto_approve_returns_approved(self, repo, project, ui):
        """Auto-approved plan returns plan_approved=True."""
        plan = _sample_plan_raw()["units"]
        state = _state(repo, project, build_plan=plan, auto_approve=True)
        result = await plan_review_node(state, ui)

        assert result == {"plan_approved": True}

    async def test_auto_approve_shows_artifact(self, repo, project, ui):
        """Auto-approved plan calls ui.show_artifact with summary."""
        plan = _sample_plan_raw()["units"]
        state = _state(repo, project, build_plan=plan, auto_approve=True)
        await plan_review_node(state, ui)

        ui.show_artifact.assert_called_once()
        title, summary = ui.show_artifact.call_args[0]
        assert title == "Build Plan"
        assert "Total build units: 3" in summary

    async def test_auto_approve_shows_cost_estimate(self, repo, project, ui):
        """Auto-approved summary includes cost estimate."""
        plan = _sample_plan_raw()["units"]
        state = _state(repo, project, build_plan=plan, auto_approve=True)
        await plan_review_node(state, ui)

        _, summary = ui.show_artifact.call_args[0]
        assert "Estimated cost:" in summary
        assert "$" in summary

    async def test_auto_approve_does_not_prompt(self, repo, project, ui):
        """Auto-approved plan never calls prompt_plan_review."""
        plan = _sample_plan_raw()["units"]
        state = _state(repo, project, build_plan=plan, auto_approve=True)
        await plan_review_node(state, ui)

        ui.prompt_plan_review.assert_not_called()

    async def test_auto_approve_shows_info_message(self, repo, project, ui):
        """Auto-approved plan shows info message about auto-approval."""
        plan = _sample_plan_raw()["units"]
        state = _state(repo, project, build_plan=plan, auto_approve=True)
        await plan_review_node(state, ui)

        ui.info.assert_called_once()
        assert "auto-approved" in ui.info.call_args[0][0].lower()


# ---------------------------------------------------------------------------
# plan_review_node — human approval
# ---------------------------------------------------------------------------


class TestPlanReviewNodeHumanApproval:
    """plan_review_node with human interaction (auto_approve=False)."""

    async def test_approved_returns_plan_approved(self, repo, project, ui):
        """Human approval returns plan_approved=True."""
        ui.prompt_plan_review.return_value = (True, "")
        plan = _sample_plan_raw()["units"]
        state = _state(repo, project, build_plan=plan, auto_approve=False)
        result = await plan_review_node(state, ui)

        assert result == {"plan_approved": True}

    async def test_approved_shows_info(self, repo, project, ui):
        """Human approval shows 'proceeding to execute' info."""
        ui.prompt_plan_review.return_value = (True, "")
        plan = _sample_plan_raw()["units"]
        state = _state(repo, project, build_plan=plan, auto_approve=False)
        await plan_review_node(state, ui)

        ui.info.assert_called_once()
        assert "approved" in ui.info.call_args[0][0].lower()

    async def test_approval_passes_summary_to_prompt(self, repo, project, ui):
        """prompt_plan_review receives the formatted summary string."""
        ui.prompt_plan_review.return_value = (True, "")
        plan = _sample_plan_raw()["units"]
        state = _state(repo, project, build_plan=plan, auto_approve=False)
        await plan_review_node(state, ui)

        ui.prompt_plan_review.assert_called_once()
        summary = ui.prompt_plan_review.call_args[0][0]
        assert "Total build units: 3" in summary
        assert "feat_01" in summary
        assert "feat_02" in summary
        assert "feat_03" in summary

    async def test_does_not_show_artifact_when_not_auto(self, repo, project, ui):
        """When not auto-approved, show_artifact is NOT called (prompt handles display)."""
        ui.prompt_plan_review.return_value = (True, "")
        plan = _sample_plan_raw()["units"]
        state = _state(repo, project, build_plan=plan, auto_approve=False)
        await plan_review_node(state, ui)

        ui.show_artifact.assert_not_called()


# ---------------------------------------------------------------------------
# plan_review_node — human feedback (rejection)
# ---------------------------------------------------------------------------


class TestPlanReviewNodeFeedback:
    """plan_review_node with human feedback (not approved)."""

    async def test_feedback_still_returns_approved(self, repo, project, ui):
        """Feedback path currently proceeds with plan_approved=True (re-plan not yet implemented)."""
        ui.prompt_plan_review.return_value = (False, "Add more tests")
        plan = _sample_plan_raw()["units"]
        state = _state(repo, project, build_plan=plan, auto_approve=False)
        result = await plan_review_node(state, ui)

        # Current implementation: re-planning not yet implemented, proceeds anyway
        assert result == {"plan_approved": True}

    async def test_feedback_shows_info_with_feedback_text(self, repo, project, ui):
        """Feedback path shows info message with the feedback content."""
        ui.prompt_plan_review.return_value = (False, "Need more unit tests")
        plan = _sample_plan_raw()["units"]
        state = _state(repo, project, build_plan=plan, auto_approve=False)
        await plan_review_node(state, ui)

        # Should have two info calls: feedback received + re-plan notice
        assert ui.info.call_count == 2
        first_info = ui.info.call_args_list[0][0][0]
        assert "Need more unit tests" in first_info


# ---------------------------------------------------------------------------
# plan_review_node — summary formatting
# ---------------------------------------------------------------------------


class TestPlanReviewSummaryFormatting:
    """Verify the plan summary string is correctly formatted."""

    async def test_summary_unit_count(self, repo, project, ui):
        """Summary includes correct unit count."""
        units = [
            {"unit_id": f"u{i}", "title": f"Unit {i}", "risk": "low"} for i in range(5)
        ]
        state = _state(repo, project, build_plan=units, auto_approve=True)
        await plan_review_node(state, ui)

        _, summary = ui.show_artifact.call_args[0]
        assert "Total build units: 5" in summary

    async def test_summary_cost_estimate(self, repo, project, ui):
        """Summary includes cost estimate from estimate_cost."""
        units = [{"risk": "high"}, {"risk": "high"}]
        state = _state(repo, project, build_plan=units, auto_approve=True)
        await plan_review_node(state, ui)

        _, summary = ui.show_artifact.call_args[0]
        # overhead (3,8) + 2 * high (1,3) = (5, 14)
        assert "$5.00" in summary
        assert "$14.00" in summary

    async def test_summary_risk_colors(self, repo, project, ui):
        """Summary includes risk-colored markup for each unit."""
        plan = _sample_plan_raw()["units"]  # low, medium, high
        state = _state(repo, project, build_plan=plan, auto_approve=True)
        await plan_review_node(state, ui)

        _, summary = ui.show_artifact.call_args[0]
        assert "[green]" in summary  # low
        assert "[yellow]" in summary  # medium
        assert "[red]" in summary  # high

    async def test_summary_tests_tag(self, repo, project, ui):
        """Units with tests_included=True get +tests tag."""
        plan = _sample_plan_raw()["units"]  # feat_02 and feat_03 have tests
        state = _state(repo, project, build_plan=plan, auto_approve=True)
        await plan_review_node(state, ui)

        _, summary = ui.show_artifact.call_args[0]
        assert "+tests" in summary

    async def test_summary_pattern_reference(self, repo, project, ui):
        """Units with pattern_reference show the reference path."""
        plan = _sample_plan_raw()["units"]
        state = _state(repo, project, build_plan=plan, auto_approve=True)
        await plan_review_node(state, ui)

        _, summary = ui.show_artifact.call_args[0]
        assert "pattern:" in summary
        assert "migrations/001_init.sql" in summary

    async def test_summary_category_and_blast_radius(self, repo, project, ui):
        """Summary lines include category and blast_radius."""
        plan = _sample_plan_raw()["units"]
        state = _state(repo, project, build_plan=plan, auto_approve=True)
        await plan_review_node(state, ui)

        _, summary = ui.show_artifact.call_args[0]
        assert "database" in summary
        assert "1 file (new migration)" in summary

    async def test_summary_empty_plan(self, repo, project, ui):
        """Empty plan produces summary with 0 units and overhead-only cost."""
        state = _state(repo, project, build_plan=[], auto_approve=True)
        await plan_review_node(state, ui)

        _, summary = ui.show_artifact.call_args[0]
        assert "Total build units: 0" in summary
        assert "$3.00" in summary  # overhead only

    async def test_summary_unit_with_missing_fields(self, repo, project, ui):
        """Units with missing optional fields don't crash formatting."""
        sparse_unit = {"unit_id": "u1"}  # minimal — no title, risk, etc.
        state = _state(repo, project, build_plan=[sparse_unit], auto_approve=True)
        await plan_review_node(state, ui)

        _, summary = ui.show_artifact.call_args[0]
        assert "Total build units: 1" in summary
        assert "u1" in summary
        assert "Untitled" in summary  # default for missing title

    async def test_summary_unknown_risk_gets_white_color(self, repo, project, ui):
        """Unit with unrecognized risk level gets white color."""
        unit = {"unit_id": "u1", "title": "Test", "risk": "extreme"}
        state = _state(repo, project, build_plan=[unit], auto_approve=True)
        await plan_review_node(state, ui)

        _, summary = ui.show_artifact.call_args[0]
        assert "[white]" in summary

    async def test_summary_no_pattern_reference(self, repo, project, ui):
        """Unit without pattern_reference omits that line."""
        unit = {"unit_id": "u1", "title": "Test", "risk": "low", "category": "api"}
        state = _state(repo, project, build_plan=[unit], auto_approve=True)
        await plan_review_node(state, ui)

        _, summary = ui.show_artifact.call_args[0]
        assert "pattern:" not in summary
