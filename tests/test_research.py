"""Tests for graft.stages.research."""

import json
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graft.stages.research import SYSTEM_PROMPT, research_node

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
    """Mock UI object exposing the methods research_node calls."""
    m = MagicMock()
    m.stage_start = MagicMock()
    m.stage_done = MagicMock()
    m.error = MagicMock()
    m.info = MagicMock()
    return m


def _state(repo, project, **kw):
    """Build a minimal FeatureState dict."""
    base = {
        "repo_path": str(repo),
        "project_dir": str(project),
        "feature_prompt": "Add dark mode",
        "codebase_profile": {"project": {"name": "acme"}},
    }
    base.update(kw)
    return base


VALID_ASSESSMENT = {
    "feature_prompt": "Add dark mode",
    "reusable_components": [
        {"path": "src/theme.ts", "reason": "existing theme system"}
    ],
    "new_artifacts_needed": [
        {"type": "component", "name": "ThemeToggle", "description": "toggle widget"}
    ],
    "pattern_to_follow": "src/features/settings/",
    "edge_cases": ["concurrent theme changes"],
    "integration_points": ["navigation bar"],
    "open_questions": [
        {
            "question": "Should dark mode persist across sessions?",
            "category": "intent",
            "recommended_answer": "Yes, store in localStorage",
        },
        {
            "question": "Should we support system preference detection?",
            "category": "preference",
            "recommended_answer": "Yes, use prefers-color-scheme",
        },
    ],
}


# ---------------------------------------------------------------------------
# Happy-path
# ---------------------------------------------------------------------------


class TestResearchNodeHappyPath:
    """Core happy-path tests where agent produces valid outputs."""

    async def test_returns_expected_keys(self, repo, project, ui):
        """research_node returns assessment, report, and current_stage."""
        (repo / "technical_assessment.json").write_text(json.dumps(VALID_ASSESSMENT))
        (repo / "research_report.md").write_text("# Research Report")

        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="fallback")
            result = await research_node(_state(repo, project), ui)

        assert set(result.keys()) == {
            "technical_assessment",
            "research_report",
            "current_stage",
        }
        assert result["current_stage"] == "research"

    async def test_parses_technical_assessment(self, repo, project, ui):
        """Valid technical_assessment.json is parsed into a dict."""
        (repo / "technical_assessment.json").write_text(json.dumps(VALID_ASSESSMENT))
        (repo / "research_report.md").write_text("# Report")

        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="fallback")
            result = await research_node(_state(repo, project), ui)

        assert result["technical_assessment"] == VALID_ASSESSMENT
        assert result["technical_assessment"]["feature_prompt"] == "Add dark mode"
        assert len(result["technical_assessment"]["reusable_components"]) == 1

    async def test_reads_research_report(self, repo, project, ui):
        """research_report.md content is returned as research_report."""
        (repo / "technical_assessment.json").write_text(json.dumps(VALID_ASSESSMENT))
        (repo / "research_report.md").write_text(
            "# Detailed Research\n\nFindings here."
        )

        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="fallback")
            result = await research_node(_state(repo, project), ui)

        assert result["research_report"] == "# Detailed Research\n\nFindings here."

    async def test_saves_artifacts(self, repo, project, ui):
        """Both research_report.md and technical_assessment.json are persisted."""
        (repo / "technical_assessment.json").write_text(json.dumps(VALID_ASSESSMENT))
        (repo / "research_report.md").write_text("# Report")

        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="fallback")
            await research_node(_state(repo, project), ui)

        art_dir = project / "artifacts"
        assert (art_dir / "research_report.md").read_text() == "# Report"
        saved_assessment = json.loads(
            (art_dir / "technical_assessment.json").read_text()
        )
        assert saved_assessment == VALID_ASSESSMENT

    async def test_marks_stage_complete(self, repo, project, ui):
        """Stage 'research' is recorded in metadata after success."""
        (repo / "technical_assessment.json").write_text(json.dumps(VALID_ASSESSMENT))
        (repo / "research_report.md").write_text("ok")

        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(_state(repo, project), ui)

        meta = json.loads((project / "metadata.json").read_text())
        assert "research" in meta["stages_completed"]

    async def test_ui_lifecycle(self, repo, project, ui):
        """stage_start and stage_done are called in order."""
        (repo / "technical_assessment.json").write_text(json.dumps(VALID_ASSESSMENT))
        (repo / "research_report.md").write_text("ok")

        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(_state(repo, project), ui)

        ui.stage_start.assert_called_once_with("research")
        ui.stage_done.assert_called_once_with("research")


# ---------------------------------------------------------------------------
# Open questions for downstream grill stage
# ---------------------------------------------------------------------------


class TestOpenQuestions:
    """Verify open_questions extraction for the Grill phase."""

    async def test_open_questions_extracted_from_assessment(self, repo, project, ui):
        """open_questions are not returned in state but info is logged."""
        (repo / "technical_assessment.json").write_text(json.dumps(VALID_ASSESSMENT))
        (repo / "research_report.md").write_text("ok")

        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            result = await research_node(_state(repo, project), ui)

        # open_questions live inside technical_assessment
        assert len(result["technical_assessment"]["open_questions"]) == 2

    async def test_open_questions_count_logged(self, repo, project, ui):
        """UI info is called with the count of open questions."""
        (repo / "technical_assessment.json").write_text(json.dumps(VALID_ASSESSMENT))
        (repo / "research_report.md").write_text("ok")

        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(_state(repo, project), ui)

        ui.info.assert_called_once()
        msg = ui.info.call_args[0][0]
        assert "2" in msg
        assert "open question" in msg
        assert "Grill" in msg

    async def test_no_open_questions_no_info(self, repo, project, ui):
        """When open_questions is empty, ui.info is not called."""
        assessment = {**VALID_ASSESSMENT, "open_questions": []}
        (repo / "technical_assessment.json").write_text(json.dumps(assessment))
        (repo / "research_report.md").write_text("ok")

        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(_state(repo, project), ui)

        ui.info.assert_not_called()

    async def test_missing_open_questions_key_no_crash(self, repo, project, ui):
        """Assessment without open_questions key doesn't crash."""
        assessment = {"feature_prompt": "Add dark mode"}
        (repo / "technical_assessment.json").write_text(json.dumps(assessment))
        (repo / "research_report.md").write_text("ok")

        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            result = await research_node(_state(repo, project), ui)

        ui.info.assert_not_called()
        assert result["technical_assessment"].get("open_questions") is None


# ---------------------------------------------------------------------------
# Agent invocation arguments
# ---------------------------------------------------------------------------


class TestRunAgentArgs:
    """Verify run_agent is invoked with the correct arguments."""

    async def test_allowed_tools(self, repo, project, ui):
        """Bash, Read, Glob, Grep should be allowed."""
        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["allowed_tools"] == ["Bash", "Read", "Glob", "Grep"]

    async def test_system_prompt_is_constant(self, repo, project, ui):
        """run_agent receives the module-level SYSTEM_PROMPT."""
        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["system_prompt"] is SYSTEM_PROMPT

    async def test_user_prompt_contains_repo_path(self, repo, project, ui):
        """The user prompt must reference the repo path."""
        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert str(repo) in kwargs["user_prompt"]

    async def test_user_prompt_contains_feature_prompt(self, repo, project, ui):
        """The user prompt includes the feature description."""
        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(
                _state(repo, project, feature_prompt="Add SSO login"), ui
            )

        _, kwargs = mock_run.call_args
        assert "FEATURE" in kwargs["user_prompt"]
        assert "Add SSO login" in kwargs["user_prompt"]

    async def test_user_prompt_contains_codebase_profile(self, repo, project, ui):
        """The user prompt includes the serialized codebase profile."""
        profile = {"project": {"name": "acme"}, "language": "typescript"}
        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(_state(repo, project, codebase_profile=profile), ui)

        _, kwargs = mock_run.call_args
        prompt = kwargs["user_prompt"]
        assert "CODEBASE PROFILE" in prompt
        assert "acme" in prompt
        assert "typescript" in prompt

    async def test_user_prompt_contains_constraints(self, repo, project, ui):
        """When constraints are provided, they appear in the prompt."""
        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(
                _state(
                    repo,
                    project,
                    constraints=["no external deps", "must be accessible"],
                ),
                ui,
            )

        _, kwargs = mock_run.call_args
        prompt = kwargs["user_prompt"]
        assert "CONSTRAINTS" in prompt
        assert "no external deps" in prompt
        assert "must be accessible" in prompt

    async def test_user_prompt_omits_constraints_when_empty(self, repo, project, ui):
        """When constraints list is empty, CONSTRAINTS section is absent."""
        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(_state(repo, project, constraints=[]), ui)

        _, kwargs = mock_run.call_args
        assert "CONSTRAINTS" not in kwargs["user_prompt"]

    async def test_cwd_uses_scope_path_when_exists(self, repo, project, ui):
        """When scope_path points to a real dir, cwd is scoped there."""
        scope = repo / "packages" / "core"
        scope.mkdir(parents=True)

        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(_state(repo, project, scope_path="packages/core"), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["cwd"] == str(scope)

    async def test_cwd_falls_back_to_repo_when_scope_missing(self, repo, project, ui):
        """When scope_path directory doesn't exist, cwd stays at repo_path."""
        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(
                _state(repo, project, scope_path="nonexistent/path"), ui
            )

        _, kwargs = mock_run.call_args
        assert kwargs["cwd"] == str(repo)

    async def test_model_forwarded(self, repo, project, ui):
        """model from state is passed through to run_agent."""
        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(
                _state(repo, project, model="claude-sonnet-4-20250514"), ui
            )

        _, kwargs = mock_run.call_args
        assert kwargs["model"] == "claude-sonnet-4-20250514"

    async def test_stage_is_research(self, repo, project, ui):
        """run_agent is called with stage='research'."""
        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["stage"] == "research"

    async def test_max_turns_is_30(self, repo, project, ui):
        """run_agent is called with max_turns=30."""
        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["max_turns"] == 30


# ---------------------------------------------------------------------------
# File-reading fallback paths
# ---------------------------------------------------------------------------


class TestFileDiscovery:
    """Verify file-reading fallback from research_cwd -> repo_path -> result.text."""

    async def test_report_falls_back_to_repo_path(self, repo, project, ui):
        """If report isn't in scoped cwd, fall back to repo_path."""
        scope = repo / "packages" / "core"
        scope.mkdir(parents=True)
        # Put file in repo root, not in scope dir
        (repo / "research_report.md").write_text("# Root report")
        (repo / "technical_assessment.json").write_text("{}")

        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="agent text")
            result = await research_node(
                _state(repo, project, scope_path="packages/core"), ui
            )

        assert result["research_report"] == "# Root report"

    async def test_report_falls_back_to_agent_text(self, repo, project, ui):
        """When no report file exists anywhere, use agent result text."""
        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="agent fallback text")
            result = await research_node(_state(repo, project), ui)

        assert result["research_report"] == "agent fallback text"

    async def test_assessment_falls_back_to_repo_path(self, repo, project, ui):
        """If assessment isn't in scoped cwd, fall back to repo_path."""
        scope = repo / "packages" / "core"
        scope.mkdir(parents=True)
        assessment = {"feature_prompt": "from root"}
        (repo / "technical_assessment.json").write_text(json.dumps(assessment))
        (repo / "research_report.md").write_text("ok")

        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            result = await research_node(
                _state(repo, project, scope_path="packages/core"), ui
            )

        assert result["technical_assessment"]["feature_prompt"] == "from root"

    async def test_assessment_prefers_scoped_dir(self, repo, project, ui):
        """When both scoped dir and repo_path have assessments, scoped one wins."""
        scope = repo / "packages" / "core"
        scope.mkdir(parents=True)
        (scope / "technical_assessment.json").write_text(json.dumps({"from": "scope"}))
        (scope / "research_report.md").write_text("scoped report")
        (repo / "technical_assessment.json").write_text(json.dumps({"from": "root"}))

        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            result = await research_node(
                _state(repo, project, scope_path="packages/core"), ui
            )

        assert result["technical_assessment"]["from"] == "scope"

    async def test_no_assessment_file_yields_empty_dict(self, repo, project, ui):
        """When no technical_assessment.json exists, assessment is empty dict."""
        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="fallback")
            result = await research_node(_state(repo, project), ui)

        assert result["technical_assessment"] == {}


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Malformed outputs and edge cases."""

    async def test_malformed_json_assessment(self, repo, project, ui):
        """Invalid JSON in assessment file -> ui.error, empty dict."""
        (repo / "technical_assessment.json").write_text("NOT VALID JSON {{{")
        (repo / "research_report.md").write_text("ok")

        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            result = await research_node(_state(repo, project), ui)

        ui.error.assert_called_once()
        assert "technical_assessment.json" in ui.error.call_args[0][0]
        assert result["technical_assessment"] == {}

    async def test_malformed_json_still_saves_empty_artifact(self, repo, project, ui):
        """Even with bad JSON, an empty assessment artifact is saved."""
        (repo / "technical_assessment.json").write_text("{broken")
        (repo / "research_report.md").write_text("ok")

        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(_state(repo, project), ui)

        saved = json.loads(
            (project / "artifacts" / "technical_assessment.json").read_text()
        )
        assert saved == {}

    async def test_malformed_json_still_completes_stage(self, repo, project, ui):
        """Malformed assessment doesn't prevent stage from completing."""
        (repo / "technical_assessment.json").write_text(">>>")
        (repo / "research_report.md").write_text("ok")

        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(_state(repo, project), ui)

        meta = json.loads((project / "metadata.json").read_text())
        assert "research" in meta["stages_completed"]

    async def test_malformed_json_still_cleans_up(self, repo, project, ui):
        """Temp files cleaned even when assessment JSON is broken."""
        (repo / "technical_assessment.json").write_text("oops")
        (repo / "research_report.md").write_text("ok")

        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(_state(repo, project), ui)

        assert not (repo / "technical_assessment.json").exists()
        assert not (repo / "research_report.md").exists()

    async def test_malformed_json_no_open_questions_logged(self, repo, project, ui):
        """Malformed assessment means no open questions are logged."""
        (repo / "technical_assessment.json").write_text("not json")
        (repo / "research_report.md").write_text("ok")

        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(_state(repo, project), ui)

        ui.info.assert_not_called()

    async def test_empty_feature_prompt_defaults_to_empty(self, repo, project, ui):
        """Missing feature_prompt defaults to empty string without error."""
        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="fallback")
            state = {
                "repo_path": str(repo),
                "project_dir": str(project),
            }
            result = await research_node(state, ui)

        assert result["current_stage"] == "research"

    async def test_empty_codebase_profile_defaults_to_empty_dict(
        self, repo, project, ui
    ):
        """Missing codebase_profile defaults to empty dict without error."""
        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="fallback")
            state = {
                "repo_path": str(repo),
                "project_dir": str(project),
            }
            result = await research_node(state, ui)

        _, kwargs = mock_run.call_args
        assert "{}" in kwargs["user_prompt"]
        assert result["current_stage"] == "research"


# ---------------------------------------------------------------------------
# Cleanup edge cases
# ---------------------------------------------------------------------------


class TestCleanup:
    """Ensure temp files are removed in all locations."""

    async def test_cleanup_both_scoped_and_root(self, repo, project, ui):
        """Files in both research_cwd and repo_path are cleaned up."""
        scope = repo / "packages" / "core"
        scope.mkdir(parents=True)
        for d in [scope, repo]:
            (d / "technical_assessment.json").write_text("{}")
            (d / "research_report.md").write_text("ok")

        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(_state(repo, project, scope_path="packages/core"), ui)

        for d in [scope, repo]:
            assert not (d / "technical_assessment.json").exists()
            assert not (d / "research_report.md").exists()

    async def test_cleanup_when_no_files_exist(self, repo, project, ui):
        """No crash when temp files were never created."""
        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="fallback")
            # No files written — should not raise
            await research_node(_state(repo, project), ui)

    async def test_cleanup_only_report_exists(self, repo, project, ui):
        """Only research_report.md exists (no assessment) — cleaned up fine."""
        (repo / "research_report.md").write_text("report only")

        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(_state(repo, project), ui)

        assert not (repo / "research_report.md").exists()

    async def test_cleanup_only_assessment_exists(self, repo, project, ui):
        """Only technical_assessment.json exists (no report) — cleaned up fine."""
        (repo / "technical_assessment.json").write_text(json.dumps(VALID_ASSESSMENT))

        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(_state(repo, project), ui)

        assert not (repo / "technical_assessment.json").exists()


# ---------------------------------------------------------------------------
# Prompt construction details
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    """Verify the exact prompt composition logic."""

    async def test_prompt_has_all_sections(self, repo, project, ui):
        """With all state fields, prompt contains repo path, feature, profile."""
        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(
                _state(
                    repo,
                    project,
                    feature_prompt="SSO login",
                    codebase_profile={"lang": "python"},
                    constraints=["no breaking changes"],
                ),
                ui,
            )

        _, kwargs = mock_run.call_args
        prompt = kwargs["user_prompt"]
        assert str(repo) in prompt
        assert "FEATURE" in prompt
        assert "SSO login" in prompt
        assert "CODEBASE PROFILE" in prompt
        assert "python" in prompt
        assert "CONSTRAINTS" in prompt
        assert "no breaking changes" in prompt
        assert "Explore the actual codebase" in prompt

    async def test_minimal_prompt_no_constraints(self, repo, project, ui):
        """Without constraints, CONSTRAINTS section is absent."""
        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        prompt = kwargs["user_prompt"]
        assert "FEATURE" in prompt
        assert "CODEBASE PROFILE" in prompt
        assert "CONSTRAINTS" not in prompt

    async def test_prompt_includes_explore_instruction(self, repo, project, ui):
        """Prompt always ends with instruction to explore the codebase."""
        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        prompt = kwargs["user_prompt"]
        assert "Explore the actual codebase" in prompt
        assert "validate and extend the profile" in prompt


# ---------------------------------------------------------------------------
# Artifact persistence details
# ---------------------------------------------------------------------------


class TestArtifactPersistence:
    """Verify that artifacts are correctly saved to project_dir."""

    async def test_report_artifact_saved(self, repo, project, ui):
        """research_report.md is saved to artifacts directory."""
        (repo / "research_report.md").write_text("detailed findings")
        (repo / "technical_assessment.json").write_text("{}")

        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(_state(repo, project), ui)

        assert (project / "artifacts" / "research_report.md").read_text() == (
            "detailed findings"
        )

    async def test_assessment_artifact_saved_formatted(self, repo, project, ui):
        """technical_assessment.json is saved with indent=2 formatting."""
        (repo / "technical_assessment.json").write_text(json.dumps(VALID_ASSESSMENT))
        (repo / "research_report.md").write_text("ok")

        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await research_node(_state(repo, project), ui)

        raw = (project / "artifacts" / "technical_assessment.json").read_text()
        assert json.loads(raw) == VALID_ASSESSMENT
        # Verify formatted with indentation (not compact)
        assert "\n" in raw

    async def test_fallback_report_saved_as_artifact(self, repo, project, ui):
        """When report falls back to agent text, that text is saved as artifact."""
        # No report file on disk — falls back to result.text
        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="agent produced this")
            await research_node(_state(repo, project), ui)

        assert (project / "artifacts" / "research_report.md").read_text() == (
            "agent produced this"
        )

    async def test_empty_assessment_artifact_on_no_file(self, repo, project, ui):
        """When no assessment file exists, an empty dict artifact is saved."""
        with patch(
            "graft.stages.research.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="fallback")
            await research_node(_state(repo, project), ui)

        saved = json.loads(
            (project / "artifacts" / "technical_assessment.json").read_text()
        )
        assert saved == {}
