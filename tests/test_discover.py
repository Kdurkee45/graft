"""Tests for graft.stages.discover."""

import json
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graft.stages.discover import SYSTEM_PROMPT, discover_node

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
    """Mock UI object exposing the methods discover_node calls."""
    m = MagicMock()
    m.stage_start = MagicMock()
    m.stage_done = MagicMock()
    m.error = MagicMock()
    m.coverage_warning = MagicMock()
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
# Happy-path
# ---------------------------------------------------------------------------


class TestDiscoverNodeHappyPath:
    """Core happy-path tests where agent produces valid outputs."""

    async def test_returns_expected_keys(self, repo, project, ui):
        """discover_node returns codebase_profile, discovery_report, current_stage."""
        profile = {"project": {"name": "acme"}}
        (repo / "codebase_profile.json").write_text(json.dumps(profile))
        (repo / "discovery_report.md").write_text("# Report")

        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="fallback")
            result = await discover_node(_state(repo, project), ui)

        assert set(result.keys()) == {
            "codebase_profile",
            "discovery_report",
            "current_stage",
        }
        assert result["current_stage"] == "discover"
        assert result["codebase_profile"] == profile
        assert result["discovery_report"] == "# Report"

    async def test_saves_artifacts(self, repo, project, ui):
        """Both discovery_report.md and codebase_profile.json are persisted."""
        profile = {"project": {"name": "acme"}}
        (repo / "codebase_profile.json").write_text(json.dumps(profile))
        (repo / "discovery_report.md").write_text("# Report")

        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="fallback")
            await discover_node(_state(repo, project), ui)

        art_dir = project / "artifacts"
        assert (art_dir / "discovery_report.md").read_text() == "# Report"
        saved_profile = json.loads((art_dir / "codebase_profile.json").read_text())
        assert saved_profile == profile

    async def test_marks_stage_complete(self, repo, project, ui):
        """Stage 'discover' is recorded in metadata after success."""
        (repo / "codebase_profile.json").write_text("{}")
        (repo / "discovery_report.md").write_text("ok")

        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await discover_node(_state(repo, project), ui)

        meta = json.loads((project / "metadata.json").read_text())
        assert "discover" in meta["stages_completed"]

    async def test_cleans_up_temp_files(self, repo, project, ui):
        """discovery_report.md and codebase_profile.json removed from repo."""
        (repo / "codebase_profile.json").write_text("{}")
        (repo / "discovery_report.md").write_text("report")

        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await discover_node(_state(repo, project), ui)

        assert not (repo / "codebase_profile.json").exists()
        assert not (repo / "discovery_report.md").exists()

    async def test_ui_lifecycle(self, repo, project, ui):
        """stage_start and stage_done are called in order."""
        (repo / "codebase_profile.json").write_text("{}")
        (repo / "discovery_report.md").write_text("ok")

        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await discover_node(_state(repo, project), ui)

        ui.stage_start.assert_called_once_with("discover")
        ui.stage_done.assert_called_once_with("discover")


# ---------------------------------------------------------------------------
# Agent invocation arguments
# ---------------------------------------------------------------------------


class TestRunAgentArgs:
    """Verify run_agent is invoked with the correct arguments."""

    async def test_allowed_tools_are_read_only(self, repo, project, ui):
        """Only Bash, Read, Glob, Grep should be allowed (read-only)."""
        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await discover_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["allowed_tools"] == ["Bash", "Read", "Glob", "Grep"]

    async def test_system_prompt_is_constant(self, repo, project, ui):
        """run_agent receives the module-level SYSTEM_PROMPT."""
        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await discover_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["system_prompt"] is SYSTEM_PROMPT

    async def test_user_prompt_contains_repo_path(self, repo, project, ui):
        """The user prompt must reference the repo path."""
        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await discover_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert str(repo) in kwargs["user_prompt"]

    async def test_user_prompt_with_scope_path(self, repo, project, ui):
        """When scope_path is set, the prompt includes a SCOPE note."""
        scope = repo / "packages" / "core"
        scope.mkdir(parents=True)

        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await discover_node(_state(repo, project, scope_path="packages/core"), ui)

        _, kwargs = mock_run.call_args
        assert "SCOPE" in kwargs["user_prompt"]
        assert "packages/core" in kwargs["user_prompt"]

    async def test_user_prompt_with_feature_prompt(self, repo, project, ui):
        """When feature_prompt is set, the prompt includes UPCOMING FEATURE."""
        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await discover_node(
                _state(repo, project, feature_prompt="Add dark mode"), ui
            )

        _, kwargs = mock_run.call_args
        assert "UPCOMING FEATURE" in kwargs["user_prompt"]
        assert "Add dark mode" in kwargs["user_prompt"]

    async def test_cwd_uses_scope_path_when_exists(self, repo, project, ui):
        """When scope_path points to a real dir, cwd is scoped there."""
        scope = repo / "packages" / "core"
        scope.mkdir(parents=True)

        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await discover_node(_state(repo, project, scope_path="packages/core"), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["cwd"] == str(scope)

    async def test_cwd_falls_back_to_repo_when_scope_missing(self, repo, project, ui):
        """When scope_path directory doesn't exist, cwd stays at repo_path."""
        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await discover_node(
                _state(repo, project, scope_path="nonexistent/path"), ui
            )

        _, kwargs = mock_run.call_args
        assert kwargs["cwd"] == str(repo)

    async def test_model_forwarded(self, repo, project, ui):
        """model from state is passed through to run_agent."""
        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await discover_node(
                _state(repo, project, model="claude-sonnet-4-20250514"), ui
            )

        _, kwargs = mock_run.call_args
        assert kwargs["model"] == "claude-sonnet-4-20250514"

    async def test_stage_is_discover(self, repo, project, ui):
        """run_agent is called with stage='discover'."""
        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await discover_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["stage"] == "discover"

    async def test_max_turns_is_40(self, repo, project, ui):
        """run_agent is called with max_turns=40."""
        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await discover_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["max_turns"] == 40


# ---------------------------------------------------------------------------
# File-reading fallback paths
# ---------------------------------------------------------------------------


class TestFileDiscovery:
    """Verify file-reading fallback from discover_cwd → repo_path → result.text."""

    async def test_report_falls_back_to_repo_path(self, repo, project, ui):
        """If report isn't in discover_cwd (scoped), fall back to repo_path."""
        scope = repo / "packages" / "core"
        scope.mkdir(parents=True)
        # Put file in repo root, not in scope dir
        (repo / "discovery_report.md").write_text("# Root report")
        (repo / "codebase_profile.json").write_text("{}")

        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="agent text")
            result = await discover_node(
                _state(repo, project, scope_path="packages/core"), ui
            )

        assert result["discovery_report"] == "# Root report"

    async def test_report_falls_back_to_agent_text(self, repo, project, ui):
        """When no report file exists anywhere, use agent result text."""
        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="agent fallback text")
            result = await discover_node(_state(repo, project), ui)

        assert result["discovery_report"] == "agent fallback text"

    async def test_profile_falls_back_to_repo_path(self, repo, project, ui):
        """If profile isn't in discover_cwd (scoped), fall back to repo_path."""
        scope = repo / "packages" / "core"
        scope.mkdir(parents=True)
        profile = {"project": {"name": "from-root"}}
        (repo / "codebase_profile.json").write_text(json.dumps(profile))
        (repo / "discovery_report.md").write_text("ok")

        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            result = await discover_node(
                _state(repo, project, scope_path="packages/core"), ui
            )

        assert result["codebase_profile"]["project"]["name"] == "from-root"

    async def test_profile_prefers_scoped_dir(self, repo, project, ui):
        """When both discover_cwd and repo_path have profiles, scoped one wins."""
        scope = repo / "packages" / "core"
        scope.mkdir(parents=True)
        (scope / "codebase_profile.json").write_text(json.dumps({"from": "scope"}))
        (scope / "discovery_report.md").write_text("scoped report")
        (repo / "codebase_profile.json").write_text(json.dumps({"from": "root"}))

        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            result = await discover_node(
                _state(repo, project, scope_path="packages/core"), ui
            )

        assert result["codebase_profile"]["from"] == "scope"

    async def test_no_profile_file_yields_empty_dict(self, repo, project, ui):
        """When no codebase_profile.json exists, profile is empty dict."""
        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="fallback")
            result = await discover_node(_state(repo, project), ui)

        assert result["codebase_profile"] == {}


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Malformed outputs and edge cases."""

    async def test_malformed_json_profile(self, repo, project, ui):
        """Invalid JSON in codebase_profile.json → ui.error, empty profile returned."""
        (repo / "codebase_profile.json").write_text("NOT VALID JSON {{{")
        (repo / "discovery_report.md").write_text("ok")

        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            result = await discover_node(_state(repo, project), ui)

        ui.error.assert_called_once()
        assert "codebase_profile.json" in ui.error.call_args[0][0]
        assert result["codebase_profile"] == {}

    async def test_malformed_json_still_saves_empty_artifact(self, repo, project, ui):
        """Even with bad JSON, an empty profile artifact is saved."""
        (repo / "codebase_profile.json").write_text("{broken")
        (repo / "discovery_report.md").write_text("ok")

        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await discover_node(_state(repo, project), ui)

        saved = json.loads(
            (project / "artifacts" / "codebase_profile.json").read_text()
        )
        assert saved == {}

    async def test_malformed_json_still_completes_stage(self, repo, project, ui):
        """Malformed profile doesn't prevent stage from completing."""
        (repo / "codebase_profile.json").write_text(">>>")
        (repo / "discovery_report.md").write_text("ok")

        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await discover_node(_state(repo, project), ui)

        meta = json.loads((project / "metadata.json").read_text())
        assert "discover" in meta["stages_completed"]

    async def test_malformed_json_still_cleans_up(self, repo, project, ui):
        """Temp files cleaned even when profile JSON is broken."""
        (repo / "codebase_profile.json").write_text("oops")
        (repo / "discovery_report.md").write_text("ok")

        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await discover_node(_state(repo, project), ui)

        assert not (repo / "codebase_profile.json").exists()
        assert not (repo / "discovery_report.md").exists()


# ---------------------------------------------------------------------------
# Coverage warnings
# ---------------------------------------------------------------------------


class TestCoverageWarnings:
    """coverage_warnings field triggers ui.coverage_warning."""

    async def test_coverage_warnings_displayed(self, repo, project, ui):
        """Non-empty coverage_warnings array triggers ui.coverage_warning."""
        warnings = [
            {"module": "src/foo.py", "coverage_pct": 5, "recommendation": "add tests"}
        ]
        profile = {"coverage_warnings": warnings}
        (repo / "codebase_profile.json").write_text(json.dumps(profile))
        (repo / "discovery_report.md").write_text("ok")

        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await discover_node(_state(repo, project), ui)

        ui.coverage_warning.assert_called_once_with(warnings)

    async def test_no_coverage_warnings_no_call(self, repo, project, ui):
        """Empty coverage_warnings array means coverage_warning is never called."""
        profile = {"coverage_warnings": []}
        (repo / "codebase_profile.json").write_text(json.dumps(profile))
        (repo / "discovery_report.md").write_text("ok")

        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await discover_node(_state(repo, project), ui)

        ui.coverage_warning.assert_not_called()

    async def test_missing_coverage_warnings_key_no_call(self, repo, project, ui):
        """Profile without coverage_warnings key → no crash, no call."""
        profile = {"project": {"name": "acme"}}
        (repo / "codebase_profile.json").write_text(json.dumps(profile))
        (repo / "discovery_report.md").write_text("ok")

        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await discover_node(_state(repo, project), ui)

        ui.coverage_warning.assert_not_called()


# ---------------------------------------------------------------------------
# Cleanup edge cases
# ---------------------------------------------------------------------------


class TestCleanup:
    """Ensure temp files are removed in all locations."""

    async def test_cleanup_both_scoped_and_root(self, repo, project, ui):
        """Files in both discover_cwd and repo_path are cleaned up."""
        scope = repo / "packages" / "core"
        scope.mkdir(parents=True)
        for d in [scope, repo]:
            (d / "codebase_profile.json").write_text("{}")
            (d / "discovery_report.md").write_text("ok")

        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await discover_node(_state(repo, project, scope_path="packages/core"), ui)

        for d in [scope, repo]:
            assert not (d / "codebase_profile.json").exists()
            assert not (d / "discovery_report.md").exists()

    async def test_cleanup_when_no_files_exist(self, repo, project, ui):
        """No crash when temp files were never created."""
        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="fallback")
            # No files written — should not raise
            await discover_node(_state(repo, project), ui)

    async def test_cleanup_only_report_exists(self, repo, project, ui):
        """Only discovery_report.md exists (no profile) — cleaned up fine."""
        (repo / "discovery_report.md").write_text("report only")

        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await discover_node(_state(repo, project), ui)

        assert not (repo / "discovery_report.md").exists()


# ---------------------------------------------------------------------------
# Prompt construction details
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    """Verify the exact prompt composition logic."""

    async def test_minimal_prompt_no_scope_no_feature(self, repo, project, ui):
        """Without scope_path or feature_prompt, prompt has only repo path and closing."""
        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await discover_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        prompt = kwargs["user_prompt"]
        assert "Discover and map the codebase at:" in prompt
        assert str(repo) in prompt
        assert "SCOPE" not in prompt
        assert "UPCOMING FEATURE" not in prompt
        assert "comprehensive discovery report" in prompt

    async def test_all_prompt_parts_present(self, repo, project, ui):
        """With scope_path and feature_prompt, all sections appear."""
        scope = repo / "web"
        scope.mkdir()

        with patch(
            "graft.stages.discover.run_agent", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await discover_node(
                _state(repo, project, scope_path="web", feature_prompt="SSO login"), ui
            )

        _, kwargs = mock_run.call_args
        prompt = kwargs["user_prompt"]
        assert "Discover and map the codebase at:" in prompt
        assert "SCOPE" in prompt
        assert "'web/'" in prompt
        assert "UPCOMING FEATURE" in prompt
        assert "SSO login" in prompt
        assert "integration-critical modules" in prompt
        assert "comprehensive discovery report" in prompt
