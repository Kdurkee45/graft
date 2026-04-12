"""Tests for graft.cli — Typer CLI layer.

Uses typer.testing.CliRunner to invoke commands while mocking out
asyncio.run and the compiled graph so we test argument parsing,
flag handling, state construction, and error paths in isolation.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from graft.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_repo(tmp_path):
    """Create a temporary directory that looks like a repo."""
    repo = tmp_path / "my-repo"
    repo.mkdir()
    return repo


@pytest.fixture
def fake_project_dir(tmp_path):
    """Create a fake project session directory with metadata + artifacts."""
    proj = tmp_path / "feat_abc12345"
    (proj / "artifacts").mkdir(parents=True)
    (proj / "logs").mkdir(parents=True)
    meta = {
        "project_id": "feat_abc12345",
        "repo_path": "/tmp/some-repo",
        "feature_prompt": "Add dark mode",
        "status": "in_progress",
        "stages_completed": ["discover", "research"],
    }
    (proj / "metadata.json").write_text(json.dumps(meta))
    return proj


@pytest.fixture
def mock_settings(monkeypatch):
    """Patch Settings.load() to return a deterministic Settings object."""
    from graft.config import Settings

    settings = Settings(
        anthropic_api_key="test-key",
        github_token="gh-token",
        model="claude-sonnet-4-20250514",
        max_agent_turns=10,
        projects_root=Path("/tmp/graft-projects"),
    )
    monkeypatch.setattr("graft.cli.Settings.load", staticmethod(lambda: settings))
    return settings


@pytest.fixture
def mock_graph():
    """Patch build_graph to return a mock compiled graph whose ainvoke returns a result dict."""
    compiled = MagicMock()
    compiled.ainvoke = MagicMock(
        return_value={"pr_url": "https://github.com/org/repo/pull/42"}
    )
    with patch("graft.cli.build_graph", return_value=compiled) as bg:
        with patch("graft.cli.asyncio.run", side_effect=lambda coro: coro) as ar:
            yield {"build_graph": bg, "asyncio_run": ar, "compiled": compiled}


@pytest.fixture
def mock_graph_no_pr():
    """Like mock_graph but ainvoke returns empty pr_url."""
    compiled = MagicMock()
    compiled.ainvoke = MagicMock(return_value={"pr_url": ""})
    with patch("graft.cli.build_graph", return_value=compiled) as bg:
        with patch("graft.cli.asyncio.run", side_effect=lambda coro: coro) as ar:
            yield {"build_graph": bg, "asyncio_run": ar, "compiled": compiled}


@pytest.fixture
def mock_create_project(tmp_path):
    """Patch create_project to return a deterministic project id and directory."""
    proj_dir = tmp_path / "feat_test1234"
    (proj_dir / "artifacts").mkdir(parents=True)
    (proj_dir / "logs").mkdir(parents=True)
    meta = {
        "project_id": "feat_test1234",
        "repo_path": "",
        "feature_prompt": "",
        "status": "in_progress",
    }
    (proj_dir / "metadata.json").write_text(json.dumps(meta))
    with patch(
        "graft.cli.create_project", return_value=("feat_test1234", proj_dir)
    ) as cp:
        yield cp


# ---------------------------------------------------------------------------
# Build command — required argument validation
# ---------------------------------------------------------------------------


class TestBuildRequiredArgs:
    """build command requires repo_path and feature_prompt."""

    def test_missing_all_args(self, mock_settings):
        result = runner.invoke(app, ["build"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output or "Usage" in result.output

    def test_missing_feature_prompt(self, mock_settings, fake_repo):
        result = runner.invoke(app, ["build", str(fake_repo)])
        assert result.exit_code != 0

    def test_help_text(self):
        result = runner.invoke(app, ["build", "--help"])
        assert result.exit_code == 0
        assert "repo_path" in result.output.lower() or "REPO_PATH" in result.output
        assert (
            "feature_prompt" in result.output.lower()
            or "FEATURE_PROMPT" in result.output
        )


# ---------------------------------------------------------------------------
# Build command — path validation
# ---------------------------------------------------------------------------


class TestBuildPathValidation:
    """build validates that repo_path exists on disk."""

    def test_nonexistent_repo_exits_1(
        self, mock_settings, mock_create_project, mock_graph
    ):
        result = runner.invoke(app, ["build", "/no/such/path", "Add tests"])
        assert result.exit_code == 1
        assert (
            "not found" in result.output.lower()
            or "Repository not found" in result.output
        )

    def test_nonexistent_scope_path_exits_1(
        self, mock_settings, mock_create_project, mock_graph, fake_repo
    ):
        result = runner.invoke(
            app,
            ["build", str(fake_repo), "Add tests", "--path", "nonexistent/sub"],
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# Build command — flag handling
# ---------------------------------------------------------------------------


class TestBuildFlags:
    """Build command parses all optional flags correctly."""

    def test_default_flags(
        self, mock_settings, fake_repo, mock_create_project, mock_graph
    ):
        """With no optional flags, defaults are applied to initial state."""
        result = runner.invoke(app, ["build", str(fake_repo), "Add dark mode"])
        assert result.exit_code == 0

        # asyncio.run was called with the coroutine from compiled.ainvoke
        mock_graph["asyncio_run"].assert_called_once()
        compiled = mock_graph["compiled"]
        compiled.ainvoke.assert_called_once()
        state = compiled.ainvoke.call_args[0][0]

        assert state["constraints"] == []
        assert state["max_units"] == 0
        assert state["auto_approve"] is False
        assert state["scope_path"] == ""

    def test_constraint_flag_single(
        self, mock_settings, fake_repo, mock_create_project, mock_graph
    ):
        result = runner.invoke(
            app,
            [
                "build",
                str(fake_repo),
                "Add tests",
                "--constraint",
                "no breaking changes",
            ],
        )
        assert result.exit_code == 0
        state = mock_graph["compiled"].ainvoke.call_args[0][0]
        assert state["constraints"] == ["no breaking changes"]

    def test_constraint_flag_multiple(
        self, mock_settings, fake_repo, mock_create_project, mock_graph
    ):
        result = runner.invoke(
            app,
            [
                "build",
                str(fake_repo),
                "Add tests",
                "--constraint",
                "no breaking changes",
                "-c",
                "must use pytest",
            ],
        )
        assert result.exit_code == 0
        state = mock_graph["compiled"].ainvoke.call_args[0][0]
        assert state["constraints"] == ["no breaking changes", "must use pytest"]

    def test_max_units_flag(
        self, mock_settings, fake_repo, mock_create_project, mock_graph
    ):
        result = runner.invoke(
            app,
            ["build", str(fake_repo), "Add tests", "--max-units", "5"],
        )
        assert result.exit_code == 0
        state = mock_graph["compiled"].ainvoke.call_args[0][0]
        assert state["max_units"] == 5

    def test_auto_approve_flag(
        self, mock_settings, fake_repo, mock_create_project, mock_graph
    ):
        result = runner.invoke(
            app,
            ["build", str(fake_repo), "Add tests", "--auto-approve"],
        )
        assert result.exit_code == 0
        state = mock_graph["compiled"].ainvoke.call_args[0][0]
        assert state["auto_approve"] is True

    def test_verbose_flag(
        self, mock_settings, fake_repo, mock_create_project, mock_graph
    ):
        """--verbose / -v is passed through to UI (and doesn't affect state directly)."""
        result = runner.invoke(
            app,
            ["build", str(fake_repo), "Add tests", "-v"],
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Build command — path resolution (expanduser + resolve)
# ---------------------------------------------------------------------------


class TestBuildPathResolution:
    """repo_path is resolved through expanduser().resolve()."""

    def test_resolved_path_in_state(
        self, mock_settings, fake_repo, mock_create_project, mock_graph
    ):
        result = runner.invoke(app, ["build", str(fake_repo), "Build feature"])
        assert result.exit_code == 0
        state = mock_graph["compiled"].ainvoke.call_args[0][0]
        # Path should be absolute and resolved
        assert Path(state["repo_path"]).is_absolute()
        assert str(fake_repo.resolve()) == state["repo_path"]

    def test_scope_path_in_state(
        self, mock_settings, fake_repo, mock_create_project, mock_graph
    ):
        """--path scope_path is stored as relative string, not resolved."""
        subdir = fake_repo / "packages" / "core"
        subdir.mkdir(parents=True)
        result = runner.invoke(
            app,
            ["build", str(fake_repo), "Build feature", "--path", "packages/core"],
        )
        assert result.exit_code == 0
        state = mock_graph["compiled"].ainvoke.call_args[0][0]
        assert state["scope_path"] == "packages/core"


# ---------------------------------------------------------------------------
# Build command — initial FeatureState construction
# ---------------------------------------------------------------------------


class TestBuildStateConstruction:
    """The initial FeatureState dict has correct defaults and derived values."""

    def test_state_has_all_required_keys(
        self, mock_settings, fake_repo, mock_create_project, mock_graph
    ):
        result = runner.invoke(app, ["build", str(fake_repo), "Add dark mode"])
        assert result.exit_code == 0
        state = mock_graph["compiled"].ainvoke.call_args[0][0]

        # Core identification
        assert state["project_id"] == "feat_test1234"
        assert state["feature_prompt"] == "Add dark mode"

        # Defaults for empty stage artifacts
        assert state["codebase_profile"] == {}
        assert state["discovery_report"] == ""
        assert state["technical_assessment"] == {}
        assert state["research_report"] == ""
        assert state["feature_spec"] == {}
        assert state["grill_transcript"] == ""
        assert state["build_plan"] == []
        assert state["feature_report"] == ""

        # Execution tracking defaults
        assert state["current_unit_index"] == 0
        assert state["units_completed"] == []
        assert state["units_reverted"] == []
        assert state["units_skipped"] == []

        # Gate defaults
        assert state["plan_approved"] is False
        assert state["grill_complete"] is False
        assert state["research_redo_needed"] is False

        # Git
        assert state["feature_branch"] == "feature/feat_test1234"
        assert state["pr_url"] == ""

        # Settings propagated
        assert state["model"] == "claude-sonnet-4-20250514"
        assert state["max_agent_turns"] == 10

        assert state["current_stage"] == ""

    def test_pr_url_displayed_when_present(
        self, mock_settings, fake_repo, mock_create_project, mock_graph
    ):
        result = runner.invoke(app, ["build", str(fake_repo), "Add dark mode"])
        assert result.exit_code == 0
        assert "PR opened" in result.output or "pull/42" in result.output

    def test_no_pr_url_fallback_message(
        self, mock_settings, fake_repo, mock_create_project, mock_graph_no_pr
    ):
        result = runner.invoke(app, ["build", str(fake_repo), "Add dark mode"])
        assert result.exit_code == 0
        assert (
            "open a PR manually" in result.output or "complete" in result.output.lower()
        )

    def test_session_artifacts_path_displayed(
        self, mock_settings, fake_repo, mock_create_project, mock_graph
    ):
        result = runner.invoke(app, ["build", str(fake_repo), "Add dark mode"])
        assert result.exit_code == 0
        assert "Session artifacts" in result.output or "feat_test1234" in result.output


# ---------------------------------------------------------------------------
# Resume command — required arguments
# ---------------------------------------------------------------------------


class TestResumeRequiredArgs:
    """resume requires project_path argument."""

    def test_missing_project_path(self, mock_settings):
        result = runner.invoke(app, ["resume"])
        assert result.exit_code != 0

    def test_help_text(self):
        result = runner.invoke(app, ["resume", "--help"])
        assert result.exit_code == 0
        assert (
            "project_path" in result.output.lower() or "PROJECT_PATH" in result.output
        )


# ---------------------------------------------------------------------------
# Resume command — validation
# ---------------------------------------------------------------------------


class TestResumeValidation:
    """resume validates directory and metadata existence."""

    def test_nonexistent_project_dir_exits_1(self, mock_settings):
        result = runner.invoke(app, ["resume", "/no/such/session"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_missing_metadata_exits_1(self, mock_settings, tmp_path):
        """Directory exists but no metadata.json → error."""
        empty_dir = tmp_path / "feat_empty"
        empty_dir.mkdir()
        result = runner.invoke(app, ["resume", str(empty_dir)])
        assert result.exit_code == 1
        assert (
            "metadata.json" in result.output or "valid Graft session" in result.output
        )


# ---------------------------------------------------------------------------
# Resume command — --from stage parameter
# ---------------------------------------------------------------------------


class TestResumeFromStage:
    """resume --from passes entry_stage to build_graph."""

    def test_default_from_stage_is_execute(
        self, mock_settings, fake_project_dir, mock_graph
    ):
        result = runner.invoke(app, ["resume", str(fake_project_dir)])
        assert result.exit_code == 0
        mock_graph["build_graph"].assert_called_once()
        _, kwargs = mock_graph["build_graph"].call_args
        assert kwargs.get("entry_stage") == "execute"

    def test_custom_from_stage(self, mock_settings, fake_project_dir, mock_graph):
        result = runner.invoke(
            app, ["resume", str(fake_project_dir), "--from", "discover"]
        )
        assert result.exit_code == 0
        _, kwargs = mock_graph["build_graph"].call_args
        assert kwargs.get("entry_stage") == "discover"

    def test_resume_from_verify(self, mock_settings, fake_project_dir, mock_graph):
        result = runner.invoke(
            app, ["resume", str(fake_project_dir), "--from", "verify"]
        )
        assert result.exit_code == 0
        _, kwargs = mock_graph["build_graph"].call_args
        assert kwargs.get("entry_stage") == "verify"


# ---------------------------------------------------------------------------
# Resume command — state reconstruction from artifacts
# ---------------------------------------------------------------------------


class TestResumeStateReconstruction:
    """resume reconstructs state from metadata and artifact files."""

    def test_state_from_metadata(self, mock_settings, fake_project_dir, mock_graph):
        result = runner.invoke(app, ["resume", str(fake_project_dir)])
        assert result.exit_code == 0
        state = mock_graph["compiled"].ainvoke.call_args[0][0]

        assert state["repo_path"] == "/tmp/some-repo"
        assert state["project_id"] == "feat_abc12345"
        assert state["feature_prompt"] == "Add dark mode"
        assert state["feature_branch"] == "feature/feat_abc12345"

    def test_resume_loads_artifacts(self, mock_settings, fake_project_dir, mock_graph):
        """When artifact files exist, they are loaded into state."""
        arts = fake_project_dir / "artifacts"
        arts.mkdir(parents=True, exist_ok=True)
        (arts / "codebase_profile.json").write_text('{"lang": "python"}')
        (arts / "discovery_report.md").write_text("# Discovery\nFound stuff")
        (arts / "technical_assessment.json").write_text('{"risk": "low"}')
        (arts / "research_report.md").write_text("# Research\nLearned things")
        (arts / "feature_spec.json").write_text('{"name": "dark-mode"}')
        (arts / "grill_transcript.md").write_text("Q: Why?\nA: Because")
        (arts / "build_plan.json").write_text('{"units": [{"id": "u1"}]}')

        result = runner.invoke(app, ["resume", str(fake_project_dir)])
        assert result.exit_code == 0
        state = mock_graph["compiled"].ainvoke.call_args[0][0]

        assert state["codebase_profile"] == {"lang": "python"}
        assert state["discovery_report"] == "# Discovery\nFound stuff"
        assert state["technical_assessment"] == {"risk": "low"}
        assert state["research_report"] == "# Research\nLearned things"
        assert state["feature_spec"] == {"name": "dark-mode"}
        assert state["grill_transcript"] == "Q: Why?\nA: Because"
        assert state["build_plan"] == [{"id": "u1"}]

    def test_resume_missing_artifacts_default_empty(
        self, mock_settings, fake_project_dir, mock_graph
    ):
        """When artifact files don't exist, state gets empty defaults."""
        result = runner.invoke(app, ["resume", str(fake_project_dir)])
        assert result.exit_code == 0
        state = mock_graph["compiled"].ainvoke.call_args[0][0]

        assert state["codebase_profile"] == {}
        assert state["discovery_report"] == ""
        assert state["technical_assessment"] == {}
        assert state["research_report"] == ""
        assert state["feature_spec"] == {}
        assert state["grill_transcript"] == ""
        assert state["build_plan"] == []

    def test_resume_gates_preset(self, mock_settings, fake_project_dir, mock_graph):
        """Resume pre-sets plan_approved and grill_complete to True."""
        result = runner.invoke(app, ["resume", str(fake_project_dir)])
        assert result.exit_code == 0
        state = mock_graph["compiled"].ainvoke.call_args[0][0]
        assert state["plan_approved"] is True
        assert state["grill_complete"] is True
        assert state["research_redo_needed"] is False

    def test_resume_auto_approve_flag(
        self, mock_settings, fake_project_dir, mock_graph
    ):
        result = runner.invoke(app, ["resume", str(fake_project_dir), "--auto-approve"])
        assert result.exit_code == 0
        state = mock_graph["compiled"].ainvoke.call_args[0][0]
        assert state["auto_approve"] is True

    def test_resume_settings_propagated(
        self, mock_settings, fake_project_dir, mock_graph
    ):
        result = runner.invoke(app, ["resume", str(fake_project_dir)])
        assert result.exit_code == 0
        state = mock_graph["compiled"].ainvoke.call_args[0][0]
        assert state["model"] == "claude-sonnet-4-20250514"
        assert state["max_agent_turns"] == 10


# ---------------------------------------------------------------------------
# Resume command — PR URL output
# ---------------------------------------------------------------------------


class TestResumePrOutput:
    """Resume displays PR URL when present."""

    def test_pr_url_shown(self, mock_settings, fake_project_dir, mock_graph):
        result = runner.invoke(app, ["resume", str(fake_project_dir)])
        assert result.exit_code == 0
        assert "PR opened" in result.output or "pull/42" in result.output

    def test_no_pr_url_no_message(
        self, mock_settings, fake_project_dir, mock_graph_no_pr
    ):
        result = runner.invoke(app, ["resume", str(fake_project_dir)])
        assert result.exit_code == 0
        assert "PR opened" not in result.output


# ---------------------------------------------------------------------------
# List command
# ---------------------------------------------------------------------------


class TestListCommand:
    """list command shows all feature sessions."""

    def test_list_no_args(self, mock_settings):
        with patch("graft.cli.list_projects", return_value=[]) as lp:
            with patch("graft.cli.UI") as MockUI:
                ui_inst = MagicMock()
                MockUI.return_value = ui_inst
                result = runner.invoke(app, ["list"])
                assert result.exit_code == 0
                lp.assert_called_once_with(mock_settings.projects_root)
                ui_inst.show_projects.assert_called_once_with([])

    def test_list_with_projects(self, mock_settings):
        projects = [
            {
                "project_id": "feat_111",
                "repo_path": "/tmp/repo1",
                "feature_prompt": "Feature A",
                "status": "in_progress",
                "stages_completed": ["discover"],
                "created_at": "2025-01-01T00:00:00",
            },
            {
                "project_id": "feat_222",
                "repo_path": "/tmp/repo2",
                "feature_prompt": "Feature B",
                "status": "completed",
                "stages_completed": [
                    "discover",
                    "research",
                    "plan",
                    "execute",
                    "verify",
                ],
                "created_at": "2025-01-02T00:00:00",
            },
        ]
        with patch("graft.cli.list_projects", return_value=projects):
            with patch("graft.cli.UI") as MockUI:
                ui_inst = MagicMock()
                MockUI.return_value = ui_inst
                result = runner.invoke(app, ["list"])
                assert result.exit_code == 0
                ui_inst.show_projects.assert_called_once_with(projects)

    def test_list_help(self):
        result = runner.invoke(app, ["list", "--help"])
        assert result.exit_code == 0
        assert "List all feature sessions" in result.output


# ---------------------------------------------------------------------------
# Top-level CLI
# ---------------------------------------------------------------------------


class TestTopLevelCli:
    """Top-level help and unknown commands."""

    def test_app_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "build" in result.output
        assert "resume" in result.output
        assert "list" in result.output
        assert "graft" in result.output.lower() or "feature" in result.output.lower()

    def test_unknown_command(self):
        result = runner.invoke(app, ["nonexistent"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Build command — build_graph called with correct args
# ---------------------------------------------------------------------------


class TestBuildGraphInvocation:
    """build calls build_graph(ui) with no entry_stage."""

    def test_build_graph_called_with_ui(
        self, mock_settings, fake_repo, mock_create_project, mock_graph
    ):
        result = runner.invoke(app, ["build", str(fake_repo), "Add feature"])
        assert result.exit_code == 0
        mock_graph["build_graph"].assert_called_once()
        args, kwargs = mock_graph["build_graph"].call_args
        # First positional arg is a UI instance
        from graft.ui import UI

        assert isinstance(args[0], UI)
        # No entry_stage kwarg for build
        assert "entry_stage" not in kwargs


# ---------------------------------------------------------------------------
# Build command — combined flags
# ---------------------------------------------------------------------------


class TestBuildCombinedFlags:
    """Test multiple flags together."""

    def test_all_flags_combined(
        self, mock_settings, fake_repo, mock_create_project, mock_graph
    ):
        subdir = fake_repo / "src"
        subdir.mkdir()
        result = runner.invoke(
            app,
            [
                "build",
                str(fake_repo),
                "Build full feature",
                "--path",
                "src",
                "--constraint",
                "no breaking changes",
                "-c",
                "use typescript",
                "--max-units",
                "3",
                "--auto-approve",
                "-v",
            ],
        )
        assert result.exit_code == 0
        state = mock_graph["compiled"].ainvoke.call_args[0][0]
        assert state["scope_path"] == "src"
        assert state["constraints"] == ["no breaking changes", "use typescript"]
        assert state["max_units"] == 3
        assert state["auto_approve"] is True
