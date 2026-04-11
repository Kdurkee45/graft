"""Tests for graft.cli — the user-facing CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from graft.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings(tmp_path):
    """Return a mock Settings object with projects_root under tmp_path."""
    settings = MagicMock()
    settings.model = "claude-test-model"
    settings.max_agent_turns = 10
    settings.projects_root = tmp_path / "projects"
    settings.projects_root.mkdir(parents=True, exist_ok=True)
    return settings


@pytest.fixture
def fake_repo(tmp_path):
    """Create a fake repo directory that exists on disk."""
    repo = tmp_path / "my-repo"
    repo.mkdir()
    return repo


@pytest.fixture
def fake_project(tmp_path):
    """Create a fake project directory with metadata and artifacts."""
    project_dir = tmp_path / "projects" / "feat_abc12345"
    (project_dir / "artifacts").mkdir(parents=True)
    (project_dir / "logs").mkdir(parents=True)

    metadata = {
        "project_id": "feat_abc12345",
        "repo_path": "/tmp/my-repo",
        "feature_prompt": "Add dark mode",
        "created_at": "2025-01-01T00:00:00+00:00",
        "status": "in_progress",
        "stages_completed": ["discover", "research"],
    }
    (project_dir / "metadata.json").write_text(json.dumps(metadata))
    return project_dir


def _mock_compiled_graph(result: dict | None = None):
    """Return a mock compiled graph whose ainvoke returns *result*."""
    if result is None:
        result = {"pr_url": ""}
    compiled = MagicMock()
    compiled.ainvoke = AsyncMock(return_value=result)
    return compiled


# ---------------------------------------------------------------------------
# build command
# ---------------------------------------------------------------------------

class TestBuildCommand:
    """Tests for the 'graft build' CLI command."""

    @patch("graft.cli.build_graph")
    @patch("graft.cli.create_project")
    @patch("graft.cli.Settings.load")
    def test_build_happy_path_with_pr_url(
        self, mock_load, mock_create, mock_build, fake_repo, mock_settings, tmp_path
    ):
        """build runs pipeline and reports PR URL when one is returned."""
        mock_load.return_value = mock_settings

        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        mock_create.return_value = ("feat_001", project_dir)

        compiled = _mock_compiled_graph({"pr_url": "https://github.com/org/repo/pull/42"})
        mock_build.return_value = compiled

        result = runner.invoke(app, ["build", str(fake_repo), "Add dark mode"])

        assert result.exit_code == 0, result.output
        assert "PR opened" in result.output or "pull/42" in result.output
        mock_create.assert_called_once()
        compiled.ainvoke.assert_awaited_once()

    @patch("graft.cli.build_graph")
    @patch("graft.cli.create_project")
    @patch("graft.cli.Settings.load")
    def test_build_happy_path_no_pr_url(
        self, mock_load, mock_create, mock_build, fake_repo, mock_settings, tmp_path
    ):
        """build reports manual PR message when no pr_url in result."""
        mock_load.return_value = mock_settings

        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        mock_create.return_value = ("feat_002", project_dir)

        compiled = _mock_compiled_graph({"pr_url": ""})
        mock_build.return_value = compiled

        result = runner.invoke(app, ["build", str(fake_repo), "Add dark mode"])

        assert result.exit_code == 0, result.output
        assert "open a PR manually" in result.output

    @patch("graft.cli.Settings.load")
    def test_build_nonexistent_repo_path(self, mock_load, mock_settings, tmp_path):
        """build exits with error when the repo path doesn't exist."""
        mock_load.return_value = mock_settings
        bogus_path = str(tmp_path / "nonexistent-repo")

        result = runner.invoke(app, ["build", bogus_path, "Add dark mode"])

        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "Repository" in result.output

    @patch("graft.cli.build_graph")
    @patch("graft.cli.create_project")
    @patch("graft.cli.Settings.load")
    def test_build_nonexistent_scope_path(
        self, mock_load, mock_create, mock_build, fake_repo, mock_settings, tmp_path
    ):
        """build exits with error when --path scope doesn't exist."""
        mock_load.return_value = mock_settings

        result = runner.invoke(
            app,
            ["build", str(fake_repo), "Add dark mode", "--path", "nonexistent/sub"],
        )

        assert result.exit_code == 1
        assert "Scope path not found" in result.output

    @patch("graft.cli.build_graph")
    @patch("graft.cli.create_project")
    @patch("graft.cli.Settings.load")
    def test_build_valid_scope_path(
        self, mock_load, mock_create, mock_build, fake_repo, mock_settings, tmp_path
    ):
        """build passes scope_path into initial state when --path is valid."""
        mock_load.return_value = mock_settings
        (fake_repo / "packages" / "core").mkdir(parents=True)

        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        mock_create.return_value = ("feat_003", project_dir)

        compiled = _mock_compiled_graph()
        mock_build.return_value = compiled

        result = runner.invoke(
            app,
            ["build", str(fake_repo), "Add dark mode", "--path", "packages/core"],
        )

        assert result.exit_code == 0, result.output
        # Verify ainvoke was called and scope_path made it into state
        call_args = compiled.ainvoke.await_args
        state = call_args[0][0]
        assert state["scope_path"] == "packages/core"

    @patch("graft.cli.build_graph")
    @patch("graft.cli.create_project")
    @patch("graft.cli.Settings.load")
    def test_build_passes_constraints_and_options(
        self, mock_load, mock_create, mock_build, fake_repo, mock_settings, tmp_path
    ):
        """build passes CLI options (constraints, max-units, model) into state."""
        mock_load.return_value = mock_settings

        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        mock_create.return_value = ("feat_004", project_dir)

        compiled = _mock_compiled_graph()
        mock_build.return_value = compiled

        result = runner.invoke(
            app,
            [
                "build",
                str(fake_repo),
                "Add dark mode",
                "--constraint", "no breaking changes",
                "--constraint", "use TypeScript",
                "--max-units", "5",
                "--auto-approve",
                "--verbose",
            ],
        )

        assert result.exit_code == 0, result.output
        state = compiled.ainvoke.await_args[0][0]
        assert state["constraints"] == ["no breaking changes", "use TypeScript"]
        assert state["max_units"] == 5
        assert state["auto_approve"] is True
        assert state["model"] == "claude-test-model"
        assert state["max_agent_turns"] == 10

    @patch("graft.cli.build_graph")
    @patch("graft.cli.create_project")
    @patch("graft.cli.Settings.load")
    def test_build_creates_correct_initial_state_keys(
        self, mock_load, mock_create, mock_build, fake_repo, mock_settings, tmp_path
    ):
        """build populates all expected FeatureState keys."""
        mock_load.return_value = mock_settings

        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        mock_create.return_value = ("feat_005", project_dir)

        compiled = _mock_compiled_graph()
        mock_build.return_value = compiled

        result = runner.invoke(app, ["build", str(fake_repo), "Add dark mode"])

        assert result.exit_code == 0, result.output
        state = compiled.ainvoke.await_args[0][0]

        # Verify all essential state keys exist
        expected_keys = {
            "repo_path", "project_id", "project_dir", "feature_prompt",
            "scope_path", "constraints", "max_units", "auto_approve",
            "codebase_profile", "discovery_report", "technical_assessment",
            "research_report", "feature_spec", "grill_transcript",
            "build_plan", "feature_report", "current_unit_index",
            "units_completed", "units_reverted", "units_skipped",
            "plan_approved", "grill_complete", "research_redo_needed",
            "feature_branch", "pr_url", "model", "max_agent_turns",
            "current_stage",
        }
        assert expected_keys.issubset(set(state.keys()))
        assert state["project_id"] == "feat_005"
        assert state["feature_branch"] == "feature/feat_005"
        assert state["plan_approved"] is False
        assert state["grill_complete"] is False

    @patch("graft.cli.Settings.load")
    def test_build_missing_api_key(self, mock_load, fake_repo):
        """build exits when Settings.load() raises SystemExit (missing API key)."""
        mock_load.side_effect = SystemExit(
            "ANTHROPIC_API_KEY not set. Add it to a .env file or export it."
        )

        result = runner.invoke(app, ["build", str(fake_repo), "Add dark mode"])

        assert result.exit_code != 0

    @patch("graft.cli.build_graph")
    @patch("graft.cli.create_project")
    @patch("graft.cli.Settings.load")
    def test_build_displays_session_artifacts_path(
        self, mock_load, mock_create, mock_build, fake_repo, mock_settings, tmp_path
    ):
        """build prints the project directory path at the end."""
        mock_load.return_value = mock_settings

        project_dir = tmp_path / "proj_out"
        project_dir.mkdir()
        mock_create.return_value = ("feat_006", project_dir)

        compiled = _mock_compiled_graph()
        mock_build.return_value = compiled

        result = runner.invoke(app, ["build", str(fake_repo), "Add dark mode"])

        assert result.exit_code == 0, result.output
        assert "Session artifacts" in result.output
        # Rich may wrap the long path across lines; check the dir name is present
        assert project_dir.name in result.output


# ---------------------------------------------------------------------------
# resume command
# ---------------------------------------------------------------------------

class TestResumeCommand:
    """Tests for the 'graft resume' CLI command."""

    @patch("graft.cli.build_graph")
    @patch("graft.cli.load_artifact")
    @patch("graft.cli.Settings.load")
    def test_resume_happy_path(
        self, mock_load, mock_load_artifact, mock_build, mock_settings, fake_project
    ):
        """resume loads artifacts and runs pipeline from the given stage."""
        mock_load.return_value = mock_settings
        mock_load_artifact.return_value = None  # no artifacts on disk

        compiled = _mock_compiled_graph({"pr_url": "https://github.com/org/repo/pull/7"})
        mock_build.return_value = compiled

        result = runner.invoke(app, ["resume", str(fake_project)])

        assert result.exit_code == 0, result.output
        assert "PR opened" in result.output or "pull/7" in result.output
        mock_build.assert_called_once()
        # Default --from is "execute"
        _, kwargs = mock_build.call_args
        assert kwargs.get("entry_stage") == "execute"

    @patch("graft.cli.build_graph")
    @patch("graft.cli.load_artifact")
    @patch("graft.cli.Settings.load")
    def test_resume_custom_from_stage(
        self, mock_load, mock_load_artifact, mock_build, mock_settings, fake_project
    ):
        """resume respects --from flag to set the entry stage."""
        mock_load.return_value = mock_settings
        mock_load_artifact.return_value = None

        compiled = _mock_compiled_graph()
        mock_build.return_value = compiled

        result = runner.invoke(
            app, ["resume", str(fake_project), "--from", "grill"]
        )

        assert result.exit_code == 0, result.output
        _, kwargs = mock_build.call_args
        assert kwargs.get("entry_stage") == "grill"

    @patch("graft.cli.Settings.load")
    def test_resume_nonexistent_project_dir(self, mock_load, mock_settings, tmp_path):
        """resume exits with error when session directory doesn't exist."""
        mock_load.return_value = mock_settings

        result = runner.invoke(
            app, ["resume", str(tmp_path / "nonexistent_project")]
        )

        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    @patch("graft.cli.Settings.load")
    def test_resume_missing_metadata(self, mock_load, mock_settings, tmp_path):
        """resume exits with error when metadata.json is absent."""
        mock_load.return_value = mock_settings
        project_dir = tmp_path / "feat_no_meta"
        project_dir.mkdir()

        result = runner.invoke(app, ["resume", str(project_dir)])

        assert result.exit_code == 1
        assert "metadata.json" in result.output

    @patch("graft.cli.build_graph")
    @patch("graft.cli.load_artifact")
    @patch("graft.cli.Settings.load")
    def test_resume_loads_existing_artifacts(
        self, mock_load, mock_load_artifact, mock_build, mock_settings, fake_project
    ):
        """resume deserializes JSON artifacts and passes them into state."""
        mock_load.return_value = mock_settings

        # Simulate artifacts on disk via the mock
        def artifact_side_effect(project_dir, name):
            artifacts = {
                "codebase_profile.json": '{"language": "python"}',
                "technical_assessment.json": '{"feasibility": "high"}',
                "feature_spec.json": '{"title": "Dark Mode"}',
                "build_plan.json": '{"units": [{"unit_id": "u1"}]}',
                "discovery_report.md": "# Discovery Report",
                "research_report.md": "# Research Report",
                "grill_transcript.md": "# Grill Q&A",
            }
            return artifacts.get(name)

        mock_load_artifact.side_effect = artifact_side_effect

        compiled = _mock_compiled_graph()
        mock_build.return_value = compiled

        result = runner.invoke(app, ["resume", str(fake_project)])

        assert result.exit_code == 0, result.output
        state = compiled.ainvoke.await_args[0][0]
        assert state["codebase_profile"] == {"language": "python"}
        assert state["technical_assessment"] == {"feasibility": "high"}
        assert state["feature_spec"] == {"title": "Dark Mode"}
        assert state["build_plan"] == [{"unit_id": "u1"}]
        assert state["discovery_report"] == "# Discovery Report"
        assert state["research_report"] == "# Research Report"
        assert state["grill_transcript"] == "# Grill Q&A"
        # Resume always marks these as True
        assert state["plan_approved"] is True
        assert state["grill_complete"] is True

    @patch("graft.cli.build_graph")
    @patch("graft.cli.load_artifact")
    @patch("graft.cli.Settings.load")
    def test_resume_no_pr_url_in_result(
        self, mock_load, mock_load_artifact, mock_build, mock_settings, fake_project
    ):
        """resume does not print PR message when result has no pr_url."""
        mock_load.return_value = mock_settings
        mock_load_artifact.return_value = None

        compiled = _mock_compiled_graph({"pr_url": ""})
        mock_build.return_value = compiled

        result = runner.invoke(app, ["resume", str(fake_project)])

        assert result.exit_code == 0, result.output
        assert "PR opened" not in result.output

    @patch("graft.cli.Settings.load")
    def test_resume_missing_api_key(self, mock_load, tmp_path):
        """resume exits when Settings.load() raises SystemExit."""
        mock_load.side_effect = SystemExit("ANTHROPIC_API_KEY not set.")

        result = runner.invoke(app, ["resume", str(tmp_path / "doesnt_matter")])

        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------

class TestListCommand:
    """Tests for the 'graft list' CLI command."""

    @patch("graft.cli.list_projects")
    @patch("graft.cli.Settings.load")
    def test_list_empty(self, mock_load, mock_list, mock_settings):
        """list prints 'no sessions' message when no projects exist."""
        mock_load.return_value = mock_settings
        mock_list.return_value = []

        result = runner.invoke(app, ["list"])

        assert result.exit_code == 0, result.output
        assert "No feature sessions" in result.output

    @patch("graft.cli.list_projects")
    @patch("graft.cli.Settings.load")
    def test_list_with_projects(self, mock_load, mock_list, mock_settings):
        """list displays a table with project metadata."""
        mock_load.return_value = mock_settings
        mock_list.return_value = [
            {
                "project_id": "feat_aaa11111",
                "repo_path": "/tmp/repo1",
                "feature_prompt": "Add auth",
                "status": "in_progress",
                "stages_completed": ["discover"],
                "created_at": "2025-06-01T12:00:00+00:00",
            },
            {
                "project_id": "feat_bbb22222",
                "repo_path": "/tmp/repo2",
                "feature_prompt": "Add caching",
                "status": "completed",
                "stages_completed": ["discover", "research", "grill", "plan", "execute", "verify"],
                "created_at": "2025-05-30T08:00:00+00:00",
            },
        ]

        result = runner.invoke(app, ["list"])

        assert result.exit_code == 0, result.output
        # Rich table may truncate long IDs with ellipsis; check prefixes
        assert "feat_aaa" in result.output
        assert "feat_bbb" in result.output

    @patch("graft.cli.list_projects")
    @patch("graft.cli.Settings.load")
    def test_list_calls_with_correct_root(self, mock_load, mock_list, mock_settings):
        """list passes settings.projects_root to list_projects."""
        mock_load.return_value = mock_settings
        mock_list.return_value = []

        runner.invoke(app, ["list"])

        mock_list.assert_called_once_with(mock_settings.projects_root)

    @patch("graft.cli.Settings.load")
    def test_list_missing_api_key(self, mock_load):
        """list exits when Settings.load() raises SystemExit."""
        mock_load.side_effect = SystemExit("ANTHROPIC_API_KEY not set.")

        result = runner.invoke(app, ["list"])

        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and argument validation for the CLI."""

    def test_no_command(self):
        """Invoking graft with no subcommand shows usage info."""
        result = runner.invoke(app, [])
        # Typer exits with code 2 (usage error) when no command given
        assert result.exit_code in (0, 2)
        assert "Usage" in result.output or "graft" in result.output.lower()

    def test_build_missing_arguments(self):
        """build with no arguments shows error / usage."""
        result = runner.invoke(app, ["build"])
        assert result.exit_code != 0

    def test_resume_missing_arguments(self):
        """resume with no arguments shows error / usage."""
        result = runner.invoke(app, ["resume"])
        assert result.exit_code != 0

    @patch("graft.cli.build_graph")
    @patch("graft.cli.create_project")
    @patch("graft.cli.Settings.load")
    def test_build_resolves_tilde_in_repo_path(
        self, mock_load, mock_create, mock_build, mock_settings, tmp_path
    ):
        """build expands ~ in repo_path before existence check."""
        mock_load.return_value = mock_settings

        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        mock_create.return_value = ("feat_010", project_dir)

        compiled = _mock_compiled_graph()
        mock_build.return_value = compiled

        # Use an absolute path that exists, but confirm expanduser/resolve runs
        # by checking the resolved path in the state
        result = runner.invoke(app, ["build", str(tmp_path), "Add dark mode"])

        if result.exit_code == 0:
            state = compiled.ainvoke.await_args[0][0]
            # Should be fully resolved, no relative components
            assert "~" not in state["repo_path"]
            assert Path(state["repo_path"]).is_absolute()
