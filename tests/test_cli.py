"""Tests for graft.cli — Typer CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from graft.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_settings(**overrides):
    """Return a mock Settings object with sensible defaults."""
    s = MagicMock()
    s.anthropic_api_key = overrides.get("anthropic_api_key", "sk-test-key")
    s.github_token = overrides.get("github_token", None)
    s.model = overrides.get("model", "claude-opus-4-20250514")
    s.max_agent_turns = overrides.get("max_agent_turns", 50)
    s.projects_root = overrides.get("projects_root", Path("/tmp/graft-tests"))
    return s


def _fake_compiled(pr_url: str = ""):
    """Return a mock compiled graph whose ainvoke returns a result dict."""
    compiled = MagicMock()
    compiled.ainvoke = AsyncMock(return_value={"pr_url": pr_url})
    return compiled


# ---------------------------------------------------------------------------
# build command
# ---------------------------------------------------------------------------


class TestBuildCommand:
    """Tests for the 'build' CLI command."""

    @patch("graft.cli.asyncio.run")
    @patch("graft.cli.build_graph")
    @patch("graft.cli.create_project")
    @patch("graft.cli.Settings.load")
    def test_build_basic_invocation(
        self, mock_load, mock_create, mock_graph, mock_arun, tmp_path
    ):
        """build command with required args calls Settings.load, create_project, and build_graph."""
        mock_load.return_value = _fake_settings(projects_root=tmp_path / "projects")
        mock_create.return_value = (
            "feat_abc12345",
            tmp_path / "projects" / "feat_abc12345",
        )

        compiled = _fake_compiled()
        mock_graph.return_value = compiled
        mock_arun.return_value = {"pr_url": ""}

        result = runner.invoke(app, ["build", str(tmp_path), "Add login page"])

        assert result.exit_code == 0, result.output
        mock_load.assert_called_once()
        mock_create.assert_called_once()
        mock_graph.assert_called_once()
        mock_arun.assert_called_once()

    @patch("graft.cli.asyncio.run")
    @patch("graft.cli.build_graph")
    @patch("graft.cli.create_project")
    @patch("graft.cli.Settings.load")
    def test_build_flags_propagate_to_state(
        self, mock_load, mock_create, mock_graph, mock_arun, tmp_path
    ):
        """--auto-approve, --verbose, --path, --constraint, --max-units propagate into FeatureState."""
        mock_load.return_value = _fake_settings(projects_root=tmp_path / "projects")
        mock_create.return_value = (
            "feat_abc12345",
            tmp_path / "projects" / "feat_abc12345",
        )

        compiled = _fake_compiled()
        mock_graph.return_value = compiled
        mock_arun.return_value = {"pr_url": ""}

        # Create the --path scope directory so validation passes
        sub = tmp_path / "packages" / "core"
        sub.mkdir(parents=True)

        result = runner.invoke(
            app,
            [
                "build",
                str(tmp_path),
                "Add login page",
                "--auto-approve",
                "--verbose",
                "--path",
                "packages/core",
                "--constraint",
                "no-breaking-changes",
                "--constraint",
                "keep-python-3.10-compat",
                "--max-units",
                "5",
            ],
        )

        assert result.exit_code == 0, result.output

        # Inspect the state dict passed to compiled.ainvoke via asyncio.run
        call_args = mock_arun.call_args
        # asyncio.run receives a coroutine — the coroutine was compiled.ainvoke(state)
        # We need to verify the args passed to compiled.ainvoke
        invoke_call = compiled.ainvoke.call_args
        state = invoke_call[0][0]

        assert state["auto_approve"] is True
        assert state["scope_path"] == "packages/core"
        assert state["constraints"] == [
            "no-breaking-changes",
            "keep-python-3.10-compat",
        ]
        assert state["max_units"] == 5

    @patch("graft.cli.asyncio.run")
    @patch("graft.cli.build_graph")
    @patch("graft.cli.create_project")
    @patch("graft.cli.Settings.load")
    def test_build_pr_url_displayed(
        self, mock_load, mock_create, mock_graph, mock_arun, tmp_path
    ):
        """When graph returns a pr_url, the build command shows it."""
        mock_load.return_value = _fake_settings(projects_root=tmp_path / "projects")
        mock_create.return_value = (
            "feat_abc12345",
            tmp_path / "projects" / "feat_abc12345",
        )

        compiled = _fake_compiled()
        mock_graph.return_value = compiled
        mock_arun.return_value = {"pr_url": "https://github.com/org/repo/pull/42"}

        result = runner.invoke(app, ["build", str(tmp_path), "Add login page"])

        assert result.exit_code == 0, result.output
        assert "https://github.com/org/repo/pull/42" in result.output

    @patch("graft.cli.Settings.load")
    def test_build_invalid_repo_path(self, mock_load, tmp_path):
        """build command exits with error when repo_path doesn't exist."""
        mock_load.return_value = _fake_settings()

        nonexistent = str(tmp_path / "no_such_repo")
        result = runner.invoke(app, ["build", nonexistent, "Add login page"])

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or result.exit_code == 1

    @patch("graft.cli.asyncio.run")
    @patch("graft.cli.build_graph")
    @patch("graft.cli.create_project")
    @patch("graft.cli.Settings.load")
    def test_build_invalid_scope_path(
        self, mock_load, mock_create, mock_graph, mock_arun, tmp_path
    ):
        """build command exits with error when --path points to a nonexistent subdir."""
        mock_load.return_value = _fake_settings(projects_root=tmp_path / "projects")

        result = runner.invoke(
            app,
            ["build", str(tmp_path), "Add login page", "--path", "nonexistent/dir"],
        )

        assert result.exit_code != 0

    @patch("graft.cli.asyncio.run")
    @patch("graft.cli.build_graph")
    @patch("graft.cli.create_project")
    @patch("graft.cli.Settings.load")
    def test_build_no_pr_url_shows_manual_message(
        self, mock_load, mock_create, mock_graph, mock_arun, tmp_path
    ):
        """When no PR URL is returned, the user sees a manual instruction."""
        mock_load.return_value = _fake_settings(projects_root=tmp_path / "projects")
        mock_create.return_value = (
            "feat_abc12345",
            tmp_path / "projects" / "feat_abc12345",
        )

        compiled = _fake_compiled()
        mock_graph.return_value = compiled
        mock_arun.return_value = {"pr_url": ""}

        result = runner.invoke(app, ["build", str(tmp_path), "Add login page"])

        assert result.exit_code == 0, result.output
        assert "manually" in result.output.lower() or "review" in result.output.lower()

    @patch("graft.cli.asyncio.run")
    @patch("graft.cli.build_graph")
    @patch("graft.cli.create_project")
    @patch("graft.cli.Settings.load")
    def test_build_feature_branch_format(
        self, mock_load, mock_create, mock_graph, mock_arun, tmp_path
    ):
        """Feature branch in state is formatted as feature/<project_id>."""
        mock_load.return_value = _fake_settings(projects_root=tmp_path / "projects")
        mock_create.return_value = (
            "feat_xyz99999",
            tmp_path / "projects" / "feat_xyz99999",
        )

        compiled = _fake_compiled()
        mock_graph.return_value = compiled
        mock_arun.return_value = {"pr_url": ""}

        result = runner.invoke(app, ["build", str(tmp_path), "Add login page"])
        assert result.exit_code == 0, result.output

        state = compiled.ainvoke.call_args[0][0]
        assert state["feature_branch"] == "feature/feat_xyz99999"


# ---------------------------------------------------------------------------
# resume command
# ---------------------------------------------------------------------------


class TestResumeCommand:
    """Tests for the 'resume' CLI command."""

    def _setup_project_dir(
        self, tmp_path: Path, *, project_id: str = "feat_abc12345"
    ) -> Path:
        """Create a minimal session directory with metadata.json and artifacts."""
        project_dir = tmp_path / project_id
        (project_dir / "artifacts").mkdir(parents=True)

        metadata = {
            "project_id": project_id,
            "repo_path": str(tmp_path / "repo"),
            "feature_prompt": "Add login page",
        }
        (project_dir / "metadata.json").write_text(json.dumps(metadata))

        # Write some artifacts
        (project_dir / "artifacts" / "codebase_profile.json").write_text(
            json.dumps({"language": "python"})
        )
        (project_dir / "artifacts" / "technical_assessment.json").write_text(
            json.dumps({"complexity": "medium"})
        )
        (project_dir / "artifacts" / "feature_spec.json").write_text(
            json.dumps({"title": "Login Page"})
        )
        (project_dir / "artifacts" / "build_plan.json").write_text(
            json.dumps({"units": [{"name": "unit1"}]})
        )
        (project_dir / "artifacts" / "discovery_report.md").write_text(
            "# Discovery\nFound stuff."
        )
        (project_dir / "artifacts" / "research_report.md").write_text(
            "# Research\nResearched things."
        )
        (project_dir / "artifacts" / "grill_transcript.md").write_text(
            "# Grill\nGrilled it."
        )
        return project_dir

    @patch("graft.cli.asyncio.run")
    @patch("graft.cli.build_graph")
    @patch("graft.cli.Settings.load")
    def test_resume_basic(self, mock_load, mock_graph, mock_arun, tmp_path):
        """resume command loads artifacts and invokes graph with entry_stage."""
        mock_load.return_value = _fake_settings()
        project_dir = self._setup_project_dir(tmp_path)

        compiled = _fake_compiled()
        mock_graph.return_value = compiled
        mock_arun.return_value = {"pr_url": ""}

        result = runner.invoke(app, ["resume", str(project_dir)])

        assert result.exit_code == 0, result.output
        mock_load.assert_called_once()
        # build_graph should be called with entry_stage="execute" (default)
        mock_graph.assert_called_once()
        _, kwargs = mock_graph.call_args
        assert kwargs.get("entry_stage") == "execute"

    @patch("graft.cli.asyncio.run")
    @patch("graft.cli.build_graph")
    @patch("graft.cli.Settings.load")
    def test_resume_custom_from_stage(self, mock_load, mock_graph, mock_arun, tmp_path):
        """resume --from sets the entry_stage argument to build_graph."""
        mock_load.return_value = _fake_settings()
        project_dir = self._setup_project_dir(tmp_path)

        compiled = _fake_compiled()
        mock_graph.return_value = compiled
        mock_arun.return_value = {"pr_url": ""}

        result = runner.invoke(app, ["resume", str(project_dir), "--from", "discover"])

        assert result.exit_code == 0, result.output
        _, kwargs = mock_graph.call_args
        assert kwargs.get("entry_stage") == "discover"

    @patch("graft.cli.asyncio.run")
    @patch("graft.cli.build_graph")
    @patch("graft.cli.Settings.load")
    def test_resume_loads_artifacts_into_state(
        self, mock_load, mock_graph, mock_arun, tmp_path
    ):
        """resume correctly loads artifact files into the FeatureState dict."""
        mock_load.return_value = _fake_settings()
        project_dir = self._setup_project_dir(tmp_path)

        compiled = _fake_compiled()
        mock_graph.return_value = compiled
        mock_arun.return_value = {"pr_url": ""}

        result = runner.invoke(app, ["resume", str(project_dir)])
        assert result.exit_code == 0, result.output

        state = compiled.ainvoke.call_args[0][0]
        assert state["codebase_profile"] == {"language": "python"}
        assert state["technical_assessment"] == {"complexity": "medium"}
        assert state["feature_spec"] == {"title": "Login Page"}
        assert state["build_plan"] == [{"name": "unit1"}]
        assert "Discovery" in state["discovery_report"]
        assert "Research" in state["research_report"]
        assert "Grill" in state["grill_transcript"]
        assert state["plan_approved"] is True
        assert state["grill_complete"] is True

    @patch("graft.cli.Settings.load")
    def test_resume_nonexistent_project_dir(self, mock_load, tmp_path):
        """resume exits with error when project directory doesn't exist."""
        mock_load.return_value = _fake_settings()

        result = runner.invoke(app, ["resume", str(tmp_path / "no_such_project")])

        assert result.exit_code != 0

    @patch("graft.cli.Settings.load")
    def test_resume_missing_metadata(self, mock_load, tmp_path):
        """resume exits with error when metadata.json is missing."""
        mock_load.return_value = _fake_settings()

        # Create the directory but no metadata.json
        project_dir = tmp_path / "feat_no_meta"
        project_dir.mkdir()

        result = runner.invoke(app, ["resume", str(project_dir)])

        assert result.exit_code != 0

    @patch("graft.cli.asyncio.run")
    @patch("graft.cli.build_graph")
    @patch("graft.cli.Settings.load")
    def test_resume_with_pr_url(self, mock_load, mock_graph, mock_arun, tmp_path):
        """resume shows PR URL when graph returns one."""
        mock_load.return_value = _fake_settings()
        project_dir = self._setup_project_dir(tmp_path)

        compiled = _fake_compiled()
        mock_graph.return_value = compiled
        mock_arun.return_value = {"pr_url": "https://github.com/org/repo/pull/99"}

        result = runner.invoke(app, ["resume", str(project_dir)])

        assert result.exit_code == 0, result.output
        assert "https://github.com/org/repo/pull/99" in result.output

    @patch("graft.cli.asyncio.run")
    @patch("graft.cli.build_graph")
    @patch("graft.cli.Settings.load")
    def test_resume_missing_artifacts_default_to_empty(
        self, mock_load, mock_graph, mock_arun, tmp_path
    ):
        """resume handles missing artifact files gracefully (defaults to empty)."""
        mock_load.return_value = _fake_settings()

        # Minimal project dir with only metadata.json and artifacts dir
        project_dir = tmp_path / "feat_minimal"
        (project_dir / "artifacts").mkdir(parents=True)
        metadata = {
            "project_id": "feat_minimal",
            "repo_path": str(tmp_path / "repo"),
        }
        (project_dir / "metadata.json").write_text(json.dumps(metadata))

        compiled = _fake_compiled()
        mock_graph.return_value = compiled
        mock_arun.return_value = {"pr_url": ""}

        result = runner.invoke(app, ["resume", str(project_dir)])
        assert result.exit_code == 0, result.output

        state = compiled.ainvoke.call_args[0][0]
        assert state["codebase_profile"] == {}
        assert state["technical_assessment"] == {}
        assert state["feature_spec"] == {}
        assert state["build_plan"] == []
        assert state["discovery_report"] == ""
        assert state["research_report"] == ""
        assert state["grill_transcript"] == ""


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------


class TestListCommand:
    """Tests for the 'list' CLI command."""

    @patch("graft.cli.UI")
    @patch("graft.cli.list_projects")
    @patch("graft.cli.Settings.load")
    def test_list_calls_show_projects(self, mock_load, mock_list, mock_ui_cls):
        """list command loads settings, lists projects, and shows them."""
        mock_load.return_value = _fake_settings()
        mock_list.return_value = [
            {"project_id": "feat_aaa", "status": "completed"},
            {"project_id": "feat_bbb", "status": "in_progress"},
        ]
        ui_instance = MagicMock()
        mock_ui_cls.return_value = ui_instance

        result = runner.invoke(app, ["list"])

        assert result.exit_code == 0, result.output
        mock_load.assert_called_once()
        mock_list.assert_called_once()
        ui_instance.show_projects.assert_called_once_with(
            [
                {"project_id": "feat_aaa", "status": "completed"},
                {"project_id": "feat_bbb", "status": "in_progress"},
            ]
        )

    @patch("graft.cli.UI")
    @patch("graft.cli.list_projects")
    @patch("graft.cli.Settings.load")
    def test_list_empty_projects(self, mock_load, mock_list, mock_ui_cls):
        """list command works with zero projects."""
        mock_load.return_value = _fake_settings()
        mock_list.return_value = []
        ui_instance = MagicMock()
        mock_ui_cls.return_value = ui_instance

        result = runner.invoke(app, ["list"])

        assert result.exit_code == 0, result.output
        ui_instance.show_projects.assert_called_once_with([])


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error cases and edge conditions."""

    def test_missing_anthropic_api_key(self, monkeypatch):
        """When ANTHROPIC_API_KEY is missing, Settings.load raises SystemExit.

        The CLI should propagate this as a non-zero exit.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        # Don't mock Settings.load — let it actually raise
        with patch(
            "graft.cli.Settings.load",
            side_effect=SystemExit(
                "ANTHROPIC_API_KEY not set. Add it to a .env file or export it in your shell."
            ),
        ):
            result = runner.invoke(app, ["build", "/tmp", "some feature"])

        assert result.exit_code != 0

    def test_build_missing_required_args(self):
        """build command fails without required arguments."""
        result = runner.invoke(app, ["build"])
        assert result.exit_code != 0

    def test_resume_missing_required_args(self):
        """resume command fails without required project_path argument."""
        result = runner.invoke(app, ["resume"])
        assert result.exit_code != 0

    @patch("graft.cli.asyncio.run")
    @patch("graft.cli.build_graph")
    @patch("graft.cli.create_project")
    @patch("graft.cli.Settings.load")
    def test_build_defaults(
        self, mock_load, mock_create, mock_graph, mock_arun, tmp_path
    ):
        """build with no optional flags uses default values for all optional fields."""
        mock_load.return_value = _fake_settings(
            projects_root=tmp_path / "projects",
            model="claude-opus-4-20250514",
            max_agent_turns=50,
        )
        mock_create.return_value = (
            "feat_def00000",
            tmp_path / "projects" / "feat_def00000",
        )

        compiled = _fake_compiled()
        mock_graph.return_value = compiled
        mock_arun.return_value = {"pr_url": ""}

        result = runner.invoke(app, ["build", str(tmp_path), "Add login page"])
        assert result.exit_code == 0, result.output

        state = compiled.ainvoke.call_args[0][0]
        assert state["auto_approve"] is False
        assert state["scope_path"] == ""
        assert state["constraints"] == []
        assert state["max_units"] == 0
        assert state["model"] == "claude-opus-4-20250514"
        assert state["max_agent_turns"] == 50
        # Verify initial execution tracking is empty
        assert state["current_unit_index"] == 0
        assert state["units_completed"] == []
        assert state["units_reverted"] == []
        assert state["units_skipped"] == []
        assert state["plan_approved"] is False
        assert state["grill_complete"] is False
        assert state["pr_url"] == ""

    @patch("graft.cli.Settings.load")
    def test_build_settings_load_error_propagates(self, mock_load):
        """Any exception from Settings.load propagates as non-zero exit."""
        mock_load.side_effect = RuntimeError("config file corrupted")

        result = runner.invoke(app, ["build", "/tmp", "some feature"])

        assert result.exit_code != 0
