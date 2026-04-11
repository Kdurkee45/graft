"""Tests for graft.cli."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from graft.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(**overrides):
    """Return a mock Settings with sensible defaults."""
    s = MagicMock()
    s.anthropic_api_key = overrides.get("anthropic_api_key", "test-key")
    s.model = overrides.get("model", "claude-opus-4-20250514")
    s.max_agent_turns = overrides.get("max_agent_turns", 50)
    s.projects_root = overrides.get("projects_root", Path("/tmp/graft/projects"))
    return s


def _make_compiled(pr_url: str = ""):
    """Return a mock compiled graph whose ainvoke returns a result dict."""
    compiled = MagicMock()
    compiled.ainvoke = AsyncMock(return_value={"pr_url": pr_url})
    return compiled


# ---------------------------------------------------------------------------
# app instance
# ---------------------------------------------------------------------------

def test_app_is_typer_instance():
    """The CLI entry point is a valid Typer application."""
    assert isinstance(app, typer.Typer)


def test_app_has_expected_commands():
    """The app registers build, resume, and list commands."""
    # Typer stores registered commands in app.registered_commands or via
    # the underlying Click group after creation.
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "build" in result.output
    assert "resume" in result.output
    assert "list" in result.output


# ---------------------------------------------------------------------------
# build command — happy path
# ---------------------------------------------------------------------------

@patch("graft.cli.asyncio.run")
@patch("graft.cli.build_graph")
@patch("graft.cli.create_project")
@patch("graft.cli.Settings.load")
def test_build_happy_path(mock_load, mock_create, mock_graph, mock_run, tmp_path):
    """build succeeds with required arguments and initialises state correctly."""
    repo = tmp_path / "repo"
    repo.mkdir()

    mock_load.return_value = _make_settings()
    mock_create.return_value = ("feat_abc123", tmp_path / "project_dir")
    compiled = _make_compiled(pr_url="https://github.com/org/repo/pull/42")
    mock_graph.return_value = compiled
    mock_run.return_value = {"pr_url": "https://github.com/org/repo/pull/42"}

    result = runner.invoke(app, ["build", str(repo), "Add dark mode"])

    assert result.exit_code == 0
    mock_load.assert_called_once()
    mock_create.assert_called_once()
    mock_graph.assert_called_once()
    mock_run.assert_called_once()

    # Verify the state passed to ainvoke
    call_args = mock_run.call_args[0][0]  # first positional arg to asyncio.run
    # asyncio.run receives the coroutine from compiled.ainvoke(state)
    compiled.ainvoke.assert_called_once()
    state = compiled.ainvoke.call_args[0][0]
    assert state["repo_path"] == str(repo.resolve())
    assert state["project_id"] == "feat_abc123"
    assert state["feature_prompt"] == "Add dark mode"
    assert state["constraints"] == []
    assert state["max_units"] == 0
    assert state["auto_approve"] is False
    assert state["model"] == "claude-opus-4-20250514"


@patch("graft.cli.asyncio.run")
@patch("graft.cli.build_graph")
@patch("graft.cli.create_project")
@patch("graft.cli.Settings.load")
def test_build_pr_url_shown(mock_load, mock_create, mock_graph, mock_run, tmp_path):
    """build prints the PR URL when the graph result contains one."""
    repo = tmp_path / "repo"
    repo.mkdir()

    mock_load.return_value = _make_settings()
    mock_create.return_value = ("feat_abc", tmp_path / "out")
    mock_graph.return_value = _make_compiled()
    mock_run.return_value = {"pr_url": "https://github.com/org/repo/pull/7"}

    result = runner.invoke(app, ["build", str(repo), "Add logging"])
    assert result.exit_code == 0
    assert "PR opened" in result.output or "artifacts" in result.output.lower()


@patch("graft.cli.asyncio.run")
@patch("graft.cli.build_graph")
@patch("graft.cli.create_project")
@patch("graft.cli.Settings.load")
def test_build_no_pr_url(mock_load, mock_create, mock_graph, mock_run, tmp_path):
    """build prints fallback message when no PR URL is returned."""
    repo = tmp_path / "repo"
    repo.mkdir()

    mock_load.return_value = _make_settings()
    mock_create.return_value = ("feat_abc", tmp_path / "out")
    mock_graph.return_value = _make_compiled(pr_url="")
    mock_run.return_value = {"pr_url": ""}

    result = runner.invoke(app, ["build", str(repo), "Improve tests"])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# build command — option parsing
# ---------------------------------------------------------------------------

@patch("graft.cli.asyncio.run")
@patch("graft.cli.build_graph")
@patch("graft.cli.create_project")
@patch("graft.cli.Settings.load")
def test_build_with_constraints(mock_load, mock_create, mock_graph, mock_run, tmp_path):
    """build passes repeated --constraint values into state."""
    repo = tmp_path / "repo"
    repo.mkdir()

    mock_load.return_value = _make_settings()
    mock_create.return_value = ("feat_c", tmp_path / "out")
    compiled = _make_compiled()
    mock_graph.return_value = compiled
    mock_run.return_value = {"pr_url": ""}

    result = runner.invoke(
        app,
        [
            "build", str(repo), "Add feature",
            "--constraint", "No new deps",
            "-c", "Keep it simple",
        ],
    )

    assert result.exit_code == 0
    state = compiled.ainvoke.call_args[0][0]
    assert state["constraints"] == ["No new deps", "Keep it simple"]


@patch("graft.cli.asyncio.run")
@patch("graft.cli.build_graph")
@patch("graft.cli.create_project")
@patch("graft.cli.Settings.load")
def test_build_with_max_units(mock_load, mock_create, mock_graph, mock_run, tmp_path):
    """build passes --max-units into the initial state."""
    repo = tmp_path / "repo"
    repo.mkdir()

    mock_load.return_value = _make_settings()
    mock_create.return_value = ("feat_m", tmp_path / "out")
    compiled = _make_compiled()
    mock_graph.return_value = compiled
    mock_run.return_value = {"pr_url": ""}

    result = runner.invoke(
        app, ["build", str(repo), "Refactor", "--max-units", "5"]
    )

    assert result.exit_code == 0
    state = compiled.ainvoke.call_args[0][0]
    assert state["max_units"] == 5


@patch("graft.cli.asyncio.run")
@patch("graft.cli.build_graph")
@patch("graft.cli.create_project")
@patch("graft.cli.Settings.load")
def test_build_auto_approve(mock_load, mock_create, mock_graph, mock_run, tmp_path):
    """build passes --auto-approve flag into state."""
    repo = tmp_path / "repo"
    repo.mkdir()

    mock_load.return_value = _make_settings()
    mock_create.return_value = ("feat_a", tmp_path / "out")
    compiled = _make_compiled()
    mock_graph.return_value = compiled
    mock_run.return_value = {"pr_url": ""}

    result = runner.invoke(
        app, ["build", str(repo), "Quickfix", "--auto-approve"]
    )

    assert result.exit_code == 0
    state = compiled.ainvoke.call_args[0][0]
    assert state["auto_approve"] is True


@patch("graft.cli.asyncio.run")
@patch("graft.cli.build_graph")
@patch("graft.cli.create_project")
@patch("graft.cli.Settings.load")
def test_build_verbose(mock_load, mock_create, mock_graph, mock_run, tmp_path):
    """build constructs UI with verbose=True when --verbose is passed."""
    repo = tmp_path / "repo"
    repo.mkdir()

    mock_load.return_value = _make_settings()
    mock_create.return_value = ("feat_v", tmp_path / "out")
    mock_graph.return_value = _make_compiled()
    mock_run.return_value = {"pr_url": ""}

    result = runner.invoke(app, ["build", str(repo), "Logging", "--verbose"])
    assert result.exit_code == 0


@patch("graft.cli.asyncio.run")
@patch("graft.cli.build_graph")
@patch("graft.cli.create_project")
@patch("graft.cli.Settings.load")
def test_build_with_scope_path(mock_load, mock_create, mock_graph, mock_run, tmp_path):
    """build resolves --path scope and sets scope_path in state."""
    repo = tmp_path / "repo"
    (repo / "packages" / "frontend").mkdir(parents=True)

    mock_load.return_value = _make_settings()
    mock_create.return_value = ("feat_s", tmp_path / "out")
    compiled = _make_compiled()
    mock_graph.return_value = compiled
    mock_run.return_value = {"pr_url": ""}

    result = runner.invoke(
        app, ["build", str(repo), "New page", "--path", "packages/frontend"]
    )

    assert result.exit_code == 0
    state = compiled.ainvoke.call_args[0][0]
    assert state["scope_path"] == "packages/frontend"


# ---------------------------------------------------------------------------
# build command — error handling
# ---------------------------------------------------------------------------

@patch("graft.cli.Settings.load", side_effect=SystemExit("ANTHROPIC_API_KEY not set"))
def test_build_missing_api_key(mock_load):
    """build exits when Settings.load() raises SystemExit (missing API key)."""
    result = runner.invoke(app, ["build", "/tmp/repo", "Feature"])
    assert result.exit_code != 0


@patch("graft.cli.Settings.load")
def test_build_repo_not_found(mock_load):
    """build exits with error when repo_path does not exist."""
    mock_load.return_value = _make_settings()

    result = runner.invoke(
        app, ["build", "/nonexistent/repo/path", "Add feature"]
    )
    assert result.exit_code != 0


@patch("graft.cli.create_project")
@patch("graft.cli.Settings.load")
def test_build_scope_path_not_found(mock_load, mock_create, tmp_path):
    """build exits with error when --path scope directory does not exist."""
    repo = tmp_path / "repo"
    repo.mkdir()

    mock_load.return_value = _make_settings()

    result = runner.invoke(
        app, ["build", str(repo), "Feature", "--path", "nonexistent/sub"]
    )
    assert result.exit_code != 0


def test_build_missing_arguments():
    """build fails when required arguments are omitted."""
    result = runner.invoke(app, ["build"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# build command — state initialisation
# ---------------------------------------------------------------------------

@patch("graft.cli.asyncio.run")
@patch("graft.cli.build_graph")
@patch("graft.cli.create_project")
@patch("graft.cli.Settings.load")
def test_build_state_defaults(mock_load, mock_create, mock_graph, mock_run, tmp_path):
    """build sets all expected default values in the initial state."""
    repo = tmp_path / "repo"
    repo.mkdir()

    settings = _make_settings(model="claude-sonnet-4-20250514", max_agent_turns=30)
    mock_load.return_value = settings
    mock_create.return_value = ("feat_d", tmp_path / "proj")
    compiled = _make_compiled()
    mock_graph.return_value = compiled
    mock_run.return_value = {"pr_url": ""}

    runner.invoke(app, ["build", str(repo), "Test defaults"])

    state = compiled.ainvoke.call_args[0][0]

    # Artifact fields should be empty
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

    # Gate flags
    assert state["plan_approved"] is False
    assert state["grill_complete"] is False
    assert state["research_redo_needed"] is False

    # Metadata fields
    assert state["feature_branch"] == "feature/feat_d"
    assert state["pr_url"] == ""
    assert state["model"] == "claude-sonnet-4-20250514"
    assert state["max_agent_turns"] == 30
    assert state["current_stage"] == ""


# ---------------------------------------------------------------------------
# resume command — happy path
# ---------------------------------------------------------------------------

@patch("graft.cli.asyncio.run")
@patch("graft.cli.build_graph")
@patch("graft.cli.load_artifact")
@patch("graft.cli.Settings.load")
def test_resume_happy_path(mock_load, mock_artifact, mock_graph, mock_run, tmp_path):
    """resume reconstructs state from disk and invokes graph."""
    project_dir = tmp_path / "feat_xyz"
    project_dir.mkdir()
    meta = {
        "project_id": "feat_xyz",
        "repo_path": "/tmp/repo",
        "feature_prompt": "Dark mode",
    }
    (project_dir / "metadata.json").write_text(json.dumps(meta))

    mock_load.return_value = _make_settings()
    mock_artifact.return_value = None  # No artifacts on disk
    compiled = _make_compiled(pr_url="https://github.com/org/repo/pull/9")
    mock_graph.return_value = compiled
    mock_run.return_value = {"pr_url": "https://github.com/org/repo/pull/9"}

    result = runner.invoke(app, ["resume", str(project_dir)])

    assert result.exit_code == 0
    mock_graph.assert_called_once()
    # Default --from is "execute"
    _, kwargs = mock_graph.call_args
    assert kwargs.get("entry_stage") == "execute"


@patch("graft.cli.asyncio.run")
@patch("graft.cli.build_graph")
@patch("graft.cli.load_artifact")
@patch("graft.cli.Settings.load")
def test_resume_from_stage(mock_load, mock_artifact, mock_graph, mock_run, tmp_path):
    """resume respects --from stage argument."""
    project_dir = tmp_path / "feat_stage"
    project_dir.mkdir()
    meta = {
        "project_id": "feat_stage",
        "repo_path": "/tmp/repo",
        "feature_prompt": "New API",
    }
    (project_dir / "metadata.json").write_text(json.dumps(meta))

    mock_load.return_value = _make_settings()
    mock_artifact.return_value = None
    compiled = _make_compiled()
    mock_graph.return_value = compiled
    mock_run.return_value = {"pr_url": ""}

    result = runner.invoke(app, ["resume", str(project_dir), "--from", "research"])

    assert result.exit_code == 0
    _, kwargs = mock_graph.call_args
    assert kwargs.get("entry_stage") == "research"


@patch("graft.cli.asyncio.run")
@patch("graft.cli.build_graph")
@patch("graft.cli.load_artifact")
@patch("graft.cli.Settings.load")
def test_resume_loads_artifacts(mock_load, mock_artifact, mock_graph, mock_run, tmp_path):
    """resume deserialises JSON artifacts from disk into state."""
    project_dir = tmp_path / "feat_art"
    project_dir.mkdir()
    meta = {
        "project_id": "feat_art",
        "repo_path": "/tmp/repo",
        "feature_prompt": "Caching",
    }
    (project_dir / "metadata.json").write_text(json.dumps(meta))

    profile = {"language": "python", "framework": "fastapi"}
    assessment = {"complexity": "medium"}
    spec = {"endpoints": ["/api/cache"]}
    plan = {"units": [{"name": "unit1"}, {"name": "unit2"}]}

    def fake_load(proj_dir, name):
        mapping = {
            "codebase_profile.json": json.dumps(profile),
            "technical_assessment.json": json.dumps(assessment),
            "feature_spec.json": json.dumps(spec),
            "build_plan.json": json.dumps(plan),
            "discovery_report.md": "# Discovery",
            "research_report.md": "# Research",
            "grill_transcript.md": "## Q1\nAnswer",
        }
        return mapping.get(name)

    mock_load.return_value = _make_settings()
    mock_artifact.side_effect = fake_load
    compiled = _make_compiled()
    mock_graph.return_value = compiled
    mock_run.return_value = {"pr_url": ""}

    result = runner.invoke(app, ["resume", str(project_dir)])
    assert result.exit_code == 0

    state = compiled.ainvoke.call_args[0][0]
    assert state["codebase_profile"] == profile
    assert state["technical_assessment"] == assessment
    assert state["feature_spec"] == spec
    assert state["build_plan"] == [{"name": "unit1"}, {"name": "unit2"}]
    assert state["discovery_report"] == "# Discovery"
    assert state["research_report"] == "# Research"
    assert state["grill_transcript"] == "## Q1\nAnswer"


@patch("graft.cli.asyncio.run")
@patch("graft.cli.build_graph")
@patch("graft.cli.load_artifact")
@patch("graft.cli.Settings.load")
def test_resume_state_flags(mock_load, mock_artifact, mock_graph, mock_run, tmp_path):
    """resume sets plan_approved and grill_complete to True."""
    project_dir = tmp_path / "feat_flags"
    project_dir.mkdir()
    meta = {
        "project_id": "feat_flags",
        "repo_path": "/tmp/repo",
        "feature_prompt": "Flags",
    }
    (project_dir / "metadata.json").write_text(json.dumps(meta))

    mock_load.return_value = _make_settings()
    mock_artifact.return_value = None
    compiled = _make_compiled()
    mock_graph.return_value = compiled
    mock_run.return_value = {"pr_url": ""}

    runner.invoke(app, ["resume", str(project_dir)])

    state = compiled.ainvoke.call_args[0][0]
    assert state["plan_approved"] is True
    assert state["grill_complete"] is True
    assert state["research_redo_needed"] is False


@patch("graft.cli.asyncio.run")
@patch("graft.cli.build_graph")
@patch("graft.cli.load_artifact")
@patch("graft.cli.Settings.load")
def test_resume_auto_approve(mock_load, mock_artifact, mock_graph, mock_run, tmp_path):
    """resume passes --auto-approve into state."""
    project_dir = tmp_path / "feat_aa"
    project_dir.mkdir()
    meta = {
        "project_id": "feat_aa",
        "repo_path": "/tmp/repo",
        "feature_prompt": "AA",
    }
    (project_dir / "metadata.json").write_text(json.dumps(meta))

    mock_load.return_value = _make_settings()
    mock_artifact.return_value = None
    compiled = _make_compiled()
    mock_graph.return_value = compiled
    mock_run.return_value = {"pr_url": ""}

    runner.invoke(app, ["resume", str(project_dir), "--auto-approve"])

    state = compiled.ainvoke.call_args[0][0]
    assert state["auto_approve"] is True


# ---------------------------------------------------------------------------
# resume command — error handling
# ---------------------------------------------------------------------------

@patch("graft.cli.Settings.load")
def test_resume_missing_project_dir(mock_load):
    """resume exits with error when project directory does not exist."""
    mock_load.return_value = _make_settings()

    result = runner.invoke(app, ["resume", "/nonexistent/feat_missing"])
    assert result.exit_code != 0


@patch("graft.cli.Settings.load")
def test_resume_missing_metadata(mock_load, tmp_path):
    """resume exits with error when metadata.json is missing."""
    project_dir = tmp_path / "feat_nometa"
    project_dir.mkdir()

    mock_load.return_value = _make_settings()

    result = runner.invoke(app, ["resume", str(project_dir)])
    assert result.exit_code != 0


@patch("graft.cli.Settings.load", side_effect=SystemExit("ANTHROPIC_API_KEY not set"))
def test_resume_missing_api_key(mock_load):
    """resume exits when Settings.load() raises SystemExit."""
    result = runner.invoke(app, ["resume", "/tmp/some/dir"])
    assert result.exit_code != 0


def test_resume_missing_arguments():
    """resume fails when required project_path argument is omitted."""
    result = runner.invoke(app, ["resume"])
    assert result.exit_code != 0


@patch("graft.cli.asyncio.run")
@patch("graft.cli.build_graph")
@patch("graft.cli.load_artifact")
@patch("graft.cli.Settings.load")
def test_resume_empty_artifacts(mock_load, mock_artifact, mock_graph, mock_run, tmp_path):
    """resume handles missing artifacts gracefully (empty defaults)."""
    project_dir = tmp_path / "feat_empty"
    project_dir.mkdir()
    meta = {
        "project_id": "feat_empty",
        "repo_path": "/tmp/repo",
        "feature_prompt": "Empty",
    }
    (project_dir / "metadata.json").write_text(json.dumps(meta))

    mock_load.return_value = _make_settings()
    mock_artifact.return_value = None  # All artifacts missing
    compiled = _make_compiled()
    mock_graph.return_value = compiled
    mock_run.return_value = {"pr_url": ""}

    result = runner.invoke(app, ["resume", str(project_dir)])
    assert result.exit_code == 0

    state = compiled.ainvoke.call_args[0][0]
    assert state["codebase_profile"] == {}
    assert state["technical_assessment"] == {}
    assert state["feature_spec"] == {}
    assert state["build_plan"] == []
    assert state["discovery_report"] == ""
    assert state["research_report"] == ""
    assert state["grill_transcript"] == ""


@patch("graft.cli.asyncio.run")
@patch("graft.cli.build_graph")
@patch("graft.cli.load_artifact")
@patch("graft.cli.Settings.load")
def test_resume_metadata_missing_feature_prompt(
    mock_load, mock_artifact, mock_graph, mock_run, tmp_path
):
    """resume handles metadata.json without feature_prompt key."""
    project_dir = tmp_path / "feat_noprompt"
    project_dir.mkdir()
    meta = {
        "project_id": "feat_noprompt",
        "repo_path": "/tmp/repo",
        # No "feature_prompt" key
    }
    (project_dir / "metadata.json").write_text(json.dumps(meta))

    mock_load.return_value = _make_settings()
    mock_artifact.return_value = None
    compiled = _make_compiled()
    mock_graph.return_value = compiled
    mock_run.return_value = {"pr_url": ""}

    result = runner.invoke(app, ["resume", str(project_dir)])
    assert result.exit_code == 0

    state = compiled.ainvoke.call_args[0][0]
    assert state["feature_prompt"] == ""


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------

@patch("graft.cli.UI")
@patch("graft.cli.list_projects")
@patch("graft.cli.Settings.load")
def test_list_empty(mock_load, mock_list, mock_ui_cls):
    """list command works when no projects exist."""
    mock_load.return_value = _make_settings()
    mock_list.return_value = []
    mock_ui = MagicMock()
    mock_ui_cls.return_value = mock_ui

    result = runner.invoke(app, ["list"])

    assert result.exit_code == 0
    mock_list.assert_called_once()
    mock_ui.show_projects.assert_called_once_with([])


@patch("graft.cli.UI")
@patch("graft.cli.list_projects")
@patch("graft.cli.Settings.load")
def test_list_populated(mock_load, mock_list, mock_ui_cls):
    """list command displays projects when they exist."""
    mock_load.return_value = _make_settings()
    projects = [
        {"project_id": "feat_a", "feature_prompt": "Feature A", "status": "in_progress"},
        {"project_id": "feat_b", "feature_prompt": "Feature B", "status": "completed"},
    ]
    mock_list.return_value = projects
    mock_ui = MagicMock()
    mock_ui_cls.return_value = mock_ui

    result = runner.invoke(app, ["list"])

    assert result.exit_code == 0
    mock_list.assert_called_once()
    mock_ui.show_projects.assert_called_once_with(projects)


@patch("graft.cli.Settings.load", side_effect=SystemExit("ANTHROPIC_API_KEY not set"))
def test_list_missing_api_key(mock_load):
    """list exits when Settings.load() raises SystemExit."""
    result = runner.invoke(app, ["list"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# build command — feature_branch derivation
# ---------------------------------------------------------------------------

@patch("graft.cli.asyncio.run")
@patch("graft.cli.build_graph")
@patch("graft.cli.create_project")
@patch("graft.cli.Settings.load")
def test_build_feature_branch_derived_from_project_id(
    mock_load, mock_create, mock_graph, mock_run, tmp_path
):
    """build derives feature_branch from the project_id."""
    repo = tmp_path / "repo"
    repo.mkdir()

    mock_load.return_value = _make_settings()
    mock_create.return_value = ("feat_hello_world", tmp_path / "out")
    compiled = _make_compiled()
    mock_graph.return_value = compiled
    mock_run.return_value = {"pr_url": ""}

    runner.invoke(app, ["build", str(repo), "Hello world"])

    state = compiled.ainvoke.call_args[0][0]
    assert state["feature_branch"] == "feature/feat_hello_world"


# ---------------------------------------------------------------------------
# resume command — feature_branch derivation
# ---------------------------------------------------------------------------

@patch("graft.cli.asyncio.run")
@patch("graft.cli.build_graph")
@patch("graft.cli.load_artifact")
@patch("graft.cli.Settings.load")
def test_resume_feature_branch_derived_from_project_id(
    mock_load, mock_artifact, mock_graph, mock_run, tmp_path
):
    """resume derives feature_branch from the project_id in metadata."""
    project_dir = tmp_path / "feat_branch"
    project_dir.mkdir()
    meta = {
        "project_id": "feat_branch",
        "repo_path": "/tmp/repo",
        "feature_prompt": "Branch test",
    }
    (project_dir / "metadata.json").write_text(json.dumps(meta))

    mock_load.return_value = _make_settings()
    mock_artifact.return_value = None
    compiled = _make_compiled()
    mock_graph.return_value = compiled
    mock_run.return_value = {"pr_url": ""}

    runner.invoke(app, ["resume", str(project_dir)])

    state = compiled.ainvoke.call_args[0][0]
    assert state["feature_branch"] == "feature/feat_branch"


# ---------------------------------------------------------------------------
# build command — settings passthrough
# ---------------------------------------------------------------------------

@patch("graft.cli.asyncio.run")
@patch("graft.cli.build_graph")
@patch("graft.cli.create_project")
@patch("graft.cli.Settings.load")
def test_build_settings_model_passthrough(
    mock_load, mock_create, mock_graph, mock_run, tmp_path
):
    """build passes settings.model and settings.max_agent_turns to state."""
    repo = tmp_path / "repo"
    repo.mkdir()

    mock_load.return_value = _make_settings(
        model="claude-sonnet-4-20250514", max_agent_turns=100
    )
    mock_create.return_value = ("feat_m2", tmp_path / "out")
    compiled = _make_compiled()
    mock_graph.return_value = compiled
    mock_run.return_value = {"pr_url": ""}

    runner.invoke(app, ["build", str(repo), "Model test"])

    state = compiled.ainvoke.call_args[0][0]
    assert state["model"] == "claude-sonnet-4-20250514"
    assert state["max_agent_turns"] == 100


# ---------------------------------------------------------------------------
# build command — graph receives UI
# ---------------------------------------------------------------------------

@patch("graft.cli.asyncio.run")
@patch("graft.cli.build_graph")
@patch("graft.cli.create_project")
@patch("graft.cli.Settings.load")
def test_build_passes_ui_to_graph(
    mock_load, mock_create, mock_graph, mock_run, tmp_path
):
    """build passes a UI instance to build_graph."""
    repo = tmp_path / "repo"
    repo.mkdir()

    mock_load.return_value = _make_settings()
    mock_create.return_value = ("feat_ui", tmp_path / "out")
    mock_graph.return_value = _make_compiled()
    mock_run.return_value = {"pr_url": ""}

    runner.invoke(app, ["build", str(repo), "UI test"])

    args, kwargs = mock_graph.call_args
    # build_graph(ui) — first positional argument should be a UI instance
    from graft.ui import UI

    assert isinstance(args[0], UI)


# ---------------------------------------------------------------------------
# resume — build_plan.json with units key
# ---------------------------------------------------------------------------

@patch("graft.cli.asyncio.run")
@patch("graft.cli.build_graph")
@patch("graft.cli.load_artifact")
@patch("graft.cli.Settings.load")
def test_resume_plan_units_extraction(
    mock_load, mock_artifact, mock_graph, mock_run, tmp_path
):
    """resume extracts 'units' key from build_plan.json."""
    project_dir = tmp_path / "feat_plan"
    project_dir.mkdir()
    meta = {
        "project_id": "feat_plan",
        "repo_path": "/tmp/repo",
        "feature_prompt": "Plan parsing",
    }
    (project_dir / "metadata.json").write_text(json.dumps(meta))

    plan_data = {"units": [{"title": "Unit A"}, {"title": "Unit B"}], "version": 1}

    def fake_load(proj_dir, name):
        if name == "build_plan.json":
            return json.dumps(plan_data)
        return None

    mock_load.return_value = _make_settings()
    mock_artifact.side_effect = fake_load
    compiled = _make_compiled()
    mock_graph.return_value = compiled
    mock_run.return_value = {"pr_url": ""}

    runner.invoke(app, ["resume", str(project_dir)])

    state = compiled.ainvoke.call_args[0][0]
    assert state["build_plan"] == [{"title": "Unit A"}, {"title": "Unit B"}]


@patch("graft.cli.asyncio.run")
@patch("graft.cli.build_graph")
@patch("graft.cli.load_artifact")
@patch("graft.cli.Settings.load")
def test_resume_plan_no_units_key(
    mock_load, mock_artifact, mock_graph, mock_run, tmp_path
):
    """resume handles build_plan.json without 'units' key gracefully."""
    project_dir = tmp_path / "feat_nounit"
    project_dir.mkdir()
    meta = {
        "project_id": "feat_nounit",
        "repo_path": "/tmp/repo",
        "feature_prompt": "No units",
    }
    (project_dir / "metadata.json").write_text(json.dumps(meta))

    def fake_load(proj_dir, name):
        if name == "build_plan.json":
            return json.dumps({"some_other_key": "value"})
        return None

    mock_load.return_value = _make_settings()
    mock_artifact.side_effect = fake_load
    compiled = _make_compiled()
    mock_graph.return_value = compiled
    mock_run.return_value = {"pr_url": ""}

    runner.invoke(app, ["resume", str(project_dir)])

    state = compiled.ainvoke.call_args[0][0]
    assert state["build_plan"] == []
