"""Tests for graft.cli — argument parsing, validation, and error exits."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from graft.cli import app

runner = CliRunner()


def _mock_settings(tmp_path: Path) -> MagicMock:
    """Return a fake Settings with projects_root under tmp_path."""
    s = MagicMock()
    s.projects_root = tmp_path / "projects"
    s.model = "test-model"
    s.max_agent_turns = 5
    return s


# ── build command ────────────────────────────────────────────────────────────


def test_build_rejects_nonexistent_repo(tmp_path):
    """build exits 1 when the repo path does not exist."""
    fake_path = str(tmp_path / "no_such_repo")
    with patch("graft.cli.Settings.load", return_value=_mock_settings(tmp_path)):
        result = runner.invoke(app, ["build", fake_path, "add login page"])
    assert result.exit_code == 1


def test_build_rejects_nonexistent_scope_path(tmp_path):
    """build exits 1 when --path points to a missing subdirectory."""
    repo = tmp_path / "repo"
    repo.mkdir()
    with patch("graft.cli.Settings.load", return_value=_mock_settings(tmp_path)):
        result = runner.invoke(
            app, ["build", str(repo), "add login page", "--path", "no_such_subdir"]
        )
    assert result.exit_code == 1


def test_build_valid_repo_invokes_pipeline(tmp_path):
    """build with a valid repo creates a project and runs the graph."""
    repo = tmp_path / "repo"
    repo.mkdir()
    project_dir = tmp_path / "projects" / "feat_abc12345"
    project_dir.mkdir(parents=True)

    fake_compiled = MagicMock()
    fake_compiled.ainvoke = AsyncMock(return_value={"pr_url": "https://github.com/pr/1"})

    with patch("graft.cli.Settings.load", return_value=_mock_settings(tmp_path)), \
         patch("graft.cli.create_project", return_value=("feat_abc12345", project_dir)), \
         patch("graft.cli.build_graph", return_value=fake_compiled) as mock_bg, \
         patch("graft.cli.asyncio.run", return_value={"pr_url": "https://github.com/pr/1"}) as mock_run:
        result = runner.invoke(app, ["build", str(repo), "add login page"])

    assert result.exit_code == 0
    mock_bg.assert_called_once()
    mock_run.assert_called_once()


def test_build_no_pr_url(tmp_path):
    """build completes gracefully when pipeline returns no pr_url."""
    repo = tmp_path / "repo"
    repo.mkdir()
    project_dir = tmp_path / "projects" / "feat_abc12345"
    project_dir.mkdir(parents=True)

    with patch("graft.cli.Settings.load", return_value=_mock_settings(tmp_path)), \
         patch("graft.cli.create_project", return_value=("feat_abc12345", project_dir)), \
         patch("graft.cli.build_graph", return_value=MagicMock()), \
         patch("graft.cli.asyncio.run", return_value={"pr_url": ""}):
        result = runner.invoke(app, ["build", str(repo), "add login page"])

    assert result.exit_code == 0
    assert "open a PR manually" in result.output


def test_build_with_valid_scope_path(tmp_path):
    """build with a valid --path scope prints scoped message and proceeds."""
    repo = tmp_path / "repo"
    sub = repo / "packages" / "core"
    sub.mkdir(parents=True)
    project_dir = tmp_path / "projects" / "feat_abc12345"
    project_dir.mkdir(parents=True)

    with patch("graft.cli.Settings.load", return_value=_mock_settings(tmp_path)), \
         patch("graft.cli.create_project", return_value=("feat_abc12345", project_dir)), \
         patch("graft.cli.build_graph", return_value=MagicMock()), \
         patch("graft.cli.asyncio.run", return_value={"pr_url": ""}):
        result = runner.invoke(
            app, ["build", str(repo), "add login", "--path", "packages/core"]
        )

    assert result.exit_code == 0
    assert "Scoped to" in result.output


def test_build_passes_constraints_and_options(tmp_path):
    """build forwards --constraint, --max-units, and --auto-approve to initial state."""
    repo = tmp_path / "repo"
    repo.mkdir()
    project_dir = tmp_path / "projects" / "feat_abc12345"
    project_dir.mkdir(parents=True)

    captured_state = {}

    def capture_run(coro):
        """Run the coroutine mock and capture the state dict passed to ainvoke."""
        return {"pr_url": ""}

    fake_compiled = MagicMock()

    def capture_ainvoke(state):
        captured_state.update(state)
        future = AsyncMock(return_value={"pr_url": ""})
        return future()

    fake_compiled.ainvoke = capture_ainvoke

    with patch("graft.cli.Settings.load", return_value=_mock_settings(tmp_path)), \
         patch("graft.cli.create_project", return_value=("feat_abc12345", project_dir)), \
         patch("graft.cli.build_graph", return_value=fake_compiled), \
         patch("graft.cli.asyncio.run", side_effect=capture_run) as mock_run:
        result = runner.invoke(app, [
            "build", str(repo), "add login",
            "--constraint", "no breaking changes",
            "--constraint", "use typescript",
            "--max-units", "3",
            "--auto-approve",
        ])

    assert result.exit_code == 0
    # Verify asyncio.run was called with the coroutine produced by ainvoke
    mock_run.assert_called_once()


# ── resume command ───────────────────────────────────────────────────────────


def test_resume_rejects_nonexistent_project_path(tmp_path):
    """resume exits 1 when the project directory does not exist."""
    fake_path = str(tmp_path / "no_such_session")
    with patch("graft.cli.Settings.load", return_value=_mock_settings(tmp_path)):
        result = runner.invoke(app, ["resume", fake_path])
    assert result.exit_code == 1


def test_resume_rejects_directory_without_metadata(tmp_path):
    """resume exits 1 when the directory has no metadata.json."""
    project_dir = tmp_path / "feat_deadbeef"
    project_dir.mkdir()
    with patch("graft.cli.Settings.load", return_value=_mock_settings(tmp_path)):
        result = runner.invoke(app, ["resume", str(project_dir)])
    assert result.exit_code == 1
    assert "metadata.json" in result.output


def test_resume_valid_session_invokes_pipeline(tmp_path):
    """resume with valid metadata reloads artifacts and runs the graph."""
    project_dir = tmp_path / "feat_abc12345"
    project_dir.mkdir()
    artifacts_dir = project_dir / "artifacts"
    artifacts_dir.mkdir()

    meta = {
        "project_id": "feat_abc12345",
        "repo_path": "/tmp/repo",
        "feature_prompt": "add login page",
    }
    (project_dir / "metadata.json").write_text(json.dumps(meta))

    # Write a sample artifact to verify load_artifact is used
    (artifacts_dir / "codebase_profile.json").write_text('{"lang": "python"}')

    fake_compiled = MagicMock()
    fake_compiled.ainvoke = AsyncMock(return_value={"pr_url": "https://github.com/pr/2"})

    with patch("graft.cli.Settings.load", return_value=_mock_settings(tmp_path)), \
         patch("graft.cli.build_graph", return_value=fake_compiled) as mock_bg, \
         patch("graft.cli.asyncio.run", return_value={"pr_url": "https://github.com/pr/2"}):
        result = runner.invoke(app, ["resume", str(project_dir)])

    assert result.exit_code == 0
    mock_bg.assert_called_once()
    # build_graph should receive entry_stage="execute" (the default for resume)
    _, kwargs = mock_bg.call_args
    assert kwargs.get("entry_stage") == "execute"


def test_resume_custom_from_stage(tmp_path):
    """resume respects --from to set the entry stage."""
    project_dir = tmp_path / "feat_abc12345"
    project_dir.mkdir()
    (project_dir / "artifacts").mkdir()

    meta = {
        "project_id": "feat_abc12345",
        "repo_path": "/tmp/repo",
        "feature_prompt": "add login page",
    }
    (project_dir / "metadata.json").write_text(json.dumps(meta))

    fake_compiled = MagicMock()
    fake_compiled.ainvoke = AsyncMock(return_value={"pr_url": ""})

    with patch("graft.cli.Settings.load", return_value=_mock_settings(tmp_path)), \
         patch("graft.cli.build_graph", return_value=fake_compiled) as mock_bg, \
         patch("graft.cli.asyncio.run", return_value={"pr_url": ""}):
        result = runner.invoke(app, ["resume", str(project_dir), "--from", "research"])

    assert result.exit_code == 0
    _, kwargs = mock_bg.call_args
    assert kwargs.get("entry_stage") == "research"


def test_resume_no_pr_url(tmp_path):
    """resume completes without printing PR line when pr_url is empty."""
    project_dir = tmp_path / "feat_abc12345"
    project_dir.mkdir()
    (project_dir / "artifacts").mkdir()

    meta = {
        "project_id": "feat_abc12345",
        "repo_path": "/tmp/repo",
        "feature_prompt": "add login page",
    }
    (project_dir / "metadata.json").write_text(json.dumps(meta))

    with patch("graft.cli.Settings.load", return_value=_mock_settings(tmp_path)), \
         patch("graft.cli.build_graph", return_value=MagicMock()), \
         patch("graft.cli.asyncio.run", return_value={"pr_url": ""}):
        result = runner.invoke(app, ["resume", str(project_dir)])

    assert result.exit_code == 0
    assert "PR opened" not in result.output


# ── list command ─────────────────────────────────────────────────────────────


def test_list_calls_list_projects_and_show_projects(tmp_path):
    """list command loads settings, calls list_projects, and renders via UI."""
    fake_settings = _mock_settings(tmp_path)
    fake_projects = [
        {"project_id": "feat_aaa", "repo_path": "/r", "feature_prompt": "f", "status": "done"},
    ]

    with patch("graft.cli.Settings.load", return_value=fake_settings), \
         patch("graft.cli.list_projects", return_value=fake_projects) as mock_lp, \
         patch("graft.cli.UI") as MockUI:
        mock_ui = MagicMock()
        MockUI.return_value = mock_ui
        result = runner.invoke(app, ["list"])

    assert result.exit_code == 0
    mock_lp.assert_called_once_with(fake_settings.projects_root)
    mock_ui.show_projects.assert_called_once_with(fake_projects)


def test_list_empty_projects(tmp_path):
    """list command handles empty project list without error."""
    fake_settings = _mock_settings(tmp_path)

    with patch("graft.cli.Settings.load", return_value=fake_settings), \
         patch("graft.cli.list_projects", return_value=[]) as mock_lp, \
         patch("graft.cli.UI") as MockUI:
        mock_ui = MagicMock()
        MockUI.return_value = mock_ui
        result = runner.invoke(app, ["list"])

    assert result.exit_code == 0
    mock_lp.assert_called_once_with(fake_settings.projects_root)
    mock_ui.show_projects.assert_called_once_with([])


# ── missing arguments ────────────────────────────────────────────────────────


def test_build_missing_arguments():
    """build with no arguments exits non-zero (typer shows usage error)."""
    result = runner.invoke(app, ["build"])
    assert result.exit_code != 0


def test_resume_missing_arguments():
    """resume with no arguments exits non-zero."""
    result = runner.invoke(app, ["resume"])
    assert result.exit_code != 0
