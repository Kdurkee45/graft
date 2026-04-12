"""Tests for graft.stages.verify."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from graft.agent import AgentResult
from graft.stages.verify import _open_pr, verify_node


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path):
    """Create a minimal repo directory for tests."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    return repo_path


@pytest.fixture
def project(tmp_path):
    """Create a minimal project directory with artifacts/ and logs/ subdirs."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "artifacts").mkdir()
    (project_dir / "logs").mkdir()
    # Create a metadata.json so mark_stage_complete / mark_project_done work
    import json

    meta = {
        "project_id": "feat_test",
        "repo_path": str(tmp_path / "repo"),
        "feature_prompt": "test feature",
        "status": "in_progress",
        "stages_completed": [],
        "created_at": "2025-01-01T00:00:00",
    }
    (project_dir / "metadata.json").write_text(json.dumps(meta))
    return project_dir


@pytest.fixture
def ui():
    """Return a mock UI with all methods used by verify_node."""
    mock = MagicMock()
    mock.stage_start = MagicMock()
    mock.stage_done = MagicMock()
    mock.pr_opened = MagicMock()
    mock.info = MagicMock()
    return mock


@pytest.fixture
def base_state(repo, project):
    """Minimal FeatureState dict for verify_node."""
    return {
        "repo_path": str(repo),
        "project_dir": str(project),
        "feature_prompt": "Add dark mode",
        "codebase_profile": {"lang": "python"},
        "feature_spec": {"feature_name": "Dark Mode"},
        "build_plan": [{"unit_id": "u1"}],
        "units_completed": [{"unit_id": "u1"}],
        "units_reverted": [],
        "units_skipped": [],
        "feature_branch": "feat/dark-mode",
        "model": "sonnet",
    }


# ---------------------------------------------------------------------------
# _open_pr tests
# ---------------------------------------------------------------------------


class TestOpenPr:
    """Tests for the _open_pr helper function."""

    def test_push_and_create_pr_returns_url(self):
        """_open_pr pushes the branch and creates a PR, returning the URL."""
        mock_push = MagicMock(returncode=0)
        mock_pr = MagicMock(
            returncode=0,
            stdout="https://github.com/org/repo/pull/42\n",
        )

        with patch("graft.stages.verify.subprocess.run") as mock_run:
            mock_run.side_effect = [mock_push, mock_pr]
            url = _open_pr("/tmp/repo", "feat/x", "Feature X", "body text")

        assert url == "https://github.com/org/repo/pull/42"

        # Verify git push was called correctly
        push_call = mock_run.call_args_list[0]
        assert push_call[0][0] == ["git", "push", "-u", "origin", "feat/x"]
        assert push_call[1]["cwd"] == "/tmp/repo"
        assert push_call[1]["check"] is True
        assert push_call[1]["timeout"] == 120

        # Verify gh pr create was called correctly
        pr_call = mock_run.call_args_list[1]
        assert pr_call[0][0] == [
            "gh",
            "pr",
            "create",
            "--title",
            "Feature X",
            "--body",
            "body text",
            "--head",
            "feat/x",
        ]
        assert pr_call[1]["cwd"] == "/tmp/repo"
        assert pr_call[1]["timeout"] == 60

    def test_gh_cli_failure_returns_none(self):
        """_open_pr returns None when gh pr create exits non-zero."""
        mock_push = MagicMock(returncode=0)
        mock_pr = MagicMock(returncode=1, stdout="", stderr="error")

        with patch("graft.stages.verify.subprocess.run") as mock_run:
            mock_run.side_effect = [mock_push, mock_pr]
            url = _open_pr("/tmp/repo", "feat/x", "Title", "Body")

        assert url is None

    def test_file_not_found_returns_none(self):
        """_open_pr returns None when gh CLI is not installed."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("gh not found")
            url = _open_pr("/tmp/repo", "feat/x", "Title", "Body")

        assert url is None

    def test_timeout_expired_returns_none(self):
        """_open_pr returns None when a subprocess times out."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd="git push", timeout=120
            )
            url = _open_pr("/tmp/repo", "feat/x", "Title", "Body")

        assert url is None

    def test_called_process_error_returns_none(self):
        """_open_pr returns None when git push fails with CalledProcessError."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=128, cmd=["git", "push"]
            )
            url = _open_pr("/tmp/repo", "feat/x", "Title", "Body")

        assert url is None

    def test_strips_trailing_whitespace_from_url(self):
        """PR URL is stripped of whitespace/newlines."""
        mock_push = MagicMock(returncode=0)
        mock_pr = MagicMock(
            returncode=0,
            stdout="  https://github.com/org/repo/pull/99  \n",
        )

        with patch("graft.stages.verify.subprocess.run") as mock_run:
            mock_run.side_effect = [mock_push, mock_pr]
            url = _open_pr("/tmp/repo", "feat/x", "Title", "Body")

        assert url == "https://github.com/org/repo/pull/99"


# ---------------------------------------------------------------------------
# verify_node tests
# ---------------------------------------------------------------------------


class TestVerifyNode:
    """Tests for the verify_node async function."""

    @pytest.fixture
    def agent_result(self):
        """Default AgentResult returned by the mocked run_agent."""
        return AgentResult(
            text="Agent fallback text",
            tool_calls=[],
            raw_messages=[],
            elapsed_seconds=10.0,
            turns_used=5,
        )

    async def test_reads_feature_report_and_saves_artifact(
        self, base_state, ui, repo, project, agent_result
    ):
        """verify_node reads feature_report.md written by agent and saves it."""
        report_content = "# Feature Report\nAll tests pass."
        (repo / "feature_report.md").write_text(report_content)

        with (
            patch(
                "graft.stages.verify.run_agent",
                new_callable=AsyncMock,
                return_value=agent_result,
            ),
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify.mark_stage_complete") as mock_mark_stage,
            patch("graft.stages.verify.mark_project_done") as mock_mark_done,
        ):
            result = await verify_node(base_state, ui)

        assert result["feature_report"] == report_content
        # Artifact should be saved
        artifact_path = project / "artifacts" / "feature_report.md"
        assert artifact_path.exists()
        assert artifact_path.read_text() == report_content

    async def test_falls_back_to_result_text_when_no_file(
        self, base_state, ui, repo, project, agent_result
    ):
        """verify_node falls back to result.text when feature_report.md not written."""
        # Do NOT create feature_report.md in repo
        assert not (repo / "feature_report.md").exists()

        with (
            patch(
                "graft.stages.verify.run_agent",
                new_callable=AsyncMock,
                return_value=agent_result,
            ),
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify.mark_stage_complete"),
            patch("graft.stages.verify.mark_project_done"),
        ):
            result = await verify_node(base_state, ui)

        assert result["feature_report"] == "Agent fallback text"

    async def test_cleans_up_feature_report_from_repo(
        self, base_state, ui, repo, agent_result
    ):
        """verify_node removes feature_report.md from the repo after reading it."""
        report_path = repo / "feature_report.md"
        report_path.write_text("# Report")

        with (
            patch(
                "graft.stages.verify.run_agent",
                new_callable=AsyncMock,
                return_value=agent_result,
            ),
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify.mark_stage_complete"),
            patch("graft.stages.verify.mark_project_done"),
        ):
            await verify_node(base_state, ui)

        assert not report_path.exists()

    async def test_opens_pr_when_branch_exists(
        self, base_state, ui, repo, agent_result
    ):
        """verify_node calls _open_pr and shows pr_opened on success."""
        pr_url = "https://github.com/org/repo/pull/1"

        with (
            patch(
                "graft.stages.verify.run_agent",
                new_callable=AsyncMock,
                return_value=agent_result,
            ),
            patch("graft.stages.verify._open_pr", return_value=pr_url) as mock_open_pr,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify.mark_stage_complete"),
            patch("graft.stages.verify.mark_project_done") as mock_mark_done,
        ):
            result = await verify_node(base_state, ui)

        assert result["pr_url"] == pr_url
        ui.pr_opened.assert_called_once_with(pr_url)
        mock_open_pr.assert_called_once_with(
            str(repo),
            "feat/dark-mode",
            "Feature: Dark Mode",
            agent_result.text,  # No file written, so fallback text is the body
        )
        mock_mark_done.assert_called_once_with(str(base_state["project_dir"]), pr_url)

    async def test_handles_missing_branch_gracefully(
        self, base_state, ui, repo, agent_result
    ):
        """verify_node skips PR opening when no feature_branch is set."""
        base_state["feature_branch"] = ""

        with (
            patch(
                "graft.stages.verify.run_agent",
                new_callable=AsyncMock,
                return_value=agent_result,
            ),
            patch("graft.stages.verify._open_pr") as mock_open_pr,
            patch("graft.stages.verify.subprocess.run") as mock_subproc,
            patch("graft.stages.verify.mark_stage_complete"),
            patch("graft.stages.verify.mark_project_done") as mock_mark_done,
        ):
            result = await verify_node(base_state, ui)

        assert result["pr_url"] == ""
        mock_open_pr.assert_not_called()
        # Should not attempt git add/commit when no branch
        mock_subproc.assert_not_called()
        ui.pr_opened.assert_not_called()
        mock_mark_done.assert_not_called()

    async def test_pr_failure_shows_manual_instructions(
        self, base_state, ui, repo, agent_result
    ):
        """verify_node shows manual PR instructions when _open_pr fails."""
        with (
            patch(
                "graft.stages.verify.run_agent",
                new_callable=AsyncMock,
                return_value=agent_result,
            ),
            patch("graft.stages.verify._open_pr", return_value=None),
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify.mark_stage_complete"),
            patch("graft.stages.verify.mark_project_done") as mock_mark_done,
        ):
            result = await verify_node(base_state, ui)

        assert result["pr_url"] == ""
        ui.pr_opened.assert_not_called()
        # Should have called ui.info twice: once for manual push hint, once for report location
        assert ui.info.call_count == 2
        manual_msg = ui.info.call_args_list[0][0][0]
        assert "feat/dark-mode" in manual_msg
        assert "manually" in manual_msg.lower()
        mock_mark_done.assert_not_called()

    async def test_calls_mark_stage_complete(self, base_state, ui, repo, agent_result):
        """verify_node always calls mark_stage_complete for 'verify'."""
        with (
            patch(
                "graft.stages.verify.run_agent",
                new_callable=AsyncMock,
                return_value=agent_result,
            ),
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify.mark_stage_complete") as mock_mark_stage,
            patch("graft.stages.verify.mark_project_done"),
        ):
            await verify_node(base_state, ui)

        mock_mark_stage.assert_called_once_with(
            str(base_state["project_dir"]), "verify"
        )

    async def test_mark_project_done_on_pr_success(
        self, base_state, ui, repo, agent_result
    ):
        """verify_node calls mark_project_done only when PR is successfully opened."""
        pr_url = "https://github.com/org/repo/pull/7"

        with (
            patch(
                "graft.stages.verify.run_agent",
                new_callable=AsyncMock,
                return_value=agent_result,
            ),
            patch("graft.stages.verify._open_pr", return_value=pr_url),
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify.mark_stage_complete"),
            patch("graft.stages.verify.mark_project_done") as mock_mark_done,
        ):
            result = await verify_node(base_state, ui)

        mock_mark_done.assert_called_once_with(str(base_state["project_dir"]), pr_url)

    async def test_does_not_mark_project_done_on_pr_failure(
        self, base_state, ui, repo, agent_result
    ):
        """mark_project_done is NOT called when PR opening fails."""
        with (
            patch(
                "graft.stages.verify.run_agent",
                new_callable=AsyncMock,
                return_value=agent_result,
            ),
            patch("graft.stages.verify._open_pr", return_value=None),
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify.mark_stage_complete"),
            patch("graft.stages.verify.mark_project_done") as mock_mark_done,
        ):
            await verify_node(base_state, ui)

        mock_mark_done.assert_not_called()

    async def test_invokes_agent_with_correct_params(
        self, base_state, ui, repo, agent_result
    ):
        """verify_node passes correct args to run_agent."""
        with (
            patch(
                "graft.stages.verify.run_agent",
                new_callable=AsyncMock,
                return_value=agent_result,
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify.mark_stage_complete"),
            patch("graft.stages.verify.mark_project_done"),
        ):
            await verify_node(base_state, ui)

        mock_agent.assert_called_once()
        kwargs = mock_agent.call_args[1]
        assert kwargs["persona"] == "Principal Quality Engineer"
        assert kwargs["cwd"] == str(repo)
        assert kwargs["project_dir"] == str(base_state["project_dir"])
        assert kwargs["stage"] == "verify"
        assert kwargs["ui"] is ui
        assert kwargs["model"] == "sonnet"
        assert kwargs["max_turns"] == 30
        assert kwargs["allowed_tools"] == ["Bash", "Read", "Glob", "Grep"]
        assert "Add dark mode" in kwargs["user_prompt"]

    async def test_return_dict_shape(self, base_state, ui, repo, agent_result):
        """verify_node returns dict with expected keys."""
        with (
            patch(
                "graft.stages.verify.run_agent",
                new_callable=AsyncMock,
                return_value=agent_result,
            ),
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify.mark_stage_complete"),
            patch("graft.stages.verify.mark_project_done"),
        ):
            result = await verify_node(base_state, ui)

        assert set(result.keys()) == {"feature_report", "pr_url", "current_stage"}
        assert result["current_stage"] == "verify"

    async def test_ui_lifecycle(self, base_state, ui, repo, agent_result):
        """verify_node calls stage_start and stage_done on the UI."""
        with (
            patch(
                "graft.stages.verify.run_agent",
                new_callable=AsyncMock,
                return_value=agent_result,
            ),
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify.mark_stage_complete"),
            patch("graft.stages.verify.mark_project_done"),
        ):
            await verify_node(base_state, ui)

        ui.stage_start.assert_called_once_with("verify")
        ui.stage_done.assert_called_once_with("verify")

    async def test_git_add_and_commit_before_pr(
        self, base_state, ui, repo, agent_result
    ):
        """verify_node runs git add -A and git commit before opening PR."""
        with (
            patch(
                "graft.stages.verify.run_agent",
                new_callable=AsyncMock,
                return_value=agent_result,
            ),
            patch("graft.stages.verify._open_pr", return_value=None),
            patch("graft.stages.verify.subprocess.run") as mock_subproc,
            patch("graft.stages.verify.mark_stage_complete"),
            patch("graft.stages.verify.mark_project_done"),
        ):
            await verify_node(base_state, ui)

        # Should have been called twice: git add -A, then git commit
        assert mock_subproc.call_count == 2
        add_call = mock_subproc.call_args_list[0]
        assert add_call[0][0] == ["git", "add", "-A"]
        assert add_call[1]["cwd"] == str(repo)

        commit_call = mock_subproc.call_args_list[1]
        assert commit_call[0][0][0] == "git"
        assert commit_call[0][0][1] == "commit"
        assert "--allow-empty" in commit_call[0][0]

    async def test_defaults_for_missing_state_keys(self, ui, tmp_path):
        """verify_node handles state with minimal keys (missing optional fields)."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        (project_dir / "artifacts").mkdir()
        (project_dir / "logs").mkdir()

        minimal_state = {
            "repo_path": str(repo_path),
            "project_dir": str(project_dir),
        }

        agent_result = AgentResult(text="minimal report")

        with (
            patch(
                "graft.stages.verify.run_agent",
                new_callable=AsyncMock,
                return_value=agent_result,
            ),
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify.mark_stage_complete"),
            patch("graft.stages.verify.mark_project_done"),
        ):
            result = await verify_node(minimal_state, ui)

        # Should not crash; branch is empty so no PR attempt
        assert result["pr_url"] == ""
        assert result["feature_report"] == "minimal report"
        assert result["current_stage"] == "verify"

    async def test_pr_title_uses_feature_name_from_spec(
        self, base_state, ui, repo, agent_result
    ):
        """PR title is constructed from feature_spec.feature_name."""
        base_state["feature_spec"] = {"feature_name": "My Cool Feature"}

        with (
            patch(
                "graft.stages.verify.run_agent",
                new_callable=AsyncMock,
                return_value=agent_result,
            ),
            patch(
                "graft.stages.verify._open_pr", return_value="https://example.com/pr/1"
            ) as mock_open_pr,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify.mark_stage_complete"),
            patch("graft.stages.verify.mark_project_done"),
        ):
            await verify_node(base_state, ui)

        pr_title = mock_open_pr.call_args[0][2]
        assert pr_title == "Feature: My Cool Feature"

    async def test_pr_title_defaults_when_no_feature_name(
        self, base_state, ui, repo, agent_result
    ):
        """PR title falls back to 'Feature' when feature_name is missing from spec."""
        base_state["feature_spec"] = {}

        with (
            patch(
                "graft.stages.verify.run_agent",
                new_callable=AsyncMock,
                return_value=agent_result,
            ),
            patch(
                "graft.stages.verify._open_pr", return_value="https://example.com/pr/1"
            ) as mock_open_pr,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify.mark_stage_complete"),
            patch("graft.stages.verify.mark_project_done"),
        ):
            await verify_node(base_state, ui)

        pr_title = mock_open_pr.call_args[0][2]
        assert pr_title == "Feature: Feature"

    async def test_save_artifact_called_with_report(
        self, base_state, ui, repo, agent_result
    ):
        """verify_node calls save_artifact with the feature report content."""
        report_text = "# Full Report\nEverything works!"
        (repo / "feature_report.md").write_text(report_text)

        with (
            patch(
                "graft.stages.verify.run_agent",
                new_callable=AsyncMock,
                return_value=agent_result,
            ),
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify.save_artifact") as mock_save,
            patch("graft.stages.verify.mark_stage_complete"),
            patch("graft.stages.verify.mark_project_done"),
        ):
            await verify_node(base_state, ui)

        mock_save.assert_called_once_with(
            str(base_state["project_dir"]),
            "feature_report.md",
            report_text,
        )
