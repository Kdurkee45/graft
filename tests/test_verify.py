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
def repo_dir(tmp_path: Path) -> Path:
    """A fake repository directory."""
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """A fake project directory with artifacts/ and metadata.json."""
    proj = tmp_path / "project"
    (proj / "artifacts").mkdir(parents=True)
    (proj / "metadata.json").write_text(
        '{"project_id":"test","stages_completed":[],"status":"in_progress"}'
    )
    return proj


@pytest.fixture
def ui() -> MagicMock:
    """A mocked UI instance."""
    mock_ui = MagicMock()
    mock_ui.stage_start = MagicMock()
    mock_ui.stage_done = MagicMock()
    mock_ui.pr_opened = MagicMock()
    mock_ui.info = MagicMock()
    return mock_ui


@pytest.fixture
def base_state(repo_dir: Path, project_dir: Path) -> dict:
    """Minimal valid FeatureState dict for verify_node."""
    return {
        "repo_path": str(repo_dir),
        "project_dir": str(project_dir),
        "feature_prompt": "Add dark mode support",
        "codebase_profile": {"language": "python"},
        "feature_spec": {"feature_name": "Dark Mode"},
        "build_plan": [{"unit_id": "u1", "title": "Add theme toggle"}],
        "units_completed": [{"unit_id": "u1"}],
        "units_reverted": [],
        "units_skipped": [],
        "feature_branch": "feat/dark-mode",
        "model": "sonnet",
    }


@pytest.fixture
def agent_result() -> AgentResult:
    """A fake AgentResult returned by run_agent."""
    return AgentResult(
        text="Agent verification output",
        tool_calls=[],
        raw_messages=[],
        elapsed_seconds=12.5,
        turns_used=5,
    )


# ---------------------------------------------------------------------------
# _open_pr tests
# ---------------------------------------------------------------------------


class TestOpenPr:
    """Tests for the _open_pr() helper function."""

    @patch("graft.stages.verify.subprocess.run")
    def test_successful_pr_creation(self, mock_run: MagicMock) -> None:
        """Returns the PR URL when git push and gh pr create both succeed."""
        push_result = MagicMock(returncode=0)
        pr_result = MagicMock(returncode=0, stdout="https://github.com/org/repo/pull/42\n")
        mock_run.side_effect = [push_result, pr_result]

        url = _open_pr("/repo", "feat/x", "Feature: X", "body text")

        assert url == "https://github.com/org/repo/pull/42"

    @patch("graft.stages.verify.subprocess.run")
    def test_git_push_called_with_correct_args(self, mock_run: MagicMock) -> None:
        """Verifies git push -u origin <branch> is called."""
        push_result = MagicMock(returncode=0)
        pr_result = MagicMock(returncode=0, stdout="https://url\n")
        mock_run.side_effect = [push_result, pr_result]

        _open_pr("/my/repo", "feat/branch", "Title", "Body")

        push_call = mock_run.call_args_list[0]
        assert push_call == call(
            ["git", "push", "-u", "origin", "feat/branch"],
            cwd="/my/repo",
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )

    @patch("graft.stages.verify.subprocess.run")
    def test_gh_pr_create_called_with_correct_args(self, mock_run: MagicMock) -> None:
        """Verifies gh pr create is called with --title, --body, --head."""
        push_result = MagicMock(returncode=0)
        pr_result = MagicMock(returncode=0, stdout="https://url\n")
        mock_run.side_effect = [push_result, pr_result]

        _open_pr("/repo", "feat/y", "Feature: Y", "report body")

        pr_call = mock_run.call_args_list[1]
        assert pr_call == call(
            [
                "gh",
                "pr",
                "create",
                "--title",
                "Feature: Y",
                "--body",
                "report body",
                "--head",
                "feat/y",
            ],
            cwd="/repo",
            capture_output=True,
            text=True,
            timeout=60,
        )

    @patch("graft.stages.verify.subprocess.run")
    def test_returns_none_when_pr_create_fails(self, mock_run: MagicMock) -> None:
        """Returns None when gh pr create exits non-zero."""
        push_result = MagicMock(returncode=0)
        pr_result = MagicMock(returncode=1, stdout="", stderr="error")
        mock_run.side_effect = [push_result, pr_result]

        url = _open_pr("/repo", "feat/z", "Title", "Body")

        assert url is None

    @patch("graft.stages.verify.subprocess.run")
    def test_returns_none_on_git_push_failure(self, mock_run: MagicMock) -> None:
        """Returns None when git push raises CalledProcessError."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "git push")

        url = _open_pr("/repo", "feat/x", "Title", "Body")

        assert url is None

    @patch("graft.stages.verify.subprocess.run")
    def test_returns_none_on_timeout(self, mock_run: MagicMock) -> None:
        """Returns None when subprocess times out."""
        mock_run.side_effect = subprocess.TimeoutExpired("git", 120)

        url = _open_pr("/repo", "feat/x", "Title", "Body")

        assert url is None

    @patch("graft.stages.verify.subprocess.run")
    def test_returns_none_on_file_not_found(self, mock_run: MagicMock) -> None:
        """Returns None when gh/git binary is not found."""
        mock_run.side_effect = FileNotFoundError("gh not found")

        url = _open_pr("/repo", "feat/x", "Title", "Body")

        assert url is None

    @patch("graft.stages.verify.subprocess.run")
    def test_strips_whitespace_from_pr_url(self, mock_run: MagicMock) -> None:
        """PR URL is stripped of trailing newlines/whitespace."""
        push_result = MagicMock(returncode=0)
        pr_result = MagicMock(
            returncode=0,
            stdout="  https://github.com/org/repo/pull/99  \n",
        )
        mock_run.side_effect = [push_result, pr_result]

        url = _open_pr("/repo", "feat/x", "Title", "Body")

        assert url == "https://github.com/org/repo/pull/99"


# ---------------------------------------------------------------------------
# verify_node tests
# ---------------------------------------------------------------------------


class TestVerifyNode:
    """Tests for the async verify_node() function."""

    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify._open_pr", return_value="https://github.com/org/repo/pull/1")
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_happy_path_returns_correct_state(
        self,
        mock_run_agent: AsyncMock,
        mock_subprocess: MagicMock,
        mock_open_pr: MagicMock,
        mock_save: MagicMock,
        mock_stage_complete: MagicMock,
        mock_project_done: MagicMock,
        base_state: dict,
        repo_dir: Path,
        project_dir: Path,
        agent_result: AgentResult,
        ui: MagicMock,
    ) -> None:
        """Full success path returns dict with feature_report, pr_url, current_stage."""
        mock_run_agent.return_value = agent_result
        # Create the report file that the agent would produce
        (repo_dir / "feature_report.md").write_text("# Feature Report\nAll good.")

        result = await verify_node(base_state, ui)

        assert result["feature_report"] == "# Feature Report\nAll good."
        assert result["pr_url"] == "https://github.com/org/repo/pull/1"
        assert result["current_stage"] == "verify"

    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify._open_pr", return_value="https://github.com/org/repo/pull/1")
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_report_file_read_and_cleaned_up(
        self,
        mock_run_agent: AsyncMock,
        mock_subprocess: MagicMock,
        mock_open_pr: MagicMock,
        mock_save: MagicMock,
        mock_stage_complete: MagicMock,
        mock_project_done: MagicMock,
        base_state: dict,
        repo_dir: Path,
        project_dir: Path,
        agent_result: AgentResult,
        ui: MagicMock,
    ) -> None:
        """feature_report.md is read from disk and then deleted."""
        mock_run_agent.return_value = agent_result
        report_path = repo_dir / "feature_report.md"
        report_path.write_text("Report contents")

        await verify_node(base_state, ui)

        # Report file should be cleaned up (unlinked)
        assert not report_path.exists()

    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify._open_pr", return_value="https://github.com/org/repo/pull/1")
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_save_artifact_called_with_report(
        self,
        mock_run_agent: AsyncMock,
        mock_subprocess: MagicMock,
        mock_open_pr: MagicMock,
        mock_save: MagicMock,
        mock_stage_complete: MagicMock,
        mock_project_done: MagicMock,
        base_state: dict,
        repo_dir: Path,
        project_dir: Path,
        agent_result: AgentResult,
        ui: MagicMock,
    ) -> None:
        """save_artifact is called with feature_report.md content."""
        mock_run_agent.return_value = agent_result
        (repo_dir / "feature_report.md").write_text("My report")

        await verify_node(base_state, ui)

        mock_save.assert_called_once_with(
            str(project_dir), "feature_report.md", "My report"
        )

    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify._open_pr", return_value="https://github.com/org/repo/pull/1")
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_fallback_to_agent_text_when_no_report_file(
        self,
        mock_run_agent: AsyncMock,
        mock_subprocess: MagicMock,
        mock_open_pr: MagicMock,
        mock_save: MagicMock,
        mock_stage_complete: MagicMock,
        mock_project_done: MagicMock,
        base_state: dict,
        repo_dir: Path,
        project_dir: Path,
        agent_result: AgentResult,
        ui: MagicMock,
    ) -> None:
        """Falls back to AgentResult.text when feature_report.md not on disk."""
        mock_run_agent.return_value = agent_result
        # Don't create feature_report.md on disk

        result = await verify_node(base_state, ui)

        assert result["feature_report"] == "Agent verification output"
        mock_save.assert_called_once_with(
            str(project_dir), "feature_report.md", "Agent verification output"
        )

    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify._open_pr", return_value="https://github.com/org/repo/pull/5")
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_git_add_and_commit_called_when_branch_set(
        self,
        mock_run_agent: AsyncMock,
        mock_subprocess: MagicMock,
        mock_open_pr: MagicMock,
        mock_save: MagicMock,
        mock_stage_complete: MagicMock,
        mock_project_done: MagicMock,
        base_state: dict,
        repo_dir: Path,
        project_dir: Path,
        agent_result: AgentResult,
        ui: MagicMock,
    ) -> None:
        """git add -A and git commit --allow-empty are called when branch is set."""
        mock_run_agent.return_value = agent_result

        await verify_node(base_state, ui)

        # There should be two subprocess.run calls: git add and git commit
        assert mock_subprocess.call_count == 2

        add_call = mock_subprocess.call_args_list[0]
        assert add_call == call(
            ["git", "add", "-A"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
        )

        commit_call = mock_subprocess.call_args_list[1]
        assert commit_call == call(
            [
                "git",
                "commit",
                "-m",
                "chore: cleanup verification artifacts",
                "--allow-empty",
            ],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
        )

    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify._open_pr")
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_no_pr_opened_when_branch_empty(
        self,
        mock_run_agent: AsyncMock,
        mock_subprocess: MagicMock,
        mock_open_pr: MagicMock,
        mock_save: MagicMock,
        mock_stage_complete: MagicMock,
        mock_project_done: MagicMock,
        base_state: dict,
        repo_dir: Path,
        project_dir: Path,
        agent_result: AgentResult,
        ui: MagicMock,
    ) -> None:
        """When feature_branch is empty, no PR is opened and no git commands run."""
        mock_run_agent.return_value = agent_result
        base_state["feature_branch"] = ""

        result = await verify_node(base_state, ui)

        mock_open_pr.assert_not_called()
        mock_subprocess.assert_not_called()
        assert result["pr_url"] == ""

    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify._open_pr", return_value=None)
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_pr_creation_failure_shows_info(
        self,
        mock_run_agent: AsyncMock,
        mock_subprocess: MagicMock,
        mock_open_pr: MagicMock,
        mock_save: MagicMock,
        mock_stage_complete: MagicMock,
        mock_project_done: MagicMock,
        base_state: dict,
        repo_dir: Path,
        project_dir: Path,
        agent_result: AgentResult,
        ui: MagicMock,
    ) -> None:
        """When _open_pr returns None, ui.info is called with guidance."""
        mock_run_agent.return_value = agent_result

        result = await verify_node(base_state, ui)

        assert result["pr_url"] == ""
        ui.pr_opened.assert_not_called()
        assert ui.info.call_count == 2
        # First info call: push manually message
        first_info = ui.info.call_args_list[0][0][0]
        assert "feat/dark-mode" in first_info
        assert "manually" in first_info.lower() or "Push branch" in first_info
        # Second info call: report saved location
        second_info = ui.info.call_args_list[1][0][0]
        assert "feature_report.md" in second_info

    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify._open_pr", return_value="https://github.com/org/repo/pull/7")
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_pr_opened_calls_ui_pr_opened(
        self,
        mock_run_agent: AsyncMock,
        mock_subprocess: MagicMock,
        mock_open_pr: MagicMock,
        mock_save: MagicMock,
        mock_stage_complete: MagicMock,
        mock_project_done: MagicMock,
        base_state: dict,
        repo_dir: Path,
        project_dir: Path,
        agent_result: AgentResult,
        ui: MagicMock,
    ) -> None:
        """When PR is successfully opened, ui.pr_opened is called with the URL."""
        mock_run_agent.return_value = agent_result

        await verify_node(base_state, ui)

        ui.pr_opened.assert_called_once_with("https://github.com/org/repo/pull/7")

    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify._open_pr", return_value="https://github.com/org/repo/pull/1")
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_mark_project_done_called_with_pr_url(
        self,
        mock_run_agent: AsyncMock,
        mock_subprocess: MagicMock,
        mock_open_pr: MagicMock,
        mock_save: MagicMock,
        mock_stage_complete: MagicMock,
        mock_project_done: MagicMock,
        base_state: dict,
        repo_dir: Path,
        project_dir: Path,
        agent_result: AgentResult,
        ui: MagicMock,
    ) -> None:
        """mark_project_done is called with the PR URL when PR succeeds."""
        mock_run_agent.return_value = agent_result

        await verify_node(base_state, ui)

        mock_project_done.assert_called_once_with(
            str(project_dir), "https://github.com/org/repo/pull/1"
        )

    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify._open_pr", return_value=None)
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_mark_project_done_not_called_when_no_pr(
        self,
        mock_run_agent: AsyncMock,
        mock_subprocess: MagicMock,
        mock_open_pr: MagicMock,
        mock_save: MagicMock,
        mock_stage_complete: MagicMock,
        mock_project_done: MagicMock,
        base_state: dict,
        repo_dir: Path,
        project_dir: Path,
        agent_result: AgentResult,
        ui: MagicMock,
    ) -> None:
        """mark_project_done is NOT called when PR creation fails."""
        mock_run_agent.return_value = agent_result

        await verify_node(base_state, ui)

        mock_project_done.assert_not_called()

    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify._open_pr", return_value="https://url")
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_mark_stage_complete_always_called(
        self,
        mock_run_agent: AsyncMock,
        mock_subprocess: MagicMock,
        mock_open_pr: MagicMock,
        mock_save: MagicMock,
        mock_stage_complete: MagicMock,
        mock_project_done: MagicMock,
        base_state: dict,
        repo_dir: Path,
        project_dir: Path,
        agent_result: AgentResult,
        ui: MagicMock,
    ) -> None:
        """mark_stage_complete is always called with 'verify'."""
        mock_run_agent.return_value = agent_result

        await verify_node(base_state, ui)

        mock_stage_complete.assert_called_once_with(str(project_dir), "verify")

    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify._open_pr", return_value="https://url")
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_ui_stage_start_and_done_called(
        self,
        mock_run_agent: AsyncMock,
        mock_subprocess: MagicMock,
        mock_open_pr: MagicMock,
        mock_save: MagicMock,
        mock_stage_complete: MagicMock,
        mock_project_done: MagicMock,
        base_state: dict,
        repo_dir: Path,
        project_dir: Path,
        agent_result: AgentResult,
        ui: MagicMock,
    ) -> None:
        """UI stage lifecycle methods are called."""
        mock_run_agent.return_value = agent_result

        await verify_node(base_state, ui)

        ui.stage_start.assert_called_once_with("verify")
        ui.stage_done.assert_called_once_with("verify")

    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify._open_pr", return_value="https://url")
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_run_agent_called_with_correct_kwargs(
        self,
        mock_run_agent: AsyncMock,
        mock_subprocess: MagicMock,
        mock_open_pr: MagicMock,
        mock_save: MagicMock,
        mock_stage_complete: MagicMock,
        mock_project_done: MagicMock,
        base_state: dict,
        repo_dir: Path,
        project_dir: Path,
        agent_result: AgentResult,
        ui: MagicMock,
    ) -> None:
        """run_agent is called with correct persona, stage, model, and tools."""
        mock_run_agent.return_value = agent_result

        await verify_node(base_state, ui)

        mock_run_agent.assert_called_once()
        kwargs = mock_run_agent.call_args.kwargs
        assert kwargs["persona"] == "Principal Quality Engineer"
        assert kwargs["stage"] == "verify"
        assert kwargs["cwd"] == str(repo_dir)
        assert kwargs["project_dir"] == str(project_dir)
        assert kwargs["model"] == "sonnet"
        assert kwargs["max_turns"] == 30
        assert kwargs["allowed_tools"] == ["Bash", "Read", "Glob", "Grep"]
        assert kwargs["ui"] is ui
        # User prompt should include the feature details
        assert "Add dark mode support" in kwargs["user_prompt"]
        assert "python" in kwargs["user_prompt"]

    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify._open_pr", return_value="https://url")
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_open_pr_called_with_feature_name_from_spec(
        self,
        mock_run_agent: AsyncMock,
        mock_subprocess: MagicMock,
        mock_open_pr: MagicMock,
        mock_save: MagicMock,
        mock_stage_complete: MagicMock,
        mock_project_done: MagicMock,
        base_state: dict,
        repo_dir: Path,
        project_dir: Path,
        agent_result: AgentResult,
        ui: MagicMock,
    ) -> None:
        """_open_pr is called with 'Feature: <name>' as the title."""
        mock_run_agent.return_value = agent_result

        await verify_node(base_state, ui)

        mock_open_pr.assert_called_once()
        args = mock_open_pr.call_args[0]
        assert args[0] == str(repo_dir)        # repo_path
        assert args[1] == "feat/dark-mode"      # branch
        assert args[2] == "Feature: Dark Mode"  # pr_title
        # body is the report content
        assert isinstance(args[3], str)

    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify._open_pr")
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_feature_name_defaults_to_feature_when_missing(
        self,
        mock_run_agent: AsyncMock,
        mock_subprocess: MagicMock,
        mock_open_pr: MagicMock,
        mock_save: MagicMock,
        mock_stage_complete: MagicMock,
        mock_project_done: MagicMock,
        base_state: dict,
        repo_dir: Path,
        project_dir: Path,
        agent_result: AgentResult,
        ui: MagicMock,
    ) -> None:
        """When feature_spec has no feature_name, default 'Feature' is used."""
        mock_run_agent.return_value = agent_result
        mock_open_pr.return_value = "https://url"
        base_state["feature_spec"] = {}  # No feature_name key

        await verify_node(base_state, ui)

        pr_title_arg = mock_open_pr.call_args[0][2]
        assert pr_title_arg == "Feature: Feature"

    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify._open_pr")
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_state_defaults_for_missing_optional_fields(
        self,
        mock_run_agent: AsyncMock,
        mock_subprocess: MagicMock,
        mock_open_pr: MagicMock,
        mock_save: MagicMock,
        mock_stage_complete: MagicMock,
        mock_project_done: MagicMock,
        repo_dir: Path,
        project_dir: Path,
        agent_result: AgentResult,
        ui: MagicMock,
    ) -> None:
        """verify_node handles minimal state with .get() defaults."""
        mock_run_agent.return_value = agent_result
        minimal_state = {
            "repo_path": str(repo_dir),
            "project_dir": str(project_dir),
        }

        result = await verify_node(minimal_state, ui)

        # Should not crash — defaults should kick in
        assert result["current_stage"] == "verify"
        assert result["pr_url"] == ""
        # _open_pr should not be called since feature_branch defaults to ""
        mock_open_pr.assert_not_called()

    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify._open_pr", return_value=None)
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_mark_stage_complete_called_even_when_pr_fails(
        self,
        mock_run_agent: AsyncMock,
        mock_subprocess: MagicMock,
        mock_open_pr: MagicMock,
        mock_save: MagicMock,
        mock_stage_complete: MagicMock,
        mock_project_done: MagicMock,
        base_state: dict,
        repo_dir: Path,
        project_dir: Path,
        agent_result: AgentResult,
        ui: MagicMock,
    ) -> None:
        """Stage is marked complete even when PR fails — verification still happened."""
        mock_run_agent.return_value = agent_result

        await verify_node(base_state, ui)

        mock_stage_complete.assert_called_once_with(str(project_dir), "verify")

    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify._open_pr", return_value="https://url")
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_prompt_includes_build_plan_and_unit_counts(
        self,
        mock_run_agent: AsyncMock,
        mock_subprocess: MagicMock,
        mock_open_pr: MagicMock,
        mock_save: MagicMock,
        mock_stage_complete: MagicMock,
        mock_project_done: MagicMock,
        base_state: dict,
        repo_dir: Path,
        project_dir: Path,
        agent_result: AgentResult,
        ui: MagicMock,
    ) -> None:
        """The prompt sent to run_agent includes unit counts and plan data."""
        mock_run_agent.return_value = agent_result
        base_state["units_reverted"] = [{"unit_id": "u2", "reason": "broken"}]
        base_state["units_skipped"] = [{"unit_id": "u3"}]

        await verify_node(base_state, ui)

        prompt = mock_run_agent.call_args.kwargs["user_prompt"]
        assert "BUILD PLAN (1 units)" in prompt
        assert "UNITS COMPLETED (1)" in prompt
        assert "UNITS REVERTED (1)" in prompt
        assert "UNITS SKIPPED (1)" in prompt
        assert "feature_report.md" in prompt

    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify._open_pr", return_value="https://url")
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_report_not_unlinked_when_not_on_disk(
        self,
        mock_run_agent: AsyncMock,
        mock_subprocess: MagicMock,
        mock_open_pr: MagicMock,
        mock_save: MagicMock,
        mock_stage_complete: MagicMock,
        mock_project_done: MagicMock,
        base_state: dict,
        repo_dir: Path,
        project_dir: Path,
        agent_result: AgentResult,
        ui: MagicMock,
    ) -> None:
        """No error if feature_report.md doesn't exist on disk."""
        mock_run_agent.return_value = agent_result
        # Don't create the file — should not raise

        result = await verify_node(base_state, ui)

        assert result["feature_report"] == "Agent verification output"
