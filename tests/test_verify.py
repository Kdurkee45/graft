"""Tests for graft.stages.verify."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from graft.stages.verify import SYSTEM_PROMPT, _open_pr, verify_node


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path):
    """Temporary directory simulating a repo working tree."""
    return tmp_path / "repo"


@pytest.fixture
def project_dir(tmp_path):
    """Temporary project directory with artifacts/ subdirectory."""
    d = tmp_path / "project"
    (d / "artifacts").mkdir(parents=True)
    (d / "metadata.json").write_text(
        json.dumps({"stages_completed": [], "status": "in_progress"})
    )
    return d


@pytest.fixture
def base_state(repo, project_dir):
    """Minimal FeatureState dict for verify_node."""
    repo.mkdir(parents=True, exist_ok=True)
    return {
        "repo_path": str(repo),
        "project_dir": str(project_dir),
        "feature_prompt": "Add dark mode",
        "codebase_profile": {"language": "python"},
        "feature_spec": {"feature_name": "Dark Mode"},
        "build_plan": [{"unit": "theme-toggle"}],
        "units_completed": [{"unit": "theme-toggle", "status": "done"}],
        "units_reverted": [],
        "units_skipped": [],
        "feature_branch": "feat/dark-mode",
        "model": "claude-sonnet-4-20250514",
    }


@pytest.fixture
def ui():
    """A mock UI that records calls."""
    m = MagicMock()
    m.stage_start = MagicMock()
    m.stage_done = MagicMock()
    m.pr_opened = MagicMock()
    m.info = MagicMock()
    return m


# ---------------------------------------------------------------------------
# _open_pr unit tests
# ---------------------------------------------------------------------------


class TestOpenPr:
    """Tests for the _open_pr helper."""

    @patch("graft.stages.verify.subprocess.run")
    def test_successful_pr(self, mock_run):
        """Happy path: git push succeeds, gh pr create succeeds, returns URL."""
        push_result = MagicMock(returncode=0)
        pr_result = MagicMock(returncode=0, stdout="https://github.com/org/repo/pull/42\n")
        mock_run.side_effect = [push_result, pr_result]

        url = _open_pr("/repo", "feat/dark-mode", "Feature: Dark Mode", "body text")

        assert url == "https://github.com/org/repo/pull/42"
        assert mock_run.call_count == 2

        # Verify git push call
        push_call = mock_run.call_args_list[0]
        assert push_call[0][0] == ["git", "push", "-u", "origin", "feat/dark-mode"]
        assert push_call[1]["cwd"] == "/repo"
        assert push_call[1]["check"] is True

        # Verify gh pr create call
        pr_call = mock_run.call_args_list[1]
        assert pr_call[0][0][:3] == ["gh", "pr", "create"]
        assert "--title" in pr_call[0][0]
        assert "Feature: Dark Mode" in pr_call[0][0]

    @patch("graft.stages.verify.subprocess.run")
    def test_gh_cli_not_installed(self, mock_run):
        """gh not installed → FileNotFoundError → returns None."""
        # git push succeeds but gh is not found
        push_result = MagicMock(returncode=0)
        mock_run.side_effect = [push_result, FileNotFoundError("gh not found")]

        # Actually FileNotFoundError can happen on the first call too;
        # but let's test it on the push itself for the "gh not installed" scenario
        mock_run.side_effect = FileNotFoundError("gh not found")

        url = _open_pr("/repo", "feat/x", "Title", "Body")
        assert url is None

    @patch("graft.stages.verify.subprocess.run")
    def test_git_push_fails(self, mock_run):
        """git push fails with CalledProcessError → returns None."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "git push")

        url = _open_pr("/repo", "feat/x", "Title", "Body")
        assert url is None

    @patch("graft.stages.verify.subprocess.run")
    def test_gh_pr_create_nonzero_exit(self, mock_run):
        """gh pr create returns non-zero → returns None."""
        push_result = MagicMock(returncode=0)
        pr_result = MagicMock(returncode=1, stdout="", stderr="already exists")
        mock_run.side_effect = [push_result, pr_result]

        url = _open_pr("/repo", "feat/x", "Title", "Body")
        assert url is None

    @patch("graft.stages.verify.subprocess.run")
    def test_timeout_returns_none(self, mock_run):
        """Timeout on either subprocess → returns None."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git push", timeout=120)

        url = _open_pr("/repo", "feat/x", "Title", "Body")
        assert url is None


# ---------------------------------------------------------------------------
# verify_node integration tests (mocked agent + subprocess)
# ---------------------------------------------------------------------------


class TestVerifyNode:
    """Tests for the verify_node LangGraph node."""

    @pytest.mark.asyncio
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_happy_path_with_report_file(
        self, mock_agent, mock_save, mock_stage, mock_done, mock_subproc, base_state, ui
    ):
        """Full success: agent writes report, PR opens, state returned."""
        # Agent returns a result
        agent_result = MagicMock()
        agent_result.text = "fallback text"
        mock_agent.return_value = agent_result

        # Write feature_report.md into the repo so verify_node can read it
        repo = Path(base_state["repo_path"])
        repo.mkdir(parents=True, exist_ok=True)
        report_content = "# Feature Report\nAll tests pass."
        (repo / "feature_report.md").write_text(report_content)

        # subprocess.run for git add, git commit, git push, gh pr create
        push_ok = MagicMock(returncode=0)
        pr_ok = MagicMock(returncode=0, stdout="https://github.com/org/repo/pull/99\n")
        mock_subproc.side_effect = [
            MagicMock(returncode=0),  # git add -A
            MagicMock(returncode=0),  # git commit
            push_ok,                  # git push
            pr_ok,                    # gh pr create
        ]

        result = await verify_node(base_state, ui)

        # Report was read from file, not agent fallback
        assert result["feature_report"] == report_content
        assert result["pr_url"] == "https://github.com/org/repo/pull/99"
        assert result["current_stage"] == "verify"

        # Report file should be cleaned up
        assert not (repo / "feature_report.md").exists()

        # Artifact saved
        mock_save.assert_called_once_with(
            str(base_state["project_dir"]),
            "feature_report.md",
            report_content,
        )

        # Stage marked complete
        mock_stage.assert_called_once_with(str(base_state["project_dir"]), "verify")

        # Project marked done with PR URL
        mock_done.assert_called_once_with(
            str(base_state["project_dir"]),
            "https://github.com/org/repo/pull/99",
        )

        # UI notifications
        ui.stage_start.assert_called_once_with("verify")
        ui.stage_done.assert_called_once_with("verify")
        ui.pr_opened.assert_called_once_with("https://github.com/org/repo/pull/99")

    @pytest.mark.asyncio
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_no_report_file_falls_back_to_agent_text(
        self, mock_agent, mock_save, mock_stage, mock_done, mock_subproc, base_state, ui
    ):
        """If agent doesn't write feature_report.md, fall back to result.text."""
        agent_result = MagicMock()
        agent_result.text = "Agent inline report"
        mock_agent.return_value = agent_result

        # No feature_report.md on disk — repo exists but empty
        repo = Path(base_state["repo_path"])
        repo.mkdir(parents=True, exist_ok=True)

        # subprocess calls for cleanup commit + PR
        mock_subproc.side_effect = [
            MagicMock(returncode=0),  # git add -A
            MagicMock(returncode=0),  # git commit
            MagicMock(returncode=0),  # git push
            MagicMock(returncode=0, stdout="https://github.com/org/repo/pull/7\n"),
        ]

        result = await verify_node(base_state, ui)

        assert result["feature_report"] == "Agent inline report"
        mock_save.assert_called_once_with(
            str(base_state["project_dir"]),
            "feature_report.md",
            "Agent inline report",
        )

    @pytest.mark.asyncio
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_no_branch_skips_pr(
        self, mock_agent, mock_save, mock_stage, mock_done, mock_subproc, base_state, ui
    ):
        """When feature_branch is empty, skip commit/push/PR entirely."""
        base_state["feature_branch"] = ""

        agent_result = MagicMock()
        agent_result.text = "Report content"
        mock_agent.return_value = agent_result

        repo = Path(base_state["repo_path"])
        repo.mkdir(parents=True, exist_ok=True)

        result = await verify_node(base_state, ui)

        assert result["pr_url"] == ""
        assert result["current_stage"] == "verify"

        # No subprocess calls for git/gh at all
        mock_subproc.assert_not_called()

        # mark_project_done should NOT be called (no PR URL)
        mock_done.assert_not_called()

        # But stage should still be marked complete
        mock_stage.assert_called_once_with(str(base_state["project_dir"]), "verify")

    @pytest.mark.asyncio
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_pr_failure_reports_manual_instructions(
        self, mock_agent, mock_save, mock_stage, mock_done, mock_subproc, base_state, ui
    ):
        """When _open_pr fails, UI shows manual push instructions."""
        agent_result = MagicMock()
        agent_result.text = "Report"
        mock_agent.return_value = agent_result

        repo = Path(base_state["repo_path"])
        repo.mkdir(parents=True, exist_ok=True)

        # git add + commit succeed, but push fails
        mock_subproc.side_effect = [
            MagicMock(returncode=0),  # git add -A
            MagicMock(returncode=0),  # git commit
            subprocess.CalledProcessError(1, "git push"),  # push fails
        ]

        result = await verify_node(base_state, ui)

        assert result["pr_url"] == ""
        # Two ui.info calls: manual push instruction + report location
        assert ui.info.call_count == 2
        first_msg = ui.info.call_args_list[0][0][0]
        assert "feat/dark-mode" in first_msg
        assert "manually" in first_msg

        # mark_project_done NOT called (no PR)
        mock_done.assert_not_called()

    @pytest.mark.asyncio
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_missing_feature_name_defaults(
        self, mock_agent, mock_save, mock_stage, mock_done, mock_subproc, base_state, ui
    ):
        """When feature_spec has no feature_name, default to 'Feature'."""
        base_state["feature_spec"] = {}  # no feature_name key

        agent_result = MagicMock()
        agent_result.text = "Report"
        mock_agent.return_value = agent_result

        repo = Path(base_state["repo_path"])
        repo.mkdir(parents=True, exist_ok=True)

        # Capture the title passed to gh pr create
        calls = []

        def track_subprocess(cmd, **kwargs):
            calls.append(cmd)
            r = MagicMock(returncode=0)
            if cmd[0] == "gh":
                r.stdout = "https://github.com/org/repo/pull/1\n"
            return r

        mock_subproc.side_effect = track_subprocess

        result = await verify_node(base_state, ui)

        # Find the gh pr create call and verify the title uses the default
        gh_calls = [c for c in calls if c[0] == "gh"]
        assert len(gh_calls) == 1
        gh_cmd = gh_calls[0]
        title_idx = gh_cmd.index("--title") + 1
        assert gh_cmd[title_idx] == "Feature: Feature"

    @pytest.mark.asyncio
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_cleanup_commit_uses_allow_empty(
        self, mock_agent, mock_save, mock_stage, mock_done, mock_subproc, base_state, ui
    ):
        """Verify that the cleanup commit uses --allow-empty flag."""
        agent_result = MagicMock()
        agent_result.text = "Report"
        mock_agent.return_value = agent_result

        repo = Path(base_state["repo_path"])
        repo.mkdir(parents=True, exist_ok=True)

        mock_subproc.side_effect = [
            MagicMock(returncode=0),  # git add -A
            MagicMock(returncode=0),  # git commit --allow-empty
            MagicMock(returncode=0),  # git push
            MagicMock(returncode=0, stdout="https://github.com/org/repo/pull/5\n"),
        ]

        await verify_node(base_state, ui)

        # Check git add -A call
        add_call = mock_subproc.call_args_list[0]
        assert add_call[0][0] == ["git", "add", "-A"]

        # Check git commit call includes --allow-empty
        commit_call = mock_subproc.call_args_list[1]
        commit_cmd = commit_call[0][0]
        assert "git" in commit_cmd[0]
        assert "commit" in commit_cmd
        assert "--allow-empty" in commit_cmd
        assert "chore: cleanup verification artifacts" in commit_cmd

    @pytest.mark.asyncio
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_agent_receives_correct_prompt(
        self, mock_agent, mock_save, mock_stage, mock_done, mock_subproc, base_state, ui
    ):
        """Verify the agent is invoked with correct persona, system prompt, and user prompt."""
        agent_result = MagicMock()
        agent_result.text = "Report"
        mock_agent.return_value = agent_result

        repo = Path(base_state["repo_path"])
        repo.mkdir(parents=True, exist_ok=True)

        # No branch → skip PR logic
        base_state["feature_branch"] = ""

        await verify_node(base_state, ui)

        mock_agent.assert_awaited_once()
        kwargs = mock_agent.call_args[1]
        assert kwargs["persona"] == "Principal Quality Engineer"
        assert kwargs["system_prompt"] == SYSTEM_PROMPT
        assert kwargs["cwd"] == str(repo)
        assert kwargs["stage"] == "verify"
        assert kwargs["max_turns"] == 30
        assert kwargs["allowed_tools"] == ["Bash", "Read", "Glob", "Grep"]
        assert kwargs["model"] == "claude-sonnet-4-20250514"

        # User prompt should contain feature details
        user_prompt = kwargs["user_prompt"]
        assert "Add dark mode" in user_prompt
        assert "theme-toggle" in user_prompt
        assert str(repo) in user_prompt

    @pytest.mark.asyncio
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_empty_optional_state_fields(
        self, mock_agent, mock_save, mock_stage, mock_done, mock_subproc, ui, tmp_path
    ):
        """verify_node handles missing optional fields via .get() defaults."""
        repo = tmp_path / "repo"
        repo.mkdir()
        project_dir = tmp_path / "project"
        (project_dir / "artifacts").mkdir(parents=True)
        (project_dir / "metadata.json").write_text(
            json.dumps({"stages_completed": [], "status": "in_progress"})
        )

        minimal_state = {
            "repo_path": str(repo),
            "project_dir": str(project_dir),
        }

        agent_result = MagicMock()
        agent_result.text = "Minimal report"
        mock_agent.return_value = agent_result

        result = await verify_node(minimal_state, ui)

        assert result["feature_report"] == "Minimal report"
        assert result["pr_url"] == ""
        assert result["current_stage"] == "verify"

        # The agent prompt should still include defaults
        user_prompt = mock_agent.call_args[1]["user_prompt"]
        assert "0 units" in user_prompt or "UNITS COMPLETED (0)" in user_prompt

    @pytest.mark.asyncio
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.mark_project_done")
    @patch("graft.stages.verify.mark_stage_complete")
    @patch("graft.stages.verify.save_artifact")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_pr_title_uses_feature_name_from_spec(
        self, mock_agent, mock_save, mock_stage, mock_done, mock_subproc, base_state, ui
    ):
        """PR title is 'Feature: {feature_name}' from the spec."""
        base_state["feature_spec"] = {"feature_name": "OAuth2 Login"}

        agent_result = MagicMock()
        agent_result.text = "Report"
        mock_agent.return_value = agent_result

        repo = Path(base_state["repo_path"])
        repo.mkdir(parents=True, exist_ok=True)

        calls = []

        def track_subprocess(cmd, **kwargs):
            calls.append(cmd)
            r = MagicMock(returncode=0)
            if cmd[0] == "gh":
                r.stdout = "https://github.com/org/repo/pull/10\n"
            return r

        mock_subproc.side_effect = track_subprocess

        result = await verify_node(base_state, ui)

        gh_calls = [c for c in calls if c[0] == "gh"]
        assert len(gh_calls) == 1
        title_idx = gh_calls[0].index("--title") + 1
        assert gh_calls[0][title_idx] == "Feature: OAuth2 Login"
        assert result["pr_url"] == "https://github.com/org/repo/pull/10"
