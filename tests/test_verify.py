"""Tests for graft.stages.verify."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from graft.stages.verify import _open_pr, verify_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeAgentResult:
    text: str = "fallback text from agent"
    tool_calls: list[dict] = field(default_factory=list)
    raw_messages: list = field(default_factory=list)
    elapsed_seconds: float = 1.0
    turns_used: int = 5


def _make_state(tmp_path, *, branch: str = "", feature_spec: dict | None = None):
    """Build a minimal FeatureState dict rooted in tmp_path."""
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    project_dir = tmp_path / "project"
    project_dir.mkdir(exist_ok=True)
    (project_dir / "artifacts").mkdir(exist_ok=True)
    (project_dir / "logs").mkdir(exist_ok=True)
    # metadata.json needed by mark_stage_complete / mark_project_done
    import json

    meta = {
        "project_id": "feat_test",
        "repo_path": str(repo),
        "feature_prompt": "test feature",
        "status": "in_progress",
        "stages_completed": [],
    }
    (project_dir / "metadata.json").write_text(json.dumps(meta))

    return {
        "repo_path": str(repo),
        "project_dir": str(project_dir),
        "feature_prompt": "Add dark mode",
        "codebase_profile": {"lang": "python"},
        "feature_spec": feature_spec or {"feature_name": "Dark Mode"},
        "build_plan": [{"unit_id": "u1"}],
        "units_completed": [{"unit_id": "u1"}],
        "units_reverted": [],
        "units_skipped": [],
        "feature_branch": branch,
        "model": "sonnet",
    }


def _make_ui():
    """Return a mock UI with all methods used by verify_node."""
    ui = MagicMock()
    ui.stage_start = MagicMock()
    ui.stage_done = MagicMock()
    ui.info = MagicMock()
    ui.pr_opened = MagicMock()
    return ui


# ---------------------------------------------------------------------------
# _open_pr tests
# ---------------------------------------------------------------------------

class TestOpenPr:
    """Tests for _open_pr helper."""

    @patch("graft.stages.verify.subprocess.run")
    def test_success_calls_git_push_and_gh_pr_create(self, mock_run):
        """_open_pr calls git push then gh pr create and returns the URL."""
        push_result = MagicMock(returncode=0)
        pr_result = MagicMock(returncode=0, stdout="https://github.com/o/r/pull/42\n")
        mock_run.side_effect = [push_result, pr_result]

        url = _open_pr("/repo", "feat-x", "Title", "Body text")

        assert url == "https://github.com/o/r/pull/42"
        assert mock_run.call_count == 2

        # First call: git push
        push_call = mock_run.call_args_list[0]
        assert push_call[0][0][:3] == ["git", "push", "-u"]
        assert push_call[1]["cwd"] == "/repo"
        assert push_call[1]["check"] is True

        # Second call: gh pr create
        pr_call = mock_run.call_args_list[1]
        assert "gh" in pr_call[0][0]
        assert "--title" in pr_call[0][0]
        assert "Title" in pr_call[0][0]

    @patch("graft.stages.verify.subprocess.run")
    def test_returns_none_on_pr_nonzero_exit(self, mock_run):
        """_open_pr returns None when gh pr create exits non-zero."""
        push_result = MagicMock(returncode=0)
        pr_result = MagicMock(returncode=1, stdout="")
        mock_run.side_effect = [push_result, pr_result]

        result = _open_pr("/repo", "feat-x", "Title", "Body")

        assert result is None

    @patch("graft.stages.verify.subprocess.run")
    def test_returns_none_on_called_process_error(self, mock_run):
        """_open_pr returns None when git push raises CalledProcessError."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "git push")

        result = _open_pr("/repo", "feat-x", "Title", "Body")

        assert result is None

    @patch("graft.stages.verify.subprocess.run")
    def test_returns_none_on_file_not_found(self, mock_run):
        """_open_pr returns None when gh binary is missing."""
        mock_run.side_effect = FileNotFoundError("gh not found")

        result = _open_pr("/repo", "feat-x", "Title", "Body")

        assert result is None

    @patch("graft.stages.verify.subprocess.run")
    def test_returns_none_on_timeout(self, mock_run):
        """_open_pr returns None when subprocess times out."""
        mock_run.side_effect = subprocess.TimeoutExpired("git push", 120)

        result = _open_pr("/repo", "feat-x", "Title", "Body")

        assert result is None


# ---------------------------------------------------------------------------
# verify_node tests
# ---------------------------------------------------------------------------

class TestVerifyNode:
    """Tests for the verify_node LangGraph node."""

    @pytest.mark.asyncio
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_calls_run_agent_with_correct_persona_and_tools(
        self, mock_agent, mock_subproc, tmp_path
    ):
        """verify_node passes the right persona, stage, and read-only tools."""
        mock_agent.return_value = FakeAgentResult()
        state = _make_state(tmp_path)
        ui = _make_ui()

        await verify_node(state, ui)

        mock_agent.assert_awaited_once()
        kwargs = mock_agent.call_args[1]
        assert kwargs["persona"] == "Principal Quality Engineer"
        assert kwargs["stage"] == "verify"
        assert set(kwargs["allowed_tools"]) == {"Bash", "Read", "Glob", "Grep"}
        assert kwargs["max_turns"] == 30
        assert kwargs["model"] == "sonnet"

    @pytest.mark.asyncio
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_reads_feature_report_and_saves_artifact(
        self, mock_agent, mock_subproc, tmp_path
    ):
        """verify_node reads feature_report.md, saves it, then deletes the source."""
        mock_agent.return_value = FakeAgentResult()
        state = _make_state(tmp_path)
        ui = _make_ui()

        # Create the report the agent would produce
        repo_path = state["repo_path"]
        report = "# Feature Report\nAll tests pass."
        (tmp_path / "repo" / "feature_report.md").write_text(report)

        result = await verify_node(state, ui)

        assert result["feature_report"] == report
        # Report file should be cleaned up
        assert not (tmp_path / "repo" / "feature_report.md").exists()
        # Artifact should be saved
        artifact_path = tmp_path / "project" / "artifacts" / "feature_report.md"
        assert artifact_path.exists()
        assert artifact_path.read_text() == report

    @pytest.mark.asyncio
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_falls_back_to_result_text_when_report_missing(
        self, mock_agent, mock_subproc, tmp_path
    ):
        """verify_node uses result.text when feature_report.md doesn't exist."""
        mock_agent.return_value = FakeAgentResult(text="Agent fallback output")
        state = _make_state(tmp_path)
        ui = _make_ui()

        # Do NOT create feature_report.md
        result = await verify_node(state, ui)

        assert result["feature_report"] == "Agent fallback output"

    @pytest.mark.asyncio
    @patch("graft.stages.verify._open_pr", return_value="https://github.com/o/r/pull/7")
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_opens_pr_when_branch_is_set(
        self, mock_agent, mock_subproc, mock_open_pr, tmp_path
    ):
        """verify_node calls _open_pr with branch, title, and body when branch is set."""
        mock_agent.return_value = FakeAgentResult(text="report body")
        state = _make_state(tmp_path, branch="feat/dark-mode")
        ui = _make_ui()

        result = await verify_node(state, ui)

        mock_open_pr.assert_called_once()
        args = mock_open_pr.call_args
        assert args[0][0] == state["repo_path"]
        assert args[0][1] == "feat/dark-mode"
        assert args[0][2] == "Feature: Dark Mode"
        assert result["pr_url"] == "https://github.com/o/r/pull/7"
        ui.pr_opened.assert_called_once_with("https://github.com/o/r/pull/7")

    @pytest.mark.asyncio
    @patch("graft.stages.verify._open_pr", return_value=None)
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_no_pr_url_when_open_pr_fails(
        self, mock_agent, mock_subproc, mock_open_pr, tmp_path
    ):
        """verify_node reports failure info when _open_pr returns None."""
        mock_agent.return_value = FakeAgentResult()
        state = _make_state(tmp_path, branch="feat/dark-mode")
        ui = _make_ui()

        result = await verify_node(state, ui)

        assert result["pr_url"] == ""
        ui.pr_opened.assert_not_called()
        # Should give helpful manual instructions
        assert ui.info.call_count == 2

    @pytest.mark.asyncio
    @patch("graft.stages.verify._open_pr", return_value="https://github.com/o/r/pull/7")
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_marks_project_done_when_pr_url_returned(
        self, mock_agent, mock_subproc, mock_open_pr, tmp_path
    ):
        """verify_node calls mark_project_done with the PR URL."""
        mock_agent.return_value = FakeAgentResult()
        state = _make_state(tmp_path, branch="feat/dark-mode")
        ui = _make_ui()

        await verify_node(state, ui)

        import json

        meta = json.loads(
            (tmp_path / "project" / "metadata.json").read_text()
        )
        assert meta["status"] == "completed"
        assert meta["pr_url"] == "https://github.com/o/r/pull/7"

    @pytest.mark.asyncio
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_does_not_open_pr_when_branch_empty(
        self, mock_agent, mock_subproc, tmp_path
    ):
        """verify_node skips PR opening when feature_branch is empty."""
        mock_agent.return_value = FakeAgentResult()
        state = _make_state(tmp_path, branch="")
        ui = _make_ui()

        result = await verify_node(state, ui)

        assert result["pr_url"] == ""
        ui.pr_opened.assert_not_called()
        # subprocess.run should NOT be called for git add / git commit
        mock_subproc.assert_not_called()

    @pytest.mark.asyncio
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_return_shape(self, mock_agent, mock_subproc, tmp_path):
        """verify_node returns dict with expected keys."""
        mock_agent.return_value = FakeAgentResult(text="report")
        state = _make_state(tmp_path)
        ui = _make_ui()

        result = await verify_node(state, ui)

        assert set(result.keys()) == {"feature_report", "pr_url", "current_stage"}
        assert result["current_stage"] == "verify"

    @pytest.mark.asyncio
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_stage_start_and_done_called(
        self, mock_agent, mock_subproc, tmp_path
    ):
        """verify_node bookends with stage_start and stage_done UI calls."""
        mock_agent.return_value = FakeAgentResult()
        state = _make_state(tmp_path)
        ui = _make_ui()

        await verify_node(state, ui)

        ui.stage_start.assert_called_once_with("verify")
        ui.stage_done.assert_called_once_with("verify")

    @pytest.mark.asyncio
    @patch("graft.stages.verify._open_pr", return_value="https://github.com/o/r/pull/1")
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_git_add_and_commit_before_pr(
        self, mock_agent, mock_subproc, mock_open_pr, tmp_path
    ):
        """verify_node runs git add -A and git commit before opening the PR."""
        mock_agent.return_value = FakeAgentResult()
        state = _make_state(tmp_path, branch="feat/x")
        ui = _make_ui()

        await verify_node(state, ui)

        assert mock_subproc.call_count == 2
        add_call = mock_subproc.call_args_list[0]
        assert add_call[0][0] == ["git", "add", "-A"]
        commit_call = mock_subproc.call_args_list[1]
        assert "commit" in commit_call[0][0]
        assert "--allow-empty" in commit_call[0][0]

    @pytest.mark.asyncio
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_prompt_includes_state_fields(
        self, mock_agent, mock_subproc, tmp_path
    ):
        """verify_node prompt includes repo_path, feature prompt, and plan info."""
        mock_agent.return_value = FakeAgentResult()
        state = _make_state(tmp_path)
        ui = _make_ui()

        await verify_node(state, ui)

        kwargs = mock_agent.call_args[1]
        prompt = kwargs["user_prompt"]
        assert state["repo_path"] in prompt
        assert "Add dark mode" in prompt
        assert "1 units" in prompt  # BUILD PLAN (1 units)

    @pytest.mark.asyncio
    @patch("graft.stages.verify.subprocess.run")
    @patch("graft.stages.verify.run_agent", new_callable=AsyncMock)
    async def test_feature_name_defaults_to_feature(
        self, mock_agent, mock_subproc, tmp_path
    ):
        """When feature_spec has no feature_name, PR title uses 'Feature'."""
        mock_agent.return_value = FakeAgentResult()
        state = _make_state(tmp_path, branch="feat/x", feature_spec={})
        ui = _make_ui()

        with patch("graft.stages.verify._open_pr", return_value=None) as mock_pr:
            await verify_node(state, ui)
            pr_title = mock_pr.call_args[0][2]
            assert pr_title == "Feature: Feature"
