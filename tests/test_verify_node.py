"""Tests for graft.stages.verify."""

import json
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from graft.agent import AgentResult
from graft.artifacts import create_project
from graft.stages.verify import _open_pr, verify_node
from graft.ui import UI


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def projects_root(tmp_path):
    root = tmp_path / "projects"
    root.mkdir()
    return root


@pytest.fixture
def project_dir(projects_root):
    """Create a real project directory with metadata on disk."""
    _, pdir = create_project(projects_root, "/tmp/repo", "Add dark mode")
    return pdir


@pytest.fixture
def repo_path(tmp_path):
    """A fake repo working directory."""
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


@pytest.fixture
def base_state(repo_path, project_dir):
    """Minimal FeatureState dict for verify_node."""
    return {
        "repo_path": str(repo_path),
        "project_dir": str(project_dir),
        "feature_prompt": "Add dark mode",
        "codebase_profile": {"language": "python"},
        "feature_spec": {"feature_name": "Dark Mode", "scope": ["toggle"]},
        "build_plan": [{"unit_id": "u1", "title": "Add toggle"}],
        "units_completed": [{"unit_id": "u1", "status": "kept"}],
        "units_reverted": [],
        "units_skipped": [],
        "feature_branch": "feat/dark-mode",
    }


@pytest.fixture
def ui():
    """A mock UI so we can verify calls without real terminal output."""
    mock = MagicMock(spec=UI)
    return mock


def _make_agent_result(text: str = "All checks passed.") -> AgentResult:
    return AgentResult(
        text=text, tool_calls=[], raw_messages=[], elapsed_seconds=1.0, turns_used=3
    )


# ---------------------------------------------------------------------------
# Tests for _open_pr helper
# ---------------------------------------------------------------------------


class TestOpenPr:
    """Tests for the _open_pr helper that wraps git push + gh pr create."""

    def test_returns_pr_url_on_success(self):
        """Happy path: git push succeeds, gh pr create prints URL."""
        push_ok = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        pr_ok = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="https://github.com/org/repo/pull/42\n",
            stderr="",
        )
        with patch(
            "graft.stages.verify.subprocess.run", side_effect=[push_ok, pr_ok]
        ) as mock_run:
            url = _open_pr("/tmp/repo", "feat/x", "Feature: X", "body text")

        assert url == "https://github.com/org/repo/pull/42"
        # Verify git push was called first, then gh pr create
        assert mock_run.call_count == 2
        push_call = mock_run.call_args_list[0]
        assert push_call[0][0][:3] == ["git", "push", "-u"]
        pr_call = mock_run.call_args_list[1]
        assert pr_call[0][0][0] == "gh"

    def test_returns_none_on_gh_failure(self):
        """gh pr create returns non-zero — should return None gracefully."""
        push_ok = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        pr_fail = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="pull request already exists",
        )
        with patch(
            "graft.stages.verify.subprocess.run", side_effect=[push_ok, pr_fail]
        ):
            url = _open_pr("/tmp/repo", "feat/x", "Title", "Body")

        assert url is None

    def test_returns_none_on_push_failure(self):
        """git push raises CalledProcessError — should return None."""
        with patch(
            "graft.stages.verify.subprocess.run",
            side_effect=subprocess.CalledProcessError(128, "git push"),
        ):
            url = _open_pr("/tmp/repo", "feat/x", "Title", "Body")

        assert url is None

    def test_returns_none_on_timeout(self):
        """Timeout during push or PR creation returns None."""
        with patch(
            "graft.stages.verify.subprocess.run",
            side_effect=subprocess.TimeoutExpired("git", 120),
        ):
            url = _open_pr("/tmp/repo", "feat/x", "Title", "Body")

        assert url is None

    def test_returns_none_when_gh_not_found(self):
        """FileNotFoundError (gh not installed) returns None."""
        with patch(
            "graft.stages.verify.subprocess.run",
            side_effect=FileNotFoundError("gh not found"),
        ):
            url = _open_pr("/tmp/repo", "feat/x", "Title", "Body")

        assert url is None

    def test_pr_url_extracted_from_stdout(self):
        """The exact URL is parsed from stdout (stripped of whitespace)."""
        push_ok = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        pr_ok = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="  https://github.com/org/repo/pull/99  \n",
            stderr="",
        )
        with patch("graft.stages.verify.subprocess.run", side_effect=[push_ok, pr_ok]):
            url = _open_pr("/tmp/repo", "feat/x", "T", "B")

        assert url == "https://github.com/org/repo/pull/99"


# ---------------------------------------------------------------------------
# Tests for verify_node (async LangGraph node)
# ---------------------------------------------------------------------------


class TestVerifyNodeHappyPath:
    """verify_node happy path — agent writes report, PR is opened."""

    async def test_happy_path_returns_pr_url(
        self, base_state, ui, repo_path, project_dir
    ):
        """Agent writes feature_report.md, node saves artifact, opens PR, returns pr_url."""
        # Simulate the agent writing feature_report.md into the repo
        report_content = "# Feature Report\nAll tests pass."
        report_file = repo_path / "feature_report.md"
        report_file.write_text(report_content)

        with (
            patch(
                "graft.stages.verify.run_agent",
                new_callable=AsyncMock,
                return_value=_make_agent_result(),
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run") as mock_subproc,
        ):
            # subprocess.run is called for: git add, git commit, git push, gh pr create
            mock_subproc.side_effect = [
                # git add -A
                subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="", stderr=""
                ),
                # git commit
                subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="", stderr=""
                ),
                # git push (inside _open_pr)
                subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="", stderr=""
                ),
                # gh pr create (inside _open_pr)
                subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout="https://github.com/org/repo/pull/7\n",
                    stderr="",
                ),
            ]

            result = await verify_node(base_state, ui)

        assert result["pr_url"] == "https://github.com/org/repo/pull/7"
        assert result["current_stage"] == "verify"
        assert "Feature Report" in result["feature_report"]

        # Artifact persisted to project_dir
        saved = (project_dir / "artifacts" / "feature_report.md").read_text()
        assert saved == report_content

        # Report file cleaned up from repo
        assert not report_file.exists()

        # UI interactions
        ui.stage_start.assert_called_once_with("verify")
        ui.stage_done.assert_called_once_with("verify")
        ui.pr_opened.assert_called_once_with("https://github.com/org/repo/pull/7")

        # run_agent was called with correct stage and cwd
        mock_agent.assert_awaited_once()
        call_kwargs = mock_agent.call_args[1]
        assert call_kwargs["stage"] == "verify"
        assert call_kwargs["cwd"] == str(repo_path)


class TestVerifyNodeNoBranch:
    """When feature_branch is not set, skip git push/PR entirely."""

    async def test_no_branch_skips_pr(self, base_state, ui, repo_path, project_dir):
        """Without feature_branch, no subprocess calls for push/PR, pr_url is empty."""
        base_state["feature_branch"] = ""

        report_file = repo_path / "feature_report.md"
        report_file.write_text("# Report\nNo branch.")

        with (
            patch(
                "graft.stages.verify.run_agent",
                new_callable=AsyncMock,
                return_value=_make_agent_result(),
            ),
            patch("graft.stages.verify.subprocess.run") as mock_subproc,
        ):
            result = await verify_node(base_state, ui)

        # No subprocess calls at all — no git add, commit, push, or PR
        mock_subproc.assert_not_called()

        assert result["pr_url"] == ""
        assert result["current_stage"] == "verify"

        # UI should NOT have called pr_opened or info about manual push
        ui.pr_opened.assert_not_called()

    async def test_no_branch_key_skips_pr(self, base_state, ui, repo_path, project_dir):
        """feature_branch key absent from state behaves like empty string."""
        del base_state["feature_branch"]

        with (
            patch(
                "graft.stages.verify.run_agent",
                new_callable=AsyncMock,
                return_value=_make_agent_result(),
            ),
            patch("graft.stages.verify.subprocess.run") as mock_subproc,
        ):
            result = await verify_node(base_state, ui)

        mock_subproc.assert_not_called()
        assert result["pr_url"] == ""


class TestVerifyNodePrFailure:
    """When gh CLI fails, verify_node degrades gracefully — pr_url is empty."""

    async def test_gh_failure_sets_empty_pr_url(
        self, base_state, ui, repo_path, project_dir
    ):
        """gh pr create failing results in pr_url='', info message shown."""
        with (
            patch(
                "graft.stages.verify.run_agent",
                new_callable=AsyncMock,
                return_value=_make_agent_result(),
            ),
            patch("graft.stages.verify.subprocess.run") as mock_subproc,
        ):
            mock_subproc.side_effect = [
                # git add -A
                subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="", stderr=""
                ),
                # git commit
                subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="", stderr=""
                ),
                # git push — fails
                subprocess.CalledProcessError(128, "git push"),
            ]

            result = await verify_node(base_state, ui)

        assert result["pr_url"] == ""
        # Should show info about manual push
        assert ui.info.call_count == 2
        first_info = ui.info.call_args_list[0][0][0]
        assert "Could not open PR" in first_info
        assert "feat/dark-mode" in first_info

        # PR opened should NOT be called
        ui.pr_opened.assert_not_called()

        # mark_project_done should NOT be called (no pr_url)
        meta = json.loads((project_dir / "metadata.json").read_text())
        assert meta["status"] == "in_progress"

    async def test_gh_returns_nonzero(self, base_state, ui, repo_path, project_dir):
        """gh pr create returns non-zero exit code — graceful degradation."""
        with (
            patch(
                "graft.stages.verify.run_agent",
                new_callable=AsyncMock,
                return_value=_make_agent_result(),
            ),
            patch("graft.stages.verify.subprocess.run") as mock_subproc,
        ):
            mock_subproc.side_effect = [
                # git add -A
                subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="", stderr=""
                ),
                # git commit
                subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="", stderr=""
                ),
                # git push (success)
                subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="", stderr=""
                ),
                # gh pr create (failure)
                subprocess.CompletedProcess(
                    args=[], returncode=1, stdout="", stderr="error"
                ),
            ]

            result = await verify_node(base_state, ui)

        assert result["pr_url"] == ""
        ui.pr_opened.assert_not_called()
        ui.info.assert_called()


class TestVerifyArtifactSaving:
    """Artifact saving and project completion marking."""

    async def test_artifact_saved_and_report_cleaned(
        self, base_state, ui, repo_path, project_dir
    ):
        """feature_report.md is saved as artifact and removed from repo."""
        report_content = "# Detailed Report\nCoverage: 95%"
        (repo_path / "feature_report.md").write_text(report_content)

        with (
            patch(
                "graft.stages.verify.run_agent",
                new_callable=AsyncMock,
                return_value=_make_agent_result(),
            ),
            patch("graft.stages.verify.subprocess.run") as mock_subproc,
        ):
            mock_subproc.side_effect = [
                subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="", stderr=""
                ),
                subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="", stderr=""
                ),
                subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="", stderr=""
                ),
                subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout="https://github.com/org/repo/pull/1\n",
                    stderr="",
                ),
            ]
            result = await verify_node(base_state, ui)

        # Artifact written to project_dir/artifacts/
        artifact_path = project_dir / "artifacts" / "feature_report.md"
        assert artifact_path.exists()
        assert artifact_path.read_text() == report_content

        # Cleaned up from repo working directory
        assert not (repo_path / "feature_report.md").exists()

        # Return value contains the report
        assert result["feature_report"] == report_content

    async def test_fallback_to_agent_text_when_no_report_file(
        self, base_state, ui, repo_path, project_dir
    ):
        """When agent doesn't write feature_report.md, fall back to agent result text."""
        agent_text = "Agent summary: all good."

        with (
            patch(
                "graft.stages.verify.run_agent",
                new_callable=AsyncMock,
                return_value=_make_agent_result(text=agent_text),
            ),
            patch("graft.stages.verify.subprocess.run") as mock_subproc,
        ):
            mock_subproc.side_effect = [
                subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="", stderr=""
                ),
                subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="", stderr=""
                ),
                subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="", stderr=""
                ),
                subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout="https://github.com/org/repo/pull/5\n",
                    stderr="",
                ),
            ]
            result = await verify_node(base_state, ui)

        assert result["feature_report"] == agent_text
        # Artifact still saved (with agent text as content)
        artifact_path = project_dir / "artifacts" / "feature_report.md"
        assert artifact_path.exists()
        assert artifact_path.read_text() == agent_text

    async def test_mark_project_done_on_pr_success(
        self, base_state, ui, repo_path, project_dir
    ):
        """mark_project_done is called with pr_url when PR creation succeeds."""
        (repo_path / "feature_report.md").write_text("report")

        with (
            patch(
                "graft.stages.verify.run_agent",
                new_callable=AsyncMock,
                return_value=_make_agent_result(),
            ),
            patch("graft.stages.verify.subprocess.run") as mock_subproc,
        ):
            mock_subproc.side_effect = [
                subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="", stderr=""
                ),
                subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="", stderr=""
                ),
                subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="", stderr=""
                ),
                subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout="https://github.com/org/repo/pull/10\n",
                    stderr="",
                ),
            ]
            await verify_node(base_state, ui)

        meta = json.loads((project_dir / "metadata.json").read_text())
        assert meta["status"] == "completed"
        assert meta["pr_url"] == "https://github.com/org/repo/pull/10"

    async def test_mark_stage_complete_always_called(
        self, base_state, ui, repo_path, project_dir
    ):
        """mark_stage_complete('verify') is always called, even without a PR."""
        base_state["feature_branch"] = ""

        with (
            patch(
                "graft.stages.verify.run_agent",
                new_callable=AsyncMock,
                return_value=_make_agent_result(),
            ),
            patch("graft.stages.verify.subprocess.run"),
        ):
            await verify_node(base_state, ui)

        meta = json.loads((project_dir / "metadata.json").read_text())
        assert "verify" in meta["stages_completed"]


class TestVerifyPromptContent:
    """Verify the prompt assembled for the agent includes all required context."""

    async def test_prompt_includes_build_plan_and_spec(
        self, base_state, ui, repo_path, project_dir
    ):
        """The user prompt sent to run_agent includes build_plan, feature_spec, and execution results."""
        captured_prompt = None

        async def capture_agent(**kwargs):
            nonlocal captured_prompt
            captured_prompt = kwargs["user_prompt"]
            return _make_agent_result()

        base_state["feature_branch"] = ""

        with (
            patch("graft.stages.verify.run_agent", side_effect=capture_agent),
            patch("graft.stages.verify.subprocess.run"),
        ):
            await verify_node(base_state, ui)

        assert captured_prompt is not None
        # Build plan content
        assert "Add toggle" in captured_prompt
        assert "BUILD PLAN (1 units)" in captured_prompt
        # Feature spec content
        assert "Dark Mode" in captured_prompt
        assert "FEATURE SPEC" in captured_prompt
        # Execution results
        assert "UNITS COMPLETED (1)" in captured_prompt
        assert "UNITS REVERTED (0)" in captured_prompt
        assert "UNITS SKIPPED (0)" in captured_prompt
        # Feature prompt
        assert "Add dark mode" in captured_prompt
        # Codebase profile
        assert "CODEBASE PROFILE" in captured_prompt
        assert "python" in captured_prompt

    async def test_prompt_with_empty_state_fields(self, ui, repo_path, project_dir):
        """When optional state fields are missing, prompt uses defaults without crashing."""
        minimal_state = {
            "repo_path": str(repo_path),
            "project_dir": str(project_dir),
        }

        captured_prompt = None

        async def capture_agent(**kwargs):
            nonlocal captured_prompt
            captured_prompt = kwargs["user_prompt"]
            return _make_agent_result()

        with (
            patch("graft.stages.verify.run_agent", side_effect=capture_agent),
            patch("graft.stages.verify.subprocess.run"),
        ):
            result = await verify_node(minimal_state, ui)

        assert captured_prompt is not None
        # Should contain zero-count sections
        assert "BUILD PLAN (0 units)" in captured_prompt
        assert "UNITS COMPLETED (0)" in captured_prompt
        assert result["current_stage"] == "verify"

    async def test_run_agent_receives_correct_kwargs(
        self, base_state, ui, repo_path, project_dir
    ):
        """run_agent is called with persona, stage, cwd, allowed_tools, and max_turns."""
        base_state["feature_branch"] = ""
        base_state["model"] = "claude-sonnet"

        with (
            patch(
                "graft.stages.verify.run_agent",
                new_callable=AsyncMock,
                return_value=_make_agent_result(),
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run"),
        ):
            await verify_node(base_state, ui)

        mock_agent.assert_awaited_once()
        kw = mock_agent.call_args[1]
        assert kw["persona"] == "Principal Quality Engineer"
        assert kw["stage"] == "verify"
        assert kw["cwd"] == str(repo_path)
        assert kw["project_dir"] == str(project_dir)
        assert kw["max_turns"] == 30
        assert kw["allowed_tools"] == ["Bash", "Read", "Glob", "Grep"]
        assert kw["model"] == "claude-sonnet"
