"""Tests for graft.stages.verify."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from graft.stages.verify import SYSTEM_PROMPT, _open_pr, verify_node

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeAgentResult:
    """Minimal stand-in for graft.agent.AgentResult."""

    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    raw_messages: list = field(default_factory=list)
    elapsed_seconds: float = 1.0
    turns_used: int = 5


def _completed(returncode: int = 0, stdout: str = "", stderr: str = ""):
    """Build a fake subprocess.CompletedProcess."""
    return subprocess.CompletedProcess(
        args=["fake"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path):
    """Temporary repo directory."""
    d = tmp_path / "repo"
    d.mkdir()
    return d


@pytest.fixture
def project(tmp_path):
    """Temporary project directory with required sub-structure."""
    d = tmp_path / "project"
    d.mkdir()
    (d / "artifacts").mkdir()
    (d / "logs").mkdir()
    meta = {"project_id": "feat_test01", "stages_completed": []}
    (d / "metadata.json").write_text(json.dumps(meta))
    return d


@pytest.fixture
def ui():
    """Mock UI object exposing the methods verify_node calls."""
    m = MagicMock()
    m.stage_start = MagicMock()
    m.stage_done = MagicMock()
    m.error = MagicMock()
    m.info = MagicMock()
    m.pr_opened = MagicMock()
    return m


def _state(repo, project, **kw):
    """Build a minimal FeatureState dict."""
    base = {
        "repo_path": str(repo),
        "project_dir": str(project),
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# _open_pr
# ---------------------------------------------------------------------------


class TestOpenPr:
    """Tests for the _open_pr helper function."""

    def test_success_returns_pr_url(self, repo):
        """Successful push + PR create returns the PR URL string."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _completed(),  # git push
                _completed(stdout="https://github.com/org/repo/pull/42\n"),
            ]
            url = _open_pr(str(repo), "feat/dark-mode", "Feature: Dark Mode", "body")

        assert url == "https://github.com/org/repo/pull/42"

    def test_push_called_with_correct_args(self, repo):
        """git push is invoked with -u origin <branch>."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _completed(),
                _completed(stdout="https://url\n"),
            ]
            _open_pr(str(repo), "feat/x", "title", "body")

        push_call = mock_run.call_args_list[0]
        assert push_call == call(
            ["git", "push", "-u", "origin", "feat/x"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )

    def test_gh_pr_create_called_with_correct_args(self, repo):
        """gh pr create is invoked with title, body, and head branch."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _completed(),
                _completed(stdout="https://url\n"),
            ]
            _open_pr(str(repo), "feat/x", "My Title", "My Body")

        pr_call = mock_run.call_args_list[1]
        assert pr_call == call(
            [
                "gh",
                "pr",
                "create",
                "--title",
                "My Title",
                "--body",
                "My Body",
                "--head",
                "feat/x",
            ],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_push_failure_returns_none(self, repo):
        """CalledProcessError on push returns None."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git push")
            url = _open_pr(str(repo), "feat/x", "title", "body")

        assert url is None

    def test_gh_not_found_returns_none(self, repo):
        """FileNotFoundError (gh not installed) returns None."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("gh not found")
            url = _open_pr(str(repo), "feat/x", "title", "body")

        assert url is None

    def test_push_timeout_returns_none(self, repo):
        """TimeoutExpired on push returns None."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("git push", 120)
            url = _open_pr(str(repo), "feat/x", "title", "body")

        assert url is None

    def test_pr_create_nonzero_returncode_returns_none(self, repo):
        """Non-zero returncode from gh pr create returns None."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _completed(),  # push succeeds
                _completed(returncode=1, stderr="already exists"),
            ]
            url = _open_pr(str(repo), "feat/x", "title", "body")

        assert url is None

    def test_stdout_stripped(self, repo):
        """PR URL is stripped of leading/trailing whitespace."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _completed(),
                _completed(stdout="  https://github.com/o/r/pull/1  \n"),
            ]
            url = _open_pr(str(repo), "feat/x", "title", "body")

        assert url == "https://github.com/o/r/pull/1"


# ---------------------------------------------------------------------------
# verify_node — happy path
# ---------------------------------------------------------------------------


class TestVerifyNodeHappyPath:
    """Core happy-path tests where agent produces valid outputs."""

    async def test_returns_expected_keys(self, repo, project, ui):
        """verify_node returns feature_report, pr_url, current_stage."""
        (repo / "feature_report.md").write_text("# Report")

        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_run.return_value = FakeAgentResult(text="fallback")
            result = await verify_node(
                _state(repo, project, feature_branch="feat/x"), ui
            )

        assert set(result.keys()) == {
            "feature_report",
            "pr_url",
            "current_stage",
        }
        assert result["current_stage"] == "verify"

    async def test_reads_feature_report_from_file(self, repo, project, ui):
        """When feature_report.md exists, its content is used."""
        (repo / "feature_report.md").write_text("# Full Report\nAll good.")

        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_run.return_value = FakeAgentResult(text="agent text")
            result = await verify_node(
                _state(repo, project, feature_branch="feat/x"), ui
            )

        assert result["feature_report"] == "# Full Report\nAll good."

    async def test_pr_opened_successfully(self, repo, project, ui):
        """When _open_pr returns a URL, it is set in pr_url."""
        pr = "https://github.com/org/repo/pull/99"
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=pr),
        ):
            mock_run.return_value = FakeAgentResult(text="report")
            result = await verify_node(
                _state(
                    repo,
                    project,
                    feature_branch="feat/x",
                    feature_spec={"feature_name": "Dark Mode"},
                ),
                ui,
            )

        assert result["pr_url"] == pr
        ui.pr_opened.assert_called_once_with(pr)

    async def test_saves_artifact(self, repo, project, ui):
        """feature_report.md is saved to the artifacts directory."""
        (repo / "feature_report.md").write_text("# Saved Report")

        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project), ui)

        art = project / "artifacts" / "feature_report.md"
        assert art.read_text() == "# Saved Report"

    async def test_marks_stage_complete(self, repo, project, ui):
        """Stage 'verify' is recorded in metadata after success."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="report")
            await verify_node(_state(repo, project), ui)

        meta = json.loads((project / "metadata.json").read_text())
        assert "verify" in meta["stages_completed"]

    async def test_marks_project_done_with_pr_url(self, repo, project, ui):
        """When PR is opened, project is marked done with the URL."""
        pr = "https://github.com/org/repo/pull/7"
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=pr),
        ):
            mock_run.return_value = FakeAgentResult(text="report")
            await verify_node(_state(repo, project, feature_branch="feat/x"), ui)

        meta = json.loads((project / "metadata.json").read_text())
        assert meta["status"] == "completed"
        assert meta["pr_url"] == pr

    async def test_ui_lifecycle(self, repo, project, ui):
        """stage_start and stage_done are called in order."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="report")
            await verify_node(_state(repo, project), ui)

        ui.stage_start.assert_called_once_with("verify")
        ui.stage_done.assert_called_once_with("verify")


# ---------------------------------------------------------------------------
# verify_node — report file handling
# ---------------------------------------------------------------------------


class TestReportFileHandling:
    """Verify report file reading, fallback, and cleanup."""

    async def test_missing_report_falls_back_to_result_text(self, repo, project, ui):
        """When feature_report.md doesn't exist, result.text is used."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="agent fallback")
            result = await verify_node(_state(repo, project), ui)

        assert result["feature_report"] == "agent fallback"

    async def test_report_file_cleaned_up(self, repo, project, ui):
        """feature_report.md is deleted after being read."""
        (repo / "feature_report.md").write_text("# Report")

        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project), ui)

        assert not (repo / "feature_report.md").exists()

    async def test_no_report_file_no_cleanup_error(self, repo, project, ui):
        """When no report file exists, no unlink error is raised."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="fallback")
            # Should not raise
            await verify_node(_state(repo, project), ui)

    async def test_fallback_report_saved_as_artifact(self, repo, project, ui):
        """Even the fallback text is saved to artifacts."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="agent text only")
            await verify_node(_state(repo, project), ui)

        art = project / "artifacts" / "feature_report.md"
        assert art.read_text() == "agent text only"


# ---------------------------------------------------------------------------
# verify_node — git staging and commit
# ---------------------------------------------------------------------------


class TestGitStagingAndCommit:
    """Verify git add and commit calls when feature_branch is present."""

    async def test_git_add_called(self, repo, project, ui):
        """subprocess.run called with ['git', 'add', '-A']."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run") as mock_sub,
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_run.return_value = FakeAgentResult(text="report")
            await verify_node(_state(repo, project, feature_branch="feat/x"), ui)

        add_call = mock_sub.call_args_list[0]
        assert add_call == call(
            ["git", "add", "-A"],
            cwd=str(repo),
            capture_output=True,
            text=True,
        )

    async def test_git_commit_called(self, repo, project, ui):
        """subprocess.run called with commit message."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run") as mock_sub,
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_run.return_value = FakeAgentResult(text="report")
            await verify_node(_state(repo, project, feature_branch="feat/x"), ui)

        commit_call = mock_sub.call_args_list[1]
        assert commit_call == call(
            [
                "git",
                "commit",
                "-m",
                "chore: cleanup verification artifacts",
                "--allow-empty",
            ],
            cwd=str(repo),
            capture_output=True,
            text=True,
        )

    async def test_no_git_calls_without_branch(self, repo, project, ui):
        """Without feature_branch, no git add/commit/push calls."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run") as mock_sub,
        ):
            mock_run.return_value = FakeAgentResult(text="report")
            await verify_node(_state(repo, project), ui)

        mock_sub.assert_not_called()


# ---------------------------------------------------------------------------
# verify_node — PR opening
# ---------------------------------------------------------------------------


class TestPrOpening:
    """Verify PR opening logic and fallback messages."""

    async def test_pr_open_failure_shows_info(self, repo, project, ui):
        """When _open_pr returns None, fallback info messages are shown."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_run.return_value = FakeAgentResult(text="report")
            result = await verify_node(
                _state(repo, project, feature_branch="feat/x"), ui
            )

        assert result["pr_url"] == ""
        ui.pr_opened.assert_not_called()
        assert ui.info.call_count == 2
        # First message: instructions to push and open manually
        first_msg = ui.info.call_args_list[0][0][0]
        assert "feat/x" in first_msg
        assert "manually" in first_msg
        # Second message: report location
        second_msg = ui.info.call_args_list[1][0][0]
        assert "feature_report.md" in second_msg

    async def test_no_branch_skips_pr(self, repo, project, ui):
        """When feature_branch is missing, PR opening is skipped entirely."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="report")
            result = await verify_node(_state(repo, project), ui)

        assert result["pr_url"] == ""
        ui.pr_opened.assert_not_called()
        ui.info.assert_not_called()

    async def test_empty_branch_skips_pr(self, repo, project, ui):
        """An empty feature_branch string skips PR opening."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run") as mock_sub,
        ):
            mock_run.return_value = FakeAgentResult(text="report")
            result = await verify_node(_state(repo, project, feature_branch=""), ui)

        assert result["pr_url"] == ""
        mock_sub.assert_not_called()

    async def test_pr_not_marked_done_without_url(self, repo, project, ui):
        """Project is NOT marked done when PR URL is empty."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="report")
            await verify_node(_state(repo, project), ui)

        meta = json.loads((project / "metadata.json").read_text())
        assert meta.get("status") != "completed"
        assert "pr_url" not in meta

    async def test_open_pr_receives_feature_name(self, repo, project, ui):
        """_open_pr is called with 'Feature: <name>' as the title."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr") as mock_pr,
        ):
            mock_pr.return_value = None
            mock_run.return_value = FakeAgentResult(text="report")
            await verify_node(
                _state(
                    repo,
                    project,
                    feature_branch="feat/x",
                    feature_spec={"feature_name": "Dark Mode"},
                ),
                ui,
            )

        mock_pr.assert_called_once()
        _, _, title, _ = mock_pr.call_args[0]
        assert title == "Feature: Dark Mode"

    async def test_default_feature_name(self, repo, project, ui):
        """When feature_spec has no feature_name, defaults to 'Feature'."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr") as mock_pr,
        ):
            mock_pr.return_value = None
            mock_run.return_value = FakeAgentResult(text="report")
            await verify_node(_state(repo, project, feature_branch="feat/x"), ui)

        _, _, title, _ = mock_pr.call_args[0]
        assert title == "Feature: Feature"


# ---------------------------------------------------------------------------
# verify_node — run_agent invocation
# ---------------------------------------------------------------------------


class TestRunAgentArgs:
    """Verify run_agent is invoked with the correct arguments."""

    async def test_allowed_tools(self, repo, project, ui):
        """Only Bash, Read, Glob, Grep should be allowed."""
        with patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["allowed_tools"] == ["Bash", "Read", "Glob", "Grep"]

    async def test_system_prompt_is_constant(self, repo, project, ui):
        """run_agent receives the module-level SYSTEM_PROMPT."""
        with patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["system_prompt"] is SYSTEM_PROMPT

    async def test_stage_is_verify(self, repo, project, ui):
        """run_agent is called with stage='verify'."""
        with patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["stage"] == "verify"

    async def test_max_turns_is_30(self, repo, project, ui):
        """run_agent is called with max_turns=30."""
        with patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["max_turns"] == 30

    async def test_model_forwarded(self, repo, project, ui):
        """model from state is passed through to run_agent."""
        with patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(
                _state(repo, project, model="claude-sonnet-4-20250514"), ui
            )

        _, kwargs = mock_run.call_args
        assert kwargs["model"] == "claude-sonnet-4-20250514"

    async def test_persona(self, repo, project, ui):
        """run_agent is called with the Quality Engineer persona."""
        with patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["persona"] == "Principal Quality Engineer"

    async def test_cwd_is_repo_path(self, repo, project, ui):
        """run_agent receives repo_path as cwd."""
        with patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["cwd"] == str(repo)


# ---------------------------------------------------------------------------
# verify_node — prompt construction
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    """Verify the user prompt includes all required state data."""

    async def test_prompt_contains_repo_path(self, repo, project, ui):
        """The user prompt references the repo path."""
        with patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert str(repo) in kwargs["user_prompt"]

    async def test_prompt_contains_feature_prompt(self, repo, project, ui):
        """The user prompt includes the feature description."""
        with patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project, feature_prompt="Add dark mode"), ui)

        _, kwargs = mock_run.call_args
        assert "Add dark mode" in kwargs["user_prompt"]

    async def test_prompt_contains_build_plan(self, repo, project, ui):
        """The user prompt includes the build plan data."""
        plan = [{"unit_id": "u1", "title": "Create API"}]
        with patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project, build_plan=plan), ui)

        _, kwargs = mock_run.call_args
        assert "Create API" in kwargs["user_prompt"]
        assert "BUILD PLAN (1 units)" in kwargs["user_prompt"]

    async def test_prompt_contains_units_completed(self, repo, project, ui):
        """The user prompt includes completed units."""
        completed = [{"unit_id": "u1", "status": "done"}]
        with patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project, units_completed=completed), ui)

        _, kwargs = mock_run.call_args
        assert "UNITS COMPLETED (1)" in kwargs["user_prompt"]

    async def test_prompt_contains_units_reverted(self, repo, project, ui):
        """The user prompt includes reverted units."""
        reverted = [{"unit_id": "u2", "reason": "tests failed"}]
        with patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project, units_reverted=reverted), ui)

        _, kwargs = mock_run.call_args
        assert "UNITS REVERTED (1)" in kwargs["user_prompt"]

    async def test_prompt_contains_units_skipped(self, repo, project, ui):
        """The user prompt includes skipped units."""
        skipped = [{"unit_id": "u3"}]
        with patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project, units_skipped=skipped), ui)

        _, kwargs = mock_run.call_args
        assert "UNITS SKIPPED (1)" in kwargs["user_prompt"]

    async def test_prompt_defaults_empty_state(self, repo, project, ui):
        """Missing optional state keys default to empty values."""
        with patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        prompt = kwargs["user_prompt"]
        assert "UNITS COMPLETED (0)" in prompt
        assert "UNITS REVERTED (0)" in prompt
        assert "UNITS SKIPPED (0)" in prompt
        assert "BUILD PLAN (0 units)" in prompt
