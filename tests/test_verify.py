"""Tests for graft.stages.verify."""

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
    m.pr_opened = MagicMock()
    m.info = MagicMock()
    m.error = MagicMock()
    return m


def _state(repo, project, **kw):
    """Build a minimal FeatureState dict."""
    base = {
        "repo_path": str(repo),
        "project_dir": str(project),
    }
    base.update(kw)
    return base


def _make_completed_result(returncode=0, stdout="", stderr=""):
    """Build a subprocess.CompletedProcess for mocking."""
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


# ---------------------------------------------------------------------------
# _open_pr — success
# ---------------------------------------------------------------------------


class TestOpenPrSuccess:
    """Happy-path tests for _open_pr."""

    def test_returns_pr_url_on_success(self):
        """When git push and gh pr create both succeed, returns the PR URL."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _make_completed_result(0),  # git push
                _make_completed_result(
                    0, stdout="https://github.com/org/repo/pull/42\n"
                ),
            ]
            url = _open_pr("/repo", "feat/thing", "Title", "Body")

        assert url == "https://github.com/org/repo/pull/42"

    def test_git_push_called_correctly(self):
        """git push is called with correct args, cwd, and timeout."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _make_completed_result(0),
                _make_completed_result(0, stdout="https://url"),
            ]
            _open_pr("/my/repo", "feat/branch", "T", "B")

        push_call = mock_run.call_args_list[0]
        assert push_call == call(
            ["git", "push", "-u", "origin", "feat/branch"],
            cwd="/my/repo",
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )

    def test_gh_pr_create_called_correctly(self):
        """gh pr create is called with correct args, cwd, and timeout."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _make_completed_result(0),
                _make_completed_result(0, stdout="https://url"),
            ]
            _open_pr("/my/repo", "feat/branch", "My Title", "My Body")

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
                "feat/branch",
            ],
            cwd="/my/repo",
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_strips_trailing_whitespace_from_url(self):
        """PR URL output is stripped of trailing newlines/spaces."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _make_completed_result(0),
                _make_completed_result(0, stdout="  https://url  \n"),
            ]
            url = _open_pr("/repo", "b", "T", "B")

        assert url == "https://url"


# ---------------------------------------------------------------------------
# _open_pr — failure cases
# ---------------------------------------------------------------------------


class TestOpenPrFailure:
    """Failure and edge-case tests for _open_pr."""

    def test_returns_none_when_git_push_raises_called_process_error(self):
        """CalledProcessError from git push → returns None."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git push")
            url = _open_pr("/repo", "b", "T", "B")

        assert url is None

    def test_returns_none_when_git_push_file_not_found(self):
        """FileNotFoundError (git not installed) → returns None."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            url = _open_pr("/repo", "b", "T", "B")

        assert url is None

    def test_returns_none_when_git_push_timeout(self):
        """TimeoutExpired from git push → returns None."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("git push", 120)
            url = _open_pr("/repo", "b", "T", "B")

        assert url is None

    def test_returns_none_when_gh_pr_create_fails(self):
        """Non-zero returncode from gh pr create → returns None."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            mock_run.side_effect = [
                _make_completed_result(0),  # git push OK
                _make_completed_result(
                    1, stderr="already exists"
                ),  # gh pr create fails
            ]
            url = _open_pr("/repo", "b", "T", "B")

        assert url is None

    def test_returns_none_when_gh_not_found(self):
        """FileNotFoundError on gh pr create (gh not installed) → returns None."""
        call_count = 0

        def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_completed_result(0)  # git push
            raise FileNotFoundError("gh not found")

        with patch("graft.stages.verify.subprocess.run", side_effect=_side_effect):
            url = _open_pr("/repo", "b", "T", "B")

        assert url is None

    def test_returns_none_when_gh_pr_create_timeout(self):
        """TimeoutExpired on gh pr create → returns None."""
        call_count = 0

        def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_completed_result(0)  # git push
            raise subprocess.TimeoutExpired("gh pr create", 60)

        with patch("graft.stages.verify.subprocess.run", side_effect=_side_effect):
            url = _open_pr("/repo", "b", "T", "B")

        assert url is None

    def test_gh_pr_create_not_called_when_push_fails(self):
        """If git push fails, gh pr create should never be called."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git push")
            _open_pr("/repo", "b", "T", "B")

        # Only git push was attempted
        assert mock_run.call_count == 1


# ---------------------------------------------------------------------------
# verify_node — happy-path
# ---------------------------------------------------------------------------


class TestVerifyNodeHappyPath:
    """Core happy-path tests for verify_node."""

    async def test_returns_expected_keys(self, repo, project, ui):
        """verify_node returns feature_report, pr_url, current_stage."""
        (repo / "feature_report.md").write_text("# Report")

        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run") as mock_sub,
        ):
            mock_run.return_value = FakeAgentResult(text="fallback")
            result = await verify_node(_state(repo, project), ui)

        assert set(result.keys()) == {"feature_report", "pr_url", "current_stage"}
        assert result["current_stage"] == "verify"

    async def test_reads_feature_report_from_file(self, repo, project, ui):
        """When feature_report.md exists in repo, its content is used."""
        (repo / "feature_report.md").write_text("# My Report\nAll good.")

        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="agent fallback")
            result = await verify_node(_state(repo, project), ui)

        assert result["feature_report"] == "# My Report\nAll good."

    async def test_falls_back_to_agent_text(self, repo, project, ui):
        """When no feature_report.md exists, use agent result text."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="agent fallback text")
            result = await verify_node(_state(repo, project), ui)

        assert result["feature_report"] == "agent fallback text"

    async def test_cleans_up_feature_report(self, repo, project, ui):
        """feature_report.md is deleted from repo after being read."""
        (repo / "feature_report.md").write_text("# Report")

        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project), ui)

        assert not (repo / "feature_report.md").exists()

    async def test_saves_report_artifact(self, repo, project, ui):
        """feature_report.md is saved as an artifact in project dir."""
        (repo / "feature_report.md").write_text("# Saved Report")

        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project), ui)

        art_path = project / "artifacts" / "feature_report.md"
        assert art_path.exists()
        assert art_path.read_text() == "# Saved Report"

    async def test_saves_fallback_text_as_artifact(self, repo, project, ui):
        """When using agent fallback text, it's still saved as artifact."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="fallback content")
            await verify_node(_state(repo, project), ui)

        art_path = project / "artifacts" / "feature_report.md"
        assert art_path.exists()
        assert art_path.read_text() == "fallback content"


# ---------------------------------------------------------------------------
# verify_node — UI lifecycle
# ---------------------------------------------------------------------------


class TestVerifyNodeUILifecycle:
    """Verify UI methods are called correctly."""

    async def test_stage_start_and_done(self, repo, project, ui):
        """stage_start('verify') and stage_done('verify') are called."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project), ui)

        ui.stage_start.assert_called_once_with("verify")
        ui.stage_done.assert_called_once_with("verify")

    async def test_pr_opened_called_on_success(self, repo, project, ui):
        """When PR is opened successfully, ui.pr_opened is called with URL."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run") as mock_sub,
            patch(
                "graft.stages.verify._open_pr", return_value="https://github.com/pr/1"
            ),
        ):
            mock_run.return_value = FakeAgentResult(text="report")
            result = await verify_node(
                _state(
                    repo,
                    project,
                    feature_branch="feat/x",
                    feature_spec={"feature_name": "X"},
                ),
                ui,
            )

        ui.pr_opened.assert_called_once_with("https://github.com/pr/1")

    async def test_info_messages_when_pr_fails(self, repo, project, ui):
        """When _open_pr returns None, info messages are displayed."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_run.return_value = FakeAgentResult(text="report")
            await verify_node(
                _state(
                    repo,
                    project,
                    feature_branch="feat/x",
                    feature_spec={"feature_name": "X"},
                ),
                ui,
            )

        assert ui.info.call_count == 2
        # First info: manual push instructions
        assert "feat/x" in ui.info.call_args_list[0][0][0]
        # Second info: report location
        assert "feature_report.md" in ui.info.call_args_list[1][0][0]

    async def test_no_pr_opened_when_no_branch(self, repo, project, ui):
        """When no feature_branch is set, pr_opened is never called."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="report")
            await verify_node(_state(repo, project), ui)

        ui.pr_opened.assert_not_called()


# ---------------------------------------------------------------------------
# verify_node — PR opening and branch handling
# ---------------------------------------------------------------------------


class TestVerifyNodePR:
    """Tests for PR opening logic within verify_node."""

    async def test_opens_pr_with_branch(self, repo, project, ui):
        """When feature_branch is set, _open_pr is called and PR URL is returned."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run") as mock_sub,
            patch(
                "graft.stages.verify._open_pr", return_value="https://github.com/pr/99"
            ) as mock_pr,
        ):
            mock_run.return_value = FakeAgentResult(text="the report")
            result = await verify_node(
                _state(
                    repo,
                    project,
                    feature_branch="feat/dark-mode",
                    feature_spec={"feature_name": "Dark Mode"},
                ),
                ui,
            )

        mock_pr.assert_called_once_with(
            str(repo), "feat/dark-mode", "Feature: Dark Mode", "the report"
        )
        assert result["pr_url"] == "https://github.com/pr/99"

    async def test_pr_url_empty_when_no_branch(self, repo, project, ui):
        """Without feature_branch, pr_url is empty string."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="report")
            result = await verify_node(_state(repo, project), ui)

        assert result["pr_url"] == ""

    async def test_pr_url_empty_when_open_pr_fails(self, repo, project, ui):
        """When _open_pr returns None, pr_url is empty string."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_run.return_value = FakeAgentResult(text="report")
            result = await verify_node(
                _state(repo, project, feature_branch="feat/x"),
                ui,
            )

        assert result["pr_url"] == ""

    async def test_git_add_and_commit_before_pr(self, repo, project, ui):
        """When branch is set, git add -A and git commit are called before _open_pr."""
        subprocess_calls = []

        def _track_subprocess(*args, **kwargs):
            subprocess_calls.append(args[0] if args else kwargs.get("args"))
            return _make_completed_result(0)

        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run", side_effect=_track_subprocess),
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_run.return_value = FakeAgentResult(text="report")
            await verify_node(
                _state(repo, project, feature_branch="feat/x"),
                ui,
            )

        # Verify git add -A was called
        assert ["git", "add", "-A"] in subprocess_calls
        # Verify git commit was called
        commit_calls = [c for c in subprocess_calls if c[0:2] == ["git", "commit"]]
        assert len(commit_calls) == 1
        assert "--allow-empty" in commit_calls[0]

    async def test_pr_title_uses_feature_name(self, repo, project, ui):
        """PR title is 'Feature: <feature_name>' from feature_spec."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value="url") as mock_pr,
        ):
            mock_run.return_value = FakeAgentResult(text="rpt")
            await verify_node(
                _state(
                    repo,
                    project,
                    feature_branch="b",
                    feature_spec={"feature_name": "SSO Login"},
                ),
                ui,
            )

        assert mock_pr.call_args[0][2] == "Feature: SSO Login"

    async def test_pr_title_defaults_to_feature(self, repo, project, ui):
        """When feature_spec has no feature_name, PR title defaults to 'Feature: Feature'."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value="url") as mock_pr,
        ):
            mock_run.return_value = FakeAgentResult(text="rpt")
            await verify_node(
                _state(repo, project, feature_branch="b", feature_spec={}),
                ui,
            )

        assert mock_pr.call_args[0][2] == "Feature: Feature"

    async def test_pr_body_is_feature_report(self, repo, project, ui):
        """The PR body is the feature_report content."""
        (repo / "feature_report.md").write_text("# Detailed Report\nLooks great.")

        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value="url") as mock_pr,
        ):
            mock_run.return_value = FakeAgentResult(text="fallback")
            await verify_node(
                _state(repo, project, feature_branch="b"),
                ui,
            )

        assert mock_pr.call_args[0][3] == "# Detailed Report\nLooks great."

    async def test_no_subprocess_calls_when_no_branch(self, repo, project, ui):
        """Without a branch, no subprocess (git add/commit) calls are made."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run") as mock_sub,
        ):
            mock_run.return_value = FakeAgentResult(text="report")
            await verify_node(_state(repo, project), ui)

        mock_sub.assert_not_called()


# ---------------------------------------------------------------------------
# verify_node — metadata and artifacts
# ---------------------------------------------------------------------------


class TestVerifyNodeMetadata:
    """Tests for stage completion and project done marking."""

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

    async def test_marks_project_done_when_pr_opened(self, repo, project, ui):
        """When PR is opened, project metadata is marked as completed with URL."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
            patch(
                "graft.stages.verify._open_pr", return_value="https://github.com/pr/7"
            ),
        ):
            mock_run.return_value = FakeAgentResult(text="report")
            await verify_node(
                _state(repo, project, feature_branch="feat/x"),
                ui,
            )

        meta = json.loads((project / "metadata.json").read_text())
        assert meta["status"] == "completed"
        assert meta["pr_url"] == "https://github.com/pr/7"

    async def test_does_not_mark_project_done_when_no_pr(self, repo, project, ui):
        """When no PR is opened, project is NOT marked as done."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="report")
            await verify_node(_state(repo, project), ui)

        meta = json.loads((project / "metadata.json").read_text())
        assert meta.get("status") != "completed"
        assert "pr_url" not in meta

    async def test_does_not_mark_project_done_when_pr_fails(self, repo, project, ui):
        """When _open_pr returns None, project is NOT marked as done."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_run.return_value = FakeAgentResult(text="report")
            await verify_node(
                _state(repo, project, feature_branch="feat/x"),
                ui,
            )

        meta = json.loads((project / "metadata.json").read_text())
        assert meta.get("status") != "completed"


# ---------------------------------------------------------------------------
# verify_node — run_agent arguments
# ---------------------------------------------------------------------------


class TestRunAgentArgs:
    """Verify run_agent is invoked with the correct arguments."""

    async def test_allowed_tools(self, repo, project, ui):
        """run_agent is called with Bash, Read, Glob, Grep tools."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["allowed_tools"] == ["Bash", "Read", "Glob", "Grep"]

    async def test_system_prompt_is_constant(self, repo, project, ui):
        """run_agent receives the module-level SYSTEM_PROMPT."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["system_prompt"] is SYSTEM_PROMPT

    async def test_stage_is_verify(self, repo, project, ui):
        """run_agent is called with stage='verify'."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["stage"] == "verify"

    async def test_max_turns_is_30(self, repo, project, ui):
        """run_agent is called with max_turns=30."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["max_turns"] == 30

    async def test_cwd_is_repo_path(self, repo, project, ui):
        """run_agent cwd is set to repo_path."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["cwd"] == str(repo)

    async def test_model_forwarded(self, repo, project, ui):
        """model from state is passed through to run_agent."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(
                _state(repo, project, model="claude-sonnet-4-20250514"), ui
            )

        _, kwargs = mock_run.call_args
        assert kwargs["model"] == "claude-sonnet-4-20250514"

    async def test_persona_is_principal_qe(self, repo, project, ui):
        """run_agent persona is 'Principal Quality Engineer'."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["persona"] == "Principal Quality Engineer"


# ---------------------------------------------------------------------------
# verify_node — prompt construction
# ---------------------------------------------------------------------------


class TestPromptConstruction:
    """Verify the prompt sent to the agent contains the right data."""

    async def test_prompt_contains_repo_path(self, repo, project, ui):
        """The user prompt must reference the repo path."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert str(repo) in kwargs["user_prompt"]

    async def test_prompt_contains_feature_prompt(self, repo, project, ui):
        """The user prompt includes the feature_prompt."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project, feature_prompt="Add dark mode"), ui)

        _, kwargs = mock_run.call_args
        assert "Add dark mode" in kwargs["user_prompt"]

    async def test_prompt_contains_build_plan(self, repo, project, ui):
        """Build plan is serialized into the prompt."""
        plan = [{"unit": "auth", "tasks": ["login", "logout"]}]
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project, build_plan=plan), ui)

        _, kwargs = mock_run.call_args
        assert "auth" in kwargs["user_prompt"]
        assert (
            "1 units" in kwargs["user_prompt"]
            or "BUILD PLAN (1 units)" in kwargs["user_prompt"]
        )

    async def test_prompt_contains_units_completed(self, repo, project, ui):
        """Completed units are included in the prompt."""
        completed = [{"unit": "auth", "status": "done"}]
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project, units_completed=completed), ui)

        _, kwargs = mock_run.call_args
        assert "UNITS COMPLETED (1)" in kwargs["user_prompt"]

    async def test_prompt_contains_units_reverted(self, repo, project, ui):
        """Reverted units are included in the prompt."""
        reverted = [{"unit": "auth", "reason": "broken"}]
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project, units_reverted=reverted), ui)

        _, kwargs = mock_run.call_args
        assert "UNITS REVERTED (1)" in kwargs["user_prompt"]

    async def test_prompt_contains_units_skipped(self, repo, project, ui):
        """Skipped units are included in the prompt."""
        skipped = [{"unit": "perf"}, {"unit": "cache"}]
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project, units_skipped=skipped), ui)

        _, kwargs = mock_run.call_args
        assert "UNITS SKIPPED (2)" in kwargs["user_prompt"]

    async def test_prompt_contains_codebase_profile(self, repo, project, ui):
        """Codebase profile is serialized into the prompt."""
        profile = {"language": "python", "framework": "fastapi"}
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project, codebase_profile=profile), ui)

        _, kwargs = mock_run.call_args
        assert "fastapi" in kwargs["user_prompt"]
        assert "CODEBASE PROFILE" in kwargs["user_prompt"]

    async def test_prompt_contains_feature_spec(self, repo, project, ui):
        """Feature spec is serialized into the prompt."""
        spec = {"feature_name": "Dark Mode", "scope": "UI only"}
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project, feature_spec=spec), ui)

        _, kwargs = mock_run.call_args
        assert "Dark Mode" in kwargs["user_prompt"]
        assert "FEATURE SPEC" in kwargs["user_prompt"]

    async def test_prompt_defaults_for_missing_state(self, repo, project, ui):
        """When optional state keys are missing, defaults are used without error."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="")
            # Minimal state — no optional keys
            result = await verify_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        prompt = kwargs["user_prompt"]
        # Defaults: empty string/list/dict
        assert "FEATURE: \n" in prompt
        assert "BUILD PLAN (0 units)" in prompt
        assert "UNITS COMPLETED (0)" in prompt
        assert "UNITS REVERTED (0)" in prompt
        assert "UNITS SKIPPED (0)" in prompt

    async def test_prompt_ends_with_instruction(self, repo, project, ui):
        """The prompt ends with the instruction to produce feature_report.md."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run"),
        ):
            mock_run.return_value = FakeAgentResult(text="")
            await verify_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["user_prompt"].endswith(
            "Run all checks and produce feature_report.md."
        )


# ---------------------------------------------------------------------------
# verify_node — empty branch edge case
# ---------------------------------------------------------------------------


class TestVerifyNodeEmptyBranch:
    """Edge cases around empty/missing branch."""

    async def test_empty_string_branch_skips_pr(self, repo, project, ui):
        """An empty string feature_branch means no PR attempt."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run") as mock_sub,
        ):
            mock_run.return_value = FakeAgentResult(text="report")
            result = await verify_node(_state(repo, project, feature_branch=""), ui)

        assert result["pr_url"] == ""
        mock_sub.assert_not_called()
        ui.pr_opened.assert_not_called()

    async def test_missing_branch_key_skips_pr(self, repo, project, ui):
        """When feature_branch is not in state at all, no PR attempt."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run") as mock_sub,
        ):
            mock_run.return_value = FakeAgentResult(text="report")
            result = await verify_node(_state(repo, project), ui)

        assert result["pr_url"] == ""
        mock_sub.assert_not_called()


# ---------------------------------------------------------------------------
# verify_node — full integration scenario
# ---------------------------------------------------------------------------


class TestVerifyNodeIntegration:
    """End-to-end scenarios combining multiple behaviors."""

    async def test_full_happy_path_with_pr(self, repo, project, ui):
        """Full scenario: report file exists, branch set, PR opens, project marked done."""
        (repo / "feature_report.md").write_text("# Full Report\nEverything passes.")

        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run") as mock_sub,
            patch(
                "graft.stages.verify._open_pr",
                return_value="https://github.com/org/repo/pull/100",
            ),
        ):
            mock_run.return_value = FakeAgentResult(text="fallback")
            result = await verify_node(
                _state(
                    repo,
                    project,
                    feature_branch="feat/awesome",
                    feature_spec={"feature_name": "Awesome Feature"},
                    feature_prompt="Make it awesome",
                    build_plan=[{"unit": "core"}],
                    units_completed=[{"unit": "core"}],
                    units_reverted=[],
                    units_skipped=[],
                ),
                ui,
            )

        # Result correctness
        assert result["feature_report"] == "# Full Report\nEverything passes."
        assert result["pr_url"] == "https://github.com/org/repo/pull/100"
        assert result["current_stage"] == "verify"

        # File cleanup
        assert not (repo / "feature_report.md").exists()

        # Artifact saved
        assert (
            project / "artifacts" / "feature_report.md"
        ).read_text() == "# Full Report\nEverything passes."

        # Metadata
        meta = json.loads((project / "metadata.json").read_text())
        assert "verify" in meta["stages_completed"]
        assert meta["status"] == "completed"
        assert meta["pr_url"] == "https://github.com/org/repo/pull/100"

        # UI lifecycle
        ui.stage_start.assert_called_once_with("verify")
        ui.stage_done.assert_called_once_with("verify")
        ui.pr_opened.assert_called_once_with("https://github.com/org/repo/pull/100")

    async def test_full_path_no_branch_no_pr(self, repo, project, ui):
        """Full scenario: no branch, no PR, report from agent text, project not marked done."""
        with (
            patch("graft.stages.verify.run_agent", new_callable=AsyncMock) as mock_run,
            patch("graft.stages.verify.subprocess.run") as mock_sub,
        ):
            mock_run.return_value = FakeAgentResult(text="Agent generated report")
            result = await verify_node(
                _state(repo, project, feature_prompt="Add login"),
                ui,
            )

        assert result["feature_report"] == "Agent generated report"
        assert result["pr_url"] == ""

        # Artifact saved
        assert (
            project / "artifacts" / "feature_report.md"
        ).read_text() == "Agent generated report"

        # Metadata — stage complete but NOT project done
        meta = json.loads((project / "metadata.json").read_text())
        assert "verify" in meta["stages_completed"]
        assert meta.get("status") != "completed"

        # No subprocess calls
        mock_sub.assert_not_called()
        ui.pr_opened.assert_not_called()
