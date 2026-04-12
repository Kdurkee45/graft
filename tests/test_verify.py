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
    """Build a minimal FeatureState dict for verify_node."""
    base = {
        "repo_path": str(repo),
        "project_dir": str(project),
        "feature_prompt": "Add dark mode",
        "codebase_profile": {"project": {"name": "acme"}},
        "feature_spec": {"feature_name": "Dark Mode"},
        "build_plan": [{"unit": "theme-toggle", "description": "Toggle button"}],
        "units_completed": [{"unit": "theme-toggle"}],
        "units_reverted": [],
        "units_skipped": [],
        "feature_branch": "feature/feat_test01",
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# _open_pr tests
# ---------------------------------------------------------------------------


class TestOpenPr:
    """Tests for the _open_pr helper that calls git push + gh pr create."""

    def test_happy_path_returns_pr_url(self):
        """Successful push and PR creation returns the PR URL."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            push_result = MagicMock(returncode=0)
            pr_result = MagicMock(
                returncode=0, stdout="https://github.com/org/repo/pull/42\n"
            )
            mock_run.side_effect = [push_result, pr_result]

            url = _open_pr("/repo", "feature/feat_01", "Feature: X", "body text")

        assert url == "https://github.com/org/repo/pull/42"

    def test_git_push_called_with_correct_args(self):
        """git push is invoked with -u origin <branch>."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            push_result = MagicMock(returncode=0)
            pr_result = MagicMock(returncode=0, stdout="https://url\n")
            mock_run.side_effect = [push_result, pr_result]

            _open_pr("/my/repo", "feature/feat_abc", "Title", "Body")

        push_call = mock_run.call_args_list[0]
        assert push_call == call(
            ["git", "push", "-u", "origin", "feature/feat_abc"],
            cwd="/my/repo",
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )

    def test_gh_pr_create_called_with_correct_args(self):
        """gh pr create is invoked with --title, --body, --head."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            push_result = MagicMock(returncode=0)
            pr_result = MagicMock(returncode=0, stdout="https://url\n")
            mock_run.side_effect = [push_result, pr_result]

            _open_pr("/repo", "feature/feat_abc", "My Title", "My Body")

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
                "feature/feat_abc",
            ],
            cwd="/repo",
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_gh_pr_create_nonzero_exit_returns_none(self):
        """Non-zero exit code from gh pr create returns None."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            push_result = MagicMock(returncode=0)
            pr_result = MagicMock(returncode=1, stdout="", stderr="error")
            mock_run.side_effect = [push_result, pr_result]

            url = _open_pr("/repo", "branch", "Title", "Body")

        assert url is None

    def test_git_push_called_process_error_returns_none(self):
        """CalledProcessError from git push returns None."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                1, "git push", stderr="rejected"
            )

            url = _open_pr("/repo", "branch", "Title", "Body")

        assert url is None

    def test_file_not_found_returns_none(self):
        """FileNotFoundError (gh/git not installed) returns None."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")

            url = _open_pr("/repo", "branch", "Title", "Body")

        assert url is None

    def test_timeout_expired_returns_none(self):
        """TimeoutExpired on git push returns None."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("git push", 120)

            url = _open_pr("/repo", "branch", "Title", "Body")

        assert url is None

    def test_pr_url_stripped_of_whitespace(self):
        """Trailing whitespace and newlines are stripped from the PR URL."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            push_result = MagicMock(returncode=0)
            pr_result = MagicMock(
                returncode=0, stdout="  https://github.com/repo/pull/7  \n"
            )
            mock_run.side_effect = [push_result, pr_result]

            url = _open_pr("/repo", "branch", "Title", "Body")

        assert url == "https://github.com/repo/pull/7"

    def test_push_timeout_is_120_seconds(self):
        """git push has a 120s timeout."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            push_result = MagicMock(returncode=0)
            pr_result = MagicMock(returncode=0, stdout="url\n")
            mock_run.side_effect = [push_result, pr_result]

            _open_pr("/repo", "branch", "Title", "Body")

        assert mock_run.call_args_list[0].kwargs["timeout"] == 120

    def test_gh_timeout_is_60_seconds(self):
        """gh pr create has a 60s timeout."""
        with patch("graft.stages.verify.subprocess.run") as mock_run:
            push_result = MagicMock(returncode=0)
            pr_result = MagicMock(returncode=0, stdout="url\n")
            mock_run.side_effect = [push_result, pr_result]

            _open_pr("/repo", "branch", "Title", "Body")

        assert mock_run.call_args_list[1].kwargs["timeout"] == 60


# ---------------------------------------------------------------------------
# verify_node — happy path
# ---------------------------------------------------------------------------


class TestVerifyNodeHappyPath:
    """Core happy-path tests for verify_node."""

    async def test_returns_expected_keys(self, repo, project, ui):
        """verify_node returns feature_report, pr_url, and current_stage."""
        (repo / "feature_report.md").write_text("# Report")

        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run") as mock_sub,
            patch("graft.stages.verify._open_pr") as mock_pr,
        ):
            mock_agent.return_value = FakeAgentResult(text="fallback")
            mock_pr.return_value = "https://github.com/org/repo/pull/99"

            result = await verify_node(_state(repo, project), ui)

        assert set(result.keys()) == {"feature_report", "pr_url", "current_stage"}

    async def test_current_stage_is_verify(self, repo, project, ui):
        """current_stage is always 'verify'."""
        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_agent.return_value = FakeAgentResult(text="fallback")

            result = await verify_node(_state(repo, project), ui)

        assert result["current_stage"] == "verify"

    async def test_reads_report_from_disk_when_present(self, repo, project, ui):
        """When feature_report.md exists in repo, its content is used."""
        (repo / "feature_report.md").write_text("# Disk Report\nAll tests pass.")

        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_agent.return_value = FakeAgentResult(text="agent fallback text")

            result = await verify_node(_state(repo, project), ui)

        assert result["feature_report"] == "# Disk Report\nAll tests pass."

    async def test_falls_back_to_agent_text_when_no_report_file(
        self, repo, project, ui
    ):
        """When feature_report.md doesn't exist, agent result text is used."""
        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_agent.return_value = FakeAgentResult(text="agent fallback text")

            result = await verify_node(_state(repo, project), ui)

        assert result["feature_report"] == "agent fallback text"

    async def test_report_file_cleaned_up_after_read(self, repo, project, ui):
        """feature_report.md is deleted from repo after being read."""
        report_path = repo / "feature_report.md"
        report_path.write_text("# Report")

        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_agent.return_value = FakeAgentResult(text="fallback")

            await verify_node(_state(repo, project), ui)

        assert not report_path.exists()

    async def test_report_saved_as_artifact(self, repo, project, ui):
        """Feature report is persisted to the project artifacts directory."""
        (repo / "feature_report.md").write_text("# Artifact Report")

        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_agent.return_value = FakeAgentResult(text="fallback")

            await verify_node(_state(repo, project), ui)

        artifact = project / "artifacts" / "feature_report.md"
        assert artifact.exists()
        assert artifact.read_text() == "# Artifact Report"


# ---------------------------------------------------------------------------
# verify_node — PR opening
# ---------------------------------------------------------------------------


class TestVerifyNodePrOpening:
    """Tests for the PR-opening flow inside verify_node."""

    async def test_pr_opened_successfully(self, repo, project, ui):
        """When _open_pr returns a URL, pr_url is set and ui.pr_opened called."""
        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr") as mock_pr,
        ):
            mock_agent.return_value = FakeAgentResult(text="report")
            mock_pr.return_value = "https://github.com/org/repo/pull/42"

            result = await verify_node(_state(repo, project), ui)

        assert result["pr_url"] == "https://github.com/org/repo/pull/42"
        ui.pr_opened.assert_called_once_with("https://github.com/org/repo/pull/42")

    async def test_pr_title_includes_feature_name(self, repo, project, ui):
        """PR title is 'Feature: <feature_name>' from spec."""
        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr") as mock_pr,
        ):
            mock_agent.return_value = FakeAgentResult(text="report")
            mock_pr.return_value = "https://url"

            await verify_node(
                _state(
                    repo,
                    project,
                    feature_spec={"feature_name": "Custom Widget"},
                ),
                ui,
            )

        mock_pr.assert_called_once()
        _, pr_title, _ = mock_pr.call_args[0][1:]  # branch, title, body
        assert pr_title == "Feature: Custom Widget"

    async def test_pr_title_defaults_to_feature_when_no_name(
        self, repo, project, ui
    ):
        """When feature_name is missing from spec, default to 'Feature'."""
        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr") as mock_pr,
        ):
            mock_agent.return_value = FakeAgentResult(text="report")
            mock_pr.return_value = "https://url"

            await verify_node(
                _state(repo, project, feature_spec={}),
                ui,
            )

        _, pr_title, _ = mock_pr.call_args[0][1:]
        assert pr_title == "Feature: Feature"

    async def test_pr_open_failure_shows_info_messages(self, repo, project, ui):
        """When _open_pr returns None, ui.info is called with manual instructions."""
        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_agent.return_value = FakeAgentResult(text="report")

            result = await verify_node(_state(repo, project), ui)

        assert result["pr_url"] == ""
        assert ui.info.call_count == 2
        first_msg = ui.info.call_args_list[0][0][0]
        assert "feature/feat_test01" in first_msg
        assert "manually" in first_msg

    async def test_no_branch_skips_pr_entirely(self, repo, project, ui):
        """When feature_branch is empty, no PR is attempted."""
        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run") as mock_sub,
            patch("graft.stages.verify._open_pr") as mock_pr,
        ):
            mock_agent.return_value = FakeAgentResult(text="report")

            result = await verify_node(
                _state(repo, project, feature_branch=""), ui
            )

        mock_pr.assert_not_called()
        assert result["pr_url"] == ""
        # git add / git commit should also be skipped
        mock_sub.assert_not_called()

    async def test_open_pr_receives_branch_name(self, repo, project, ui):
        """_open_pr is called with the feature_branch from state."""
        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr") as mock_pr,
        ):
            mock_agent.return_value = FakeAgentResult(text="report")
            mock_pr.return_value = None

            await verify_node(
                _state(repo, project, feature_branch="feature/feat_xyz"),
                ui,
            )

        assert mock_pr.call_args[0][1] == "feature/feat_xyz"


# ---------------------------------------------------------------------------
# verify_node — git staging and commit
# ---------------------------------------------------------------------------


class TestVerifyNodeGitStaging:
    """Tests for git add/commit before PR opening."""

    async def test_git_add_and_commit_called_when_branch_present(
        self, repo, project, ui
    ):
        """git add -A and git commit --allow-empty are called when branch exists."""
        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run") as mock_sub,
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_agent.return_value = FakeAgentResult(text="report")

            await verify_node(_state(repo, project), ui)

        # Two subprocess calls: git add, git commit
        assert mock_sub.call_count == 2

        add_call = mock_sub.call_args_list[0]
        assert add_call[0][0] == ["git", "add", "-A"]
        assert add_call[1]["cwd"] == str(repo)

        commit_call = mock_sub.call_args_list[1]
        assert commit_call[0][0] == [
            "git",
            "commit",
            "-m",
            "chore: cleanup verification artifacts",
            "--allow-empty",
        ]
        assert commit_call[1]["cwd"] == str(repo)

    async def test_allow_empty_flag_present_in_commit(self, repo, project, ui):
        """The commit uses --allow-empty so it succeeds even with no changes."""
        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run") as mock_sub,
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_agent.return_value = FakeAgentResult(text="report")

            await verify_node(_state(repo, project), ui)

        commit_cmd = mock_sub.call_args_list[1][0][0]
        assert "--allow-empty" in commit_cmd


# ---------------------------------------------------------------------------
# verify_node — UI lifecycle
# ---------------------------------------------------------------------------


class TestVerifyNodeUILifecycle:
    """Tests that verify_node calls UI methods in the right order."""

    async def test_stage_start_and_done_called(self, repo, project, ui):
        """stage_start('verify') and stage_done('verify') bracket the work."""
        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_agent.return_value = FakeAgentResult(text="report")

            await verify_node(_state(repo, project), ui)

        ui.stage_start.assert_called_once_with("verify")
        ui.stage_done.assert_called_once_with("verify")

    async def test_pr_opened_not_called_when_no_url(self, repo, project, ui):
        """ui.pr_opened is not called when _open_pr returns None."""
        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_agent.return_value = FakeAgentResult(text="report")

            await verify_node(_state(repo, project), ui)

        ui.pr_opened.assert_not_called()


# ---------------------------------------------------------------------------
# verify_node — metadata / artifact persistence
# ---------------------------------------------------------------------------


class TestVerifyNodePersistence:
    """Tests that verify_node persists metadata and artifacts correctly."""

    async def test_mark_stage_complete_called(self, repo, project, ui):
        """verify stage is marked complete in project metadata."""
        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_agent.return_value = FakeAgentResult(text="report")

            await verify_node(_state(repo, project), ui)

        meta = json.loads((project / "metadata.json").read_text())
        assert "verify" in meta["stages_completed"]

    async def test_mark_project_done_on_pr_success(self, repo, project, ui):
        """Project is marked done with PR URL when PR opens successfully."""
        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run"),
            patch(
                "graft.stages.verify._open_pr",
                return_value="https://github.com/org/repo/pull/99",
            ),
        ):
            mock_agent.return_value = FakeAgentResult(text="report")

            await verify_node(_state(repo, project), ui)

        meta = json.loads((project / "metadata.json").read_text())
        assert meta["status"] == "completed"
        assert meta["pr_url"] == "https://github.com/org/repo/pull/99"

    async def test_project_not_marked_done_when_pr_fails(self, repo, project, ui):
        """Project status stays in_progress when no PR URL."""
        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_agent.return_value = FakeAgentResult(text="report")

            await verify_node(_state(repo, project), ui)

        meta = json.loads((project / "metadata.json").read_text())
        assert meta["status"] == "in_progress"
        assert "pr_url" not in meta


# ---------------------------------------------------------------------------
# verify_node — agent invocation
# ---------------------------------------------------------------------------


class TestVerifyNodeAgentInvocation:
    """Tests that run_agent is called with correct parameters."""

    async def test_agent_called_with_verify_stage(self, repo, project, ui):
        """run_agent is called with stage='verify'."""
        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_agent.return_value = FakeAgentResult(text="report")

            await verify_node(_state(repo, project), ui)

        kwargs = mock_agent.call_args[1]
        assert kwargs["stage"] == "verify"

    async def test_agent_called_with_correct_persona(self, repo, project, ui):
        """run_agent persona is Principal Quality Engineer."""
        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_agent.return_value = FakeAgentResult(text="report")

            await verify_node(_state(repo, project), ui)

        kwargs = mock_agent.call_args[1]
        assert kwargs["persona"] == "Principal Quality Engineer"

    async def test_agent_allowed_tools(self, repo, project, ui):
        """run_agent is given Bash, Read, Glob, Grep tools only."""
        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_agent.return_value = FakeAgentResult(text="report")

            await verify_node(_state(repo, project), ui)

        kwargs = mock_agent.call_args[1]
        assert kwargs["allowed_tools"] == ["Bash", "Read", "Glob", "Grep"]

    async def test_agent_receives_system_prompt(self, repo, project, ui):
        """run_agent uses the module-level SYSTEM_PROMPT."""
        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_agent.return_value = FakeAgentResult(text="report")

            await verify_node(_state(repo, project), ui)

        kwargs = mock_agent.call_args[1]
        assert kwargs["system_prompt"] == SYSTEM_PROMPT

    async def test_agent_max_turns_is_30(self, repo, project, ui):
        """run_agent max_turns is 30."""
        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_agent.return_value = FakeAgentResult(text="report")

            await verify_node(_state(repo, project), ui)

        kwargs = mock_agent.call_args[1]
        assert kwargs["max_turns"] == 30

    async def test_agent_prompt_includes_build_plan_count(self, repo, project, ui):
        """User prompt includes the number of build plan units."""
        plan = [{"unit": "a"}, {"unit": "b"}, {"unit": "c"}]
        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_agent.return_value = FakeAgentResult(text="report")

            await verify_node(_state(repo, project, build_plan=plan), ui)

        kwargs = mock_agent.call_args[1]
        assert "BUILD PLAN (3 units)" in kwargs["user_prompt"]

    async def test_agent_prompt_includes_units_completed_count(
        self, repo, project, ui
    ):
        """User prompt includes the count of completed units."""
        completed = [{"unit": "x"}, {"unit": "y"}]
        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_agent.return_value = FakeAgentResult(text="report")

            await verify_node(
                _state(repo, project, units_completed=completed), ui
            )

        kwargs = mock_agent.call_args[1]
        assert "UNITS COMPLETED (2)" in kwargs["user_prompt"]

    async def test_agent_model_forwarded_from_state(self, repo, project, ui):
        """The model setting from state is forwarded to run_agent."""
        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_agent.return_value = FakeAgentResult(text="report")

            await verify_node(
                _state(repo, project, model="claude-sonnet-4-20250514"), ui
            )

        kwargs = mock_agent.call_args[1]
        assert kwargs["model"] == "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# verify_node — state defaults
# ---------------------------------------------------------------------------


class TestVerifyNodeStateDefaults:
    """Tests for missing/empty state fields."""

    async def test_missing_feature_prompt_defaults_to_empty(
        self, repo, project, ui
    ):
        """feature_prompt defaults to empty string when not in state."""
        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_agent.return_value = FakeAgentResult(text="report")

            state = {
                "repo_path": str(repo),
                "project_dir": str(project),
            }
            result = await verify_node(state, ui)

        kwargs = mock_agent.call_args[1]
        assert "FEATURE: \n" in kwargs["user_prompt"]

    async def test_missing_optional_state_fields(self, repo, project, ui):
        """verify_node handles missing optional state fields gracefully."""
        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run"),
            patch("graft.stages.verify._open_pr", return_value=None),
        ):
            mock_agent.return_value = FakeAgentResult(text="report")

            state = {
                "repo_path": str(repo),
                "project_dir": str(project),
            }
            # Should not raise
            result = await verify_node(state, ui)

        assert result["current_stage"] == "verify"
        assert result["pr_url"] == ""

    async def test_no_feature_branch_in_state(self, repo, project, ui):
        """When feature_branch is absent, pr_url is empty."""
        with (
            patch(
                "graft.stages.verify.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.verify.subprocess.run") as mock_sub,
            patch("graft.stages.verify._open_pr") as mock_pr,
        ):
            mock_agent.return_value = FakeAgentResult(text="report")

            state = {
                "repo_path": str(repo),
                "project_dir": str(project),
            }
            result = await verify_node(state, ui)

        mock_pr.assert_not_called()
        mock_sub.assert_not_called()
        assert result["pr_url"] == ""
