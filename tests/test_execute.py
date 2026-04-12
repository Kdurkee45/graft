"""Tests for graft.stages.execute."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graft.stages.execute import (
    VERIFY_SCRIPT,
    _git,
    _order_by_dependencies,
    _run_lint,
    _run_tests,
    execute_node,
)

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
    """Mock UI object exposing the methods execute_node calls."""
    m = MagicMock()
    m.stage_start = MagicMock()
    m.stage_done = MagicMock()
    m.error = MagicMock()
    m.unit_start = MagicMock()
    m.unit_kept = MagicMock()
    m.unit_reverted = MagicMock()
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
# Existing _order_by_dependencies tests
# ---------------------------------------------------------------------------


def test_order_by_dependencies_no_deps():
    """Units without dependencies maintain order."""
    plan = [
        {"unit_id": "a"},
        {"unit_id": "b"},
        {"unit_id": "c"},
    ]
    result = _order_by_dependencies(plan)
    assert [u["unit_id"] for u in result] == ["a", "b", "c"]


def test_order_by_dependencies_simple_chain():
    """Units with linear dependencies are ordered correctly."""
    plan = [
        {"unit_id": "c", "depends_on": ["b"]},
        {"unit_id": "a", "depends_on": []},
        {"unit_id": "b", "depends_on": ["a"]},
    ]
    result = _order_by_dependencies(plan)
    ids = [u["unit_id"] for u in result]
    assert ids.index("a") < ids.index("b") < ids.index("c")


def test_order_by_dependencies_circular():
    """Circular dependencies don't infinite loop — remaining units appended."""
    plan = [
        {"unit_id": "a", "depends_on": ["b"]},
        {"unit_id": "b", "depends_on": ["a"]},
    ]
    result = _order_by_dependencies(plan)
    assert len(result) == 2


def test_order_by_dependencies_multiple_deps():
    """Unit with multiple dependencies waits for all."""
    plan = [
        {"unit_id": "c", "depends_on": ["a", "b"]},
        {"unit_id": "a", "depends_on": []},
        {"unit_id": "b", "depends_on": []},
    ]
    result = _order_by_dependencies(plan)
    ids = [u["unit_id"] for u in result]
    assert ids.index("c") > ids.index("a")
    assert ids.index("c") > ids.index("b")


# ---------------------------------------------------------------------------
# _git
# ---------------------------------------------------------------------------


class TestGit:
    """Tests for the _git subprocess wrapper."""

    @patch("graft.stages.execute.subprocess.run")
    def test_passes_correct_args(self, mock_run):
        mock_run.return_value = _completed()
        _git("/some/repo", "status")

        mock_run.assert_called_once_with(
            ["git", "status"],
            cwd="/some/repo",
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )

    @patch("graft.stages.execute.subprocess.run")
    def test_multiple_args(self, mock_run):
        mock_run.return_value = _completed()
        _git("/repo", "commit", "-m", "feat: title")

        mock_run.assert_called_once_with(
            ["git", "commit", "-m", "feat: title"],
            cwd="/repo",
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )

    @patch("graft.stages.execute.subprocess.run")
    def test_check_false(self, mock_run):
        mock_run.return_value = _completed(returncode=1)
        result = _git("/repo", "checkout", "-b", "branch", check=False)

        assert result.returncode == 1
        mock_run.assert_called_once_with(
            ["git", "checkout", "-b", "branch"],
            cwd="/repo",
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    @patch("graft.stages.execute.subprocess.run")
    def test_check_true_raises_on_failure(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")
        with pytest.raises(subprocess.CalledProcessError):
            _git("/repo", "add", "-A")

    @patch("graft.stages.execute.subprocess.run")
    def test_returns_completed_process(self, mock_run):
        expected = _completed(stdout="on branch main")
        mock_run.return_value = expected
        result = _git("/repo", "status")
        assert result is expected


# ---------------------------------------------------------------------------
# _run_tests
# ---------------------------------------------------------------------------


class TestRunTests:
    """Tests for the _run_tests helper."""

    @patch("graft.stages.execute.subprocess.run")
    def test_passing_tests(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="5 passed", stderr="")
        passed, output = _run_tests("/repo")

        assert passed is True
        assert "5 passed" in output
        mock_run.assert_called_once_with(
            ["bash", "-c", VERIFY_SCRIPT],
            cwd="/repo",
            capture_output=True,
            text=True,
            timeout=300,
        )

    @patch("graft.stages.execute.subprocess.run")
    def test_failing_tests(self, mock_run):
        mock_run.return_value = _completed(
            returncode=1, stdout="", stderr="FAILED test_foo"
        )
        passed, output = _run_tests("/repo")

        assert passed is False
        assert "FAILED test_foo" in output

    @patch("graft.stages.execute.subprocess.run")
    def test_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="bash", timeout=300)
        passed, output = _run_tests("/repo")

        assert passed is False
        assert "timed out" in output.lower()
        assert "300s" in output

    @patch("graft.stages.execute.subprocess.run")
    def test_file_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError("bash not found")
        passed, output = _run_tests("/repo")

        assert passed is True
        assert "skipping" in output.lower()

    @patch("graft.stages.execute.subprocess.run")
    def test_output_truncation(self, mock_run):
        """Output longer than 2000 chars is truncated to last 2000."""
        long_output = "x" * 3000
        mock_run.return_value = _completed(returncode=0, stdout=long_output, stderr="")
        passed, output = _run_tests("/repo")

        assert passed is True
        assert len(output) == 2000

    @patch("graft.stages.execute.subprocess.run")
    def test_combines_stdout_and_stderr(self, mock_run):
        mock_run.return_value = _completed(
            returncode=0, stdout="stdout part", stderr="\nstderr part"
        )
        passed, output = _run_tests("/repo")

        assert passed is True
        assert "stdout part" in output
        assert "stderr part" in output


# ---------------------------------------------------------------------------
# _run_lint
# ---------------------------------------------------------------------------


class TestRunLint:
    """Tests for the _run_lint helper."""

    @patch("graft.stages.execute.subprocess.run")
    def test_first_linter_succeeds(self, mock_run):
        """eslint succeeds on first try — no further linters attempted."""
        mock_run.return_value = _completed(returncode=0)
        passed, output = _run_lint("/repo")

        assert passed is True
        assert output == "Lint passed"
        # Only eslint was called (first in the list)
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "npx"
        assert "eslint" in args

    @patch("graft.stages.execute.subprocess.run")
    def test_second_linter_succeeds(self, mock_run):
        """eslint fails (non-zero), ruff succeeds."""
        mock_run.side_effect = [
            _completed(returncode=1),  # eslint fails
            _completed(returncode=0),  # ruff succeeds
        ]
        passed, output = _run_lint("/repo")

        assert passed is True
        assert output == "Lint passed"
        assert mock_run.call_count == 2

    @patch("graft.stages.execute.subprocess.run")
    def test_third_linter_succeeds(self, mock_run):
        """eslint and ruff fail, prettier succeeds."""
        mock_run.side_effect = [
            _completed(returncode=1),  # eslint fails
            _completed(returncode=1),  # ruff fails
            _completed(returncode=0),  # prettier succeeds
        ]
        passed, output = _run_lint("/repo")

        assert passed is True
        assert output == "Lint passed"
        assert mock_run.call_count == 3

    @patch("graft.stages.execute.subprocess.run")
    def test_all_linters_fail(self, mock_run):
        """All three linters return non-zero — fallback message."""
        mock_run.side_effect = [
            _completed(returncode=1),
            _completed(returncode=1),
            _completed(returncode=1),
        ]
        passed, output = _run_lint("/repo")

        assert passed is True
        assert "No linter found" in output

    @patch("graft.stages.execute.subprocess.run")
    def test_file_not_found_skips(self, mock_run):
        """FileNotFoundError on all linters → fallback."""
        mock_run.side_effect = FileNotFoundError("npx not found")
        passed, output = _run_lint("/repo")

        assert passed is True
        assert "No linter found" in output

    @patch("graft.stages.execute.subprocess.run")
    def test_timeout_skips(self, mock_run):
        """TimeoutExpired on all linters → fallback."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="npx", timeout=60)
        passed, output = _run_lint("/repo")

        assert passed is True
        assert "No linter found" in output

    @patch("graft.stages.execute.subprocess.run")
    def test_mixed_errors_then_success(self, mock_run):
        """FileNotFoundError, then TimeoutExpired, then success."""
        mock_run.side_effect = [
            FileNotFoundError("npx"),
            subprocess.TimeoutExpired(cmd="ruff", timeout=60),
            _completed(returncode=0),  # prettier succeeds
        ]
        passed, output = _run_lint("/repo")

        assert passed is True
        assert output == "Lint passed"

    @patch("graft.stages.execute.subprocess.run")
    def test_linter_commands_are_correct(self, mock_run):
        """Verify the exact commands attempted."""
        mock_run.side_effect = [
            _completed(returncode=1),
            _completed(returncode=1),
            _completed(returncode=1),
        ]
        _run_lint("/repo")

        calls = mock_run.call_args_list
        assert calls[0][0][0] == ["npx", "eslint", ".", "--fix"]
        assert calls[1][0][0] == ["python", "-m", "ruff", "check", ".", "--fix"]
        assert calls[2][0][0] == ["npx", "prettier", "--write", "."]
        # All should use cwd and timeout=60
        for c in calls:
            assert c[1]["cwd"] == "/repo"
            assert c[1]["timeout"] == 60


# ---------------------------------------------------------------------------
# execute_node — empty plan
# ---------------------------------------------------------------------------


class TestExecuteNodeEmptyPlan:
    """Early return when there is no build plan."""

    async def test_empty_plan_returns_early(self, repo, project, ui):
        result = await execute_node(_state(repo, project, build_plan=[]), ui)

        assert result["current_stage"] == "execute"
        ui.error.assert_called_once()
        assert "No build plan" in ui.error.call_args[0][0]

    async def test_empty_plan_marks_stage_complete(self, repo, project, ui):
        await execute_node(_state(repo, project, build_plan=[]), ui)

        meta = json.loads((project / "metadata.json").read_text())
        assert "execute" in meta["stages_completed"]

    async def test_missing_build_plan_key(self, repo, project, ui):
        """State without build_plan key defaults to empty list."""
        result = await execute_node(_state(repo, project), ui)

        assert result["current_stage"] == "execute"
        ui.error.assert_called_once()


# ---------------------------------------------------------------------------
# execute_node — feature branch creation
# ---------------------------------------------------------------------------


class TestExecuteNodeBranch:
    """Feature branch creation and checkout logic."""

    async def test_creates_new_branch(self, repo, project, ui):
        """When checkout -b succeeds, we get a new branch."""
        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run") as mock_run,
        ):
            mock_agent.return_value = FakeAgentResult()
            # checkout -b succeeds, add ok, commit no changes, rest ok
            mock_run.return_value = _completed(returncode=0)
            result = await execute_node(
                _state(
                    repo,
                    project,
                    build_plan=[{"unit_id": "u1", "title": "T"}],
                    project_id="sess01",
                ),
                ui,
            )

        assert result["feature_branch"] == "feature/sess01"

    async def test_checkout_existing_branch(self, repo, project, ui):
        """When checkout -b fails, falls back to checkout existing."""
        call_count = 0

        def _side_effect(args, **kwargs):
            nonlocal call_count
            call_count += 1
            if args == ["git", "checkout", "-b", "feature/sess01"]:
                return _completed(returncode=1)
            return _completed(returncode=0)

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch(
                "graft.stages.execute.subprocess.run",
                side_effect=_side_effect,
            ),
        ):
            mock_agent.return_value = FakeAgentResult()
            result = await execute_node(
                _state(
                    repo,
                    project,
                    build_plan=[{"unit_id": "u1", "title": "T"}],
                    project_id="sess01",
                ),
                ui,
            )

        assert result["feature_branch"] == "feature/sess01"

    async def test_uses_state_feature_branch(self, repo, project, ui):
        """feature_branch from state overrides the default name."""
        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run") as mock_run,
        ):
            mock_agent.return_value = FakeAgentResult()
            mock_run.return_value = _completed(returncode=0)
            result = await execute_node(
                _state(
                    repo,
                    project,
                    build_plan=[{"unit_id": "u1", "title": "T"}],
                    feature_branch="feature/custom-name",
                ),
                ui,
            )

        assert result["feature_branch"] == "feature/custom-name"


# ---------------------------------------------------------------------------
# execute_node — successful unit execution
# ---------------------------------------------------------------------------


class TestExecuteNodeSuccess:
    """Full successful execution: agent → commit → test → lint → keep."""

    async def test_single_unit_success(self, repo, project, ui):
        """One unit flows through the full pipeline successfully."""
        plan = [{"unit_id": "u1", "title": "Add widget"}]

        def _run_side_effect(args, **kwargs):
            # git commit returns 0 (has changes)
            if args[:2] == ["git", "commit"]:
                return _completed(returncode=0)
            # test suite passes
            if args[0] == "bash":
                return _completed(returncode=0, stdout="3 passed")
            # linters: first succeeds
            if args[0] == "npx":
                return _completed(returncode=0)
            return _completed(returncode=0)

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch(
                "graft.stages.execute.subprocess.run",
                side_effect=_run_side_effect,
            ),
        ):
            mock_agent.return_value = FakeAgentResult()
            result = await execute_node(_state(repo, project, build_plan=plan), ui)

        assert len(result["units_completed"]) == 1
        assert result["units_completed"][0]["unit_id"] == "u1"
        assert result["units_completed"][0]["title"] == "Add widget"
        assert result["units_reverted"] == []
        assert result["units_skipped"] == []
        ui.unit_kept.assert_called_once_with("u1", "Implemented and passing")

    async def test_multi_unit_success(self, repo, project, ui):
        """Multiple units all succeed."""
        plan = [
            {"unit_id": "u1", "title": "First"},
            {"unit_id": "u2", "title": "Second"},
        ]

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run") as mock_run,
        ):
            mock_agent.return_value = FakeAgentResult()
            mock_run.return_value = _completed(returncode=0)
            result = await execute_node(_state(repo, project, build_plan=plan), ui)

        assert len(result["units_completed"]) == 2
        ids = [u["unit_id"] for u in result["units_completed"]]
        assert ids == ["u1", "u2"]

    async def test_completed_unit_has_correct_shape(self, repo, project, ui):
        """Each completed unit dict has expected keys."""
        plan = [
            {
                "unit_id": "u1",
                "title": "Widget",
                "category": "ui",
                "tests_included": True,
            }
        ]

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run") as mock_run,
        ):
            mock_agent.return_value = FakeAgentResult()
            mock_run.return_value = _completed(returncode=0)
            result = await execute_node(_state(repo, project, build_plan=plan), ui)

        completed = result["units_completed"][0]
        assert completed == {
            "unit_id": "u1",
            "title": "Widget",
            "category": "ui",
            "tests_included": True,
        }


# ---------------------------------------------------------------------------
# execute_node — unit revert on test failure
# ---------------------------------------------------------------------------


class TestExecuteNodeTestFailure:
    """Unit is reverted when tests fail after commit."""

    async def test_revert_on_test_failure(self, repo, project, ui):
        plan = [{"unit_id": "u1", "title": "Bad change"}]
        call_seq = []

        def _run_side_effect(args, **kwargs):
            call_seq.append(args)
            # commit succeeds
            if args[:2] == ["git", "commit"]:
                return _completed(returncode=0)
            # tests fail
            if args[0] == "bash":
                return _completed(returncode=1, stderr="FAIL")
            # revert succeeds
            if args[:2] == ["git", "revert"]:
                return _completed(returncode=0)
            return _completed(returncode=0)

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch(
                "graft.stages.execute.subprocess.run",
                side_effect=_run_side_effect,
            ),
        ):
            mock_agent.return_value = FakeAgentResult()
            result = await execute_node(_state(repo, project, build_plan=plan), ui)

        assert len(result["units_reverted"]) == 1
        assert result["units_reverted"][0]["unit_id"] == "u1"
        assert "Tests failed" in result["units_reverted"][0]["reason"]
        assert result["units_completed"] == []
        ui.unit_reverted.assert_called_once_with("u1", "Tests failed")

        # Verify git revert was called
        revert_calls = [c for c in call_seq if c[:2] == ["git", "revert"]]
        assert len(revert_calls) == 1
        assert revert_calls[0] == ["git", "revert", "HEAD", "--no-edit"]


# ---------------------------------------------------------------------------
# execute_node — no changes produced
# ---------------------------------------------------------------------------


class TestExecuteNodeNoChanges:
    """Unit is reverted when commit returns non-zero (no changes)."""

    async def test_no_changes_path(self, repo, project, ui):
        plan = [{"unit_id": "u1", "title": "No-op"}]

        def _run_side_effect(args, **kwargs):
            if args[:2] == ["git", "commit"]:
                return _completed(returncode=1, stdout="nothing to commit")
            return _completed(returncode=0)

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch(
                "graft.stages.execute.subprocess.run",
                side_effect=_run_side_effect,
            ),
        ):
            mock_agent.return_value = FakeAgentResult()
            result = await execute_node(_state(repo, project, build_plan=plan), ui)

        assert len(result["units_reverted"]) == 1
        assert result["units_reverted"][0]["reason"] == "No changes produced"
        ui.unit_reverted.assert_called_once_with("u1", "No changes made")


# ---------------------------------------------------------------------------
# execute_node — unmet dependencies (skip)
# ---------------------------------------------------------------------------


class TestExecuteNodeSkip:
    """Units with unmet dependencies are skipped."""

    async def test_skip_unmet_deps(self, repo, project, ui):
        plan = [
            {"unit_id": "u2", "title": "Depends on u1", "depends_on": ["u1"]},
        ]

        def _run_side_effect(args, **kwargs):
            return _completed(returncode=0)

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch(
                "graft.stages.execute.subprocess.run",
                side_effect=_run_side_effect,
            ),
        ):
            mock_agent.return_value = FakeAgentResult()
            result = await execute_node(_state(repo, project, build_plan=plan), ui)

        assert len(result["units_skipped"]) == 1
        assert result["units_skipped"][0]["unit_id"] == "u2"
        assert "u1" in result["units_skipped"][0]["reason"]
        ui.unit_reverted.assert_called_once()

    async def test_skip_when_dep_was_reverted(self, repo, project, ui):
        """If dep was reverted (not completed), dependent is skipped."""
        plan = [
            {"unit_id": "u1", "title": "First"},
            {"unit_id": "u2", "title": "Depends", "depends_on": ["u1"]},
        ]

        def _run_side_effect(args, **kwargs):
            # u1 commit returns non-zero (no changes → reverted)
            if args[:2] == ["git", "commit"]:
                return _completed(returncode=1)
            return _completed(returncode=0)

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch(
                "graft.stages.execute.subprocess.run",
                side_effect=_run_side_effect,
            ),
        ):
            mock_agent.return_value = FakeAgentResult()
            result = await execute_node(_state(repo, project, build_plan=plan), ui)

        assert len(result["units_reverted"]) == 1
        assert result["units_reverted"][0]["unit_id"] == "u1"
        assert len(result["units_skipped"]) == 1
        assert result["units_skipped"][0]["unit_id"] == "u2"

    async def test_met_deps_proceed(self, repo, project, ui):
        """When dependency IS completed, dependent unit proceeds."""
        plan = [
            {"unit_id": "u1", "title": "Foundation"},
            {"unit_id": "u2", "title": "Builds on u1", "depends_on": ["u1"]},
        ]

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run") as mock_run,
        ):
            mock_agent.return_value = FakeAgentResult()
            mock_run.return_value = _completed(returncode=0)
            result = await execute_node(_state(repo, project, build_plan=plan), ui)

        assert len(result["units_completed"]) == 2


# ---------------------------------------------------------------------------
# execute_node — agent RuntimeError
# ---------------------------------------------------------------------------


class TestExecuteNodeAgentError:
    """Agent raising RuntimeError causes unit revert."""

    async def test_runtime_error_reverts_unit(self, repo, project, ui):
        plan = [{"unit_id": "u1", "title": "Broken agent"}]

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run") as mock_run,
        ):
            mock_agent.side_effect = RuntimeError("Agent crashed")
            mock_run.return_value = _completed(returncode=0)
            result = await execute_node(_state(repo, project, build_plan=plan), ui)

        assert len(result["units_reverted"]) == 1
        assert "Agent crashed" in result["units_reverted"][0]["reason"]
        ui.unit_reverted.assert_called_once()
        assert "Agent failed" in ui.unit_reverted.call_args[0][1]

    async def test_agent_error_does_not_stop_other_units(self, repo, project, ui):
        """After agent error on u1, u2 (no deps) still executes."""
        plan = [
            {"unit_id": "u1", "title": "Fails"},
            {"unit_id": "u2", "title": "Works"},
        ]
        call_count = 0

        async def _agent_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("boom")
            return FakeAgentResult()

        with (
            patch(
                "graft.stages.execute.run_agent",
                new_callable=AsyncMock,
                side_effect=_agent_side_effect,
            ),
            patch("graft.stages.execute.subprocess.run") as mock_run,
        ):
            mock_run.return_value = _completed(returncode=0)
            result = await execute_node(_state(repo, project, build_plan=plan), ui)

        assert len(result["units_reverted"]) == 1
        assert len(result["units_completed"]) == 1
        assert result["units_completed"][0]["unit_id"] == "u2"


# ---------------------------------------------------------------------------
# execute_node — execution log artifact
# ---------------------------------------------------------------------------


class TestExecuteNodeArtifacts:
    """Verify execution log artifact is saved."""

    async def test_saves_execution_log(self, repo, project, ui):
        plan = [{"unit_id": "u1", "title": "Task"}]

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run") as mock_run,
        ):
            mock_agent.return_value = FakeAgentResult()
            mock_run.return_value = _completed(returncode=0)
            await execute_node(_state(repo, project, build_plan=plan), ui)

        log_path = project / "artifacts" / "execution_log.json"
        assert log_path.exists()
        log = json.loads(log_path.read_text())
        assert log["units_completed"] == 1
        assert log["total_planned"] == 1
        assert len(log["completed"]) == 1

    async def test_execution_log_counts_reverted(self, repo, project, ui):
        plan = [{"unit_id": "u1", "title": "Fail"}]

        def _run_side_effect(args, **kwargs):
            if args[:2] == ["git", "commit"]:
                return _completed(returncode=1)
            return _completed(returncode=0)

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch(
                "graft.stages.execute.subprocess.run",
                side_effect=_run_side_effect,
            ),
        ):
            mock_agent.return_value = FakeAgentResult()
            await execute_node(_state(repo, project, build_plan=plan), ui)

        log_path = project / "artifacts" / "execution_log.json"
        log = json.loads(log_path.read_text())
        assert log["units_reverted"] == 1
        assert log["units_completed"] == 0

    async def test_execution_log_counts_skipped(self, repo, project, ui):
        plan = [
            {"unit_id": "u1", "title": "Has dep", "depends_on": ["missing"]},
        ]

        with (
            patch("graft.stages.execute.run_agent", new_callable=AsyncMock),
            patch("graft.stages.execute.subprocess.run") as mock_run,
        ):
            mock_run.return_value = _completed(returncode=0)
            await execute_node(_state(repo, project, build_plan=plan), ui)

        log_path = project / "artifacts" / "execution_log.json"
        log = json.loads(log_path.read_text())
        assert log["units_skipped"] == 1


# ---------------------------------------------------------------------------
# execute_node — return dict shape
# ---------------------------------------------------------------------------


class TestExecuteNodeReturnShape:
    """Verify the shape of the returned dict."""

    async def test_return_keys(self, repo, project, ui):
        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run") as mock_run,
        ):
            mock_agent.return_value = FakeAgentResult()
            mock_run.return_value = _completed(returncode=0)
            result = await execute_node(
                _state(
                    repo,
                    project,
                    build_plan=[{"unit_id": "u1", "title": "T"}],
                ),
                ui,
            )

        expected_keys = {
            "units_completed",
            "units_reverted",
            "units_skipped",
            "feature_branch",
            "current_stage",
        }
        assert set(result.keys()) == expected_keys
        assert result["current_stage"] == "execute"

    async def test_empty_plan_return_shape(self, repo, project, ui):
        """Empty plan early-return still has current_stage."""
        result = await execute_node(_state(repo, project, build_plan=[]), ui)
        assert result == {"current_stage": "execute"}


# ---------------------------------------------------------------------------
# execute_node — UI lifecycle
# ---------------------------------------------------------------------------


class TestExecuteNodeUI:
    """Verify UI callbacks are invoked correctly."""

    async def test_stage_start_and_done(self, repo, project, ui):
        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run") as mock_run,
        ):
            mock_agent.return_value = FakeAgentResult()
            mock_run.return_value = _completed(returncode=0)
            await execute_node(
                _state(
                    repo,
                    project,
                    build_plan=[{"unit_id": "u1", "title": "T"}],
                ),
                ui,
            )

        ui.stage_start.assert_called_once_with("execute")
        ui.stage_done.assert_called_once_with("execute")

    async def test_unit_start_called(self, repo, project, ui):
        plan = [
            {"unit_id": "u1", "title": "First"},
            {"unit_id": "u2", "title": "Second"},
        ]

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run") as mock_run,
        ):
            mock_agent.return_value = FakeAgentResult()
            mock_run.return_value = _completed(returncode=0)
            await execute_node(_state(repo, project, build_plan=plan), ui)

        assert ui.unit_start.call_count == 2
        ui.unit_start.assert_any_call("u1", "First", 1, 2)
        ui.unit_start.assert_any_call("u2", "Second", 2, 2)

    async def test_marks_stage_complete(self, repo, project, ui):
        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run") as mock_run,
        ):
            mock_agent.return_value = FakeAgentResult()
            mock_run.return_value = _completed(returncode=0)
            await execute_node(
                _state(
                    repo,
                    project,
                    build_plan=[{"unit_id": "u1", "title": "T"}],
                ),
                ui,
            )

        meta = json.loads((project / "metadata.json").read_text())
        assert "execute" in meta["stages_completed"]


# ---------------------------------------------------------------------------
# execute_node — prompt construction
# ---------------------------------------------------------------------------


class TestExecuteNodePrompt:
    """Verify the prompt sent to run_agent."""

    async def test_prompt_contains_title_and_description(self, repo, project, ui):
        plan = [
            {
                "unit_id": "u1",
                "title": "Add auth",
                "description": "Implement OAuth flow",
            }
        ]

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run") as mock_run,
        ):
            mock_agent.return_value = FakeAgentResult()
            mock_run.return_value = _completed(returncode=0)
            await execute_node(_state(repo, project, build_plan=plan), ui)

        prompt = mock_agent.call_args[1]["user_prompt"]
        assert "Add auth" in prompt
        assert "Implement OAuth flow" in prompt

    async def test_prompt_includes_pattern_reference(self, repo, project, ui):
        plan = [
            {
                "unit_id": "u1",
                "title": "T",
                "pattern_reference": "src/routes/users.ts",
            }
        ]

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run") as mock_run,
        ):
            mock_agent.return_value = FakeAgentResult()
            mock_run.return_value = _completed(returncode=0)
            await execute_node(_state(repo, project, build_plan=plan), ui)

        prompt = mock_agent.call_args[1]["user_prompt"]
        assert "src/routes/users.ts" in prompt
        assert "Read this file FIRST" in prompt

    async def test_prompt_includes_tests_flag(self, repo, project, ui):
        plan = [{"unit_id": "u1", "title": "T", "tests_included": True}]

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run") as mock_run,
        ):
            mock_agent.return_value = FakeAgentResult()
            mock_run.return_value = _completed(returncode=0)
            await execute_node(_state(repo, project, build_plan=plan), ui)

        prompt = mock_agent.call_args[1]["user_prompt"]
        assert "TESTS" in prompt
        assert "co-located tests" in prompt

    async def test_prompt_includes_acceptance_criteria(self, repo, project, ui):
        plan = [
            {
                "unit_id": "u1",
                "title": "T",
                "acceptance_criteria": ["Must handle errors", "Returns 200"],
            }
        ]

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run") as mock_run,
        ):
            mock_agent.return_value = FakeAgentResult()
            mock_run.return_value = _completed(returncode=0)
            await execute_node(_state(repo, project, build_plan=plan), ui)

        prompt = mock_agent.call_args[1]["user_prompt"]
        assert "Must handle errors" in prompt
        assert "Returns 200" in prompt

    async def test_agent_called_with_correct_kwargs(self, repo, project, ui):
        plan = [{"unit_id": "u1", "title": "Task"}]

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run") as mock_run,
        ):
            mock_agent.return_value = FakeAgentResult()
            mock_run.return_value = _completed(returncode=0)
            await execute_node(
                _state(
                    repo,
                    project,
                    build_plan=plan,
                    model="claude-sonnet-4-20250514",
                    max_agent_turns=30,
                ),
                ui,
            )

        kwargs = mock_agent.call_args[1]
        assert kwargs["cwd"] == str(repo)
        assert kwargs["project_dir"] == str(project)
        assert kwargs["stage"] == "execute_u1"
        assert kwargs["model"] == "claude-sonnet-4-20250514"
        assert kwargs["max_turns"] == 30


# ---------------------------------------------------------------------------
# execute_node — lint amend flow
# ---------------------------------------------------------------------------


class TestExecuteNodeLint:
    """Lint auto-fix and commit amend flow."""

    async def test_lint_amend_on_success(self, repo, project, ui):
        """When lint passes, an amend commit is attempted."""
        plan = [{"unit_id": "u1", "title": "T"}]
        git_calls = []

        def _run_side_effect(args, **kwargs):
            if args[0] == "git":
                git_calls.append(args)
            if args[:2] == ["git", "commit"] and "--amend" not in args:
                return _completed(returncode=0)
            if args[0] == "bash":
                return _completed(returncode=0, stdout="ok")
            return _completed(returncode=0)

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch(
                "graft.stages.execute.subprocess.run",
                side_effect=_run_side_effect,
            ),
        ):
            mock_agent.return_value = FakeAgentResult()
            await execute_node(_state(repo, project, build_plan=plan), ui)

        # Should see git commit --amend --no-edit in the calls
        amend_calls = [c for c in git_calls if "commit" in c and "--amend" in c]
        assert len(amend_calls) == 1
        assert "--no-edit" in amend_calls[0]


# ---------------------------------------------------------------------------
# execute_node — default unit_id fallback
# ---------------------------------------------------------------------------


class TestExecuteNodeDefaults:
    """Verify default values for missing unit fields."""

    async def test_missing_unit_id_gets_default(self, repo, project, ui):
        """Unit without unit_id gets 'feat_01' as default."""
        plan = [{"title": "No ID unit"}]

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run") as mock_run,
        ):
            mock_agent.return_value = FakeAgentResult()
            mock_run.return_value = _completed(returncode=0)
            result = await execute_node(_state(repo, project, build_plan=plan), ui)

        assert result["units_completed"][0]["unit_id"] == "feat_01"

    async def test_missing_title_uses_untitled(self, repo, project, ui):
        plan = [{"unit_id": "u1"}]

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run") as mock_run,
        ):
            mock_agent.return_value = FakeAgentResult()
            mock_run.return_value = _completed(returncode=0)
            result = await execute_node(_state(repo, project, build_plan=plan), ui)

        assert result["units_completed"][0]["title"] == "Untitled"
