"""Tests for graft.stages.execute."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from graft.stages.execute import (
    _git,
    _order_by_dependencies,
    _run_lint,
    _run_tests,
    execute_node,
)


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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ui():
    """A fully stubbed UI instance."""
    ui = MagicMock()
    ui.stage_start = MagicMock()
    ui.stage_done = MagicMock()
    ui.error = MagicMock()
    ui.unit_start = MagicMock()
    ui.unit_kept = MagicMock()
    ui.unit_reverted = MagicMock()
    return ui


def _make_state(tmp_path, plan=None, **overrides):
    """Build a minimal FeatureState dict for testing."""
    project_dir = tmp_path / "project"
    (project_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    state = {
        "repo_path": str(tmp_path / "repo"),
        "project_dir": str(project_dir),
        "project_id": "test_session",
        "build_plan": plan or [],
        "codebase_profile": {},
        "feature_spec": {},
        "max_agent_turns": 5,
    }
    state.update(overrides)
    return state


def _ok(returncode=0, stdout="ok", stderr=""):
    """Construct a fake CompletedProcess."""
    return subprocess.CompletedProcess(
        args=[],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


# ---------------------------------------------------------------------------
# _git() helper
# ---------------------------------------------------------------------------


class TestGitHelper:
    """Tests for the _git() subprocess wrapper."""

    @patch("graft.stages.execute.subprocess.run")
    def test_git_calls_subprocess_with_correct_args(self, mock_run):
        mock_run.return_value = _ok()
        result = _git("/my/repo", "status")

        mock_run.assert_called_once_with(
            ["git", "status"],
            cwd="/my/repo",
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        assert result.returncode == 0

    @patch("graft.stages.execute.subprocess.run")
    def test_git_passes_multiple_args(self, mock_run):
        mock_run.return_value = _ok()
        _git("/repo", "commit", "-m", "msg")
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == ["git", "commit", "-m", "msg"]

    @patch("graft.stages.execute.subprocess.run")
    def test_git_check_false(self, mock_run):
        mock_run.return_value = _ok(returncode=1)
        result = _git("/repo", "checkout", "-b", "branch", check=False)
        assert mock_run.call_args[1]["check"] is False
        assert result.returncode == 1

    @patch("graft.stages.execute.subprocess.run")
    def test_git_propagates_calledprocesserror(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(128, "git")
        with pytest.raises(subprocess.CalledProcessError):
            _git("/repo", "bad-command")


# ---------------------------------------------------------------------------
# _run_tests()
# ---------------------------------------------------------------------------


class TestRunTests:
    """Tests for the _run_tests() helper."""

    @patch("graft.stages.execute.subprocess.run")
    def test_success_returns_true(self, mock_run):
        mock_run.return_value = _ok(returncode=0, stdout="3 passed", stderr="")
        passed, output = _run_tests("/repo")
        assert passed is True
        assert "3 passed" in output

    @patch("graft.stages.execute.subprocess.run")
    def test_failure_returns_false(self, mock_run):
        mock_run.return_value = _ok(returncode=1, stdout="", stderr="FAILED test_x")
        passed, output = _run_tests("/repo")
        assert passed is False
        assert "FAILED" in output

    @patch("graft.stages.execute.subprocess.run")
    def test_timeout_300_seconds(self, mock_run):
        mock_run.return_value = _ok()
        _run_tests("/repo")
        assert mock_run.call_args[1]["timeout"] == 300

    @patch("graft.stages.execute.subprocess.run")
    def test_timeout_expired_returns_false(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="bash", timeout=300)
        passed, output = _run_tests("/repo")
        assert passed is False
        assert "timed out" in output.lower()

    @patch("graft.stages.execute.subprocess.run")
    def test_file_not_found_returns_true(self, mock_run):
        """No test runner available → pass (don't block)."""
        mock_run.side_effect = FileNotFoundError
        passed, output = _run_tests("/repo")
        assert passed is True
        assert "skipping" in output.lower()

    @patch("graft.stages.execute.subprocess.run")
    def test_output_truncated_to_2000_chars(self, mock_run):
        long_output = "x" * 5000
        mock_run.return_value = _ok(stdout=long_output, stderr="")
        _, output = _run_tests("/repo")
        assert len(output) <= 2000


# ---------------------------------------------------------------------------
# _run_lint()
# ---------------------------------------------------------------------------


class TestRunLint:
    """Tests for linter detection logic."""

    @patch("graft.stages.execute.subprocess.run")
    def test_first_linter_succeeds(self, mock_run):
        """eslint succeeds on first try — returns immediately."""
        mock_run.return_value = _ok(returncode=0)
        passed, output = _run_lint("/repo")
        assert passed is True
        assert "passed" in output.lower()
        # Should have been called only once since the first linter passed
        assert mock_run.call_count == 1

    @patch("graft.stages.execute.subprocess.run")
    def test_falls_through_to_ruff(self, mock_run):
        """eslint not found → ruff succeeds."""

        def side_effect(cmd, **kwargs):
            if "eslint" in cmd:
                raise FileNotFoundError
            return _ok(returncode=0)

        mock_run.side_effect = side_effect
        passed, output = _run_lint("/repo")
        assert passed is True
        assert mock_run.call_count == 2

    @patch("graft.stages.execute.subprocess.run")
    def test_all_linters_not_found(self, mock_run):
        """No linter installed → passes with skip message."""
        mock_run.side_effect = FileNotFoundError
        passed, output = _run_lint("/repo")
        assert passed is True
        assert "skipping" in output.lower()

    @patch("graft.stages.execute.subprocess.run")
    def test_timeout_skips_to_next(self, mock_run):
        """Timed-out linter is skipped."""

        def side_effect(cmd, **kwargs):
            if "eslint" in cmd:
                raise subprocess.TimeoutExpired(cmd="eslint", timeout=60)
            if "ruff" in cmd:
                return _ok(returncode=0)
            raise FileNotFoundError

        mock_run.side_effect = side_effect
        passed, output = _run_lint("/repo")
        assert passed is True

    @patch("graft.stages.execute.subprocess.run")
    def test_linter_nonzero_continues(self, mock_run):
        """Linter returning non-zero → tries next linter."""

        def side_effect(cmd, **kwargs):
            if "eslint" in cmd:
                return _ok(returncode=1)
            if "ruff" in cmd:
                return _ok(returncode=0)
            raise FileNotFoundError

        mock_run.side_effect = side_effect
        passed, output = _run_lint("/repo")
        assert passed is True
        assert mock_run.call_count == 2


# ---------------------------------------------------------------------------
# execute_node — happy path
# ---------------------------------------------------------------------------


class TestExecuteNodeHappyPath:
    """execute_node processes units, commits, tests, marks complete."""

    @pytest.mark.asyncio
    async def test_single_unit_completed(self, tmp_path, mock_ui):
        """One unit → agent runs, commit succeeds, tests pass, unit completed."""
        plan = [
            {
                "unit_id": "u1",
                "title": "Add widget",
                "description": "Add the widget module",
                "pattern_reference": "src/foo.py",
                "tests_included": True,
                "acceptance_criteria": ["widget works"],
            }
        ]
        state = _make_state(tmp_path, plan=plan)

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute._git") as mock_git,
            patch("graft.stages.execute._run_tests") as mock_tests,
            patch("graft.stages.execute._run_lint") as mock_lint,
            patch("graft.stages.execute.save_artifact") as mock_save,
            patch("graft.stages.execute.mark_stage_complete") as mock_mark,
        ):
            # checkout -b succeeds
            mock_git.return_value = _ok()
            # commit succeeds (returncode 0)
            mock_git.return_value = _ok(returncode=0)
            mock_tests.return_value = (True, "all passed")
            mock_lint.return_value = (True, "lint passed")

            result = await execute_node(state, mock_ui)

        assert len(result["units_completed"]) == 1
        assert result["units_completed"][0]["unit_id"] == "u1"
        assert result["units_reverted"] == []
        assert result["units_skipped"] == []
        assert "feature/" in result["feature_branch"]
        mock_agent.assert_awaited_once()
        mock_ui.unit_kept.assert_called_once()
        mock_save.assert_called_once()
        mock_mark.assert_called_once_with(state["project_dir"], "execute")

    @pytest.mark.asyncio
    async def test_multiple_units_all_completed(self, tmp_path, mock_ui):
        """Two independent units both complete successfully."""
        plan = [
            {"unit_id": "u1", "title": "First"},
            {"unit_id": "u2", "title": "Second"},
        ]
        state = _make_state(tmp_path, plan=plan)

        with (
            patch("graft.stages.execute.run_agent", new_callable=AsyncMock),
            patch("graft.stages.execute._git", return_value=_ok()),
            patch("graft.stages.execute._run_tests", return_value=(True, "ok")),
            patch("graft.stages.execute._run_lint", return_value=(True, "ok")),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            result = await execute_node(state, mock_ui)

        assert len(result["units_completed"]) == 2
        assert mock_ui.unit_kept.call_count == 2


# ---------------------------------------------------------------------------
# execute_node — test failure → revert
# ---------------------------------------------------------------------------


class TestExecuteNodeTestFailure:
    """When tests fail, the commit is reverted."""

    @pytest.mark.asyncio
    async def test_unit_reverted_on_test_failure(self, tmp_path, mock_ui):
        plan = [{"unit_id": "u1", "title": "Broken unit"}]
        state = _make_state(tmp_path, plan=plan)

        with (
            patch("graft.stages.execute.run_agent", new_callable=AsyncMock),
            patch("graft.stages.execute._git") as mock_git,
            patch("graft.stages.execute._run_tests") as mock_tests,
            patch("graft.stages.execute._run_lint"),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            mock_git.return_value = _ok()
            mock_tests.return_value = (False, "FAILED test_widget.py::test_x")

            result = await execute_node(state, mock_ui)

        assert len(result["units_reverted"]) == 1
        assert "Tests failed" in result["units_reverted"][0]["reason"]
        assert result["units_completed"] == []

        # Verify git revert was called
        revert_calls = [
            c for c in mock_git.call_args_list if len(c[0]) >= 2 and c[0][1] == "revert"
        ]
        assert len(revert_calls) == 1
        assert "HEAD" in revert_calls[0][0]
        assert "--no-edit" in revert_calls[0][0]

        mock_ui.unit_reverted.assert_called_once()


# ---------------------------------------------------------------------------
# execute_node — dependency skip
# ---------------------------------------------------------------------------


class TestExecuteNodeDependencySkip:
    """Units with unmet dependencies are skipped."""

    @pytest.mark.asyncio
    async def test_unit_skipped_when_dep_not_completed(self, tmp_path, mock_ui):
        """Unit depends on another that was reverted → skipped."""
        plan = [
            {"unit_id": "u1", "title": "Base", "depends_on": []},
            {"unit_id": "u2", "title": "Depends on u1", "depends_on": ["u1"]},
        ]
        state = _make_state(tmp_path, plan=plan)

        with (
            patch("graft.stages.execute.run_agent", new_callable=AsyncMock),
            patch("graft.stages.execute._git") as mock_git,
            patch("graft.stages.execute._run_tests") as mock_tests,
            patch("graft.stages.execute._run_lint"),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            # u1 fails tests → reverted, so u2 should be skipped
            mock_git.return_value = _ok()
            mock_tests.return_value = (False, "FAIL")

            result = await execute_node(state, mock_ui)

        assert len(result["units_reverted"]) == 1
        assert result["units_reverted"][0]["unit_id"] == "u1"
        assert len(result["units_skipped"]) == 1
        assert result["units_skipped"][0]["unit_id"] == "u2"
        assert "Dependencies not met" in result["units_skipped"][0]["reason"]

    @pytest.mark.asyncio
    async def test_independent_unit_after_skipped_still_runs(self, tmp_path, mock_ui):
        """An independent unit can succeed even after a prior skip."""
        plan = [
            {"unit_id": "u1", "title": "Base"},
            {"unit_id": "u2", "title": "Depends on u1", "depends_on": ["u1"]},
            {"unit_id": "u3", "title": "Independent"},
        ]
        state = _make_state(tmp_path, plan=plan)

        call_count = 0

        async def mock_agent(**kwargs):
            pass

        with (
            patch("graft.stages.execute.run_agent", side_effect=mock_agent),
            patch("graft.stages.execute._git") as mock_git,
            patch("graft.stages.execute._run_tests") as mock_tests,
            patch("graft.stages.execute._run_lint", return_value=(True, "ok")),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            # u1 tests fail; u2 depends on u1 (skipped); u3 passes
            test_results = iter([(False, "FAIL"), (True, "ok")])
            mock_git.return_value = _ok()
            mock_tests.side_effect = lambda path: next(test_results)

            result = await execute_node(state, mock_ui)

        assert result["units_reverted"][0]["unit_id"] == "u1"
        assert result["units_skipped"][0]["unit_id"] == "u2"
        assert len(result["units_completed"]) == 1
        assert result["units_completed"][0]["unit_id"] == "u3"


# ---------------------------------------------------------------------------
# execute_node — empty plan
# ---------------------------------------------------------------------------


class TestExecuteNodeEmptyPlan:
    """No build plan → early return."""

    @pytest.mark.asyncio
    async def test_empty_plan_returns_early(self, tmp_path, mock_ui):
        state = _make_state(tmp_path, plan=[])

        with (
            patch("graft.stages.execute.mark_stage_complete") as mock_mark,
        ):
            result = await execute_node(state, mock_ui)

        assert result == {"current_stage": "execute"}
        mock_ui.error.assert_called_once()
        mock_mark.assert_called_once()


# ---------------------------------------------------------------------------
# execute_node — agent failure
# ---------------------------------------------------------------------------


class TestExecuteNodeAgentFailure:
    """When run_agent raises RuntimeError, unit is reverted."""

    @pytest.mark.asyncio
    async def test_agent_runtime_error_reverts_unit(self, tmp_path, mock_ui):
        plan = [{"unit_id": "u1", "title": "Crasher"}]
        state = _make_state(tmp_path, plan=plan)

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute._git", return_value=_ok()),
            patch("graft.stages.execute._run_tests"),
            patch("graft.stages.execute._run_lint"),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            mock_agent.side_effect = RuntimeError("Agent crashed")

            result = await execute_node(state, mock_ui)

        assert len(result["units_reverted"]) == 1
        assert "Agent crashed" in result["units_reverted"][0]["reason"]
        assert result["units_completed"] == []


# ---------------------------------------------------------------------------
# execute_node — commit produces no changes
# ---------------------------------------------------------------------------


class TestExecuteNodeNoChanges:
    """When git commit fails (nothing to commit), unit is reverted."""

    @pytest.mark.asyncio
    async def test_no_changes_reverts_unit(self, tmp_path, mock_ui):
        plan = [{"unit_id": "u1", "title": "No-op"}]
        state = _make_state(tmp_path, plan=plan)

        with (
            patch("graft.stages.execute.run_agent", new_callable=AsyncMock),
            patch("graft.stages.execute._git") as mock_git,
            patch("graft.stages.execute._run_tests"),
            patch("graft.stages.execute._run_lint"),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            # Most calls succeed, but commit returns non-zero
            def git_side_effect(repo_path, *args, check=True):
                if args and args[0] == "commit":
                    return _ok(returncode=1)
                return _ok()

            mock_git.side_effect = git_side_effect

            result = await execute_node(state, mock_ui)

        assert len(result["units_reverted"]) == 1
        assert "No changes" in result["units_reverted"][0]["reason"]
        mock_ui.unit_reverted.assert_called_once()


# ---------------------------------------------------------------------------
# execute_node — execution_log.json artifact
# ---------------------------------------------------------------------------


class TestExecutionLog:
    """Verify the execution_log.json artifact structure."""

    @pytest.mark.asyncio
    async def test_execution_log_written_with_correct_counts(self, tmp_path, mock_ui):
        plan = [
            {"unit_id": "u1", "title": "Good"},
            {"unit_id": "u2", "title": "Bad"},
            {"unit_id": "u3", "title": "Depends on u2", "depends_on": ["u2"]},
        ]
        state = _make_state(tmp_path, plan=plan)

        with (
            patch("graft.stages.execute.run_agent", new_callable=AsyncMock),
            patch("graft.stages.execute._git", return_value=_ok()),
            patch("graft.stages.execute._run_tests") as mock_tests,
            patch("graft.stages.execute._run_lint", return_value=(True, "ok")),
            patch("graft.stages.execute.save_artifact") as mock_save,
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            # u1 passes, u2 fails tests, u3 skipped
            test_results = iter([(True, "ok"), (False, "FAIL")])
            mock_tests.side_effect = lambda path: next(test_results)

            result = await execute_node(state, mock_ui)

        # Verify save_artifact was called with execution_log.json
        mock_save.assert_called_once()
        call_args = mock_save.call_args
        assert call_args[0][1] == "execution_log.json"

        log = json.loads(call_args[0][2])
        assert log["units_completed"] == 1
        assert log["units_reverted"] == 1
        assert log["units_skipped"] == 1
        assert log["total_planned"] == 3
        assert len(log["completed"]) == 1
        assert len(log["reverted"]) == 1
        assert len(log["skipped"]) == 1


# ---------------------------------------------------------------------------
# execute_node — feature branch creation
# ---------------------------------------------------------------------------


class TestExecuteNodeBranch:
    """Verify branch creation/checkout logic."""

    @pytest.mark.asyncio
    async def test_uses_existing_branch_when_create_fails(self, tmp_path, mock_ui):
        plan = [{"unit_id": "u1", "title": "Unit"}]
        state = _make_state(tmp_path, plan=plan, feature_branch="feature/existing")

        with (
            patch("graft.stages.execute.run_agent", new_callable=AsyncMock),
            patch("graft.stages.execute._git") as mock_git,
            patch("graft.stages.execute._run_tests", return_value=(True, "ok")),
            patch("graft.stages.execute._run_lint", return_value=(True, "ok")),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            # checkout -b fails (branch exists), then checkout succeeds
            call_num = 0

            def git_side_effect(repo_path, *args, check=True):
                nonlocal call_num
                call_num += 1
                if args and args[0] == "checkout" and "-b" in args:
                    return _ok(returncode=1)
                return _ok()

            mock_git.side_effect = git_side_effect

            result = await execute_node(state, mock_ui)

        assert result["feature_branch"] == "feature/existing"
        # Verify both checkout attempts were made
        checkout_calls = [
            c
            for c in mock_git.call_args_list
            if len(c[0]) >= 2 and c[0][1] == "checkout"
        ]
        assert len(checkout_calls) >= 2


# ---------------------------------------------------------------------------
# _order_by_dependencies — additional edge case
# ---------------------------------------------------------------------------


def test_order_by_dependencies_empty_plan():
    """Empty plan returns empty list."""
    assert _order_by_dependencies([]) == []
