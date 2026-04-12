"""Tests for graft.stages.execute."""

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
# _order_by_dependencies (existing tests preserved)
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
    """Tests for the _git helper."""

    def test_git_runs_subprocess_with_correct_args(self, monkeypatch):
        """_git calls subprocess.run with ['git', *args] in the given cwd."""
        mock_run = MagicMock(return_value=subprocess.CompletedProcess([], 0))
        monkeypatch.setattr(subprocess, "run", mock_run)

        _git("/tmp/repo", "status")

        mock_run.assert_called_once_with(
            ["git", "status"],
            cwd="/tmp/repo",
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )

    def test_git_passes_multiple_args(self, monkeypatch):
        """_git forwards all positional args to git."""
        mock_run = MagicMock(return_value=subprocess.CompletedProcess([], 0))
        monkeypatch.setattr(subprocess, "run", mock_run)

        _git("/repo", "commit", "-m", "feat: hello")

        args_called = mock_run.call_args[0][0]
        assert args_called == ["git", "commit", "-m", "feat: hello"]

    def test_git_check_false(self, monkeypatch):
        """_git passes check=False when requested."""
        mock_run = MagicMock(return_value=subprocess.CompletedProcess([], 1))
        monkeypatch.setattr(subprocess, "run", mock_run)

        result = _git("/repo", "checkout", "-b", "feature/x", check=False)

        assert mock_run.call_args[1]["check"] is False
        assert result.returncode == 1

    def test_git_check_true_raises_on_failure(self, monkeypatch):
        """_git with check=True (default) raises CalledProcessError on nonzero exit."""
        monkeypatch.setattr(
            subprocess,
            "run",
            MagicMock(side_effect=subprocess.CalledProcessError(128, "git")),
        )

        with pytest.raises(subprocess.CalledProcessError):
            _git("/repo", "checkout", "nonexistent-branch")

    def test_git_returns_completed_process(self, monkeypatch):
        """_git returns the CompletedProcess from subprocess.run."""
        cp = subprocess.CompletedProcess([], 0, stdout="abc\n", stderr="")
        monkeypatch.setattr(subprocess, "run", MagicMock(return_value=cp))

        result = _git("/repo", "log", "--oneline")
        assert result.stdout == "abc\n"


# ---------------------------------------------------------------------------
# _run_tests
# ---------------------------------------------------------------------------


class TestRunTests:
    """Tests for the _run_tests helper."""

    def test_returns_true_on_success(self, monkeypatch):
        """_run_tests returns (True, output) when tests pass (rc=0)."""
        cp = subprocess.CompletedProcess([], 0, stdout="3 passed", stderr="")
        monkeypatch.setattr(subprocess, "run", MagicMock(return_value=cp))

        passed, output = _run_tests("/repo")

        assert passed is True
        assert "3 passed" in output

    def test_returns_false_on_failure(self, monkeypatch):
        """_run_tests returns (False, output) when tests fail (rc!=0)."""
        cp = subprocess.CompletedProcess([], 1, stdout="FAIL", stderr="error detail")
        monkeypatch.setattr(subprocess, "run", MagicMock(return_value=cp))

        passed, output = _run_tests("/repo")

        assert passed is False
        assert "FAIL" in output

    def test_handles_timeout(self, monkeypatch):
        """_run_tests returns (False, timeout message) on TimeoutExpired."""
        monkeypatch.setattr(
            subprocess,
            "run",
            MagicMock(side_effect=subprocess.TimeoutExpired("bash", 300)),
        )

        passed, output = _run_tests("/repo")

        assert passed is False
        assert "timed out" in output.lower()

    def test_handles_file_not_found(self, monkeypatch):
        """_run_tests returns (True, skip message) when bash not found."""
        monkeypatch.setattr(
            subprocess,
            "run",
            MagicMock(side_effect=FileNotFoundError("bash")),
        )

        passed, output = _run_tests("/repo")

        assert passed is True
        assert "skipping" in output.lower()

    def test_output_truncated_to_2000_chars(self, monkeypatch):
        """_run_tests truncates combined output to last 2000 chars."""
        long_output = "x" * 5000
        cp = subprocess.CompletedProcess([], 0, stdout=long_output, stderr="")
        monkeypatch.setattr(subprocess, "run", MagicMock(return_value=cp))

        passed, output = _run_tests("/repo")

        assert passed is True
        assert len(output) == 2000

    def test_combines_stdout_and_stderr(self, monkeypatch):
        """_run_tests concatenates stdout + stderr."""
        cp = subprocess.CompletedProcess([], 0, stdout="OUT", stderr="ERR")
        monkeypatch.setattr(subprocess, "run", MagicMock(return_value=cp))

        passed, output = _run_tests("/repo")

        assert "OUT" in output
        assert "ERR" in output

    def test_uses_correct_cwd_and_timeout(self, monkeypatch):
        """_run_tests runs in the repo directory with 300s timeout."""
        mock_run = MagicMock(
            return_value=subprocess.CompletedProcess([], 0, stdout="", stderr="")
        )
        monkeypatch.setattr(subprocess, "run", mock_run)

        _run_tests("/my/repo")

        assert mock_run.call_args[1]["cwd"] == "/my/repo"
        assert mock_run.call_args[1]["timeout"] == 300


# ---------------------------------------------------------------------------
# _run_lint
# ---------------------------------------------------------------------------


class TestRunLint:
    """Tests for the _run_lint helper."""

    def test_returns_true_on_first_linter_success(self, monkeypatch):
        """_run_lint returns (True, 'Lint passed') on first successful linter."""
        mock_run = MagicMock(
            return_value=subprocess.CompletedProcess([], 0, stdout="", stderr="")
        )
        monkeypatch.setattr(subprocess, "run", mock_run)

        passed, output = _run_lint("/repo")

        assert passed is True
        assert output == "Lint passed"
        # Should have only been called once since first linter succeeded
        assert mock_run.call_count == 1

    def test_tries_next_linter_on_failure(self, monkeypatch):
        """_run_lint tries subsequent linters when earlier ones fail."""
        # First call fails (rc=1), second succeeds (rc=0)
        mock_run = MagicMock(
            side_effect=[
                subprocess.CompletedProcess([], 1, stdout="", stderr=""),
                subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            ]
        )
        monkeypatch.setattr(subprocess, "run", mock_run)

        passed, output = _run_lint("/repo")

        assert passed is True
        assert output == "Lint passed"
        assert mock_run.call_count == 2

    def test_skips_linters_with_file_not_found(self, monkeypatch):
        """_run_lint skips linters that raise FileNotFoundError."""
        mock_run = MagicMock(
            side_effect=[
                FileNotFoundError("npx"),
                subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            ]
        )
        monkeypatch.setattr(subprocess, "run", mock_run)

        passed, output = _run_lint("/repo")

        assert passed is True
        assert output == "Lint passed"

    def test_skips_linters_with_timeout(self, monkeypatch):
        """_run_lint skips linters that time out."""
        mock_run = MagicMock(
            side_effect=[
                subprocess.TimeoutExpired("npx", 60),
                subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            ]
        )
        monkeypatch.setattr(subprocess, "run", mock_run)

        passed, output = _run_lint("/repo")

        assert passed is True
        assert output == "Lint passed"

    def test_all_linters_fail_returns_skip_message(self, monkeypatch):
        """_run_lint returns (True, skip message) when no linter works."""
        mock_run = MagicMock(
            side_effect=[
                FileNotFoundError("npx"),
                FileNotFoundError("python"),
                FileNotFoundError("npx"),
            ]
        )
        monkeypatch.setattr(subprocess, "run", mock_run)

        passed, output = _run_lint("/repo")

        assert passed is True
        assert "skipping" in output.lower()

    def test_all_linters_nonzero_returns_skip(self, monkeypatch):
        """When all linters return nonzero, _run_lint still returns True with skip."""
        mock_run = MagicMock(
            return_value=subprocess.CompletedProcess([], 1, stdout="", stderr="")
        )
        monkeypatch.setattr(subprocess, "run", mock_run)

        passed, output = _run_lint("/repo")

        assert passed is True
        assert "skipping" in output.lower()
        # All 3 linters tried
        assert mock_run.call_count == 3

    def test_tries_eslint_ruff_prettier_in_order(self, monkeypatch):
        """_run_lint attempts eslint, then ruff, then prettier."""
        calls = []

        def track_run(cmd, **kwargs):
            calls.append(cmd)
            raise FileNotFoundError()

        monkeypatch.setattr(subprocess, "run", track_run)

        _run_lint("/repo")

        assert calls[0][0] == "npx" and "eslint" in calls[0][1]
        assert calls[1] == ["python", "-m", "ruff", "check", ".", "--fix"]
        assert calls[2][0] == "npx" and "prettier" in calls[2][1]


# ---------------------------------------------------------------------------
# execute_node (async)
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> dict:
    """Build a minimal FeatureState dict for testing."""
    base = {
        "repo_path": "/tmp/test-repo",
        "project_dir": "/tmp/test-project",
        "project_id": "feat_abc123",
        "build_plan": [],
        "codebase_profile": {},
        "feature_spec": {},
        "max_agent_turns": 10,
    }
    base.update(overrides)
    return base


def _make_ui() -> MagicMock:
    """Build a mock UI with all methods used by execute_node."""
    ui = MagicMock()
    ui.stage_start = MagicMock()
    ui.stage_done = MagicMock()
    ui.error = MagicMock()
    ui.unit_start = MagicMock()
    ui.unit_kept = MagicMock()
    ui.unit_reverted = MagicMock()
    return ui


def _simple_plan(count=1, **unit_overrides):
    """Generate a simple build plan with N units."""
    units = []
    for i in range(1, count + 1):
        unit = {
            "unit_id": f"unit_{i}",
            "title": f"Unit {i}",
            "description": f"Implement unit {i}",
            "pattern_reference": "src/example.py",
            "tests_included": True,
            "acceptance_criteria": ["It works"],
            "depends_on": [],
        }
        unit.update(unit_overrides)
        units.append(unit)
    return units


class TestExecuteNodeEmptyPlan:
    """execute_node with an empty build plan."""

    @pytest.mark.asyncio
    async def test_empty_plan_returns_early(self):
        """Empty plan logs error and returns without executing anything."""
        state = _make_state(build_plan=[])
        ui = _make_ui()

        with (
            patch("graft.stages.execute.mark_stage_complete") as mock_mark,
            patch("graft.stages.execute.save_artifact"),
        ):
            result = await execute_node(state, ui)

        ui.error.assert_called_once()
        assert "nothing" in ui.error.call_args[0][0].lower()
        mock_mark.assert_called_once_with("/tmp/test-project", "execute")
        assert result["current_stage"] == "execute"

    @pytest.mark.asyncio
    async def test_empty_plan_does_not_call_git_or_agent(self):
        """Empty plan should not attempt any git or agent operations."""
        state = _make_state(build_plan=[])
        ui = _make_ui()

        with (
            patch("graft.stages.execute._git") as mock_git,
            patch("graft.stages.execute.run_agent") as mock_agent,
            patch("graft.stages.execute.mark_stage_complete"),
            patch("graft.stages.execute.save_artifact"),
        ):
            await execute_node(state, ui)

        mock_git.assert_not_called()
        mock_agent.assert_not_called()


class TestExecuteNodeBranch:
    """execute_node branch creation logic."""

    @pytest.mark.asyncio
    async def test_creates_feature_branch(self):
        """execute_node creates a feature branch from state."""
        state = _make_state(
            build_plan=_simple_plan(1),
            feature_branch="feature/my-feat",
        )
        ui = _make_ui()
        git_calls = []

        def fake_git(repo_path, *args, check=True):
            git_calls.append(args)
            return subprocess.CompletedProcess([], 0, stdout="", stderr="")

        with (
            patch("graft.stages.execute._git", side_effect=fake_git),
            patch("graft.stages.execute.run_agent", new_callable=AsyncMock),
            patch("graft.stages.execute._run_tests", return_value=(True, "ok")),
            patch("graft.stages.execute._run_lint", return_value=(True, "ok")),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            result = await execute_node(state, ui)

        # First git call should be checkout -b
        assert git_calls[0] == ("checkout", "-b", "feature/my-feat")
        assert result["feature_branch"] == "feature/my-feat"

    @pytest.mark.asyncio
    async def test_checks_out_existing_branch_on_create_failure(self):
        """If branch already exists, execute_node checks it out instead."""
        state = _make_state(
            build_plan=_simple_plan(1),
            feature_branch="feature/existing",
        )
        ui = _make_ui()
        git_calls = []

        def fake_git(repo_path, *args, check=True):
            git_calls.append(args)
            if args == ("checkout", "-b", "feature/existing"):
                return subprocess.CompletedProcess(
                    [], 1, stdout="", stderr="already exists"
                )
            return subprocess.CompletedProcess([], 0, stdout="", stderr="")

        with (
            patch("graft.stages.execute._git", side_effect=fake_git),
            patch("graft.stages.execute.run_agent", new_callable=AsyncMock),
            patch("graft.stages.execute._run_tests", return_value=(True, "ok")),
            patch("graft.stages.execute._run_lint", return_value=(True, "ok")),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            await execute_node(state, ui)

        # Second git call should be plain checkout
        assert git_calls[1] == ("checkout", "feature/existing")

    @pytest.mark.asyncio
    async def test_default_branch_name_from_project_id(self):
        """Uses project_id when feature_branch is not in state."""
        state = _make_state(build_plan=_simple_plan(1), project_id="feat_xyz")
        # Remove feature_branch from state so default is used
        state.pop("feature_branch", None)
        ui = _make_ui()
        git_calls = []

        def fake_git(repo_path, *args, check=True):
            git_calls.append(args)
            return subprocess.CompletedProcess([], 0, stdout="", stderr="")

        with (
            patch("graft.stages.execute._git", side_effect=fake_git),
            patch("graft.stages.execute.run_agent", new_callable=AsyncMock),
            patch("graft.stages.execute._run_tests", return_value=(True, "ok")),
            patch("graft.stages.execute._run_lint", return_value=(True, "ok")),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            result = await execute_node(state, ui)

        assert git_calls[0] == ("checkout", "-b", "feature/feat_xyz")
        assert result["feature_branch"] == "feature/feat_xyz"


class TestExecuteNodeUnitFlow:
    """execute_node unit implementation, commit, test, lint flow."""

    @pytest.mark.asyncio
    async def test_successful_unit_committed_and_kept(self):
        """A unit that passes tests and lint is committed and kept."""
        state = _make_state(build_plan=_simple_plan(1))
        ui = _make_ui()
        git_calls = []

        def fake_git(repo_path, *args, check=True):
            git_calls.append(args)
            return subprocess.CompletedProcess([], 0, stdout="", stderr="")

        with (
            patch("graft.stages.execute._git", side_effect=fake_git),
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute._run_tests", return_value=(True, "3 passed")),
            patch("graft.stages.execute._run_lint", return_value=(True, "Lint passed")),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            result = await execute_node(state, ui)

        mock_agent.assert_called_once()
        assert len(result["units_completed"]) == 1
        assert result["units_completed"][0]["unit_id"] == "unit_1"
        assert result["units_reverted"] == []
        assert result["units_skipped"] == []
        ui.unit_kept.assert_called_once()

    @pytest.mark.asyncio
    async def test_commit_add_and_message(self):
        """After agent runs, execute_node stages all files and commits with title."""
        state = _make_state(build_plan=_simple_plan(1))
        ui = _make_ui()
        git_calls = []

        def fake_git(repo_path, *args, check=True):
            git_calls.append(args)
            return subprocess.CompletedProcess([], 0, stdout="", stderr="")

        with (
            patch("graft.stages.execute._git", side_effect=fake_git),
            patch("graft.stages.execute.run_agent", new_callable=AsyncMock),
            patch("graft.stages.execute._run_tests", return_value=(True, "ok")),
            patch("graft.stages.execute._run_lint", return_value=(True, "ok")),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            await execute_node(state, ui)

        # After branch checkout, expect: add -A, commit -m "feat: Unit 1"
        assert ("add", "-A") in git_calls
        commit_call = [c for c in git_calls if c[0] == "commit" and "-m" in c]
        assert len(commit_call) >= 1
        assert "feat: Unit 1" in commit_call[0]

    @pytest.mark.asyncio
    async def test_reverts_on_test_failure(self):
        """When tests fail, the commit is reverted and the unit is marked reverted."""
        state = _make_state(build_plan=_simple_plan(1))
        ui = _make_ui()
        git_calls = []

        def fake_git(repo_path, *args, check=True):
            git_calls.append(args)
            return subprocess.CompletedProcess([], 0, stdout="", stderr="")

        with (
            patch("graft.stages.execute._git", side_effect=fake_git),
            patch("graft.stages.execute.run_agent", new_callable=AsyncMock),
            patch(
                "graft.stages.execute._run_tests", return_value=(False, "FAIL: test_x")
            ),
            patch("graft.stages.execute._run_lint", return_value=(True, "ok")),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            result = await execute_node(state, ui)

        assert len(result["units_reverted"]) == 1
        assert "Tests failed" in result["units_reverted"][0]["reason"]
        assert result["units_completed"] == []
        # Should have called git revert HEAD --no-edit
        assert ("revert", "HEAD", "--no-edit") in git_calls
        ui.unit_reverted.assert_called_once()

    @pytest.mark.asyncio
    async def test_reverts_when_no_changes_committed(self):
        """When commit returns nonzero (nothing to commit), unit is reverted."""
        state = _make_state(build_plan=_simple_plan(1))
        ui = _make_ui()

        def fake_git(repo_path, *args, check=True):
            if args[0] == "commit":
                return subprocess.CompletedProcess(
                    [], 1, stdout="", stderr="nothing to commit"
                )
            return subprocess.CompletedProcess([], 0, stdout="", stderr="")

        with (
            patch("graft.stages.execute._git", side_effect=fake_git),
            patch("graft.stages.execute.run_agent", new_callable=AsyncMock),
            patch("graft.stages.execute._run_tests", return_value=(True, "ok")),
            patch("graft.stages.execute._run_lint", return_value=(True, "ok")),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            result = await execute_node(state, ui)

        assert len(result["units_reverted"]) == 1
        assert "No changes" in result["units_reverted"][0]["reason"]

    @pytest.mark.asyncio
    async def test_amends_commit_after_lint_fix(self):
        """After lint passes, execute_node stages fixes and amends the commit."""
        state = _make_state(build_plan=_simple_plan(1))
        ui = _make_ui()
        git_calls = []

        def fake_git(repo_path, *args, check=True):
            git_calls.append(args)
            return subprocess.CompletedProcess([], 0, stdout="", stderr="")

        with (
            patch("graft.stages.execute._git", side_effect=fake_git),
            patch("graft.stages.execute.run_agent", new_callable=AsyncMock),
            patch("graft.stages.execute._run_tests", return_value=(True, "ok")),
            patch("graft.stages.execute._run_lint", return_value=(True, "Lint passed")),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            await execute_node(state, ui)

        # After lint, should stage and amend
        # Find the amend call
        amend_calls = [c for c in git_calls if "amend" in str(c)]
        assert len(amend_calls) >= 1
        assert ("commit", "--amend", "--no-edit") in git_calls

    @pytest.mark.asyncio
    async def test_agent_failure_reverts_unit(self):
        """When run_agent raises RuntimeError, the unit is reverted."""
        state = _make_state(build_plan=_simple_plan(1))
        ui = _make_ui()

        with (
            patch(
                "graft.stages.execute._git",
                return_value=subprocess.CompletedProcess([], 0),
            ),
            patch(
                "graft.stages.execute.run_agent",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Agent crashed"),
            ),
            patch("graft.stages.execute._run_tests"),
            patch("graft.stages.execute._run_lint"),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            result = await execute_node(state, ui)

        assert len(result["units_reverted"]) == 1
        assert "Agent crashed" in result["units_reverted"][0]["reason"]
        ui.unit_reverted.assert_called_once()


class TestExecuteNodeDependencies:
    """execute_node dependency handling."""

    @pytest.mark.asyncio
    async def test_skips_unit_with_unmet_dependencies(self):
        """Units whose dependencies were not completed are skipped."""
        plan = [
            {
                "unit_id": "unit_1",
                "title": "Base Unit",
                "depends_on": [],
            },
            {
                "unit_id": "unit_2",
                "title": "Depends on Base",
                "depends_on": ["unit_1"],
            },
        ]
        state = _make_state(build_plan=plan)
        ui = _make_ui()

        # unit_1 agent fails, so unit_2 dep is unmet
        call_count = 0

        async def failing_agent(**kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        with (
            patch(
                "graft.stages.execute._git",
                return_value=subprocess.CompletedProcess([], 0),
            ),
            patch("graft.stages.execute.run_agent", side_effect=failing_agent),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            result = await execute_node(state, ui)

        # unit_1 reverted because agent failed, unit_2 skipped because dep unmet
        assert len(result["units_reverted"]) == 1
        assert result["units_reverted"][0]["unit_id"] == "unit_1"
        assert len(result["units_skipped"]) == 1
        assert result["units_skipped"][0]["unit_id"] == "unit_2"
        assert "unit_1" in result["units_skipped"][0]["reason"]

    @pytest.mark.asyncio
    async def test_met_dependencies_proceed(self):
        """Units with met dependencies proceed normally."""
        plan = [
            {
                "unit_id": "unit_1",
                "title": "Base",
                "depends_on": [],
            },
            {
                "unit_id": "unit_2",
                "title": "Dependent",
                "depends_on": ["unit_1"],
            },
        ]
        state = _make_state(build_plan=plan)
        ui = _make_ui()

        def fake_git(repo_path, *args, check=True):
            return subprocess.CompletedProcess([], 0, stdout="", stderr="")

        with (
            patch("graft.stages.execute._git", side_effect=fake_git),
            patch("graft.stages.execute.run_agent", new_callable=AsyncMock),
            patch("graft.stages.execute._run_tests", return_value=(True, "ok")),
            patch("graft.stages.execute._run_lint", return_value=(True, "ok")),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            result = await execute_node(state, ui)

        assert len(result["units_completed"]) == 2
        assert result["units_skipped"] == []


class TestExecuteNodeArtifacts:
    """execute_node artifact and logging."""

    @pytest.mark.asyncio
    async def test_saves_execution_log(self):
        """execute_node saves execution_log.json with correct structure."""
        plan = _simple_plan(2)
        state = _make_state(build_plan=plan)
        ui = _make_ui()

        def fake_git(repo_path, *args, check=True):
            return subprocess.CompletedProcess([], 0, stdout="", stderr="")

        saved_artifacts = {}

        def capture_artifact(project_dir, name, content):
            saved_artifacts[name] = content

        with (
            patch("graft.stages.execute._git", side_effect=fake_git),
            patch("graft.stages.execute.run_agent", new_callable=AsyncMock),
            patch("graft.stages.execute._run_tests", return_value=(True, "ok")),
            patch("graft.stages.execute._run_lint", return_value=(True, "ok")),
            patch("graft.stages.execute.save_artifact", side_effect=capture_artifact),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            await execute_node(state, ui)

        assert "execution_log.json" in saved_artifacts
        log = json.loads(saved_artifacts["execution_log.json"])
        assert log["units_completed"] == 2
        assert log["units_reverted"] == 0
        assert log["units_skipped"] == 0
        assert log["total_planned"] == 2
        assert len(log["completed"]) == 2
        assert len(log["reverted"]) == 0
        assert len(log["skipped"]) == 0

    @pytest.mark.asyncio
    async def test_marks_stage_complete(self):
        """execute_node calls mark_stage_complete at the end."""
        state = _make_state(build_plan=_simple_plan(1))
        ui = _make_ui()

        def fake_git(repo_path, *args, check=True):
            return subprocess.CompletedProcess([], 0, stdout="", stderr="")

        with (
            patch("graft.stages.execute._git", side_effect=fake_git),
            patch("graft.stages.execute.run_agent", new_callable=AsyncMock),
            patch("graft.stages.execute._run_tests", return_value=(True, "ok")),
            patch("graft.stages.execute._run_lint", return_value=(True, "ok")),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete") as mock_mark,
        ):
            await execute_node(state, ui)

        mock_mark.assert_called_once_with("/tmp/test-project", "execute")

    @pytest.mark.asyncio
    async def test_execution_log_mixed_results(self):
        """Execution log tracks completed, reverted, and skipped units."""
        plan = [
            {"unit_id": "ok_unit", "title": "OK", "depends_on": []},
            {"unit_id": "fail_unit", "title": "Fail", "depends_on": []},
            {"unit_id": "skip_unit", "title": "Skip", "depends_on": ["fail_unit"]},
        ]
        state = _make_state(build_plan=plan)
        ui = _make_ui()

        call_idx = 0

        async def selective_agent(**kwargs):
            nonlocal call_idx
            call_idx += 1
            # Second call (fail_unit) raises
            if call_idx == 2:
                raise RuntimeError("boom")

        def fake_git(repo_path, *args, check=True):
            return subprocess.CompletedProcess([], 0, stdout="", stderr="")

        saved_artifacts = {}

        def capture_artifact(project_dir, name, content):
            saved_artifacts[name] = content

        with (
            patch("graft.stages.execute._git", side_effect=fake_git),
            patch("graft.stages.execute.run_agent", side_effect=selective_agent),
            patch("graft.stages.execute._run_tests", return_value=(True, "ok")),
            patch("graft.stages.execute._run_lint", return_value=(True, "ok")),
            patch("graft.stages.execute.save_artifact", side_effect=capture_artifact),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            result = await execute_node(state, ui)

        log = json.loads(saved_artifacts["execution_log.json"])
        assert log["units_completed"] == 1
        assert log["units_reverted"] == 1
        assert log["units_skipped"] == 1


class TestExecuteNodeAgentPrompt:
    """execute_node passes correct prompt to run_agent."""

    @pytest.mark.asyncio
    async def test_agent_called_with_unit_details(self):
        """run_agent receives correct persona, system prompt, and user prompt."""
        plan = [
            {
                "unit_id": "feat_01",
                "title": "Add login endpoint",
                "description": "Create POST /login",
                "pattern_reference": "src/routes/auth.py",
                "tests_included": True,
                "acceptance_criteria": ["Returns 200 on valid creds"],
                "depends_on": [],
            }
        ]
        state = _make_state(build_plan=plan, model="sonnet")
        ui = _make_ui()

        def fake_git(repo_path, *args, check=True):
            return subprocess.CompletedProcess([], 0, stdout="", stderr="")

        with (
            patch("graft.stages.execute._git", side_effect=fake_git),
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute._run_tests", return_value=(True, "ok")),
            patch("graft.stages.execute._run_lint", return_value=(True, "ok")),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            await execute_node(state, ui)

        mock_agent.assert_called_once()
        kwargs = mock_agent.call_args[1]
        assert "feat_01" in kwargs["persona"]
        assert "Add login endpoint" in kwargs["user_prompt"]
        assert "Create POST /login" in kwargs["user_prompt"]
        assert "src/routes/auth.py" in kwargs["user_prompt"]
        assert "Returns 200 on valid creds" in kwargs["user_prompt"]
        assert "tests" in kwargs["user_prompt"].lower()
        assert kwargs["model"] == "sonnet"
        assert kwargs["stage"] == "execute_feat_01"
        assert kwargs["cwd"] == "/tmp/test-repo"

    @pytest.mark.asyncio
    async def test_agent_prompt_without_pattern_reference(self):
        """When pattern_reference is empty, it's not in the prompt."""
        plan = [
            {
                "unit_id": "unit_1",
                "title": "Simple unit",
                "description": "Do something",
                "pattern_reference": "",
                "tests_included": False,
                "acceptance_criteria": [],
                "depends_on": [],
            }
        ]
        state = _make_state(build_plan=plan)
        ui = _make_ui()

        def fake_git(repo_path, *args, check=True):
            return subprocess.CompletedProcess([], 0, stdout="", stderr="")

        with (
            patch("graft.stages.execute._git", side_effect=fake_git),
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute._run_tests", return_value=(True, "ok")),
            patch("graft.stages.execute._run_lint", return_value=(True, "ok")),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            await execute_node(state, ui)

        prompt = mock_agent.call_args[1]["user_prompt"]
        assert "PATTERN REFERENCE" not in prompt

    @pytest.mark.asyncio
    async def test_ui_lifecycle_calls(self):
        """execute_node calls stage_start, unit_start, unit_kept, stage_done."""
        state = _make_state(build_plan=_simple_plan(1))
        ui = _make_ui()

        def fake_git(repo_path, *args, check=True):
            return subprocess.CompletedProcess([], 0, stdout="", stderr="")

        with (
            patch("graft.stages.execute._git", side_effect=fake_git),
            patch("graft.stages.execute.run_agent", new_callable=AsyncMock),
            patch("graft.stages.execute._run_tests", return_value=(True, "ok")),
            patch("graft.stages.execute._run_lint", return_value=(True, "ok")),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            await execute_node(state, ui)

        ui.stage_start.assert_called_once_with("execute")
        ui.unit_start.assert_called_once_with("unit_1", "Unit 1", 1, 1)
        ui.unit_kept.assert_called_once()
        ui.stage_done.assert_called_once_with("execute")


class TestExecuteNodeMultipleUnits:
    """execute_node with multiple units."""

    @pytest.mark.asyncio
    async def test_iterates_all_units(self):
        """execute_node processes each unit in order."""
        state = _make_state(build_plan=_simple_plan(3))
        ui = _make_ui()

        agent_calls = []

        async def tracking_agent(**kwargs):
            agent_calls.append(kwargs["stage"])

        def fake_git(repo_path, *args, check=True):
            return subprocess.CompletedProcess([], 0, stdout="", stderr="")

        with (
            patch("graft.stages.execute._git", side_effect=fake_git),
            patch("graft.stages.execute.run_agent", side_effect=tracking_agent),
            patch("graft.stages.execute._run_tests", return_value=(True, "ok")),
            patch("graft.stages.execute._run_lint", return_value=(True, "ok")),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            result = await execute_node(state, ui)

        assert len(agent_calls) == 3
        assert agent_calls == ["execute_unit_1", "execute_unit_2", "execute_unit_3"]
        assert len(result["units_completed"]) == 3

    @pytest.mark.asyncio
    async def test_continues_after_one_unit_fails(self):
        """If one unit fails, remaining independent units still execute."""
        plan = [
            {"unit_id": "a", "title": "A", "depends_on": []},
            {"unit_id": "b", "title": "B", "depends_on": []},
            {"unit_id": "c", "title": "C", "depends_on": []},
        ]
        state = _make_state(build_plan=plan)
        ui = _make_ui()

        call_idx = 0

        async def selective_agent(**kwargs):
            nonlocal call_idx
            call_idx += 1
            if call_idx == 2:
                raise RuntimeError("boom")

        def fake_git(repo_path, *args, check=True):
            return subprocess.CompletedProcess([], 0, stdout="", stderr="")

        with (
            patch("graft.stages.execute._git", side_effect=fake_git),
            patch("graft.stages.execute.run_agent", side_effect=selective_agent),
            patch("graft.stages.execute._run_tests", return_value=(True, "ok")),
            patch("graft.stages.execute._run_lint", return_value=(True, "ok")),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            result = await execute_node(state, ui)

        assert len(result["units_completed"]) == 2
        assert len(result["units_reverted"]) == 1

    @pytest.mark.asyncio
    async def test_completed_unit_metadata(self):
        """Completed unit entries include correct metadata."""
        plan = [
            {
                "unit_id": "u1",
                "title": "My Unit",
                "category": "backend",
                "tests_included": True,
                "depends_on": [],
            }
        ]
        state = _make_state(build_plan=plan)
        ui = _make_ui()

        def fake_git(repo_path, *args, check=True):
            return subprocess.CompletedProcess([], 0, stdout="", stderr="")

        with (
            patch("graft.stages.execute._git", side_effect=fake_git),
            patch("graft.stages.execute.run_agent", new_callable=AsyncMock),
            patch("graft.stages.execute._run_tests", return_value=(True, "ok")),
            patch("graft.stages.execute._run_lint", return_value=(True, "ok")),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            result = await execute_node(state, ui)

        completed = result["units_completed"][0]
        assert completed["unit_id"] == "u1"
        assert completed["title"] == "My Unit"
        assert completed["category"] == "backend"
        assert completed["tests_included"] is True
