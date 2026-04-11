"""Tests for graft.stages.execute."""

import subprocess
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
# Existing dependency-ordering tests
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
# Additional dependency-ordering edge cases
# ---------------------------------------------------------------------------


def test_order_by_dependencies_empty_plan():
    """Empty plan returns empty list."""
    assert _order_by_dependencies([]) == []


def test_order_by_dependencies_single_unit():
    """Single unit is returned as-is."""
    plan = [{"unit_id": "only"}]
    result = _order_by_dependencies(plan)
    assert len(result) == 1
    assert result[0]["unit_id"] == "only"


def test_order_by_dependencies_diamond():
    """Diamond dependency graph: a → b, a → c, b → d, c → d."""
    plan = [
        {"unit_id": "d", "depends_on": ["b", "c"]},
        {"unit_id": "b", "depends_on": ["a"]},
        {"unit_id": "c", "depends_on": ["a"]},
        {"unit_id": "a", "depends_on": []},
    ]
    result = _order_by_dependencies(plan)
    ids = [u["unit_id"] for u in result]
    assert ids.index("a") < ids.index("b")
    assert ids.index("a") < ids.index("c")
    assert ids.index("b") < ids.index("d")
    assert ids.index("c") < ids.index("d")


def test_order_by_dependencies_missing_unit_id():
    """Units without unit_id still get processed without error."""
    plan = [
        {"depends_on": []},
        {"unit_id": "b", "depends_on": []},
    ]
    result = _order_by_dependencies(plan)
    assert len(result) == 2


def test_order_by_dependencies_unresolvable_dep():
    """A unit depending on a non-existent ID is appended after max passes."""
    plan = [
        {"unit_id": "a", "depends_on": ["nonexistent"]},
        {"unit_id": "b", "depends_on": []},
    ]
    result = _order_by_dependencies(plan)
    ids = [u["unit_id"] for u in result]
    # b should come first (no deps), then a is forced in
    assert ids == ["b", "a"]


# ---------------------------------------------------------------------------
# _git() helper
# ---------------------------------------------------------------------------


class TestGitHelper:
    """Tests for the _git() subprocess wrapper."""

    @patch("graft.stages.execute.subprocess.run")
    def test_git_basic_command(self, mock_run):
        """_git invokes git with correct args and cwd."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git", "status"], returncode=0, stdout="clean", stderr=""
        )
        result = _git("/some/repo", "status")
        mock_run.assert_called_once_with(
            ["git", "status"],
            cwd="/some/repo",
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        assert result.returncode == 0

    @patch("graft.stages.execute.subprocess.run")
    def test_git_multiple_args(self, mock_run):
        """_git passes through all positional arguments."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        _git("/repo", "commit", "-m", "feat: something")
        mock_run.assert_called_once_with(
            ["git", "commit", "-m", "feat: something"],
            cwd="/repo",
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )

    @patch("graft.stages.execute.subprocess.run")
    def test_git_check_false(self, mock_run):
        """_git with check=False does not raise on non-zero returncode."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="error"
        )
        result = _git("/repo", "checkout", "-b", "branch", check=False)
        mock_run.assert_called_once_with(
            ["git", "checkout", "-b", "branch"],
            cwd="/repo",
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 1

    @patch("graft.stages.execute.subprocess.run")
    def test_git_check_true_raises(self, mock_run):
        """_git with check=True raises CalledProcessError on failure."""
        mock_run.side_effect = subprocess.CalledProcessError(
            128, ["git", "bad"]
        )
        with pytest.raises(subprocess.CalledProcessError):
            _git("/repo", "bad")


# ---------------------------------------------------------------------------
# _run_tests()
# ---------------------------------------------------------------------------


class TestRunTests:
    """Tests for _run_tests() with mocked subprocess."""

    @patch("graft.stages.execute.subprocess.run")
    def test_tests_pass(self, mock_run):
        """Passing tests return (True, output)."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="5 passed", stderr=""
        )
        passed, output = _run_tests("/repo")
        assert passed is True
        assert "5 passed" in output

    @patch("graft.stages.execute.subprocess.run")
    def test_tests_fail(self, mock_run):
        """Failing tests return (False, output)."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="FAILED", stderr="2 errors"
        )
        passed, output = _run_tests("/repo")
        assert passed is False
        assert "FAILED" in output

    @patch("graft.stages.execute.subprocess.run")
    def test_tests_timeout(self, mock_run):
        """Timeout returns (False, timeout message)."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="bash", timeout=300)
        passed, output = _run_tests("/repo")
        assert passed is False
        assert "timed out" in output.lower()

    @patch("graft.stages.execute.subprocess.run")
    def test_tests_no_runner(self, mock_run):
        """FileNotFoundError returns (True, skip message)."""
        mock_run.side_effect = FileNotFoundError("bash not found")
        passed, output = _run_tests("/repo")
        assert passed is True
        assert "skipping" in output.lower()

    @patch("graft.stages.execute.subprocess.run")
    def test_tests_output_truncated(self, mock_run):
        """Long output is truncated to last 2000 chars."""
        long_output = "x" * 5000
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=long_output, stderr=""
        )
        passed, output = _run_tests("/repo")
        assert passed is True
        assert len(output) <= 2000

    @patch("graft.stages.execute.subprocess.run")
    def test_tests_uses_verify_script(self, mock_run):
        """_run_tests passes the VERIFY_SCRIPT to bash."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )
        _run_tests("/my/repo")
        mock_run.assert_called_once_with(
            ["bash", "-c", VERIFY_SCRIPT],
            cwd="/my/repo",
            capture_output=True,
            text=True,
            timeout=300,
        )


# ---------------------------------------------------------------------------
# _run_lint()
# ---------------------------------------------------------------------------


class TestRunLint:
    """Tests for _run_lint() with mocked subprocess."""

    @patch("graft.stages.execute.subprocess.run")
    def test_lint_first_linter_passes(self, mock_run):
        """If the first linter passes, return immediately."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )
        passed, output = _run_lint("/repo")
        assert passed is True
        assert output == "Lint passed"
        # Only the first linter was tried
        assert mock_run.call_count == 1

    @patch("graft.stages.execute.subprocess.run")
    def test_lint_falls_through_on_failure(self, mock_run):
        """If a linter returns non-zero, try the next one."""
        mock_run.side_effect = [
            subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="err"),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
        ]
        passed, output = _run_lint("/repo")
        assert passed is True
        assert mock_run.call_count == 2

    @patch("graft.stages.execute.subprocess.run")
    def test_lint_all_fail(self, mock_run):
        """If all linters fail, return (True, 'No linter found')."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr=""
        )
        passed, output = _run_lint("/repo")
        assert passed is True
        assert "No linter found" in output

    @patch("graft.stages.execute.subprocess.run")
    def test_lint_file_not_found_skips(self, mock_run):
        """FileNotFoundError for a linter moves to the next."""
        mock_run.side_effect = [
            FileNotFoundError("npx not found"),
            FileNotFoundError("python not found"),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
        ]
        passed, output = _run_lint("/repo")
        assert passed is True
        assert output == "Lint passed"

    @patch("graft.stages.execute.subprocess.run")
    def test_lint_timeout_skips(self, mock_run):
        """TimeoutExpired for a linter moves to the next."""
        mock_run.side_effect = [
            subprocess.TimeoutExpired(cmd="npx", timeout=60),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
        ]
        passed, output = _run_lint("/repo")
        assert passed is True

    @patch("graft.stages.execute.subprocess.run")
    def test_lint_all_not_found(self, mock_run):
        """If all linters raise FileNotFoundError, return skip message."""
        mock_run.side_effect = FileNotFoundError("not found")
        passed, output = _run_lint("/repo")
        assert passed is True
        assert "No linter found" in output


# ---------------------------------------------------------------------------
# VERIFY_SCRIPT content checks
# ---------------------------------------------------------------------------


class TestVerifyScript:
    """Verify the VERIFY_SCRIPT detects project types correctly."""

    def test_npm_detection(self):
        assert "package.json" in VERIFY_SCRIPT
        assert "npm test" in VERIFY_SCRIPT

    def test_python_detection(self):
        assert "pyproject.toml" in VERIFY_SCRIPT
        assert "setup.py" in VERIFY_SCRIPT
        assert "requirements.txt" in VERIFY_SCRIPT
        assert "pytest" in VERIFY_SCRIPT

    def test_cargo_detection(self):
        assert "Cargo.toml" in VERIFY_SCRIPT
        assert "cargo test" in VERIFY_SCRIPT

    def test_go_detection(self):
        assert "go.mod" in VERIFY_SCRIPT
        assert "go test" in VERIFY_SCRIPT


# ---------------------------------------------------------------------------
# execute_node() — full async integration tests with mocks
# ---------------------------------------------------------------------------


def _make_ui() -> MagicMock:
    """Create a mock UI object with all methods used by execute_node."""
    ui = MagicMock()
    ui.stage_start = MagicMock()
    ui.stage_done = MagicMock()
    ui.error = MagicMock()
    ui.unit_start = MagicMock()
    ui.unit_kept = MagicMock()
    ui.unit_reverted = MagicMock()
    return ui


def _make_state(
    repo_path: str = "/tmp/repo",
    project_dir: str = "/tmp/project",
    build_plan: list | None = None,
    **overrides,
) -> dict:
    """Create a minimal FeatureState dict."""
    state = {
        "repo_path": repo_path,
        "project_dir": project_dir,
        "project_id": "test_session",
        "build_plan": build_plan or [],
        "codebase_profile": {},
        "feature_spec": {},
        "max_agent_turns": 10,
    }
    state.update(overrides)
    return state


def _success_git(*args, **kwargs):
    """Simulate a successful git call."""
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr="")


def _fail_git(*args, **kwargs):
    """Simulate a failed git call (no changes to commit)."""
    return subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="nothing to commit"
    )


class TestExecuteNode:
    """Tests for the main execute_node() async function."""

    @pytest.mark.asyncio
    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    async def test_empty_plan(self, mock_save, mock_mark):
        """Empty build plan logs error and returns early."""
        ui = _make_ui()
        state = _make_state(build_plan=[])

        result = await execute_node(state, ui)

        ui.error.assert_called_once()
        assert "No build plan" in ui.error.call_args[0][0]
        mock_mark.assert_called_once_with("/tmp/project", "execute")
        assert result["current_stage"] == "execute"

    @pytest.mark.asyncio
    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute._run_lint", return_value=(True, "Lint passed"))
    @patch("graft.stages.execute._run_tests", return_value=(True, "5 passed"))
    @patch("graft.stages.execute._git")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    async def test_single_unit_success(
        self, mock_agent, mock_git, mock_tests, mock_lint, mock_save, mock_mark
    ):
        """Happy path: one unit is implemented, committed, tested, and kept."""
        mock_git.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )
        plan = [
            {
                "unit_id": "feat_01",
                "title": "Add button",
                "description": "Add a submit button",
                "pattern_reference": "src/components/Button.tsx",
                "tests_included": True,
                "acceptance_criteria": ["Button renders"],
            }
        ]
        ui = _make_ui()
        state = _make_state(build_plan=plan)

        result = await execute_node(state, ui)

        # Agent was called
        mock_agent.assert_called_once()
        # Git operations: checkout -b, add -A, commit, add -A (lint), commit --amend
        assert mock_git.call_count >= 3
        # Unit was kept
        assert len(result["units_completed"]) == 1
        assert result["units_completed"][0]["unit_id"] == "feat_01"
        assert len(result["units_reverted"]) == 0
        assert len(result["units_skipped"]) == 0
        ui.unit_kept.assert_called_once()

    @pytest.mark.asyncio
    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute._git")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    async def test_agent_failure_reverts_unit(
        self, mock_agent, mock_git, mock_save, mock_mark
    ):
        """If the agent raises RuntimeError, the unit is reverted."""
        mock_git.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        mock_agent.side_effect = RuntimeError("API timeout")

        plan = [{"unit_id": "feat_01", "title": "Broken unit"}]
        ui = _make_ui()
        state = _make_state(build_plan=plan)

        result = await execute_node(state, ui)

        assert len(result["units_reverted"]) == 1
        assert "API timeout" in result["units_reverted"][0]["reason"]
        assert len(result["units_completed"]) == 0
        ui.unit_reverted.assert_called_once()

    @pytest.mark.asyncio
    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute._git")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    async def test_no_changes_reverts_unit(
        self, mock_agent, mock_git, mock_save, mock_mark
    ):
        """If commit produces no changes, the unit is reverted."""

        def git_side_effect(repo_path, *args, check=True):
            # Make "commit" fail (nothing to commit) but others succeed
            if args and args[0] == "commit":
                return subprocess.CompletedProcess(
                    args=[], returncode=1, stdout="", stderr="nothing to commit"
                )
            return subprocess.CompletedProcess(
                args=[], returncode=0, stdout="ok", stderr=""
            )

        mock_git.side_effect = git_side_effect

        plan = [{"unit_id": "feat_01", "title": "No-op unit"}]
        ui = _make_ui()
        state = _make_state(build_plan=plan)

        result = await execute_node(state, ui)

        assert len(result["units_reverted"]) == 1
        assert "No changes" in result["units_reverted"][0]["reason"]

    @pytest.mark.asyncio
    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute._run_tests", return_value=(False, "FAILED: assert 1==2"))
    @patch("graft.stages.execute._git")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    async def test_test_failure_triggers_revert(
        self, mock_agent, mock_git, mock_tests, mock_save, mock_mark
    ):
        """If tests fail after commit, the commit is reverted with git revert."""
        mock_git.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        plan = [{"unit_id": "feat_01", "title": "Fails tests"}]
        ui = _make_ui()
        state = _make_state(build_plan=plan)

        result = await execute_node(state, ui)

        assert len(result["units_reverted"]) == 1
        assert "Tests failed" in result["units_reverted"][0]["reason"]
        # Verify git revert HEAD --no-edit was called
        revert_calls = [
            c for c in mock_git.call_args_list
            if len(c[0]) >= 2 and c[0][1] == "revert"
        ]
        assert len(revert_calls) == 1
        assert "HEAD" in revert_calls[0][0]
        assert "--no-edit" in revert_calls[0][0]

    @pytest.mark.asyncio
    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute._run_lint", return_value=(True, "Lint passed"))
    @patch("graft.stages.execute._run_tests", return_value=(True, "ok"))
    @patch("graft.stages.execute._git")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    async def test_lint_autofix_amends_commit(
        self, mock_agent, mock_git, mock_tests, mock_lint, mock_save, mock_mark
    ):
        """After lint passes, staged files are amended into the commit."""
        mock_git.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        plan = [{"unit_id": "feat_01", "title": "Lint-fixed unit"}]
        ui = _make_ui()
        state = _make_state(build_plan=plan)

        await execute_node(state, ui)

        # Look for the amend commit call
        amend_calls = [
            c for c in mock_git.call_args_list
            if len(c[0]) >= 2 and "--amend" in c[0]
        ]
        assert len(amend_calls) == 1
        assert "--no-edit" in amend_calls[0][0]

    @pytest.mark.asyncio
    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute._run_lint", return_value=(True, "Lint passed"))
    @patch("graft.stages.execute._run_tests", return_value=(True, "ok"))
    @patch("graft.stages.execute._git")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    async def test_unmet_deps_skip_unit(
        self, mock_agent, mock_git, mock_tests, mock_lint, mock_save, mock_mark
    ):
        """Units with unmet dependencies are skipped."""
        mock_git.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        plan = [
            {"unit_id": "feat_02", "title": "Depends on missing", "depends_on": ["feat_01"]},
        ]
        ui = _make_ui()
        state = _make_state(build_plan=plan)

        result = await execute_node(state, ui)

        assert len(result["units_skipped"]) == 1
        assert "feat_01" in result["units_skipped"][0]["reason"]
        assert len(result["units_completed"]) == 0
        # Agent should never be called for skipped units
        mock_agent.assert_not_called()

    @pytest.mark.asyncio
    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute._run_lint", return_value=(True, "Lint passed"))
    @patch("graft.stages.execute._run_tests", return_value=(True, "ok"))
    @patch("graft.stages.execute._git")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    async def test_multiple_units_ordered_execution(
        self, mock_agent, mock_git, mock_tests, mock_lint, mock_save, mock_mark
    ):
        """Multiple units are executed in dependency order, tracking completed IDs."""
        mock_git.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        plan = [
            {"unit_id": "feat_02", "title": "Second", "depends_on": ["feat_01"]},
            {"unit_id": "feat_01", "title": "First", "depends_on": []},
        ]
        ui = _make_ui()
        state = _make_state(build_plan=plan)

        result = await execute_node(state, ui)

        assert len(result["units_completed"]) == 2
        ids = [u["unit_id"] for u in result["units_completed"]]
        assert ids == ["feat_01", "feat_02"]

    @pytest.mark.asyncio
    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute._run_lint", return_value=(True, "Lint passed"))
    @patch("graft.stages.execute._run_tests", return_value=(True, "ok"))
    @patch("graft.stages.execute._git")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    async def test_branch_creation_fallback(
        self, mock_agent, mock_git, mock_tests, mock_lint, mock_save, mock_mark
    ):
        """If branch already exists, falls back to git checkout."""
        call_count = 0

        def git_side_effect(repo_path, *args, check=True):
            nonlocal call_count
            call_count += 1
            # First call is checkout -b (branch creation) — fail
            if args and args[0] == "checkout" and "-b" in args:
                return subprocess.CompletedProcess(
                    args=[], returncode=128, stdout="", stderr="branch already exists"
                )
            return subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            )

        mock_git.side_effect = git_side_effect

        plan = [{"unit_id": "feat_01", "title": "On existing branch"}]
        ui = _make_ui()
        state = _make_state(build_plan=plan)

        result = await execute_node(state, ui)

        # Should see checkout -b fail, then checkout (without -b) as fallback
        checkout_calls = [
            c for c in mock_git.call_args_list
            if len(c[0]) >= 2 and c[0][1] == "checkout"
        ]
        assert len(checkout_calls) == 2
        # First: checkout -b
        assert "-b" in checkout_calls[0][0]
        # Second: plain checkout (fallback)
        assert "-b" not in checkout_calls[1][0]

    @pytest.mark.asyncio
    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute._run_lint", return_value=(True, "Lint passed"))
    @patch("graft.stages.execute._run_tests", return_value=(True, "ok"))
    @patch("graft.stages.execute._git")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    async def test_feature_branch_from_state(
        self, mock_agent, mock_git, mock_tests, mock_lint, mock_save, mock_mark
    ):
        """Branch name comes from state['feature_branch'] when set."""
        mock_git.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        plan = [{"unit_id": "feat_01", "title": "Custom branch"}]
        ui = _make_ui()
        state = _make_state(
            build_plan=plan,
            feature_branch="feature/custom-branch",
        )

        result = await execute_node(state, ui)

        assert result["feature_branch"] == "feature/custom-branch"
        # First git call should use the custom branch name
        first_call = mock_git.call_args_list[0]
        assert "feature/custom-branch" in first_call[0]

    @pytest.mark.asyncio
    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute._run_lint", return_value=(True, "Lint passed"))
    @patch("graft.stages.execute._run_tests", return_value=(True, "ok"))
    @patch("graft.stages.execute._git")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    async def test_execution_log_saved(
        self, mock_agent, mock_git, mock_tests, mock_lint, mock_save, mock_mark
    ):
        """Execution log artifact is saved with correct counts."""
        mock_git.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        plan = [
            {"unit_id": "feat_01", "title": "Unit A"},
            {"unit_id": "feat_02", "title": "Unit B"},
        ]
        ui = _make_ui()
        state = _make_state(build_plan=plan)

        await execute_node(state, ui)

        # save_artifact should be called with execution_log.json
        mock_save.assert_called_once()
        call_args = mock_save.call_args
        assert call_args[0][0] == "/tmp/project"
        assert call_args[0][1] == "execution_log.json"
        import json
        log = json.loads(call_args[0][2])
        assert log["units_completed"] == 2
        assert log["total_planned"] == 2

    @pytest.mark.asyncio
    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute._run_lint", return_value=(True, "Lint passed"))
    @patch("graft.stages.execute._run_tests", return_value=(True, "ok"))
    @patch("graft.stages.execute._git")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    async def test_reverted_unit_not_in_completed_ids(
        self, mock_agent, mock_git, mock_tests, mock_lint, mock_save, mock_mark
    ):
        """A reverted unit's ID is NOT added to completed_ids, so dependents are skipped."""
        call_index = 0

        def agent_side_effect(**kwargs):
            nonlocal call_index
            call_index += 1
            if call_index == 1:
                raise RuntimeError("Agent crash")
            return MagicMock()

        mock_agent.side_effect = agent_side_effect
        mock_git.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        plan = [
            {"unit_id": "feat_01", "title": "Will fail", "depends_on": []},
            {"unit_id": "feat_02", "title": "Depends on failed", "depends_on": ["feat_01"]},
        ]
        ui = _make_ui()
        state = _make_state(build_plan=plan)

        result = await execute_node(state, ui)

        # feat_01 reverted, feat_02 skipped because dep not met
        assert len(result["units_reverted"]) == 1
        assert result["units_reverted"][0]["unit_id"] == "feat_01"
        assert len(result["units_skipped"]) == 1
        assert result["units_skipped"][0]["unit_id"] == "feat_02"

    @pytest.mark.asyncio
    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute._run_lint", return_value=(True, "Lint passed"))
    @patch("graft.stages.execute._run_tests", return_value=(True, "ok"))
    @patch("graft.stages.execute._git")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    async def test_unit_id_defaults_to_index(
        self, mock_agent, mock_git, mock_tests, mock_lint, mock_save, mock_mark
    ):
        """Units without unit_id get default feat_NN naming."""
        mock_git.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        plan = [{"title": "No ID unit"}]  # No unit_id key
        ui = _make_ui()
        state = _make_state(build_plan=plan)

        result = await execute_node(state, ui)

        assert len(result["units_completed"]) == 1
        # The unit_id should default to "feat_01"
        assert result["units_completed"][0]["unit_id"] == "feat_01"

    @pytest.mark.asyncio
    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute._run_lint", return_value=(True, "Lint passed"))
    @patch("graft.stages.execute._run_tests", return_value=(True, "ok"))
    @patch("graft.stages.execute._git")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    async def test_agent_prompt_includes_pattern_reference(
        self, mock_agent, mock_git, mock_tests, mock_lint, mock_save, mock_mark
    ):
        """The agent prompt includes the pattern_reference and instructions."""
        mock_git.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        plan = [
            {
                "unit_id": "feat_01",
                "title": "Styled component",
                "description": "Create a styled button",
                "pattern_reference": "src/components/Card.tsx",
                "tests_included": True,
                "acceptance_criteria": ["Renders correctly"],
            }
        ]
        ui = _make_ui()
        state = _make_state(build_plan=plan)

        await execute_node(state, ui)

        agent_kwargs = mock_agent.call_args[1]
        prompt = agent_kwargs["user_prompt"]
        assert "src/components/Card.tsx" in prompt
        assert "Read this file FIRST" in prompt
        assert "Write co-located tests" in prompt
        assert "Renders correctly" in prompt
        assert "Styled component" in prompt

    @pytest.mark.asyncio
    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute._run_lint", return_value=(True, "Lint passed"))
    @patch("graft.stages.execute._run_tests", return_value=(True, "ok"))
    @patch("graft.stages.execute._git")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    async def test_commit_message_includes_title(
        self, mock_agent, mock_git, mock_tests, mock_lint, mock_save, mock_mark
    ):
        """The git commit message uses 'feat: <title>'."""
        mock_git.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        plan = [{"unit_id": "feat_01", "title": "Add dark mode toggle"}]
        ui = _make_ui()
        state = _make_state(build_plan=plan)

        await execute_node(state, ui)

        commit_calls = [
            c for c in mock_git.call_args_list
            if len(c[0]) >= 2 and c[0][1] == "commit" and "--amend" not in c[0]
        ]
        assert len(commit_calls) == 1
        assert "feat: Add dark mode toggle" in commit_calls[0][0]

    @pytest.mark.asyncio
    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute._run_lint", return_value=(True, "Lint passed"))
    @patch("graft.stages.execute._run_tests", return_value=(True, "ok"))
    @patch("graft.stages.execute._git")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    async def test_stage_lifecycle_signals(
        self, mock_agent, mock_git, mock_tests, mock_lint, mock_save, mock_mark
    ):
        """execute_node calls ui.stage_start and ui.stage_done."""
        mock_git.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        plan = [{"unit_id": "feat_01", "title": "X"}]
        ui = _make_ui()
        state = _make_state(build_plan=plan)

        await execute_node(state, ui)

        ui.stage_start.assert_called_once_with("execute")
        ui.stage_done.assert_called_once_with("execute")
        mock_mark.assert_called_once_with("/tmp/project", "execute")

    @pytest.mark.asyncio
    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute._run_lint", return_value=(True, "Lint passed"))
    @patch("graft.stages.execute._run_tests", return_value=(True, "ok"))
    @patch("graft.stages.execute._git")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    async def test_atomic_commit_per_unit(
        self, mock_agent, mock_git, mock_tests, mock_lint, mock_save, mock_mark
    ):
        """Each unit gets its own git add + commit cycle."""
        mock_git.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        plan = [
            {"unit_id": "feat_01", "title": "Unit A"},
            {"unit_id": "feat_02", "title": "Unit B"},
            {"unit_id": "feat_03", "title": "Unit C"},
        ]
        ui = _make_ui()
        state = _make_state(build_plan=plan)

        await execute_node(state, ui)

        # Each unit should have: add -A, commit, add -A (lint amend), commit --amend
        # Plus the branch checkout at the start
        commit_calls = [
            c for c in mock_git.call_args_list
            if len(c[0]) >= 2 and c[0][1] == "commit" and "--amend" not in c[0]
        ]
        assert len(commit_calls) == 3  # One commit per unit
        assert "feat: Unit A" in commit_calls[0][0]
        assert "feat: Unit B" in commit_calls[1][0]
        assert "feat: Unit C" in commit_calls[2][0]

    @pytest.mark.asyncio
    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute._run_lint", return_value=(True, "Lint passed"))
    @patch("graft.stages.execute._run_tests", return_value=(True, "ok"))
    @patch("graft.stages.execute._git")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    async def test_default_branch_name_from_project_id(
        self, mock_agent, mock_git, mock_tests, mock_lint, mock_save, mock_mark
    ):
        """Without feature_branch in state, branch name uses project_id."""
        mock_git.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        plan = [{"unit_id": "feat_01", "title": "X"}]
        ui = _make_ui()
        state = _make_state(build_plan=plan)
        # No feature_branch key in state — should default to feature/<project_id>

        result = await execute_node(state, ui)

        assert result["feature_branch"] == "feature/test_session"

    @pytest.mark.asyncio
    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute._run_lint", return_value=(True, "Lint passed"))
    @patch("graft.stages.execute._run_tests", return_value=(True, "ok"))
    @patch("graft.stages.execute._git")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    async def test_unit_without_tests_included(
        self, mock_agent, mock_git, mock_tests, mock_lint, mock_save, mock_mark
    ):
        """Unit with tests_included=False does not include test instructions in prompt."""
        mock_git.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        plan = [
            {
                "unit_id": "feat_01",
                "title": "Config only",
                "tests_included": False,
            }
        ]
        ui = _make_ui()
        state = _make_state(build_plan=plan)

        await execute_node(state, ui)

        agent_kwargs = mock_agent.call_args[1]
        prompt = agent_kwargs["user_prompt"]
        assert "Write co-located tests" not in prompt

    @pytest.mark.asyncio
    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute._run_lint", return_value=(True, "Lint passed"))
    @patch("graft.stages.execute._run_tests", return_value=(True, "ok"))
    @patch("graft.stages.execute._git")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    async def test_unit_without_pattern_reference(
        self, mock_agent, mock_git, mock_tests, mock_lint, mock_save, mock_mark
    ):
        """Unit without pattern_reference omits pattern instructions from prompt."""
        mock_git.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        plan = [
            {
                "unit_id": "feat_01",
                "title": "No pattern",
                "pattern_reference": "",
            }
        ]
        ui = _make_ui()
        state = _make_state(build_plan=plan)

        await execute_node(state, ui)

        agent_kwargs = mock_agent.call_args[1]
        prompt = agent_kwargs["user_prompt"]
        assert "PATTERN REFERENCE" not in prompt
        assert "Read this file FIRST" not in prompt

    @pytest.mark.asyncio
    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute._run_lint", return_value=(True, "Lint passed"))
    @patch("graft.stages.execute._run_tests", return_value=(True, "ok"))
    @patch("graft.stages.execute._git")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    async def test_mixed_success_and_failure(
        self, mock_agent, mock_git, mock_tests, mock_lint, mock_save, mock_mark
    ):
        """Mix of passing and failing units are tracked correctly in results."""
        call_index = 0

        def agent_side_effect(**kwargs):
            nonlocal call_index
            call_index += 1
            # Second unit's agent fails
            if call_index == 2:
                raise RuntimeError("Crash on unit 2")
            return MagicMock()

        mock_agent.side_effect = agent_side_effect
        mock_git.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        plan = [
            {"unit_id": "feat_01", "title": "Works"},
            {"unit_id": "feat_02", "title": "Crashes"},
            {"unit_id": "feat_03", "title": "Also works"},
        ]
        ui = _make_ui()
        state = _make_state(build_plan=plan)

        result = await execute_node(state, ui)

        assert len(result["units_completed"]) == 2
        assert len(result["units_reverted"]) == 1
        completed_ids = [u["unit_id"] for u in result["units_completed"]]
        assert "feat_01" in completed_ids
        assert "feat_03" in completed_ids
        assert result["units_reverted"][0]["unit_id"] == "feat_02"
