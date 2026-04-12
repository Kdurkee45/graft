"""Tests for graft.stages.execute."""

from __future__ import annotations

import subprocess
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from graft.stages.execute import (
    SYSTEM_PROMPT,
    _git,
    _order_by_dependencies,
    _run_lint,
    _run_tests,
    execute_node,
)


# ---------------------------------------------------------------------------
# Existing _order_by_dependencies tests (preserved)
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
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> dict:
    """Build a minimal FeatureState dict for tests."""
    base = {
        "repo_path": "/tmp/repo",
        "project_dir": "/tmp/project",
        "project_id": "test-session",
        "build_plan": [],
        "codebase_profile": {},
        "feature_spec": {},
        "max_agent_turns": 10,
    }
    base.update(overrides)
    return base


def _ok(stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    """Create a successful CompletedProcess."""
    return subprocess.CompletedProcess(
        args=[], returncode=0, stdout=stdout, stderr=stderr
    )


def _fail(stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    """Create a failed CompletedProcess."""
    return subprocess.CompletedProcess(
        args=[], returncode=1, stdout=stdout, stderr=stderr
    )


def _make_ui() -> MagicMock:
    """Build a mock UI with all required methods."""
    ui = MagicMock()
    ui.stage_start = MagicMock()
    ui.stage_done = MagicMock()
    ui.error = MagicMock()
    ui.unit_start = MagicMock()
    ui.unit_kept = MagicMock()
    ui.unit_reverted = MagicMock()
    return ui


# ---------------------------------------------------------------------------
# _git tests
# ---------------------------------------------------------------------------


class TestGit:
    """Tests for the _git helper."""

    @patch("graft.stages.execute.subprocess.run")
    def test_git_runs_in_repo_directory(self, mock_run):
        mock_run.return_value = _ok()
        _git("/my/repo", "status")
        mock_run.assert_called_once_with(
            ["git", "status"],
            cwd="/my/repo",
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )

    @patch("graft.stages.execute.subprocess.run")
    def test_git_passes_multiple_args(self, mock_run):
        mock_run.return_value = _ok()
        _git("/repo", "checkout", "-b", "my-branch")
        mock_run.assert_called_once_with(
            ["git", "checkout", "-b", "my-branch"],
            cwd="/repo",
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )

    @patch("graft.stages.execute.subprocess.run")
    def test_git_check_true_raises_on_failure(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")
        with pytest.raises(subprocess.CalledProcessError):
            _git("/repo", "bad-command", check=True)

    @patch("graft.stages.execute.subprocess.run")
    def test_git_check_false_does_not_raise(self, mock_run):
        mock_run.return_value = _fail()
        result = _git("/repo", "checkout", "-b", "existing", check=False)
        assert result.returncode == 1

    @patch("graft.stages.execute.subprocess.run")
    def test_git_returns_completed_process(self, mock_run):
        mock_run.return_value = _ok(stdout="abc123")
        result = _git("/repo", "rev-parse", "HEAD")
        assert result.stdout == "abc123"
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# _run_tests tests
# ---------------------------------------------------------------------------


class TestRunTests:
    """Tests for the _run_tests helper."""

    @patch("graft.stages.execute.subprocess.run")
    def test_tests_pass(self, mock_run):
        mock_run.return_value = _ok(stdout="5 passed")
        passed, output = _run_tests("/repo")
        assert passed is True
        assert "5 passed" in output

    @patch("graft.stages.execute.subprocess.run")
    def test_tests_fail(self, mock_run):
        mock_run.return_value = _fail(stderr="FAIL: test_foo")
        passed, output = _run_tests("/repo")
        assert passed is False
        assert "FAIL" in output

    @patch("graft.stages.execute.subprocess.run")
    def test_tests_combine_stdout_and_stderr(self, mock_run):
        mock_run.return_value = _ok(stdout="out\n", stderr="err\n")
        passed, output = _run_tests("/repo")
        assert passed is True
        assert "out" in output
        assert "err" in output

    @patch("graft.stages.execute.subprocess.run")
    def test_tests_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="bash", timeout=300)
        passed, output = _run_tests("/repo")
        assert passed is False
        assert "timed out" in output.lower()

    @patch("graft.stages.execute.subprocess.run")
    def test_tests_file_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError("bash not found")
        passed, output = _run_tests("/repo")
        assert passed is True
        assert "skipping" in output.lower()

    @patch("graft.stages.execute.subprocess.run")
    def test_tests_output_truncated_to_2000_chars(self, mock_run):
        long_output = "x" * 5000
        mock_run.return_value = _ok(stdout=long_output)
        passed, output = _run_tests("/repo")
        assert len(output) <= 2000

    @patch("graft.stages.execute.subprocess.run")
    def test_tests_runs_verify_script_with_bash(self, mock_run):
        mock_run.return_value = _ok()
        _run_tests("/myrepo")
        args, kwargs = mock_run.call_args
        assert args[0][0] == "bash"
        assert args[0][1] == "-c"
        assert kwargs["cwd"] == "/myrepo"
        assert kwargs["timeout"] == 300


# ---------------------------------------------------------------------------
# _run_lint tests
# ---------------------------------------------------------------------------


class TestRunLint:
    """Tests for the _run_lint helper."""

    @patch("graft.stages.execute.subprocess.run")
    def test_lint_eslint_passes(self, mock_run):
        mock_run.return_value = _ok()
        passed, output = _run_lint("/repo")
        assert passed is True
        assert output == "Lint passed"
        # eslint is tried first
        first_call_cmd = mock_run.call_args_list[0][0][0]
        assert "eslint" in first_call_cmd

    @patch("graft.stages.execute.subprocess.run")
    def test_lint_falls_through_to_ruff(self, mock_run):
        """If eslint fails, ruff is tried next."""
        mock_run.side_effect = [_fail(), _ok()]
        passed, output = _run_lint("/repo")
        assert passed is True
        assert output == "Lint passed"
        assert len(mock_run.call_args_list) == 2
        second_cmd = mock_run.call_args_list[1][0][0]
        assert "ruff" in second_cmd

    @patch("graft.stages.execute.subprocess.run")
    def test_lint_falls_through_to_prettier(self, mock_run):
        """If eslint and ruff fail, prettier is tried."""
        mock_run.side_effect = [_fail(), _fail(), _ok()]
        passed, output = _run_lint("/repo")
        assert passed is True
        assert output == "Lint passed"
        third_cmd = mock_run.call_args_list[2][0][0]
        assert "prettier" in third_cmd

    @patch("graft.stages.execute.subprocess.run")
    def test_lint_all_fail_returns_skipping(self, mock_run):
        """If all linters fail, returns True with skip message."""
        mock_run.side_effect = [_fail(), _fail(), _fail()]
        passed, output = _run_lint("/repo")
        assert passed is True
        assert "skipping" in output.lower()

    @patch("graft.stages.execute.subprocess.run")
    def test_lint_file_not_found_skips(self, mock_run):
        """FileNotFoundError for a linter just tries the next one."""
        mock_run.side_effect = [
            FileNotFoundError("npx"),
            FileNotFoundError("python"),
            FileNotFoundError("npx"),
        ]
        passed, output = _run_lint("/repo")
        assert passed is True
        assert "skipping" in output.lower()

    @patch("graft.stages.execute.subprocess.run")
    def test_lint_timeout_skips(self, mock_run):
        """TimeoutExpired for a linter just tries the next one."""
        mock_run.side_effect = [
            subprocess.TimeoutExpired("npx", 60),
            _ok(),
        ]
        passed, output = _run_lint("/repo")
        assert passed is True
        assert output == "Lint passed"

    @patch("graft.stages.execute.subprocess.run")
    def test_lint_runs_in_repo_directory(self, mock_run):
        mock_run.return_value = _ok()
        _run_lint("/my/repo")
        _, kwargs = mock_run.call_args
        assert kwargs["cwd"] == "/my/repo"
        assert kwargs["timeout"] == 60


# ---------------------------------------------------------------------------
# execute_node — async tests
# ---------------------------------------------------------------------------


class TestExecuteNode:
    """Async tests for the execute_node LangGraph node."""

    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    async def test_empty_plan_returns_early(self, mock_save, mock_mark):
        """No build plan → error message, no git or agent calls."""
        ui = _make_ui()
        state = _make_state(build_plan=[])

        result = await execute_node(state, ui)

        ui.error.assert_called_once()
        assert "nothing to execute" in ui.error.call_args[0][0].lower()
        mock_mark.assert_called_once_with("/tmp/project", "execute")
        assert result["current_stage"] == "execute"

    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_feature_branch_created(
        self, mock_run, mock_agent, mock_save, mock_mark
    ):
        """execute_node creates (or checks out) the feature branch."""
        ui = _make_ui()
        plan = [{"unit_id": "u1", "title": "Add widget"}]
        state = _make_state(build_plan=plan)

        # Branch creation succeeds, then git add/commit/test/lint all succeed
        mock_run.return_value = _ok()

        result = await execute_node(state, ui)

        # First git call is checkout -b
        first_call = mock_run.call_args_list[0]
        args = first_call[0][0]
        assert args[:3] == ["git", "checkout", "-b"]
        assert "feature/" in args[3]

    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_existing_branch_fallback(
        self, mock_run, mock_agent, mock_save, mock_mark
    ):
        """If branch already exists, falls back to checkout without -b."""
        ui = _make_ui()
        plan = [{"unit_id": "u1", "title": "Widget"}]
        state = _make_state(build_plan=plan)

        # checkout -b fails (branch exists), then checkout succeeds, rest ok
        branch_fail = _fail(stderr="already exists")
        mock_run.side_effect = [branch_fail, _ok(), _ok(), _ok(), _ok(), _ok(), _ok()]

        await execute_node(state, ui)

        # Second call is checkout without -b
        second_call = mock_run.call_args_list[1]
        args = second_call[0][0]
        assert args[:2] == ["git", "checkout"]
        assert "-b" not in args

    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_successful_unit_flow(
        self, mock_run, mock_agent, mock_save, mock_mark
    ):
        """Happy path: agent → git add → commit → test pass → lint pass → kept."""
        ui = _make_ui()
        plan = [
            {
                "unit_id": "feat_01",
                "title": "Add parser",
                "description": "Implement the main parser",
                "pattern_reference": "src/existing.py",
                "tests_included": True,
                "acceptance_criteria": ["Parses input", "Returns AST"],
            }
        ]
        state = _make_state(build_plan=plan)

        # All subprocess calls succeed
        mock_run.return_value = _ok()

        result = await execute_node(state, ui)

        # Agent was called
        mock_agent.assert_called_once()
        agent_kwargs = mock_agent.call_args[1]
        assert agent_kwargs["cwd"] == "/tmp/repo"
        assert "feat_01" in agent_kwargs["stage"]

        # UI callbacks
        ui.unit_start.assert_called_once_with("feat_01", "Add parser", 1, 1)
        ui.unit_kept.assert_called_once_with("feat_01", "Implemented and passing")
        ui.unit_reverted.assert_not_called()

        # Result tracking
        assert len(result["units_completed"]) == 1
        assert result["units_completed"][0]["unit_id"] == "feat_01"
        assert result["units_completed"][0]["tests_included"] is True
        assert len(result["units_reverted"]) == 0
        assert "feature/" in result["feature_branch"]

    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_agent_failure_reverts_unit(
        self, mock_run, mock_agent, mock_save, mock_mark
    ):
        """RuntimeError from agent → unit reverted, no commit attempted."""
        ui = _make_ui()
        plan = [{"unit_id": "u1", "title": "Broken unit"}]
        state = _make_state(build_plan=plan)

        mock_run.return_value = _ok()  # branch creation
        mock_agent.side_effect = RuntimeError("Agent crashed")

        result = await execute_node(state, ui)

        ui.unit_reverted.assert_called_once()
        assert "Agent failed" in ui.unit_reverted.call_args[0][1]
        assert len(result["units_reverted"]) == 1
        assert len(result["units_completed"]) == 0

    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_no_changes_reverts_unit(
        self, mock_run, mock_agent, mock_save, mock_mark
    ):
        """If commit produces no changes → unit reverted."""
        ui = _make_ui()
        plan = [{"unit_id": "u1", "title": "No-op unit"}]
        state = _make_state(build_plan=plan)

        # Branch ok, git add ok, commit fails (nothing to commit)
        def run_side_effect(cmd, **kwargs):
            if cmd == ["git", "commit", "-m", "feat: No-op unit"]:
                return _fail(stdout="nothing to commit")
            return _ok()

        mock_run.side_effect = run_side_effect

        result = await execute_node(state, ui)

        ui.unit_reverted.assert_called_once_with("u1", "No changes made")
        assert len(result["units_reverted"]) == 1
        assert result["units_reverted"][0]["reason"] == "No changes produced"

    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_test_failure_reverts_commit(
        self, mock_run, mock_agent, mock_save, mock_mark
    ):
        """Tests fail after commit → git revert HEAD, unit reverted."""
        ui = _make_ui()
        plan = [{"unit_id": "u1", "title": "Failing tests"}]
        state = _make_state(build_plan=plan)

        call_count = 0

        def run_side_effect(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            # The test runner (bash -c ...) should fail
            if cmd[0] == "bash" and cmd[1] == "-c":
                return _fail(stderr="AssertionError: test_foo")
            return _ok()

        mock_run.side_effect = run_side_effect

        result = await execute_node(state, ui)

        # Check that revert was called
        revert_calls = [
            c
            for c in mock_run.call_args_list
            if c[0][0][:2] == ["git", "revert"]
        ]
        assert len(revert_calls) == 1
        assert revert_calls[0][0][0] == ["git", "revert", "HEAD", "--no-edit"]

        ui.unit_reverted.assert_called_once_with("u1", "Tests failed")
        assert len(result["units_reverted"]) == 1
        assert "Tests failed" in result["units_reverted"][0]["reason"]

    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_lint_autofix_amends_commit(
        self, mock_run, mock_agent, mock_save, mock_mark
    ):
        """Lint passes → git add -A and commit --amend --no-edit."""
        ui = _make_ui()
        plan = [{"unit_id": "u1", "title": "Lint me"}]
        state = _make_state(build_plan=plan)

        mock_run.return_value = _ok()

        await execute_node(state, ui)

        # Find the amend commit call
        all_cmds = [c[0][0] for c in mock_run.call_args_list]
        amend_calls = [c for c in all_cmds if "commit" in c and "--amend" in c]
        assert len(amend_calls) == 1
        assert "--no-edit" in amend_calls[0]

    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_multiple_units_sequential(
        self, mock_run, mock_agent, mock_save, mock_mark
    ):
        """Multiple units are executed sequentially with proper tracking."""
        ui = _make_ui()
        plan = [
            {"unit_id": "u1", "title": "First"},
            {"unit_id": "u2", "title": "Second"},
            {"unit_id": "u3", "title": "Third"},
        ]
        state = _make_state(build_plan=plan)
        mock_run.return_value = _ok()

        result = await execute_node(state, ui)

        assert mock_agent.call_count == 3
        assert len(result["units_completed"]) == 3
        assert ui.unit_start.call_count == 3
        # Verify index/total passed to ui.unit_start
        ui.unit_start.assert_any_call("u1", "First", 1, 3)
        ui.unit_start.assert_any_call("u2", "Second", 2, 3)
        ui.unit_start.assert_any_call("u3", "Third", 3, 3)

    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_unmet_dependencies_skip_unit(
        self, mock_run, mock_agent, mock_save, mock_mark
    ):
        """Unit with unmet dependencies is skipped."""
        ui = _make_ui()
        plan = [
            {"unit_id": "u2", "depends_on": ["u1"]},  # u1 not in plan → unmet
        ]
        state = _make_state(build_plan=plan)
        mock_run.return_value = _ok()

        result = await execute_node(state, ui)

        mock_agent.assert_not_called()
        ui.unit_reverted.assert_called_once()
        assert "Unmet dependencies" in ui.unit_reverted.call_args[0][1]
        assert len(result["units_skipped"]) == 1

    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_dependent_unit_skipped_when_dep_reverted(
        self, mock_run, mock_agent, mock_save, mock_mark
    ):
        """If u1 is reverted, u2 depending on u1 is skipped."""
        ui = _make_ui()
        plan = [
            {"unit_id": "u1", "title": "Fails"},
            {"unit_id": "u2", "title": "Depends on u1", "depends_on": ["u1"]},
        ]
        state = _make_state(build_plan=plan)

        # Agent for u1 crashes
        mock_agent.side_effect = [RuntimeError("crash"), AsyncMock()]
        mock_run.return_value = _ok()

        result = await execute_node(state, ui)

        assert len(result["units_reverted"]) == 1
        assert result["units_reverted"][0]["unit_id"] == "u1"
        assert len(result["units_skipped"]) == 1
        assert result["units_skipped"][0]["unit_id"] == "u2"

    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_prompt_includes_pattern_reference(
        self, mock_run, mock_agent, mock_save, mock_mark
    ):
        """Pattern reference is included in the agent prompt."""
        ui = _make_ui()
        plan = [
            {
                "unit_id": "u1",
                "title": "Add widget",
                "pattern_reference": "src/models/user.py",
            }
        ]
        state = _make_state(build_plan=plan)
        mock_run.return_value = _ok()

        await execute_node(state, ui)

        prompt = mock_agent.call_args[1]["user_prompt"]
        assert "src/models/user.py" in prompt
        assert "PATTERN REFERENCE" in prompt
        assert "Read this file FIRST" in prompt

    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_prompt_omits_pattern_reference_when_empty(
        self, mock_run, mock_agent, mock_save, mock_mark
    ):
        """No pattern reference → no PATTERN REFERENCE block in prompt."""
        ui = _make_ui()
        plan = [{"unit_id": "u1", "title": "Simple", "pattern_reference": ""}]
        state = _make_state(build_plan=plan)
        mock_run.return_value = _ok()

        await execute_node(state, ui)

        prompt = mock_agent.call_args[1]["user_prompt"]
        assert "PATTERN REFERENCE" not in prompt

    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_prompt_includes_tests_section_when_flagged(
        self, mock_run, mock_agent, mock_save, mock_mark
    ):
        """tests_included=True → TESTS block in prompt."""
        ui = _make_ui()
        plan = [{"unit_id": "u1", "title": "With tests", "tests_included": True}]
        state = _make_state(build_plan=plan)
        mock_run.return_value = _ok()

        await execute_node(state, ui)

        prompt = mock_agent.call_args[1]["user_prompt"]
        assert "TESTS:" in prompt
        assert "co-located tests" in prompt.lower()

    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_prompt_omits_tests_section_when_not_included(
        self, mock_run, mock_agent, mock_save, mock_mark
    ):
        """tests_included=False → no TESTS block."""
        ui = _make_ui()
        plan = [{"unit_id": "u1", "title": "No tests", "tests_included": False}]
        state = _make_state(build_plan=plan)
        mock_run.return_value = _ok()

        await execute_node(state, ui)

        prompt = mock_agent.call_args[1]["user_prompt"]
        assert "TESTS:" not in prompt

    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_prompt_includes_acceptance_criteria(
        self, mock_run, mock_agent, mock_save, mock_mark
    ):
        """Acceptance criteria are listed in the prompt."""
        ui = _make_ui()
        plan = [
            {
                "unit_id": "u1",
                "title": "Widget",
                "acceptance_criteria": ["Must parse JSON", "Must handle errors"],
            }
        ]
        state = _make_state(build_plan=plan)
        mock_run.return_value = _ok()

        await execute_node(state, ui)

        prompt = mock_agent.call_args[1]["user_prompt"]
        assert "- Must parse JSON" in prompt
        assert "- Must handle errors" in prompt
        assert "ACCEPTANCE CRITERIA" in prompt

    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_agent_receives_system_prompt(
        self, mock_run, mock_agent, mock_save, mock_mark
    ):
        """Agent is called with the module's SYSTEM_PROMPT."""
        ui = _make_ui()
        plan = [{"unit_id": "u1", "title": "T"}]
        state = _make_state(build_plan=plan)
        mock_run.return_value = _ok()

        await execute_node(state, ui)

        assert mock_agent.call_args[1]["system_prompt"] == SYSTEM_PROMPT

    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_execution_log_saved(
        self, mock_run, mock_agent, mock_save, mock_mark
    ):
        """Execution log artifact is saved with correct counts."""
        ui = _make_ui()
        plan = [
            {"unit_id": "u1", "title": "Good"},
            {"unit_id": "u2", "title": "Bad"},
        ]
        state = _make_state(build_plan=plan)

        call_idx = 0

        def run_side_effect(cmd, **kwargs):
            nonlocal call_idx
            call_idx += 1
            # Make the commit for u2 fail (no changes)
            if cmd[:2] == ["git", "commit"] and "feat: Bad" in cmd:
                return _fail()
            return _ok()

        mock_run.side_effect = run_side_effect

        result = await execute_node(state, ui)

        # save_artifact should have been called with execution_log.json
        mock_save.assert_called_once()
        call_args = mock_save.call_args
        assert call_args[0][0] == "/tmp/project"
        assert call_args[0][1] == "execution_log.json"
        # Parse the JSON to verify structure
        import json

        log = json.loads(call_args[0][2])
        assert log["total_planned"] == 2
        assert log["units_completed"] == 1
        assert log["units_reverted"] == 1

    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_stage_start_and_done_called(
        self, mock_run, mock_agent, mock_save, mock_mark
    ):
        """UI stage lifecycle: stage_start at beginning, stage_done at end."""
        ui = _make_ui()
        plan = [{"unit_id": "u1", "title": "T"}]
        state = _make_state(build_plan=plan)
        mock_run.return_value = _ok()

        await execute_node(state, ui)

        ui.stage_start.assert_called_once_with("execute")
        ui.stage_done.assert_called_once_with("execute")

    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_custom_feature_branch(
        self, mock_run, mock_agent, mock_save, mock_mark
    ):
        """Custom feature_branch in state is used instead of default."""
        ui = _make_ui()
        plan = [{"unit_id": "u1", "title": "T"}]
        state = _make_state(build_plan=plan, feature_branch="custom/my-branch")
        mock_run.return_value = _ok()

        result = await execute_node(state, ui)

        first_call = mock_run.call_args_list[0]
        assert first_call[0][0] == ["git", "checkout", "-b", "custom/my-branch"]
        assert result["feature_branch"] == "custom/my-branch"

    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_unit_id_defaults_to_feat_nn(
        self, mock_run, mock_agent, mock_save, mock_mark
    ):
        """Units without explicit unit_id get feat_01, feat_02, etc."""
        ui = _make_ui()
        plan = [{"title": "No ID unit"}]  # no unit_id key
        state = _make_state(build_plan=plan)
        mock_run.return_value = _ok()

        result = await execute_node(state, ui)

        ui.unit_start.assert_called_once_with("feat_01", "No ID unit", 1, 1)
        assert result["units_completed"][0]["unit_id"] == "feat_01"

    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_max_agent_turns_passed_to_agent(
        self, mock_run, mock_agent, mock_save, mock_mark
    ):
        """max_agent_turns from state is forwarded to run_agent."""
        ui = _make_ui()
        plan = [{"unit_id": "u1", "title": "T"}]
        state = _make_state(build_plan=plan, max_agent_turns=25)
        mock_run.return_value = _ok()

        await execute_node(state, ui)

        assert mock_agent.call_args[1]["max_turns"] == 25

    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_model_forwarded_to_agent(
        self, mock_run, mock_agent, mock_save, mock_mark
    ):
        """Model from state is forwarded to run_agent."""
        ui = _make_ui()
        plan = [{"unit_id": "u1", "title": "T"}]
        state = _make_state(build_plan=plan, model="claude-sonnet-4-20250514")
        mock_run.return_value = _ok()

        await execute_node(state, ui)

        assert mock_agent.call_args[1]["model"] == "claude-sonnet-4-20250514"

    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_prompt_includes_working_directory(
        self, mock_run, mock_agent, mock_save, mock_mark
    ):
        """Prompt includes the WORKING DIRECTORY line."""
        ui = _make_ui()
        plan = [{"unit_id": "u1", "title": "T"}]
        state = _make_state(build_plan=plan, repo_path="/custom/repo/path")
        mock_run.return_value = _ok()

        await execute_node(state, ui)

        prompt = mock_agent.call_args[1]["user_prompt"]
        assert "WORKING DIRECTORY: /custom/repo/path" in prompt

    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_mark_stage_complete_called(
        self, mock_run, mock_agent, mock_save, mock_mark
    ):
        """mark_stage_complete is called at the end."""
        ui = _make_ui()
        plan = [{"unit_id": "u1", "title": "T"}]
        state = _make_state(build_plan=plan)
        mock_run.return_value = _ok()

        await execute_node(state, ui)

        mock_mark.assert_called_once_with("/tmp/project", "execute")

    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_mixed_success_and_failure(
        self, mock_run, mock_agent, mock_save, mock_mark
    ):
        """Mix of passing and failing units tracked correctly."""
        ui = _make_ui()
        plan = [
            {"unit_id": "u1", "title": "Good"},
            {"unit_id": "u2", "title": "Agent fails"},
            {"unit_id": "u3", "title": "Also good"},
        ]
        state = _make_state(build_plan=plan)

        # u2 agent raises
        mock_agent.side_effect = [None, RuntimeError("boom"), None]
        mock_run.return_value = _ok()

        result = await execute_node(state, ui)

        assert len(result["units_completed"]) == 2
        assert result["units_completed"][0]["unit_id"] == "u1"
        assert result["units_completed"][1]["unit_id"] == "u3"
        assert len(result["units_reverted"]) == 1
        assert result["units_reverted"][0]["unit_id"] == "u2"

    @patch("graft.stages.execute.mark_stage_complete")
    @patch("graft.stages.execute.save_artifact")
    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_commit_message_includes_title(
        self, mock_run, mock_agent, mock_save, mock_mark
    ):
        """Commit message follows 'feat: {title}' format."""
        ui = _make_ui()
        plan = [{"unit_id": "u1", "title": "Add API endpoint"}]
        state = _make_state(build_plan=plan)
        mock_run.return_value = _ok()

        await execute_node(state, ui)

        commit_calls = [
            c for c in mock_run.call_args_list if c[0][0][:2] == ["git", "commit"]
        ]
        # First commit call (not the amend)
        main_commit = [c for c in commit_calls if "--amend" not in c[0][0]]
        assert len(main_commit) >= 1
        assert main_commit[0][0][0] == [
            "git", "commit", "-m", "feat: Add API endpoint"
        ]
