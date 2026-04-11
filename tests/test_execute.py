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


def test_order_by_dependencies_empty_plan():
    """Empty plan returns empty list."""
    assert _order_by_dependencies([]) == []


def test_order_by_dependencies_single_unit():
    """Single unit with no deps comes through unchanged."""
    plan = [{"unit_id": "only"}]
    result = _order_by_dependencies(plan)
    assert len(result) == 1
    assert result[0]["unit_id"] == "only"


def test_order_by_dependencies_missing_unit_id():
    """Units without unit_id still get processed (empty string id)."""
    plan = [{"title": "no id"}, {"title": "also no id"}]
    result = _order_by_dependencies(plan)
    assert len(result) == 2


def test_order_by_dependencies_unresolvable_dep():
    """Unit depending on non-existent id is appended in fallback pass."""
    plan = [
        {"unit_id": "a", "depends_on": ["nonexistent"]},
        {"unit_id": "b", "depends_on": []},
    ]
    result = _order_by_dependencies(plan)
    assert len(result) == 2
    ids = [u["unit_id"] for u in result]
    # b has no deps so resolves first; a is appended in the fallback
    assert ids[0] == "b"


# ---------------------------------------------------------------------------
# _git helper
# ---------------------------------------------------------------------------


class TestGit:
    """Tests for the _git subprocess helper."""

    @patch("graft.stages.execute.subprocess.run")
    def test_git_passes_correct_args(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git", "status"], returncode=0, stdout="clean", stderr=""
        )
        result = _git("/repo", "status")
        mock_run.assert_called_once_with(
            ["git", "status"],
            cwd="/repo",
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        assert result.returncode == 0

    @patch("graft.stages.execute.subprocess.run")
    def test_git_add_all(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git", "add", "-A"], returncode=0, stdout="", stderr=""
        )
        _git("/repo", "add", "-A")
        mock_run.assert_called_once_with(
            ["git", "add", "-A"],
            cwd="/repo",
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )

    @patch("graft.stages.execute.subprocess.run")
    def test_git_commit(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git", "commit", "-m", "feat: something"],
            returncode=0,
            stdout="1 file changed",
            stderr="",
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
    def test_git_checkout_branch(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git", "checkout", "-b", "feature/xyz"],
            returncode=0,
            stdout="",
            stderr="Switched to a new branch",
        )
        _git("/repo", "checkout", "-b", "feature/xyz")
        mock_run.assert_called_once_with(
            ["git", "checkout", "-b", "feature/xyz"],
            cwd="/repo",
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )

    @patch("graft.stages.execute.subprocess.run")
    def test_git_revert_head(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git", "revert", "HEAD", "--no-edit"],
            returncode=0,
            stdout="",
            stderr="",
        )
        _git("/repo", "revert", "HEAD", "--no-edit")
        mock_run.assert_called_once_with(
            ["git", "revert", "HEAD", "--no-edit"],
            cwd="/repo",
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )

    @patch("graft.stages.execute.subprocess.run")
    def test_git_check_false_does_not_raise(self, mock_run):
        """check=False suppresses CalledProcessError."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["git", "checkout", "-b", "exists"],
            returncode=128,
            stdout="",
            stderr="fatal: branch already exists",
        )
        result = _git("/repo", "checkout", "-b", "exists", check=False)
        assert result.returncode == 128
        mock_run.assert_called_once_with(
            ["git", "checkout", "-b", "exists"],
            cwd="/repo",
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )


# ---------------------------------------------------------------------------
# _run_tests
# ---------------------------------------------------------------------------


class TestRunTests:
    """Tests for the _run_tests function."""

    @patch("graft.stages.execute.subprocess.run")
    def test_tests_pass(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["bash", "-c", "..."],
            returncode=0,
            stdout="4 passed",
            stderr="",
        )
        passed, output = _run_tests("/repo")
        assert passed is True
        assert "4 passed" in output

    @patch("graft.stages.execute.subprocess.run")
    def test_tests_fail(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["bash", "-c", "..."],
            returncode=1,
            stdout="FAILED test_foo.py",
            stderr="AssertionError",
        )
        passed, output = _run_tests("/repo")
        assert passed is False
        assert "FAILED" in output

    @patch("graft.stages.execute.subprocess.run")
    def test_tests_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="bash", timeout=300)
        passed, output = _run_tests("/repo")
        assert passed is False
        assert "timed out" in output.lower()
        assert "300s" in output

    @patch("graft.stages.execute.subprocess.run")
    def test_tests_no_runner_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError("bash not found")
        passed, output = _run_tests("/repo")
        assert passed is True
        assert "skipping" in output.lower()

    @patch("graft.stages.execute.subprocess.run")
    def test_tests_output_truncated_to_2000_chars(self, mock_run):
        long_output = "x" * 5000
        mock_run.return_value = subprocess.CompletedProcess(
            args=["bash", "-c", "..."],
            returncode=0,
            stdout=long_output,
            stderr="",
        )
        passed, output = _run_tests("/repo")
        assert passed is True
        assert len(output) <= 2000

    @patch("graft.stages.execute.subprocess.run")
    def test_tests_combines_stdout_stderr(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["bash", "-c", "..."],
            returncode=0,
            stdout="stdout part",
            stderr="stderr part",
        )
        passed, output = _run_tests("/repo")
        assert "stdout part" in output
        assert "stderr part" in output

    @patch("graft.stages.execute.subprocess.run")
    def test_tests_uses_300s_timeout(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["bash", "-c", "..."],
            returncode=0,
            stdout="ok",
            stderr="",
        )
        _run_tests("/repo")
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs["timeout"] == 300


# ---------------------------------------------------------------------------
# _run_lint
# ---------------------------------------------------------------------------


class TestRunLint:
    """Tests for the _run_lint function."""

    @patch("graft.stages.execute.subprocess.run")
    def test_lint_first_linter_passes(self, mock_run):
        """If eslint passes, returns immediately."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["npx", "eslint", ".", "--fix"],
            returncode=0,
            stdout="",
            stderr="",
        )
        passed, output = _run_lint("/repo")
        assert passed is True
        assert output == "Lint passed"
        # Should only call the first linter since it passed
        mock_run.assert_called_once()

    @patch("graft.stages.execute.subprocess.run")
    def test_lint_falls_through_to_ruff(self, mock_run):
        """If eslint fails, tries ruff next."""
        def side_effect(cmd, **kwargs):
            if cmd[0] == "npx" and cmd[1] == "eslint":
                return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="error")
            if cmd[0] == "python":
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
            return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="")

        mock_run.side_effect = side_effect
        passed, output = _run_lint("/repo")
        assert passed is True
        assert output == "Lint passed"
        assert mock_run.call_count == 2

    @patch("graft.stages.execute.subprocess.run")
    def test_lint_all_fail_returns_true(self, mock_run):
        """If all linters fail, returns True with 'No linter found' message."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["cmd"], returncode=1, stdout="", stderr=""
        )
        passed, output = _run_lint("/repo")
        assert passed is True
        assert "No linter found" in output

    @patch("graft.stages.execute.subprocess.run")
    def test_lint_file_not_found_skips(self, mock_run):
        """FileNotFoundError for each linter → falls through gracefully."""
        mock_run.side_effect = FileNotFoundError("not found")
        passed, output = _run_lint("/repo")
        assert passed is True
        assert "No linter found" in output
        # All three linters attempted
        assert mock_run.call_count == 3

    @patch("graft.stages.execute.subprocess.run")
    def test_lint_timeout_skips(self, mock_run):
        """TimeoutExpired for a linter continues to next."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="npx", timeout=60)
        passed, output = _run_lint("/repo")
        assert passed is True
        assert "No linter found" in output

    @patch("graft.stages.execute.subprocess.run")
    def test_lint_tries_three_linters(self, mock_run):
        """All three linters are attempted when each fails."""
        mock_run.side_effect = FileNotFoundError("nope")
        _run_lint("/repo")
        assert mock_run.call_count == 3
        cmds = [c.args[0] for c in mock_run.call_args_list]
        # First is eslint, second is ruff, third is prettier
        assert cmds[0][0] == "npx"
        assert cmds[1][0] == "python"
        assert cmds[2][0] == "npx"

    @patch("graft.stages.execute.subprocess.run")
    def test_lint_prettier_passes_as_fallback(self, mock_run):
        """If eslint & ruff fail but prettier succeeds, lint passes."""
        call_count = [0]

        def side_effect(cmd, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise FileNotFoundError("not found")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        mock_run.side_effect = side_effect
        passed, output = _run_lint("/repo")
        assert passed is True
        assert output == "Lint passed"


# ---------------------------------------------------------------------------
# execute_node — full async integration tests
# ---------------------------------------------------------------------------


def _make_state(
    tmp_path,
    build_plan=None,
    feature_branch=None,
    model=None,
    max_agent_turns=50,
) -> dict:
    """Build a minimal FeatureState dict for testing."""
    project_dir = tmp_path / "project"
    (project_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    # Create metadata.json needed by mark_stage_complete
    meta = {
        "project_id": "test_feat",
        "repo_path": str(tmp_path / "repo"),
        "feature_prompt": "test",
        "status": "in_progress",
        "stages_completed": [],
    }
    (project_dir / "metadata.json").write_text(json.dumps(meta))

    state = {
        "repo_path": str(tmp_path / "repo"),
        "project_id": "test_feat",
        "project_dir": str(project_dir),
        "feature_prompt": "test feature",
        "build_plan": build_plan or [],
        "codebase_profile": {},
        "feature_spec": {},
        "max_agent_turns": max_agent_turns,
    }
    if feature_branch:
        state["feature_branch"] = feature_branch
    if model:
        state["model"] = model
    return state


def _mock_ui():
    """Create a mock UI with all methods used by execute_node."""
    ui = MagicMock(spec=[
        "stage_start", "stage_done", "error",
        "unit_start", "unit_kept", "unit_reverted",
    ])
    return ui


def _successful_git_run(cmd, **kwargs):
    """Default mock for subprocess.run that succeeds for everything."""
    return subprocess.CompletedProcess(
        args=cmd, returncode=0, stdout="ok", stderr=""
    )


class TestExecuteNode:
    """Tests for the execute_node async function."""

    async def test_empty_plan_returns_early(self, tmp_path):
        """No build plan → marks stage complete, returns early."""
        state = _make_state(tmp_path, build_plan=[])
        ui = _mock_ui()

        result = await execute_node(state, ui)

        ui.stage_start.assert_called_once_with("execute")
        ui.error.assert_called_once()
        assert "nothing to execute" in ui.error.call_args[0][0].lower()
        assert result["current_stage"] == "execute"

    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_single_unit_happy_path(self, mock_subprocess, mock_agent, tmp_path):
        """One unit: implement → commit → test pass → lint pass → kept."""
        plan = [
            {
                "unit_id": "feat_01",
                "title": "Add widget",
                "description": "Add a widget component",
                "pattern_reference": "src/components/Button.tsx",
                "tests_included": True,
                "acceptance_criteria": ["Widget renders"],
            }
        ]
        state = _make_state(tmp_path, build_plan=plan)
        ui = _mock_ui()

        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=["cmd"], returncode=0, stdout="ok", stderr=""
        )

        result = await execute_node(state, ui)

        assert len(result["units_completed"]) == 1
        assert len(result["units_reverted"]) == 0
        assert result["units_completed"][0]["unit_id"] == "feat_01"
        assert result["units_completed"][0]["tests_included"] is True
        ui.unit_kept.assert_called_once()
        mock_agent.assert_awaited_once()

    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_unit_reverted_on_test_failure(self, mock_subprocess, mock_agent, tmp_path):
        """Tests failing causes git revert and unit marked as reverted."""
        plan = [
            {
                "unit_id": "feat_01",
                "title": "Bad unit",
                "description": "This will fail tests",
            }
        ]
        state = _make_state(tmp_path, build_plan=plan)
        ui = _mock_ui()

        call_index = [0]

        def subprocess_side_effect(cmd, **kwargs):
            call_index[0] += 1
            # git checkout -b: succeeds
            if cmd[:2] == ["git", "checkout"]:
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
            # git add: succeeds
            if cmd[:2] == ["git", "add"]:
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
            # git commit: succeeds (changes made)
            if cmd[:2] == ["git", "commit"]:
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="1 file changed", stderr="")
            # bash -c (test script): fails
            if cmd[0] == "bash":
                return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="FAILED", stderr="AssertionError")
            # git revert: succeeds
            if cmd[:2] == ["git", "revert"]:
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        mock_subprocess.side_effect = subprocess_side_effect

        result = await execute_node(state, ui)

        assert len(result["units_completed"]) == 0
        assert len(result["units_reverted"]) == 1
        assert result["units_reverted"][0]["unit_id"] == "feat_01"
        assert "Tests failed" in result["units_reverted"][0]["reason"]
        ui.unit_reverted.assert_called_once()

        # Verify git revert HEAD --no-edit was called
        revert_calls = [
            c for c in mock_subprocess.call_args_list
            if len(c.args[0]) >= 2 and c.args[0][:2] == ["git", "revert"]
        ]
        assert len(revert_calls) == 1
        assert "HEAD" in revert_calls[0].args[0]
        assert "--no-edit" in revert_calls[0].args[0]

    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_unit_reverted_when_no_changes(self, mock_subprocess, mock_agent, tmp_path):
        """If commit produces nothing (returncode != 0), unit is reverted."""
        plan = [{"unit_id": "feat_01", "title": "Empty unit"}]
        state = _make_state(tmp_path, build_plan=plan)
        ui = _mock_ui()

        def subprocess_side_effect(cmd, **kwargs):
            if cmd[:2] == ["git", "commit"]:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=1, stdout="", stderr="nothing to commit"
                )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        mock_subprocess.side_effect = subprocess_side_effect

        result = await execute_node(state, ui)

        assert len(result["units_reverted"]) == 1
        assert "No changes" in result["units_reverted"][0]["reason"]

    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_unit_reverted_on_agent_failure(self, mock_subprocess, mock_agent, tmp_path):
        """RuntimeError from run_agent causes unit revert."""
        plan = [{"unit_id": "feat_01", "title": "Agent crash"}]
        state = _make_state(tmp_path, build_plan=plan)
        ui = _mock_ui()
        mock_agent.side_effect = RuntimeError("Agent exploded")
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=["git"], returncode=0, stdout="", stderr=""
        )

        result = await execute_node(state, ui)

        assert len(result["units_reverted"]) == 1
        assert "Agent exploded" in result["units_reverted"][0]["reason"]

    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_lint_fix_amend_flow(self, mock_subprocess, mock_agent, tmp_path):
        """After lint passes, git add + commit --amend --no-edit is called."""
        plan = [{"unit_id": "feat_01", "title": "Lint fix unit"}]
        state = _make_state(tmp_path, build_plan=plan)
        ui = _mock_ui()

        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=["cmd"], returncode=0, stdout="ok", stderr=""
        )

        result = await execute_node(state, ui)

        assert len(result["units_completed"]) == 1

        # Find the commit --amend call
        amend_calls = [
            c for c in mock_subprocess.call_args_list
            if len(c.args[0]) >= 3
            and c.args[0][0] == "git"
            and c.args[0][1] == "commit"
            and "--amend" in c.args[0]
        ]
        assert len(amend_calls) == 1
        assert "--no-edit" in amend_calls[0].args[0]

    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_skipped_unit_unmet_dependencies(self, mock_subprocess, mock_agent, tmp_path):
        """Unit with unmet deps (dep was reverted) gets skipped."""
        plan = [
            {"unit_id": "feat_01", "title": "Base unit"},
            {"unit_id": "feat_02", "title": "Depends on base", "depends_on": ["feat_01"]},
        ]
        state = _make_state(tmp_path, build_plan=plan)
        ui = _mock_ui()

        # Make the first unit fail (agent error) so feat_02's dep is unmet
        mock_agent.side_effect = RuntimeError("Agent failed")
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=["git"], returncode=0, stdout="", stderr=""
        )

        result = await execute_node(state, ui)

        assert len(result["units_reverted"]) == 1
        assert result["units_reverted"][0]["unit_id"] == "feat_01"
        assert len(result["units_skipped"]) == 1
        assert result["units_skipped"][0]["unit_id"] == "feat_02"
        assert "Dependencies not met" in result["units_skipped"][0]["reason"]

    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_execution_log_artifact_saved(self, mock_subprocess, mock_agent, tmp_path):
        """Execution log JSON artifact is written with correct counts."""
        plan = [
            {"unit_id": "feat_01", "title": "Good unit"},
            {"unit_id": "feat_02", "title": "Bad unit"},
        ]
        state = _make_state(tmp_path, build_plan=plan)
        ui = _mock_ui()

        agent_call_count = [0]

        async def agent_side_effect(**kwargs):
            agent_call_count[0] += 1
            if agent_call_count[0] == 2:
                raise RuntimeError("fail")

        mock_agent.side_effect = agent_side_effect
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=["cmd"], returncode=0, stdout="ok", stderr=""
        )

        await execute_node(state, ui)

        # Check the execution_log.json was written
        project_dir = state["project_dir"]
        log_path = tmp_path / "project" / "artifacts" / "execution_log.json"
        assert log_path.exists()
        log = json.loads(log_path.read_text())
        assert log["units_completed"] == 1
        assert log["units_reverted"] == 1
        assert log["total_planned"] == 2
        assert len(log["completed"]) == 1
        assert len(log["reverted"]) == 1

    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_feature_branch_created(self, mock_subprocess, mock_agent, tmp_path):
        """Feature branch name is returned and checkout -b is attempted."""
        plan = [{"unit_id": "feat_01", "title": "Unit"}]
        state = _make_state(tmp_path, build_plan=plan, feature_branch="feature/my-feature")
        ui = _mock_ui()
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=["cmd"], returncode=0, stdout="ok", stderr=""
        )

        result = await execute_node(state, ui)

        assert result["feature_branch"] == "feature/my-feature"

        # Verify checkout -b was called with the branch name
        checkout_calls = [
            c for c in mock_subprocess.call_args_list
            if len(c.args[0]) >= 4
            and c.args[0][:2] == ["git", "checkout"]
            and "-b" in c.args[0]
        ]
        assert len(checkout_calls) == 1
        assert "feature/my-feature" in checkout_calls[0].args[0]

    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_existing_branch_falls_back_to_checkout(self, mock_subprocess, mock_agent, tmp_path):
        """If branch already exists, falls back to plain checkout."""
        plan = [{"unit_id": "feat_01", "title": "Unit"}]
        state = _make_state(tmp_path, build_plan=plan, feature_branch="feature/existing")
        ui = _mock_ui()

        def subprocess_side_effect(cmd, **kwargs):
            # checkout -b fails (branch exists), plain checkout succeeds
            if cmd[:2] == ["git", "checkout"] and "-b" in cmd:
                return subprocess.CompletedProcess(
                    args=cmd, returncode=128, stdout="", stderr="already exists"
                )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

        mock_subprocess.side_effect = subprocess_side_effect

        result = await execute_node(state, ui)

        # Should have called checkout without -b as fallback
        plain_checkout_calls = [
            c for c in mock_subprocess.call_args_list
            if c.args[0][:2] == ["git", "checkout"]
            and "-b" not in c.args[0]
            and "feature/existing" in c.args[0]
        ]
        assert len(plain_checkout_calls) >= 1

    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_default_branch_name_from_project_id(self, mock_subprocess, mock_agent, tmp_path):
        """Without explicit feature_branch, uses feature/{project_id}."""
        plan = [{"unit_id": "feat_01", "title": "Unit"}]
        state = _make_state(tmp_path, build_plan=plan)
        # No feature_branch in state
        ui = _mock_ui()
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=["cmd"], returncode=0, stdout="ok", stderr=""
        )

        result = await execute_node(state, ui)

        assert result["feature_branch"] == "feature/test_feat"

    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_multiple_units_all_pass(self, mock_subprocess, mock_agent, tmp_path):
        """Three units all pass → all in units_completed."""
        plan = [
            {"unit_id": "feat_01", "title": "Unit A", "category": "core"},
            {"unit_id": "feat_02", "title": "Unit B", "category": "api"},
            {"unit_id": "feat_03", "title": "Unit C", "category": "ui"},
        ]
        state = _make_state(tmp_path, build_plan=plan)
        ui = _mock_ui()
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=["cmd"], returncode=0, stdout="ok", stderr=""
        )

        result = await execute_node(state, ui)

        assert len(result["units_completed"]) == 3
        assert len(result["units_reverted"]) == 0
        assert len(result["units_skipped"]) == 0
        # Verify categories preserved
        categories = [u["category"] for u in result["units_completed"]]
        assert categories == ["core", "api", "ui"]
        # Agent called 3 times
        assert mock_agent.await_count == 3

    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_unit_without_unit_id_gets_default(self, mock_subprocess, mock_agent, tmp_path):
        """Units without unit_id get feat_NN default ids."""
        plan = [{"title": "No ID unit"}]
        state = _make_state(tmp_path, build_plan=plan)
        ui = _mock_ui()
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=["cmd"], returncode=0, stdout="ok", stderr=""
        )

        result = await execute_node(state, ui)

        assert len(result["units_completed"]) == 1
        assert result["units_completed"][0]["unit_id"] == "feat_01"

    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_agent_receives_correct_prompt_parts(self, mock_subprocess, mock_agent, tmp_path):
        """Verify the agent prompt includes task, pattern ref, criteria, tests flag."""
        plan = [
            {
                "unit_id": "feat_01",
                "title": "Add button",
                "description": "Create button component",
                "pattern_reference": "src/components/Input.tsx",
                "tests_included": True,
                "acceptance_criteria": ["Button renders", "Has click handler"],
            }
        ]
        state = _make_state(tmp_path, build_plan=plan)
        ui = _mock_ui()
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=["cmd"], returncode=0, stdout="ok", stderr=""
        )

        await execute_node(state, ui)

        # Inspect the user_prompt passed to run_agent
        call_kwargs = mock_agent.call_args[1]
        prompt = call_kwargs["user_prompt"]
        assert "Add button" in prompt
        assert "Create button component" in prompt
        assert "src/components/Input.tsx" in prompt
        assert "Read this file FIRST" in prompt
        assert "Button renders" in prompt
        assert "Has click handler" in prompt
        assert "tests" in prompt.lower()

    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_commit_message_includes_title(self, mock_subprocess, mock_agent, tmp_path):
        """Commit message uses 'feat: {title}' format."""
        plan = [{"unit_id": "feat_01", "title": "Add dark mode"}]
        state = _make_state(tmp_path, build_plan=plan)
        ui = _mock_ui()
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=["cmd"], returncode=0, stdout="ok", stderr=""
        )

        await execute_node(state, ui)

        commit_calls = [
            c for c in mock_subprocess.call_args_list
            if len(c.args[0]) >= 3
            and c.args[0][:2] == ["git", "commit"]
            and "-m" in c.args[0]
            and "--amend" not in c.args[0]
        ]
        assert len(commit_calls) == 1
        # The message arg follows -m
        msg_idx = commit_calls[0].args[0].index("-m") + 1
        assert commit_calls[0].args[0][msg_idx] == "feat: Add dark mode"

    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_stage_done_called_at_end(self, mock_subprocess, mock_agent, tmp_path):
        """stage_done('execute') is called after all units processed."""
        plan = [{"unit_id": "feat_01", "title": "Unit"}]
        state = _make_state(tmp_path, build_plan=plan)
        ui = _mock_ui()
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=["cmd"], returncode=0, stdout="ok", stderr=""
        )

        await execute_node(state, ui)

        ui.stage_done.assert_called_once_with("execute")

    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_mixed_results_tracking(self, mock_subprocess, mock_agent, tmp_path):
        """Mix of completed, reverted, and skipped units are tracked correctly."""
        plan = [
            {"unit_id": "feat_01", "title": "Good"},
            {"unit_id": "feat_02", "title": "Will fail tests"},
            {"unit_id": "feat_03", "title": "Depends on failed", "depends_on": ["feat_02"]},
        ]
        state = _make_state(tmp_path, build_plan=plan)
        ui = _mock_ui()

        agent_call_count = [0]

        async def agent_side_effect(**kwargs):
            agent_call_count[0] += 1

        mock_agent.side_effect = agent_side_effect

        call_index = [0]

        def subprocess_side_effect(cmd, **kwargs):
            call_index[0] += 1
            # Make commit for feat_02 succeed but tests fail
            if cmd[0] == "bash":
                # Check if we're on second unit by counting bash calls
                bash_count = sum(
                    1 for c in mock_subprocess.call_args_list
                    if c.args[0][0] == "bash"
                )
                if bash_count == 1:  # Second bash call (0-indexed count at call time)
                    return subprocess.CompletedProcess(
                        args=cmd, returncode=1, stdout="FAILED", stderr=""
                    )
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

        mock_subprocess.side_effect = subprocess_side_effect

        result = await execute_node(state, ui)

        assert len(result["units_completed"]) >= 1
        # feat_03 should be skipped because feat_02 was reverted
        assert len(result["units_skipped"]) == 1
        assert result["units_skipped"][0]["unit_id"] == "feat_03"

    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_model_passed_to_agent(self, mock_subprocess, mock_agent, tmp_path):
        """The model from state is passed through to run_agent."""
        plan = [{"unit_id": "feat_01", "title": "Unit"}]
        state = _make_state(tmp_path, build_plan=plan, model="claude-sonnet-4-20250514")
        ui = _mock_ui()
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=["cmd"], returncode=0, stdout="ok", stderr=""
        )

        await execute_node(state, ui)

        call_kwargs = mock_agent.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"

    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_max_turns_passed_to_agent(self, mock_subprocess, mock_agent, tmp_path):
        """max_agent_turns from state is forwarded to run_agent."""
        plan = [{"unit_id": "feat_01", "title": "Unit"}]
        state = _make_state(tmp_path, build_plan=plan, max_agent_turns=25)
        ui = _mock_ui()
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=["cmd"], returncode=0, stdout="ok", stderr=""
        )

        await execute_node(state, ui)

        call_kwargs = mock_agent.call_args[1]
        assert call_kwargs["max_turns"] == 25

    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_unit_without_pattern_reference(self, mock_subprocess, mock_agent, tmp_path):
        """Unit with no pattern_reference doesn't include 'Read this file FIRST'."""
        plan = [
            {
                "unit_id": "feat_01",
                "title": "Simple unit",
                "description": "Do something",
            }
        ]
        state = _make_state(tmp_path, build_plan=plan)
        ui = _mock_ui()
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=["cmd"], returncode=0, stdout="ok", stderr=""
        )

        await execute_node(state, ui)

        call_kwargs = mock_agent.call_args[1]
        prompt = call_kwargs["user_prompt"]
        assert "Read this file FIRST" not in prompt

    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_unit_without_tests_included(self, mock_subprocess, mock_agent, tmp_path):
        """Unit with tests_included=False doesn't include test instructions."""
        plan = [
            {
                "unit_id": "feat_01",
                "title": "No-test unit",
                "tests_included": False,
            }
        ]
        state = _make_state(tmp_path, build_plan=plan)
        ui = _mock_ui()
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=["cmd"], returncode=0, stdout="ok", stderr=""
        )

        await execute_node(state, ui)

        call_kwargs = mock_agent.call_args[1]
        prompt = call_kwargs["user_prompt"]
        assert "Write co-located tests" not in prompt

    @patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
    @patch("graft.stages.execute.subprocess.run")
    async def test_execution_log_counts_match(self, mock_subprocess, mock_agent, tmp_path):
        """Execution log has correct total_planned matching input plan length."""
        plan = [
            {"unit_id": "a", "title": "A"},
            {"unit_id": "b", "title": "B"},
            {"unit_id": "c", "title": "C"},
        ]
        state = _make_state(tmp_path, build_plan=plan)
        ui = _mock_ui()
        mock_subprocess.return_value = subprocess.CompletedProcess(
            args=["cmd"], returncode=0, stdout="ok", stderr=""
        )

        await execute_node(state, ui)

        log_path = tmp_path / "project" / "artifacts" / "execution_log.json"
        log = json.loads(log_path.read_text())
        assert log["total_planned"] == 3
        assert log["units_completed"] == 3
        assert log["units_reverted"] == 0
        assert log["units_skipped"] == 0
