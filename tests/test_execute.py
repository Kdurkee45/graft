"""Tests for graft.stages.execute."""

import asyncio
import json
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

from graft.stages.execute import (
    _git,
    _order_by_dependencies,
    _run_lint,
    _run_tests,
    execute_node,
)
from graft.ui import UI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_completed_process(
    returncode: int = 0, stdout: str = "", stderr: str = "",
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr,
    )


def _make_state(
    build_plan: list[dict] | None = None, **overrides
) -> dict:
    """Return a minimal FeatureState dict suitable for execute_node."""
    base = {
        "repo_path": "/tmp/repo",
        "project_dir": "/tmp/project",
        "project_id": "test123",
        "build_plan": build_plan or [],
        "codebase_profile": {},
        "feature_spec": {},
        "max_agent_turns": 5,
    }
    base.update(overrides)
    return base


def _make_ui() -> UI:
    ui = UI(auto_approve=True)
    # Silence all rich output during tests
    ui.stage_start = MagicMock()
    ui.stage_done = MagicMock()
    ui.error = MagicMock()
    ui.unit_start = MagicMock()
    ui.unit_kept = MagicMock()
    ui.unit_reverted = MagicMock()
    ui.info = MagicMock()
    return ui


# ---------------------------------------------------------------------------
# _order_by_dependencies (existing tests)
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
# _git() wrapper
# ---------------------------------------------------------------------------


@patch("graft.stages.execute.subprocess.run")
def test_git_calls_subprocess_with_correct_args(mock_run):
    """_git() passes the right positional and keyword args to subprocess.run."""
    mock_run.return_value = _make_completed_process()
    _git("/my/repo", "checkout", "-b", "feature/x")

    mock_run.assert_called_once_with(
        ["git", "checkout", "-b", "feature/x"],
        cwd="/my/repo",
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )


@patch("graft.stages.execute.subprocess.run")
def test_git_check_false_propagated(mock_run):
    """_git(check=False) propagates the flag to subprocess.run."""
    mock_run.return_value = _make_completed_process(returncode=1)
    result = _git("/repo", "status", check=False)

    assert result.returncode == 1
    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["check"] is False


# ---------------------------------------------------------------------------
# _run_tests()
# ---------------------------------------------------------------------------


@patch("graft.stages.execute.subprocess.run")
def test_run_tests_success(mock_run):
    """Passing tests return (True, output)."""
    mock_run.return_value = _make_completed_process(
        returncode=0, stdout="3 passed\n", stderr="",
    )
    passed, output = _run_tests("/repo")

    assert passed is True
    assert "3 passed" in output


@patch("graft.stages.execute.subprocess.run")
def test_run_tests_failure_truncates_output(mock_run):
    """Failing tests return (False, truncated_output) capped at 2000 chars."""
    long_output = "x" * 5000
    mock_run.return_value = _make_completed_process(
        returncode=1, stdout=long_output, stderr="",
    )
    passed, output = _run_tests("/repo")

    assert passed is False
    assert len(output) == 2000


@patch("graft.stages.execute.subprocess.run")
def test_run_tests_timeout(mock_run):
    """TimeoutExpired returns (False, timeout message)."""
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="bash", timeout=300)

    passed, output = _run_tests("/repo")

    assert passed is False
    assert "timed out" in output.lower()


@patch("graft.stages.execute.subprocess.run")
def test_run_tests_file_not_found(mock_run):
    """FileNotFoundError returns (True, skip message)."""
    mock_run.side_effect = FileNotFoundError("bash not found")

    passed, output = _run_tests("/repo")

    assert passed is True
    assert "skipping" in output.lower()


# ---------------------------------------------------------------------------
# _run_lint()
# ---------------------------------------------------------------------------


@patch("graft.stages.execute.subprocess.run")
def test_run_lint_first_linter_succeeds(mock_run):
    """When the first linter passes, return immediately."""
    mock_run.return_value = _make_completed_process(returncode=0)

    passed, output = _run_lint("/repo")

    assert passed is True
    assert output == "Lint passed"
    # Only the first linter command should have been tried
    assert mock_run.call_count == 1


@patch("graft.stages.execute.subprocess.run")
def test_run_lint_falls_through_to_second_linter(mock_run):
    """When the first linter fails, tries the next one."""
    mock_run.side_effect = [
        _make_completed_process(returncode=1),  # eslint fails
        _make_completed_process(returncode=0),  # ruff passes
    ]
    passed, output = _run_lint("/repo")

    assert passed is True
    assert output == "Lint passed"
    assert mock_run.call_count == 2


@patch("graft.stages.execute.subprocess.run")
def test_run_lint_no_linter_found(mock_run):
    """When all linters throw FileNotFoundError, return success with skip."""
    mock_run.side_effect = FileNotFoundError("not found")

    passed, output = _run_lint("/repo")

    assert passed is True
    assert "no linter found" in output.lower()


@patch("graft.stages.execute.subprocess.run")
def test_run_lint_timeout_continues_to_next(mock_run):
    """TimeoutExpired on a linter continues to the next one."""
    mock_run.side_effect = [
        subprocess.TimeoutExpired(cmd="npx", timeout=60),  # eslint timeout
        subprocess.TimeoutExpired(cmd="python", timeout=60),  # ruff timeout
        _make_completed_process(returncode=0),  # prettier passes
    ]
    passed, output = _run_lint("/repo")

    assert passed is True
    assert output == "Lint passed"
    assert mock_run.call_count == 3


# ---------------------------------------------------------------------------
# execute_node() — async integration-level tests
# ---------------------------------------------------------------------------


@patch("graft.stages.execute.save_artifact")
@patch("graft.stages.execute.mark_stage_complete")
def test_execute_node_empty_plan(mock_mark, mock_save):
    """Empty build plan returns early without executing anything."""
    ui = _make_ui()
    state = _make_state(build_plan=[])

    result = asyncio.run(execute_node(state, ui))

    assert result["current_stage"] == "execute"
    ui.error.assert_called_once()
    mock_mark.assert_called_once_with("/tmp/project", "execute")


@patch("graft.stages.execute.save_artifact")
@patch("graft.stages.execute.mark_stage_complete")
@patch("graft.stages.execute.subprocess.run")
@patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
def test_execute_node_branch_creation(mock_agent, mock_run, mock_mark, mock_save):
    """execute_node creates a feature branch via checkout -b."""
    # Make checkout -b succeed, then make every other git call succeed
    mock_run.return_value = _make_completed_process(returncode=0)
    # Agent succeeds
    mock_agent.return_value = MagicMock()

    ui = _make_ui()
    plan = [{"unit_id": "u1", "title": "Add widget"}]
    state = _make_state(build_plan=plan, feature_branch="feature/test")

    asyncio.run(execute_node(state, ui))

    # First subprocess call should be checkout -b
    first_call = mock_run.call_args_list[0]
    assert first_call[0][0] == ["git", "checkout", "-b", "feature/test"]


@patch("graft.stages.execute.save_artifact")
@patch("graft.stages.execute.mark_stage_complete")
@patch("graft.stages.execute.subprocess.run")
@patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
def test_execute_node_successful_unit_flow(mock_agent, mock_run, mock_mark, mock_save):
    """Successful flow: agent → commit → test pass → lint pass → keep."""
    mock_agent.return_value = MagicMock()

    # Build a side_effect list that handles each subprocess call:
    # 1. checkout -b (branch creation)
    # 2. git add -A (stage changes)
    # 3. git commit (succeeds - returncode 0)
    # 4. bash -c VERIFY_SCRIPT (tests pass)
    # 5. lint: eslint (pass)
    # 6. git add -A (amend lint fixes)
    # 7. git commit --amend --no-edit
    mock_run.side_effect = [
        _make_completed_process(returncode=0),  # checkout -b
        _make_completed_process(returncode=0),  # git add -A
        _make_completed_process(returncode=0, stdout="1 file changed"),  # commit
        _make_completed_process(returncode=0, stdout="3 passed"),  # tests
        _make_completed_process(returncode=0),  # eslint
        _make_completed_process(returncode=0),  # git add -A (amend)
        _make_completed_process(returncode=0),  # commit --amend
    ]

    ui = _make_ui()
    plan = [{"unit_id": "u1", "title": "Add widget", "tests_included": True}]
    state = _make_state(build_plan=plan)

    result = asyncio.run(execute_node(state, ui))

    assert len(result["units_completed"]) == 1
    assert result["units_completed"][0]["unit_id"] == "u1"
    assert result["units_completed"][0]["tests_included"] is True
    assert len(result["units_reverted"]) == 0
    assert len(result["units_skipped"]) == 0
    ui.unit_kept.assert_called_once_with("u1", "Implemented and passing")


@patch("graft.stages.execute.save_artifact")
@patch("graft.stages.execute.mark_stage_complete")
@patch("graft.stages.execute.subprocess.run")
@patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
def test_execute_node_failed_tests_revert(mock_agent, mock_run, mock_mark, mock_save):
    """Failed tests trigger git revert and unit is recorded as reverted."""
    mock_agent.return_value = MagicMock()

    mock_run.side_effect = [
        _make_completed_process(returncode=0),  # checkout -b
        _make_completed_process(returncode=0),  # git add -A
        _make_completed_process(returncode=0),  # commit succeeds
        _make_completed_process(returncode=1, stderr="FAILED test_x"),  # tests fail
        _make_completed_process(returncode=0),  # git revert HEAD --no-edit
    ]

    ui = _make_ui()
    plan = [{"unit_id": "u1", "title": "Broken widget"}]
    state = _make_state(build_plan=plan)

    result = asyncio.run(execute_node(state, ui))

    assert len(result["units_completed"]) == 0
    assert len(result["units_reverted"]) == 1
    assert result["units_reverted"][0]["unit_id"] == "u1"
    assert "Tests failed" in result["units_reverted"][0]["reason"]
    ui.unit_reverted.assert_called_once_with("u1", "Tests failed")

    # Verify git revert was called
    revert_call = mock_run.call_args_list[4]
    assert revert_call[0][0] == ["git", "revert", "HEAD", "--no-edit"]


@patch("graft.stages.execute.save_artifact")
@patch("graft.stages.execute.mark_stage_complete")
@patch("graft.stages.execute.subprocess.run")
@patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
def test_execute_node_unmet_dependency_skip(mock_agent, mock_run, mock_mark, mock_save):
    """Unit with unmet dependencies is skipped."""
    mock_agent.return_value = MagicMock()

    # First unit succeeds, second depends on "missing_unit"
    mock_run.side_effect = [
        _make_completed_process(returncode=0),  # checkout -b
        _make_completed_process(returncode=0),  # git add -A
        _make_completed_process(returncode=0),  # commit
        _make_completed_process(returncode=0, stdout="ok"),  # tests
        _make_completed_process(returncode=0),  # lint
        _make_completed_process(returncode=0),  # git add -A (amend)
        _make_completed_process(returncode=0),  # commit --amend
    ]

    ui = _make_ui()
    plan = [
        {"unit_id": "u1", "title": "Base widget"},
        {"unit_id": "u2", "title": "Widget extension", "depends_on": ["missing_unit"]},
    ]
    state = _make_state(build_plan=plan)

    result = asyncio.run(execute_node(state, ui))

    assert len(result["units_completed"]) == 1
    assert len(result["units_skipped"]) == 1
    assert result["units_skipped"][0]["unit_id"] == "u2"
    assert "missing_unit" in result["units_skipped"][0]["reason"]


@patch("graft.stages.execute.save_artifact")
@patch("graft.stages.execute.mark_stage_complete")
@patch("graft.stages.execute.subprocess.run")
@patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
def test_execute_node_saves_execution_log(mock_agent, mock_run, mock_mark, mock_save):
    """execution_log.json is saved with correct summary counts."""
    mock_agent.return_value = MagicMock()
    mock_run.return_value = _make_completed_process(returncode=0)

    ui = _make_ui()
    plan = [{"unit_id": "u1", "title": "Widget"}]
    state = _make_state(build_plan=plan)

    asyncio.run(execute_node(state, ui))

    # save_artifact should be called with execution_log.json
    mock_save.assert_called_once()
    call_args = mock_save.call_args
    assert call_args[0][0] == "/tmp/project"
    assert call_args[0][1] == "execution_log.json"

    log = json.loads(call_args[0][2])
    assert log["total_planned"] == 1
    assert log["units_completed"] == 1
    assert log["units_reverted"] == 0
    assert log["units_skipped"] == 0
    assert len(log["completed"]) == 1


@patch("graft.stages.execute.save_artifact")
@patch("graft.stages.execute.mark_stage_complete")
@patch("graft.stages.execute.subprocess.run")
@patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
def test_execute_node_agent_failure_reverts_unit(mock_agent, mock_run, mock_mark, mock_save):
    """RuntimeError from run_agent records the unit as reverted."""
    mock_agent.side_effect = RuntimeError("Agent crashed")
    mock_run.return_value = _make_completed_process(returncode=0)

    ui = _make_ui()
    plan = [{"unit_id": "u1", "title": "Crasher"}]
    state = _make_state(build_plan=plan)

    result = asyncio.run(execute_node(state, ui))

    assert len(result["units_reverted"]) == 1
    assert "Agent crashed" in result["units_reverted"][0]["reason"]
    ui.unit_reverted.assert_called_once()


@patch("graft.stages.execute.save_artifact")
@patch("graft.stages.execute.mark_stage_complete")
@patch("graft.stages.execute.subprocess.run")
@patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
def test_execute_node_no_changes_reverts_unit(mock_agent, mock_run, mock_mark, mock_save):
    """Commit with no changes (returncode != 0) records unit as reverted."""
    mock_agent.return_value = MagicMock()

    mock_run.side_effect = [
        _make_completed_process(returncode=0),  # checkout -b
        _make_completed_process(returncode=0),  # git add -A
        _make_completed_process(returncode=1, stderr="nothing to commit"),  # commit fails
    ]

    ui = _make_ui()
    plan = [{"unit_id": "u1", "title": "No-op"}]
    state = _make_state(build_plan=plan)

    result = asyncio.run(execute_node(state, ui))

    assert len(result["units_reverted"]) == 1
    assert "No changes" in result["units_reverted"][0]["reason"]
    ui.unit_reverted.assert_called_once_with("u1", "No changes made")


@patch("graft.stages.execute.save_artifact")
@patch("graft.stages.execute.mark_stage_complete")
@patch("graft.stages.execute.subprocess.run")
@patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
def test_execute_node_branch_fallback_checkout(mock_agent, mock_run, mock_mark, mock_save):
    """If checkout -b fails (branch exists), falls back to checkout."""
    mock_agent.return_value = MagicMock()

    mock_run.side_effect = [
        _make_completed_process(returncode=1, stderr="already exists"),  # checkout -b fails
        _make_completed_process(returncode=0),  # checkout (fallback)
        _make_completed_process(returncode=0),  # git add -A
        _make_completed_process(returncode=0),  # commit
        _make_completed_process(returncode=0, stdout="ok"),  # tests
        _make_completed_process(returncode=0),  # lint
        _make_completed_process(returncode=0),  # git add -A (amend)
        _make_completed_process(returncode=0),  # commit --amend
    ]

    ui = _make_ui()
    plan = [{"unit_id": "u1", "title": "Widget"}]
    state = _make_state(build_plan=plan, feature_branch="feature/existing")

    result = asyncio.run(execute_node(state, ui))

    # First call: checkout -b, second call: checkout (fallback)
    first_call = mock_run.call_args_list[0]
    second_call = mock_run.call_args_list[1]
    assert first_call[0][0] == ["git", "checkout", "-b", "feature/existing"]
    assert second_call[0][0] == ["git", "checkout", "feature/existing"]
    assert len(result["units_completed"]) == 1


@patch("graft.stages.execute.save_artifact")
@patch("graft.stages.execute.mark_stage_complete")
@patch("graft.stages.execute.subprocess.run")
@patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
def test_execute_node_default_branch_name(mock_agent, mock_run, mock_mark, mock_save):
    """When no feature_branch is in state, a default branch name is derived."""
    mock_agent.return_value = MagicMock()
    mock_run.return_value = _make_completed_process(returncode=0)

    ui = _make_ui()
    plan = [{"unit_id": "u1", "title": "Widget"}]
    # No feature_branch key — should fall back to "feature/{project_id}"
    state = _make_state(build_plan=plan)

    result = asyncio.run(execute_node(state, ui))

    assert result["feature_branch"] == "feature/test123"
    # Verify the checkout -b used the default
    first_call = mock_run.call_args_list[0]
    assert first_call[0][0] == ["git", "checkout", "-b", "feature/test123"]


@patch("graft.stages.execute.save_artifact")
@patch("graft.stages.execute.mark_stage_complete")
@patch("graft.stages.execute.subprocess.run")
@patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
def test_execute_node_marks_stage_complete(mock_agent, mock_run, mock_mark, mock_save):
    """execute_node calls mark_stage_complete and stage_done."""
    mock_agent.return_value = MagicMock()
    mock_run.return_value = _make_completed_process(returncode=0)

    ui = _make_ui()
    plan = [{"unit_id": "u1", "title": "Widget"}]
    state = _make_state(build_plan=plan)

    asyncio.run(execute_node(state, ui))

    mock_mark.assert_called_once_with("/tmp/project", "execute")
    ui.stage_done.assert_called_once_with("execute")


@patch("graft.stages.execute.save_artifact")
@patch("graft.stages.execute.mark_stage_complete")
@patch("graft.stages.execute.subprocess.run")
@patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
def test_execute_node_prompt_includes_pattern_ref(mock_agent, mock_run, mock_mark, mock_save):
    """When unit has pattern_reference, the prompt includes it."""
    mock_agent.return_value = MagicMock()
    mock_run.return_value = _make_completed_process(returncode=0)

    ui = _make_ui()
    plan = [{
        "unit_id": "u1",
        "title": "Add endpoint",
        "pattern_reference": "src/routes/users.py",
        "acceptance_criteria": ["Returns 200"],
    }]
    state = _make_state(build_plan=plan)

    asyncio.run(execute_node(state, ui))

    # Inspect the user_prompt passed to run_agent
    agent_call_kwargs = mock_agent.call_args[1]
    prompt = agent_call_kwargs["user_prompt"]
    assert "src/routes/users.py" in prompt
    assert "Read this file FIRST" in prompt
    assert "Returns 200" in prompt


@patch("graft.stages.execute.save_artifact")
@patch("graft.stages.execute.mark_stage_complete")
@patch("graft.stages.execute.subprocess.run")
@patch("graft.stages.execute.run_agent", new_callable=AsyncMock)
def test_execute_node_multiple_units_mixed_results(mock_agent, mock_run, mock_mark, mock_save):
    """Multiple units: first succeeds, second fails tests — correct tallies."""
    mock_agent.return_value = MagicMock()

    mock_run.side_effect = [
        _make_completed_process(returncode=0),  # checkout -b
        # --- Unit 1 (succeeds) ---
        _make_completed_process(returncode=0),  # git add -A
        _make_completed_process(returncode=0),  # commit
        _make_completed_process(returncode=0, stdout="ok"),  # tests pass
        _make_completed_process(returncode=0),  # lint
        _make_completed_process(returncode=0),  # git add -A (amend)
        _make_completed_process(returncode=0),  # commit --amend
        # --- Unit 2 (test failure) ---
        _make_completed_process(returncode=0),  # git add -A
        _make_completed_process(returncode=0),  # commit
        _make_completed_process(returncode=1, stderr="FAIL"),  # tests fail
        _make_completed_process(returncode=0),  # git revert
    ]

    ui = _make_ui()
    plan = [
        {"unit_id": "u1", "title": "Good unit"},
        {"unit_id": "u2", "title": "Bad unit"},
    ]
    state = _make_state(build_plan=plan)

    result = asyncio.run(execute_node(state, ui))

    assert len(result["units_completed"]) == 1
    assert len(result["units_reverted"]) == 1
    assert result["units_completed"][0]["unit_id"] == "u1"
    assert result["units_reverted"][0]["unit_id"] == "u2"

    # Verify execution_log.json has correct counts
    log = json.loads(mock_save.call_args[0][2])
    assert log["units_completed"] == 1
    assert log["units_reverted"] == 1
    assert log["total_planned"] == 2
