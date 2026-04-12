"""Tests for graft.stages.execute."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graft.stages.execute import (
    _git,
    _order_by_dependencies,
    _run_lint,
    _run_tests,
    execute_node,
)

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeAgentResult:
    """Minimal stand-in for graft.agent.AgentResult."""

    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    raw_messages: list = field(default_factory=list)
    elapsed_seconds: float = 1.0
    turns_used: int = 5


def _state(repo, project, **kw):
    """Build a minimal FeatureState dict."""
    base = {
        "repo_path": str(repo),
        "project_dir": str(project),
        "project_id": "feat_test01",
    }
    base.update(kw)
    return base


def _make_plan(*unit_defs):
    """Build a list of build-plan unit dicts from compact tuples.

    Each tuple: (unit_id, title[, depends_on])
    """
    units = []
    for ud in unit_defs:
        unit = {
            "unit_id": ud[0],
            "title": ud[1],
            "description": ud[1],
            "pattern_reference": "src/example.py",
            "tests_included": True,
            "acceptance_criteria": ["it works"],
        }
        if len(ud) > 2:
            unit["depends_on"] = ud[2]
        units.append(unit)
    return units


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
    """Mock UI exposing methods execute.py calls."""
    m = MagicMock()
    m.stage_start = MagicMock()
    m.stage_done = MagicMock()
    m.error = MagicMock()
    m.unit_start = MagicMock()
    m.unit_kept = MagicMock()
    m.unit_reverted = MagicMock()
    return m


def _success_proc(stdout="", stderr=""):
    """Return a CompletedProcess that looks like success."""
    return subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=stdout,
        stderr=stderr,
    )


def _fail_proc(stdout="", stderr="error"):
    """Return a CompletedProcess that looks like failure."""
    return subprocess.CompletedProcess(
        args=[],
        returncode=1,
        stdout=stdout,
        stderr=stderr,
    )


# ===================================================================
# Existing _order_by_dependencies tests (preserved verbatim)
# ===================================================================


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


# ===================================================================
# _git helper tests
# ===================================================================


class TestGit:
    """Tests for the _git subprocess helper."""

    def test_git_success(self):
        """_git returns CompletedProcess on success."""
        with patch(
            "graft.stages.execute.subprocess.run", return_value=_success_proc("ok\n")
        ) as mock_run:
            result = _git("/tmp/repo", "status")

        assert result.returncode == 0
        assert result.stdout == "ok\n"
        mock_run.assert_called_once_with(
            ["git", "status"],
            cwd="/tmp/repo",
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )

    def test_git_passes_check_false(self):
        """_git passes check=False through to subprocess."""
        with patch(
            "graft.stages.execute.subprocess.run", return_value=_fail_proc()
        ) as mock_run:
            result = _git("/tmp/repo", "checkout", "-b", "feat", check=False)

        assert result.returncode == 1
        mock_run.assert_called_once_with(
            ["git", "checkout", "-b", "feat"],
            cwd="/tmp/repo",
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def test_git_raises_on_failure_with_check_true(self):
        """_git raises CalledProcessError when check=True and command fails."""
        with patch(
            "graft.stages.execute.subprocess.run",
            side_effect=subprocess.CalledProcessError(128, "git"),
        ):
            with pytest.raises(subprocess.CalledProcessError):
                _git("/tmp/repo", "status")

    def test_git_multiple_args(self):
        """_git passes multiple arguments correctly."""
        with patch(
            "graft.stages.execute.subprocess.run", return_value=_success_proc()
        ) as mock_run:
            _git("/tmp/repo", "commit", "-m", "feat: add thing")

        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == ["git", "commit", "-m", "feat: add thing"]


# ===================================================================
# _run_tests tests
# ===================================================================


class TestRunTests:
    """Tests for the _run_tests helper."""

    def test_tests_pass(self):
        """Returns (True, output) when test suite passes."""
        with patch(
            "graft.stages.execute.subprocess.run",
            return_value=_success_proc(stdout="3 passed", stderr=""),
        ):
            passed, output = _run_tests("/repo")

        assert passed is True
        assert "3 passed" in output

    def test_tests_fail(self):
        """Returns (False, output) when test suite fails."""
        with patch(
            "graft.stages.execute.subprocess.run",
            return_value=_fail_proc(stdout="FAILED test_foo.py", stderr="1 failed"),
        ):
            passed, output = _run_tests("/repo")

        assert passed is False
        assert "FAILED" in output or "failed" in output

    def test_tests_timeout(self):
        """Returns (False, timeout message) when tests exceed 300s."""
        with patch(
            "graft.stages.execute.subprocess.run",
            side_effect=subprocess.TimeoutExpired("bash", 300),
        ):
            passed, output = _run_tests("/repo")

        assert passed is False
        assert "timed out" in output.lower()

    def test_no_test_runner_found(self):
        """Returns (True, skip message) when bash not found."""
        with patch(
            "graft.stages.execute.subprocess.run",
            side_effect=FileNotFoundError("bash"),
        ):
            passed, output = _run_tests("/repo")

        assert passed is True
        assert "skipping" in output.lower()

    def test_output_truncated_to_2000_chars(self):
        """Long test output is truncated to last 2000 characters."""
        long_output = "x" * 5000
        with patch(
            "graft.stages.execute.subprocess.run",
            return_value=_success_proc(stdout=long_output, stderr=""),
        ):
            passed, output = _run_tests("/repo")

        assert passed is True
        assert len(output) == 2000

    def test_combines_stdout_and_stderr(self):
        """Output includes both stdout and stderr."""
        with patch(
            "graft.stages.execute.subprocess.run",
            return_value=_success_proc(stdout="PASS", stderr="warnings here"),
        ):
            passed, output = _run_tests("/repo")

        assert "PASS" in output
        assert "warnings here" in output

    def test_runs_bash_with_verify_script(self):
        """Verifies that the VERIFY_SCRIPT is passed to bash."""
        with patch(
            "graft.stages.execute.subprocess.run", return_value=_success_proc()
        ) as mock_run:
            _run_tests("/my/repo")

        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0][0] == "bash"
        assert args[0][1] == "-c"
        assert kwargs["cwd"] == "/my/repo"
        assert kwargs["timeout"] == 300


# ===================================================================
# _run_lint tests
# ===================================================================


class TestRunLint:
    """Tests for the _run_lint helper."""

    def test_first_linter_succeeds(self):
        """Returns success when first linter (eslint) passes."""
        with patch(
            "graft.stages.execute.subprocess.run",
            return_value=_success_proc(),
        ):
            passed, output = _run_lint("/repo")

        assert passed is True
        assert "lint passed" in output.lower()

    def test_first_fails_second_succeeds(self):
        """Falls through to ruff when eslint fails."""
        side_effects = [
            _fail_proc(),  # eslint fails
            _success_proc(),  # ruff succeeds
        ]
        with patch(
            "graft.stages.execute.subprocess.run",
            side_effect=side_effects,
        ):
            passed, output = _run_lint("/repo")

        assert passed is True
        assert "lint passed" in output.lower()

    def test_all_linters_fail(self):
        """Returns (True, 'No linter found') when all linters fail."""
        side_effects = [
            _fail_proc(),  # eslint
            _fail_proc(),  # ruff
            _fail_proc(),  # prettier
        ]
        with patch(
            "graft.stages.execute.subprocess.run",
            side_effect=side_effects,
        ):
            passed, output = _run_lint("/repo")

        assert passed is True
        assert "no linter" in output.lower()

    def test_all_linters_not_found(self):
        """Returns (True, skip) when all linters raise FileNotFoundError."""
        with patch(
            "graft.stages.execute.subprocess.run",
            side_effect=FileNotFoundError("not installed"),
        ):
            passed, output = _run_lint("/repo")

        assert passed is True
        assert "no linter" in output.lower()

    def test_linter_timeout_continues(self):
        """TimeoutExpired on one linter continues to next."""
        side_effects = [
            subprocess.TimeoutExpired("npx", 60),  # eslint timeout
            _success_proc(),  # ruff succeeds
        ]
        with patch(
            "graft.stages.execute.subprocess.run",
            side_effect=side_effects,
        ):
            passed, output = _run_lint("/repo")

        assert passed is True
        assert "lint passed" in output.lower()

    def test_mixed_errors(self):
        """Mix of FileNotFoundError, TimeoutExpired, and failures."""
        side_effects = [
            FileNotFoundError("npx"),  # eslint not found
            subprocess.TimeoutExpired("ruff", 60),  # ruff timeout
            _fail_proc(),  # prettier fails
        ]
        with patch(
            "graft.stages.execute.subprocess.run",
            side_effect=side_effects,
        ):
            passed, output = _run_lint("/repo")

        assert passed is True
        assert "no linter" in output.lower()

    def test_tries_three_linters(self):
        """All three linters are attempted when they all raise FileNotFoundError."""
        with patch(
            "graft.stages.execute.subprocess.run",
            side_effect=FileNotFoundError("not found"),
        ) as mock_run:
            _run_lint("/repo")

        assert mock_run.call_count == 3


# ===================================================================
# execute_node tests
# ===================================================================


class TestExecuteNodeEmptyPlan:
    """execute_node with no build_plan."""

    async def test_empty_plan_returns_early(self, repo, project, ui):
        """Empty build_plan logs an error and marks stage complete."""
        state = _state(repo, project, build_plan=[])
        result = await execute_node(state, ui)

        ui.error.assert_called_once()
        assert "No build plan" in ui.error.call_args[0][0]
        assert result["current_stage"] == "execute"

    async def test_no_build_plan_key(self, repo, project, ui):
        """Missing build_plan key treated as empty."""
        state = _state(repo, project)
        # No build_plan key at all
        result = await execute_node(state, ui)

        ui.error.assert_called_once()
        assert result["current_stage"] == "execute"


class TestExecuteNodeBranch:
    """Feature branch creation logic."""

    async def test_creates_feature_branch(self, repo, project, ui):
        """execute_node creates a feature branch from project_id."""
        plan = _make_plan(("u1", "Add button"))

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run") as mock_sub,
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            mock_agent.return_value = FakeAgentResult(text="done")
            # checkout -b succeeds, then add, commit succeed, tests pass, lint pass
            mock_sub.return_value = _success_proc()

            result = await execute_node(_state(repo, project, build_plan=plan), ui)

        assert result["feature_branch"] == "feature/feat_test01"
        # First subprocess call should be git checkout -b
        first_call = mock_sub.call_args_list[0]
        assert first_call[0][0] == ["git", "checkout", "-b", "feature/feat_test01"]

    async def test_switches_to_existing_branch(self, repo, project, ui):
        """If branch already exists, falls back to checking it out."""
        plan = _make_plan(("u1", "Add button"))

        call_idx = [0]

        def _side_effect(cmd, **kwargs):
            call_idx[0] += 1
            # First call: checkout -b fails (branch exists)
            if call_idx[0] == 1:
                return _fail_proc(stderr="already exists")
            # All others: succeed
            return _success_proc()

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch(
                "graft.stages.execute.subprocess.run", side_effect=_side_effect
            ) as mock_sub,
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            mock_agent.return_value = FakeAgentResult(text="done")
            await execute_node(_state(repo, project, build_plan=plan), ui)

        # Second call should be git checkout (no -b) of the same branch
        second_call = mock_sub.call_args_list[1]
        assert second_call[0][0] == ["git", "checkout", "feature/feat_test01"]

    async def test_uses_explicit_feature_branch(self, repo, project, ui):
        """Explicit feature_branch in state is used instead of default."""
        plan = _make_plan(("u1", "Add button"))

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run") as mock_sub,
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            mock_agent.return_value = FakeAgentResult(text="done")
            mock_sub.return_value = _success_proc()

            result = await execute_node(
                _state(repo, project, build_plan=plan, feature_branch="my-branch"),
                ui,
            )

        assert result["feature_branch"] == "my-branch"
        first_call = mock_sub.call_args_list[0]
        assert first_call[0][0] == ["git", "checkout", "-b", "my-branch"]


class TestExecuteNodeHappyPath:
    """Happy-path execute_node with successful units."""

    async def test_two_unit_plan(self, repo, project, ui):
        """Two units both pass — both end up in units_completed."""
        plan = _make_plan(("u1", "Add model"), ("u2", "Add view"))

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run") as mock_sub,
            patch("graft.stages.execute.save_artifact") as mock_save,
            patch("graft.stages.execute.mark_stage_complete") as mock_mark,
        ):
            mock_agent.return_value = FakeAgentResult(text="done")
            mock_sub.return_value = _success_proc()

            result = await execute_node(_state(repo, project, build_plan=plan), ui)

        assert len(result["units_completed"]) == 2
        assert len(result["units_reverted"]) == 0
        assert len(result["units_skipped"]) == 0
        assert result["units_completed"][0]["unit_id"] == "u1"
        assert result["units_completed"][1]["unit_id"] == "u2"

    async def test_agent_called_per_unit(self, repo, project, ui):
        """run_agent is invoked once per unit."""
        plan = _make_plan(("u1", "Add model"), ("u2", "Add view"))

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run") as mock_sub,
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            mock_agent.return_value = FakeAgentResult(text="done")
            mock_sub.return_value = _success_proc()

            await execute_node(_state(repo, project, build_plan=plan), ui)

        assert mock_agent.call_count == 2
        # Verify each call has correct stage
        stages = [c.kwargs["stage"] for c in mock_agent.call_args_list]
        assert stages == ["execute_u1", "execute_u2"]

    async def test_unit_kept_ui_called(self, repo, project, ui):
        """UI.unit_kept is called for each successful unit."""
        plan = _make_plan(("u1", "Add model"))

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run", return_value=_success_proc()),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            mock_agent.return_value = FakeAgentResult(text="done")
            await execute_node(_state(repo, project, build_plan=plan), ui)

        ui.unit_start.assert_called_once_with("u1", "Add model", 1, 1)
        ui.unit_kept.assert_called_once_with("u1", "Implemented and passing")

    async def test_commit_message_contains_title(self, repo, project, ui):
        """Git commit message includes the unit title."""
        plan = _make_plan(("u1", "Add button component"))

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch(
                "graft.stages.execute.subprocess.run", return_value=_success_proc()
            ) as mock_sub,
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            mock_agent.return_value = FakeAgentResult(text="done")
            await execute_node(_state(repo, project, build_plan=plan), ui)

        # Find the commit call
        commit_calls = [
            c
            for c in mock_sub.call_args_list
            if len(c[0][0]) >= 2 and c[0][0][1] == "commit"
        ]
        assert len(commit_calls) >= 1
        # First commit should contain the title
        assert "feat: Add button component" in commit_calls[0][0][0]

    async def test_completed_unit_has_metadata(self, repo, project, ui):
        """Completed unit dicts contain expected metadata fields."""
        plan = [
            {
                "unit_id": "u1",
                "title": "Add model",
                "description": "Build the model",
                "pattern_reference": "src/models.py",
                "tests_included": True,
                "acceptance_criteria": ["it works"],
                "category": "backend",
            }
        ]

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run", return_value=_success_proc()),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            mock_agent.return_value = FakeAgentResult(text="done")
            result = await execute_node(_state(repo, project, build_plan=plan), ui)

        completed = result["units_completed"][0]
        assert completed["unit_id"] == "u1"
        assert completed["title"] == "Add model"
        assert completed["category"] == "backend"
        assert completed["tests_included"] is True


class TestExecuteNodeRevert:
    """Unit revert when tests fail."""

    async def test_revert_on_test_failure(self, repo, project, ui):
        """When tests fail, the commit is reverted and unit goes to reverted."""
        plan = _make_plan(("u1", "Add thing"))

        call_idx = [0]

        def _side_effect(cmd, **kwargs):
            call_idx[0] += 1
            cmd_list = cmd
            # git checkout -b => success
            # git add -A => success
            # git commit => success
            # bash -c (tests) => FAIL
            if cmd_list[0] == "bash":
                return _fail_proc(stdout="FAILED test_thing.py", stderr="1 failed")
            return _success_proc()

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch(
                "graft.stages.execute.subprocess.run", side_effect=_side_effect
            ) as mock_sub,
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            mock_agent.return_value = FakeAgentResult(text="done")
            result = await execute_node(_state(repo, project, build_plan=plan), ui)

        assert len(result["units_completed"]) == 0
        assert len(result["units_reverted"]) == 1
        assert result["units_reverted"][0]["unit_id"] == "u1"
        assert "Tests failed" in result["units_reverted"][0]["reason"]

        # Verify git revert HEAD --no-edit was called
        revert_calls = [
            c
            for c in mock_sub.call_args_list
            if len(c[0][0]) >= 3 and c[0][0][:2] == ["git", "revert"]
        ]
        assert len(revert_calls) == 1
        assert revert_calls[0][0][0] == ["git", "revert", "HEAD", "--no-edit"]

    async def test_revert_ui_notification(self, repo, project, ui):
        """UI.unit_reverted is called when tests fail."""
        plan = _make_plan(("u1", "Add thing"))

        def _side_effect(cmd, **kwargs):
            if cmd[0] == "bash":
                return _fail_proc(stdout="FAIL")
            return _success_proc()

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run", side_effect=_side_effect),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            mock_agent.return_value = FakeAgentResult(text="done")
            await execute_node(_state(repo, project, build_plan=plan), ui)

        ui.unit_reverted.assert_called_once_with("u1", "Tests failed")

    async def test_no_changes_made_reverts(self, repo, project, ui):
        """Unit with no changes (empty commit) is reverted."""
        plan = _make_plan(("u1", "Add thing"))

        call_idx = [0]

        def _side_effect(cmd, **kwargs):
            call_idx[0] += 1
            cmd_list = cmd
            # git commit => FAIL (nothing to commit)
            if (
                len(cmd_list) >= 2
                and cmd_list[1] == "commit"
                and "--amend" not in cmd_list
            ):
                return _fail_proc(stderr="nothing to commit")
            return _success_proc()

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run", side_effect=_side_effect),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            mock_agent.return_value = FakeAgentResult(text="done")
            result = await execute_node(_state(repo, project, build_plan=plan), ui)

        assert len(result["units_reverted"]) == 1
        assert "No changes" in result["units_reverted"][0]["reason"]
        ui.unit_reverted.assert_called_once_with("u1", "No changes made")


class TestExecuteNodeAgentFailure:
    """Error handling when agent fails mid-unit."""

    async def test_agent_runtime_error(self, repo, project, ui):
        """RuntimeError from agent → unit reverted, execution continues."""
        plan = _make_plan(("u1", "Broken unit"), ("u2", "Good unit"))

        agent_call_count = [0]

        async def _agent_side_effect(**kwargs):
            agent_call_count[0] += 1
            if agent_call_count[0] == 1:
                raise RuntimeError("Claude API error")
            return FakeAgentResult(text="done")

        with (
            patch(
                "graft.stages.execute.run_agent",
                new_callable=AsyncMock,
                side_effect=_agent_side_effect,
            ),
            patch("graft.stages.execute.subprocess.run", return_value=_success_proc()),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            result = await execute_node(_state(repo, project, build_plan=plan), ui)

        assert len(result["units_reverted"]) == 1
        assert result["units_reverted"][0]["unit_id"] == "u1"
        assert "Claude API error" in result["units_reverted"][0]["reason"]
        assert len(result["units_completed"]) == 1
        assert result["units_completed"][0]["unit_id"] == "u2"

    async def test_agent_failure_ui_notification(self, repo, project, ui):
        """UI.unit_reverted is called when agent raises."""
        plan = _make_plan(("u1", "Broken"))

        with (
            patch(
                "graft.stages.execute.run_agent",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
            patch("graft.stages.execute.subprocess.run", return_value=_success_proc()),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            await execute_node(_state(repo, project, build_plan=plan), ui)

        ui.unit_reverted.assert_called_once_with("u1", "Agent failed: boom")


class TestExecuteNodeDependencies:
    """Dependency tracking and skipping."""

    async def test_skips_unit_with_unmet_deps(self, repo, project, ui):
        """Unit whose dependency was reverted is skipped."""
        plan = _make_plan(
            ("u1", "Base unit"),
            ("u2", "Depends on u1", ["u1"]),
        )

        # u1's agent fails, so u2's dependency is unmet
        async def _agent_side_effect(**kwargs):
            if "u1" in kwargs.get("stage", ""):
                raise RuntimeError("agent crash")
            return FakeAgentResult(text="done")

        with (
            patch(
                "graft.stages.execute.run_agent",
                new_callable=AsyncMock,
                side_effect=_agent_side_effect,
            ),
            patch("graft.stages.execute.subprocess.run", return_value=_success_proc()),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            result = await execute_node(_state(repo, project, build_plan=plan), ui)

        assert len(result["units_reverted"]) == 1
        assert result["units_reverted"][0]["unit_id"] == "u1"
        assert len(result["units_skipped"]) == 1
        assert result["units_skipped"][0]["unit_id"] == "u2"
        assert "u1" in result["units_skipped"][0]["reason"]
        assert len(result["units_completed"]) == 0

    async def test_skipped_unit_ui_notification(self, repo, project, ui):
        """UI.unit_reverted is called for skipped units (with unmet dep info)."""
        plan = _make_plan(
            ("u1", "Base"),
            ("u2", "Child", ["u1"]),
        )

        with (
            patch(
                "graft.stages.execute.run_agent",
                new_callable=AsyncMock,
                side_effect=RuntimeError("fail"),
            ),
            patch("graft.stages.execute.subprocess.run", return_value=_success_proc()),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            await execute_node(_state(repo, project, build_plan=plan), ui)

        # u1 reverted due to agent failure, u2 reverted (UI call) due to unmet deps
        reverted_calls = ui.unit_reverted.call_args_list
        assert len(reverted_calls) == 2
        # u2's call mentions unmet deps
        u2_call = [c for c in reverted_calls if c[0][0] == "u2"]
        assert len(u2_call) == 1
        assert "Unmet dependencies" in u2_call[0][0][1]


class TestExecuteNodeArtifacts:
    """Execution log and artifact saving."""

    async def test_saves_execution_log(self, repo, project, ui):
        """execution_log.json is saved with correct counts."""
        plan = _make_plan(("u1", "Unit one"))

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run", return_value=_success_proc()),
            patch("graft.stages.execute.save_artifact") as mock_save,
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            mock_agent.return_value = FakeAgentResult(text="done")
            await execute_node(_state(repo, project, build_plan=plan), ui)

        mock_save.assert_called_once()
        call_args = mock_save.call_args
        assert call_args[0][1] == "execution_log.json"

        log = json.loads(call_args[0][2])
        assert log["units_completed"] == 1
        assert log["units_reverted"] == 0
        assert log["units_skipped"] == 0
        assert log["total_planned"] == 1
        assert len(log["completed"]) == 1
        assert log["completed"][0]["unit_id"] == "u1"

    async def test_execution_log_with_mixed_results(self, repo, project, ui):
        """Execution log reflects mix of completed, reverted, skipped units."""
        plan = _make_plan(
            ("u1", "Good unit"),
            ("u2", "Broken unit"),
            ("u3", "Depends on u2", ["u2"]),
        )

        agent_count = [0]

        async def _agent_side_effect(**kwargs):
            agent_count[0] += 1
            if "u2" in kwargs.get("stage", ""):
                raise RuntimeError("crash")
            return FakeAgentResult(text="done")

        with (
            patch(
                "graft.stages.execute.run_agent",
                new_callable=AsyncMock,
                side_effect=_agent_side_effect,
            ),
            patch("graft.stages.execute.subprocess.run", return_value=_success_proc()),
            patch("graft.stages.execute.save_artifact") as mock_save,
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            result = await execute_node(_state(repo, project, build_plan=plan), ui)

        log = json.loads(mock_save.call_args[0][2])
        assert log["units_completed"] == 1
        assert log["units_reverted"] == 1
        assert log["units_skipped"] == 1
        assert log["total_planned"] == 3

    async def test_marks_stage_complete(self, repo, project, ui):
        """mark_stage_complete is called with 'execute'."""
        plan = _make_plan(("u1", "Unit"))

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run", return_value=_success_proc()),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete") as mock_mark,
        ):
            mock_agent.return_value = FakeAgentResult(text="done")
            await execute_node(_state(repo, project, build_plan=plan), ui)

        mock_mark.assert_called_once_with(str(project), "execute")

    async def test_stage_ui_lifecycle(self, repo, project, ui):
        """stage_start and stage_done are called."""
        plan = _make_plan(("u1", "Unit"))

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run", return_value=_success_proc()),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            mock_agent.return_value = FakeAgentResult(text="done")
            await execute_node(_state(repo, project, build_plan=plan), ui)

        ui.stage_start.assert_called_once_with("execute")
        ui.stage_done.assert_called_once_with("execute")


class TestExecuteNodeLint:
    """Lint fix and commit amend behavior."""

    async def test_lint_pass_amends_commit(self, repo, project, ui):
        """When lint passes, a git add + commit --amend --no-edit follows."""
        plan = _make_plan(("u1", "Unit"))

        subprocess_calls = []

        def _side_effect(cmd, **kwargs):
            subprocess_calls.append(cmd)
            return _success_proc()

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run", side_effect=_side_effect),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            mock_agent.return_value = FakeAgentResult(text="done")
            await execute_node(_state(repo, project, build_plan=plan), ui)

        # Find amend commit call
        amend_calls = [
            c
            for c in subprocess_calls
            if isinstance(c, list) and len(c) >= 3 and c[0] == "git" and "--amend" in c
        ]
        assert len(amend_calls) == 1
        assert "--no-edit" in amend_calls[0]


class TestExecuteNodePromptConstruction:
    """Verify the prompts sent to run_agent."""

    async def test_prompt_includes_pattern_reference(self, repo, project, ui):
        """Prompt includes the pattern reference instruction."""
        plan = [
            {
                "unit_id": "u1",
                "title": "Add button",
                "description": "Build a button component",
                "pattern_reference": "src/components/Button.tsx",
                "tests_included": False,
                "acceptance_criteria": ["renders correctly"],
            }
        ]

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run", return_value=_success_proc()),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            mock_agent.return_value = FakeAgentResult(text="done")
            await execute_node(_state(repo, project, build_plan=plan), ui)

        prompt = mock_agent.call_args.kwargs["user_prompt"]
        assert "src/components/Button.tsx" in prompt
        assert "Read this file FIRST" in prompt

    async def test_prompt_includes_tests_instruction_when_enabled(
        self, repo, project, ui
    ):
        """Prompt includes test instruction when tests_included is True."""
        plan = _make_plan(("u1", "Add model"))

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run", return_value=_success_proc()),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            mock_agent.return_value = FakeAgentResult(text="done")
            await execute_node(_state(repo, project, build_plan=plan), ui)

        prompt = mock_agent.call_args.kwargs["user_prompt"]
        assert "TESTS:" in prompt
        assert "co-located tests" in prompt

    async def test_prompt_omits_tests_when_disabled(self, repo, project, ui):
        """Prompt does NOT include test instruction when tests_included is False."""
        plan = [
            {
                "unit_id": "u1",
                "title": "Config change",
                "description": "Update config",
                "pattern_reference": "",
                "tests_included": False,
                "acceptance_criteria": [],
            }
        ]

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run", return_value=_success_proc()),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            mock_agent.return_value = FakeAgentResult(text="done")
            await execute_node(_state(repo, project, build_plan=plan), ui)

        prompt = mock_agent.call_args.kwargs["user_prompt"]
        assert "TESTS:" not in prompt

    async def test_prompt_omits_pattern_ref_when_empty(self, repo, project, ui):
        """No pattern reference instruction when pattern_reference is empty."""
        plan = [
            {
                "unit_id": "u1",
                "title": "Add thing",
                "description": "Add a thing",
                "pattern_reference": "",
                "tests_included": False,
                "acceptance_criteria": [],
            }
        ]

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run", return_value=_success_proc()),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            mock_agent.return_value = FakeAgentResult(text="done")
            await execute_node(_state(repo, project, build_plan=plan), ui)

        prompt = mock_agent.call_args.kwargs["user_prompt"]
        assert "PATTERN REFERENCE:" not in prompt

    async def test_agent_receives_correct_kwargs(self, repo, project, ui):
        """run_agent kwargs include persona, system_prompt, cwd, etc."""
        plan = _make_plan(("u1", "Unit"))

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run", return_value=_success_proc()),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            mock_agent.return_value = FakeAgentResult(text="done")
            await execute_node(
                _state(
                    repo, project, build_plan=plan, model="opus", max_agent_turns=25
                ),
                ui,
            )

        kwargs = mock_agent.call_args.kwargs
        assert "Principal Software Engineer" in kwargs["persona"]
        assert kwargs["cwd"] == str(repo)
        assert kwargs["project_dir"] == str(project)
        assert kwargs["stage"] == "execute_u1"
        assert kwargs["model"] == "opus"
        assert kwargs["max_turns"] == 25


class TestExecuteNodeUnitIdFallback:
    """Unit ID defaults when unit_id key is missing."""

    async def test_auto_generated_unit_id(self, repo, project, ui):
        """Units without unit_id get auto-generated IDs feat_01, feat_02."""
        plan = [
            {"title": "First thing", "description": "do it"},
            {"title": "Second thing", "description": "do it too"},
        ]

        with (
            patch(
                "graft.stages.execute.run_agent", new_callable=AsyncMock
            ) as mock_agent,
            patch("graft.stages.execute.subprocess.run", return_value=_success_proc()),
            patch("graft.stages.execute.save_artifact"),
            patch("graft.stages.execute.mark_stage_complete"),
        ):
            mock_agent.return_value = FakeAgentResult(text="done")
            result = await execute_node(_state(repo, project, build_plan=plan), ui)

        # The auto-generated IDs should be feat_01, feat_02
        stages = [c.kwargs["stage"] for c in mock_agent.call_args_list]
        assert stages == ["execute_feat_01", "execute_feat_02"]
