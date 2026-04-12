"""Tests for graft.ui."""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from graft.ui import MAX_DISPLAY_CHARS, STAGE_LABELS, STAGE_ORDER, UI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ui(auto_approve: bool = True, verbose: bool = False) -> UI:
    """Create a UI with a string-buffered console for output capture."""
    ui = UI.__new__(UI)
    ui._current_stage = None
    ui.auto_approve = auto_approve
    ui.verbose = verbose
    ui.console = Console(file=StringIO(), force_terminal=True, width=120)
    return ui


def _output(ui: UI) -> str:
    """Extract captured text from the UI's string-backed console."""
    ui.console.file.seek(0)
    return ui.console.file.read()


# ---------------------------------------------------------------------------
# Original tests (preserved)
# ---------------------------------------------------------------------------


def test_stage_labels_match_pipeline():
    """Stage labels cover all pipeline stages."""
    expected = {
        "discover",
        "research",
        "grill",
        "plan",
        "plan_review",
        "execute",
        "verify",
    }
    assert set(STAGE_LABELS.keys()) == expected


def test_ui_auto_approve_default():
    """UI defaults to non-auto-approve in interactive mode."""
    ui = UI()
    # In test context stdin may not be a tty, so auto_approve could be True
    assert isinstance(ui.auto_approve, bool)


def test_ui_verbose_default():
    """UI defaults to non-verbose."""
    ui = UI()
    assert ui.verbose is False


def test_ui_stage_log_suppressed_when_not_verbose(capsys):
    """stage_log is suppressed when verbose=False."""
    ui = UI(verbose=False)
    ui.stage_log("discover", "should not appear")
    captured = capsys.readouterr()
    assert "should not appear" not in captured.out


# ---------------------------------------------------------------------------
# Constants & initialization
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify module-level constants."""

    def test_stage_order_matches_labels(self):
        assert list(STAGE_ORDER) == list(STAGE_LABELS.keys())

    def test_max_display_chars_is_positive_int(self):
        assert isinstance(MAX_DISPLAY_CHARS, int) and MAX_DISPLAY_CHARS > 0


class TestUIInit:
    """UI.__init__ behavior."""

    def test_auto_approve_explicit_true(self):
        """Passing auto_approve=True forces it on regardless of tty."""
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            ui = UI(auto_approve=True)
            assert ui.auto_approve is True

    def test_auto_approve_non_tty(self):
        """When stdin is not a tty, auto_approve is forced True."""
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            ui = UI(auto_approve=False)
            assert ui.auto_approve is True

    def test_auto_approve_false_when_tty(self):
        """When stdin is a tty and auto_approve not requested, stays False."""
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            ui = UI(auto_approve=False)
            assert ui.auto_approve is False

    def test_verbose_flag(self):
        ui = UI(verbose=True)
        assert ui.verbose is True

    def test_initial_current_stage_is_none(self):
        ui = UI()
        assert ui._current_stage is None


# ---------------------------------------------------------------------------
# banner()
# ---------------------------------------------------------------------------


class TestBanner:
    def test_banner_contains_repo_path(self):
        ui = _make_ui()
        ui.banner("/tmp/my-repo", "sess_abc123", "Add dark mode")
        out = _output(ui)
        assert "/tmp/my-repo" in out

    def test_banner_contains_session_id(self):
        ui = _make_ui()
        ui.banner("/tmp/repo", "sess_xyz789", "feature prompt")
        out = _output(ui)
        assert "sess_xyz789" in out

    def test_banner_contains_feature_prompt(self):
        ui = _make_ui()
        ui.banner("/tmp/repo", "sess_1", "Implement OAuth login")
        out = _output(ui)
        assert "Implement OAuth login" in out

    def test_banner_truncates_long_prompt(self):
        long_prompt = "A" * 200
        ui = _make_ui()
        ui.banner("/tmp/repo", "sess_1", long_prompt)
        out = _output(ui)
        # The first 120 chars should appear; the full 200 should not
        assert "A" * 120 in out
        assert "…" in out

    def test_banner_no_ellipsis_for_short_prompt(self):
        short_prompt = "Short"
        ui = _make_ui()
        ui.banner("/tmp/repo", "sess_1", short_prompt)
        out = _output(ui)
        assert "Short" in out
        # Ellipsis character should not appear inside the panel content
        # (it may appear in box-drawing; check the specific Unicode char)
        # Just ensure no truncation happened
        assert short_prompt in out

    def test_banner_contains_title(self):
        ui = _make_ui()
        ui.banner("/tmp/repo", "sess_1", "prompt")
        out = _output(ui)
        assert "Graft" in out


# ---------------------------------------------------------------------------
# stage_start() / stage_done()
# ---------------------------------------------------------------------------


class TestStageTransitions:
    def test_stage_start_sets_current_stage(self):
        ui = _make_ui()
        ui.stage_start("discover")
        assert ui._current_stage == "discover"

    def test_stage_start_displays_label(self):
        ui = _make_ui()
        ui.stage_start("research")
        out = _output(ui)
        assert "Research" in out

    def test_stage_start_unknown_stage_uses_raw_name(self):
        ui = _make_ui()
        ui.stage_start("custom_stage")
        out = _output(ui)
        assert "custom_stage" in out

    def test_stage_done_displays_complete(self):
        ui = _make_ui()
        ui.stage_done("grill")
        out = _output(ui)
        assert "Grill" in out
        assert "complete" in out

    def test_stage_done_unknown_stage(self):
        ui = _make_ui()
        ui.stage_done("unknown_stage")
        out = _output(ui)
        assert "unknown_stage" in out
        assert "complete" in out

    @pytest.mark.parametrize("stage", list(STAGE_LABELS.keys()))
    def test_stage_start_all_stages(self, stage):
        """Every registered stage can be started without error."""
        ui = _make_ui()
        ui.stage_start(stage)
        assert ui._current_stage == stage

    @pytest.mark.parametrize("stage", list(STAGE_LABELS.keys()))
    def test_stage_done_all_stages(self, stage):
        """Every registered stage can be marked done without error."""
        ui = _make_ui()
        ui.stage_done(stage)
        out = _output(ui)
        assert "complete" in out


# ---------------------------------------------------------------------------
# stage_log()
# ---------------------------------------------------------------------------


class TestStageLog:
    def test_stage_log_visible_when_verbose(self):
        ui = _make_ui(verbose=True)
        ui.stage_log("discover", "found 12 files")
        out = _output(ui)
        assert "found 12 files" in out
        assert "Discover" in out

    def test_stage_log_suppressed_when_not_verbose(self):
        ui = _make_ui(verbose=False)
        ui.stage_log("discover", "should not appear")
        out = _output(ui)
        assert "should not appear" not in out

    def test_stage_log_unknown_stage_verbose(self):
        ui = _make_ui(verbose=True)
        ui.stage_log("my_custom", "hello")
        out = _output(ui)
        assert "my_custom" in out
        assert "hello" in out


# ---------------------------------------------------------------------------
# show_artifact()
# ---------------------------------------------------------------------------


class TestShowArtifact:
    def test_short_content_displayed_fully(self):
        ui = _make_ui()
        ui.show_artifact("My Artifact", "This is the content.")
        out = _output(ui)
        assert "My Artifact" in out
        assert "This is the content." in out

    def test_long_content_truncated(self):
        long_content = "x" * (MAX_DISPLAY_CHARS + 500)
        ui = _make_ui()
        ui.show_artifact("Big File", long_content)
        out = _output(ui)
        assert "truncated" in out
        # First MAX_DISPLAY_CHARS characters should appear
        assert "x" * 100 in out

    def test_exact_boundary_not_truncated(self):
        exact = "y" * MAX_DISPLAY_CHARS
        ui = _make_ui()
        ui.show_artifact("Exact", exact)
        out = _output(ui)
        assert "truncated" not in out

    def test_one_over_boundary_truncated(self):
        over = "z" * (MAX_DISPLAY_CHARS + 1)
        ui = _make_ui()
        ui.show_artifact("Over", over)
        out = _output(ui)
        assert "truncated" in out


# ---------------------------------------------------------------------------
# grill_question()
# ---------------------------------------------------------------------------


class TestGrillQuestion:
    def test_returns_user_input(self):
        ui = _make_ui()
        ui.console.input = MagicMock(return_value="My custom answer")
        result = ui.grill_question("What framework?", "React", "tech", 1)
        assert result == "My custom answer"

    def test_empty_input_returns_recommended(self):
        ui = _make_ui()
        ui.console.input = MagicMock(return_value="")
        result = ui.grill_question("What framework?", "React", "tech", 1)
        assert result == "React"

    def test_whitespace_only_returns_recommended(self):
        ui = _make_ui()
        ui.console.input = MagicMock(return_value="   ")
        result = ui.grill_question("Q?", "default", "cat", 2)
        assert result == "default"

    def test_eof_returns_recommended(self):
        ui = _make_ui()
        ui.console.input = MagicMock(side_effect=EOFError)
        result = ui.grill_question("Q?", "fallback", "cat", 1)
        assert result == "fallback"

    def test_keyboard_interrupt_returns_recommended(self):
        ui = _make_ui()
        ui.console.input = MagicMock(side_effect=KeyboardInterrupt)
        result = ui.grill_question("Q?", "safe", "cat", 3)
        assert result == "safe"

    def test_displays_question_and_metadata(self):
        ui = _make_ui()
        ui.console.input = MagicMock(return_value="answer")
        ui.grill_question("How many?", "42", "sizing", 5)
        out = _output(ui)
        assert "How many?" in out
        assert "sizing" in out
        assert "42" in out
        assert "Question 5" in out


# ---------------------------------------------------------------------------
# prompt_plan_review()
# ---------------------------------------------------------------------------


class TestPromptPlanReview:
    def test_auto_approve_returns_true_no_input(self):
        ui = _make_ui(auto_approve=True)
        approved, feedback = ui.prompt_plan_review("Plan summary here")
        assert approved is True
        assert feedback == ""

    def test_auto_approve_displays_message(self):
        ui = _make_ui(auto_approve=True)
        ui.prompt_plan_review("Plan content")
        out = _output(ui)
        assert "auto-approved" in out

    def test_approve_keyword(self):
        ui = _make_ui(auto_approve=False)
        ui.console.input = MagicMock(return_value="approve")
        approved, feedback = ui.prompt_plan_review("Build plan")
        assert approved is True
        assert feedback == ""

    @pytest.mark.parametrize("response", ["yes", "y", "lgtm", ""])
    def test_approval_synonyms(self, response):
        ui = _make_ui(auto_approve=False)
        ui.console.input = MagicMock(return_value=response)
        approved, feedback = ui.prompt_plan_review("Plan")
        assert approved is True
        assert feedback == ""

    def test_feedback_returns_false_with_text(self):
        ui = _make_ui(auto_approve=False)
        ui.console.input = MagicMock(return_value="Add more tests please")
        approved, feedback = ui.prompt_plan_review("Plan")
        assert approved is False
        assert feedback == "Add more tests please"

    def test_eof_auto_approves(self):
        ui = _make_ui(auto_approve=False)
        ui.console.input = MagicMock(side_effect=EOFError)
        approved, feedback = ui.prompt_plan_review("Plan")
        assert approved is True
        assert feedback == ""

    def test_keyboard_interrupt_auto_approves(self):
        ui = _make_ui(auto_approve=False)
        ui.console.input = MagicMock(side_effect=KeyboardInterrupt)
        approved, feedback = ui.prompt_plan_review("Plan")
        assert approved is True
        assert feedback == ""

    def test_plan_summary_shown_in_artifact(self):
        ui = _make_ui(auto_approve=True)
        ui.prompt_plan_review("Step 1: scaffold\nStep 2: implement")
        out = _output(ui)
        assert "Build Plan" in out
        assert "Step 1" in out

    def test_case_insensitive_approve(self):
        ui = _make_ui(auto_approve=False)
        ui.console.input = MagicMock(return_value="APPROVE")
        approved, _ = ui.prompt_plan_review("Plan")
        assert approved is True

    def test_case_insensitive_yes(self):
        ui = _make_ui(auto_approve=False)
        ui.console.input = MagicMock(return_value="YES")
        approved, _ = ui.prompt_plan_review("Plan")
        assert approved is True


# ---------------------------------------------------------------------------
# unit_start() / unit_kept() / unit_reverted()
# ---------------------------------------------------------------------------


class TestUnitProgress:
    def test_unit_start_displays_info(self):
        ui = _make_ui()
        ui.unit_start("unit_001", "Add login form", 1, 5)
        out = _output(ui)
        assert "unit_001" in out
        assert "Add login form" in out
        assert "1" in out
        assert "5" in out

    def test_unit_kept_displays_delta(self):
        ui = _make_ui()
        ui.unit_kept("unit_001", "+35 -2")
        out = _output(ui)
        assert "kept" in out
        assert "+35 -2" in out

    def test_unit_reverted_displays_reason(self):
        ui = _make_ui()
        ui.unit_reverted("unit_002", "tests failed")
        out = _output(ui)
        assert "reverted" in out
        assert "tests failed" in out


# ---------------------------------------------------------------------------
# pr_opened()
# ---------------------------------------------------------------------------


class TestPrOpened:
    def test_pr_url_displayed(self):
        ui = _make_ui()
        ui.pr_opened("https://github.com/org/repo/pull/42")
        out = _output(ui)
        assert "https://github.com/org/repo/pull/42" in out

    def test_pr_opened_title(self):
        ui = _make_ui()
        ui.pr_opened("https://example.com/pr/1")
        out = _output(ui)
        assert "PR Opened" in out


# ---------------------------------------------------------------------------
# coverage_warning()
# ---------------------------------------------------------------------------


class TestCoverageWarning:
    def test_single_warning(self):
        warnings = [
            {
                "module": "auth.py",
                "coverage_pct": 45,
                "recommendation": "Add integration tests",
            }
        ]
        ui = _make_ui()
        ui.coverage_warning(warnings)
        out = _output(ui)
        assert "auth.py" in out
        assert "45" in out
        assert "Add integration tests" in out
        assert "Coverage Warning" in out

    def test_multiple_warnings(self):
        warnings = [
            {
                "module": "auth.py",
                "coverage_pct": 45,
                "recommendation": "Add tests for auth",
            },
            {
                "module": "db.py",
                "coverage_pct": 30,
                "recommendation": "Add tests for db",
            },
        ]
        ui = _make_ui()
        ui.coverage_warning(warnings)
        out = _output(ui)
        assert "auth.py" in out
        assert "db.py" in out
        assert "30" in out

    def test_empty_warnings_list(self):
        """Even with no warnings, the panel renders without error."""
        ui = _make_ui()
        ui.coverage_warning([])
        out = _output(ui)
        assert "Coverage Warning" in out


# ---------------------------------------------------------------------------
# error() and info()
# ---------------------------------------------------------------------------


class TestErrorAndInfo:
    def test_error_message_displayed(self):
        ui = _make_ui()
        ui.error("Something went wrong")
        out = _output(ui)
        assert "Error" in out
        assert "Something went wrong" in out

    def test_info_message_displayed(self):
        ui = _make_ui()
        ui.info("Processing 5 files")
        out = _output(ui)
        assert "Processing 5 files" in out

    def test_error_with_special_chars(self):
        ui = _make_ui()
        ui.error("File not found: /tmp/[test].py")
        out = _output(ui)
        assert "File not found" in out

    def test_info_with_empty_message(self):
        ui = _make_ui()
        ui.info("")
        # Should not raise


# ---------------------------------------------------------------------------
# _safe_print()
# ---------------------------------------------------------------------------


class TestSafePrint:
    def test_safe_print_normal(self):
        ui = _make_ui()
        ui._safe_print("hello world")
        out = _output(ui)
        assert "hello world" in out

    def test_safe_print_swallows_blocking_io_error(self):
        ui = _make_ui()
        ui.console.print = MagicMock(side_effect=BlockingIOError)
        # Should not raise
        ui._safe_print("test")

    def test_safe_print_swallows_broken_pipe_error(self):
        ui = _make_ui()
        ui.console.print = MagicMock(side_effect=BrokenPipeError)
        ui._safe_print("test")

    def test_safe_print_swallows_os_error(self):
        ui = _make_ui()
        ui.console.print = MagicMock(side_effect=OSError("write failed"))
        ui._safe_print("test")

    def test_safe_print_does_not_swallow_value_error(self):
        ui = _make_ui()
        ui.console.print = MagicMock(side_effect=ValueError("unexpected"))
        with pytest.raises(ValueError, match="unexpected"):
            ui._safe_print("test")


# ---------------------------------------------------------------------------
# stage_start() error handling (mirroring _safe_print for the rule call)
# ---------------------------------------------------------------------------


class TestStageStartErrorHandling:
    def test_stage_start_swallows_broken_pipe(self):
        ui = _make_ui()
        ui.console.rule = MagicMock(side_effect=BrokenPipeError)
        # Should not raise
        ui.stage_start("discover")
        assert ui._current_stage == "discover"

    def test_stage_start_swallows_blocking_io(self):
        ui = _make_ui()
        ui.console.rule = MagicMock(side_effect=BlockingIOError)
        ui.stage_start("plan")
        assert ui._current_stage == "plan"

    def test_stage_start_swallows_os_error(self):
        ui = _make_ui()
        ui.console.rule = MagicMock(side_effect=OSError)
        ui.stage_start("execute")
        assert ui._current_stage == "execute"


# ---------------------------------------------------------------------------
# show_projects()
# ---------------------------------------------------------------------------


class TestShowProjects:
    def test_empty_projects_list(self):
        ui = _make_ui()
        ui.show_projects([])
        out = _output(ui)
        assert "No feature sessions found" in out

    def test_single_project(self):
        projects = [
            {
                "project_id": "feat_abc123",
                "repo_path": "/tmp/my-repo",
                "feature_prompt": "Add dark mode",
                "status": "completed",
                "stages_completed": ["discover", "research"],
                "created_at": "2024-01-15T10:30:00Z",
            }
        ]
        ui = _make_ui()
        ui.show_projects(projects)
        out = _output(ui)
        assert "feat_abc123" in out
        assert "/tmp/my-repo" in out
        assert "Add dark mode" in out
        assert "completed" in out
        assert "discover" in out
        assert "research" in out

    def test_multiple_projects(self):
        projects = [
            {
                "project_id": "feat_001",
                "repo_path": "/repo1",
                "feature_prompt": "Feature A",
                "status": "in_progress",
                "stages_completed": ["discover"],
                "created_at": "2024-01-01T00:00:00Z",
            },
            {
                "project_id": "feat_002",
                "repo_path": "/repo2",
                "feature_prompt": "Feature B",
                "status": "completed",
                "stages_completed": [],
                "created_at": "2024-02-01T00:00:00Z",
            },
        ]
        ui = _make_ui()
        ui.show_projects(projects)
        out = _output(ui)
        assert "feat_001" in out
        assert "feat_002" in out
        assert "Feature A" in out
        assert "Feature B" in out

    def test_project_with_missing_optional_fields(self):
        """Projects with minimal data should render with defaults."""
        projects = [
            {
                "project_id": "feat_minimal",
            }
        ]
        ui = _make_ui()
        ui.show_projects(projects)
        out = _output(ui)
        assert "feat_minimal" in out
        assert "unknown" in out  # default status

    def test_project_truncates_long_repo_path(self):
        long_path = "/very/long/path/" + "x" * 100
        projects = [
            {
                "project_id": "feat_long",
                "repo_path": long_path,
                "feature_prompt": "Some feature",
                "status": "ok",
                "stages_completed": [],
                "created_at": "2024-01-01T00:00:00",
            }
        ]
        ui = _make_ui()
        ui.show_projects(projects)
        out = _output(ui)
        # Repo path truncated to 40 chars
        assert "feat_long" in out

    def test_project_no_stages_shows_dash(self):
        projects = [
            {
                "project_id": "feat_none",
                "repo_path": "/repo",
                "feature_prompt": "Nothing yet",
                "status": "pending",
                "stages_completed": [],
                "created_at": "2024-01-01T00:00:00",
            }
        ]
        ui = _make_ui()
        ui.show_projects(projects)
        out = _output(ui)
        assert "—" in out

    def test_show_projects_table_title(self):
        projects = [
            {
                "project_id": "feat_t",
                "repo_path": "/r",
                "feature_prompt": "f",
                "status": "ok",
                "stages_completed": [],
                "created_at": "2024-01-01T00:00:00",
            }
        ]
        ui = _make_ui()
        ui.show_projects(projects)
        out = _output(ui)
        assert "Feature Sessions" in out
