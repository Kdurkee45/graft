"""Tests for graft.ui."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from graft.ui import MAX_DISPLAY_CHARS, STAGE_LABELS, STAGE_ORDER, UI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ui(*, auto_approve: bool = False, verbose: bool = False) -> tuple[UI, io.StringIO]:
    """Create a UI whose output goes to a StringIO buffer for assertions.

    Returns (ui, buffer).  Read the buffer with ``buf.getvalue()``.
    """
    ui = UI.__new__(UI)
    buf = io.StringIO()
    ui.console = Console(file=buf, force_terminal=False, width=120)
    ui._current_stage = None
    ui.auto_approve = auto_approve
    ui.verbose = verbose
    return ui, buf


# ---------------------------------------------------------------------------
# Existing tests (kept intact)
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
# STAGE_ORDER
# ---------------------------------------------------------------------------


def test_stage_order_matches_labels():
    """STAGE_ORDER should list exactly the keys of STAGE_LABELS in order."""
    assert list(STAGE_ORDER) == list(STAGE_LABELS.keys())


# ---------------------------------------------------------------------------
# banner()
# ---------------------------------------------------------------------------


class TestBanner:
    def test_banner_contains_repo_path(self):
        ui, buf = _make_ui()
        ui.banner("/home/user/repo", "proj-123", "Add login feature")
        output = buf.getvalue()
        assert "/home/user/repo" in output

    def test_banner_contains_project_id(self):
        ui, buf = _make_ui()
        ui.banner("/repo", "session-abc-42", "prompt text")
        output = buf.getvalue()
        assert "session-abc-42" in output

    def test_banner_contains_feature_prompt(self):
        ui, buf = _make_ui()
        ui.banner("/repo", "p1", "Build a REST API")
        output = buf.getvalue()
        assert "Build a REST API" in output

    def test_banner_truncates_long_prompt(self):
        long_prompt = "x" * 200
        ui, buf = _make_ui()
        ui.banner("/repo", "p1", long_prompt)
        output = buf.getvalue()
        # The first 120 chars should be present
        assert "x" * 120 in output
        # The trailing ellipsis character should be present
        assert "…" in output
        # Full 200-char string should NOT appear
        assert "x" * 200 not in output

    def test_banner_no_truncation_for_short_prompt(self):
        short_prompt = "y" * 120
        ui, buf = _make_ui()
        ui.banner("/repo", "p1", short_prompt)
        output = buf.getvalue()
        assert "y" * 120 in output
        # No ellipsis when exactly at boundary
        assert "…" not in output

    def test_banner_contains_title(self):
        ui, buf = _make_ui()
        ui.banner("/repo", "p1", "prompt")
        output = buf.getvalue()
        assert "Graft" in output
        assert "Feature Factory" in output


# ---------------------------------------------------------------------------
# stage_start() / stage_done()
# ---------------------------------------------------------------------------


class TestStageStartDone:
    def test_stage_start_sets_current_stage(self):
        ui, _ = _make_ui()
        assert ui._current_stage is None
        ui.stage_start("discover")
        assert ui._current_stage == "discover"

    def test_stage_start_prints_label(self):
        ui, buf = _make_ui()
        ui.stage_start("discover")
        output = buf.getvalue()
        assert "Discover" in output

    def test_stage_start_unknown_stage_uses_raw_name(self):
        ui, buf = _make_ui()
        ui.stage_start("custom_stage")
        output = buf.getvalue()
        assert "custom_stage" in output

    def test_stage_done_prints_complete(self):
        ui, buf = _make_ui()
        ui.stage_done("research")
        output = buf.getvalue()
        assert "Research" in output
        assert "complete" in output

    def test_stage_done_unknown_stage_uses_raw_name(self):
        ui, buf = _make_ui()
        ui.stage_done("unknown_stage")
        output = buf.getvalue()
        assert "unknown_stage" in output
        assert "complete" in output

    @pytest.mark.parametrize("stage", list(STAGE_LABELS.keys()))
    def test_stage_start_all_known_stages(self, stage):
        ui, buf = _make_ui()
        ui.stage_start(stage)
        output = buf.getvalue()
        # The human-readable portion of the label (without emoji) should appear
        label_text = STAGE_LABELS[stage].split(" ", 1)[1]
        assert label_text in output

    @pytest.mark.parametrize("stage", list(STAGE_LABELS.keys()))
    def test_stage_done_all_known_stages(self, stage):
        ui, buf = _make_ui()
        ui.stage_done(stage)
        output = buf.getvalue()
        label_text = STAGE_LABELS[stage].split(" ", 1)[1]
        assert label_text in output


# ---------------------------------------------------------------------------
# stage_log()
# ---------------------------------------------------------------------------


class TestStageLog:
    def test_stage_log_verbose_prints_message(self):
        ui, buf = _make_ui(verbose=True)
        ui.stage_log("discover", "found 12 files")
        output = buf.getvalue()
        assert "found 12 files" in output
        assert "Discover" in output

    def test_stage_log_non_verbose_suppresses(self):
        ui, buf = _make_ui(verbose=False)
        ui.stage_log("discover", "should not appear")
        output = buf.getvalue()
        assert output == ""

    def test_stage_log_unknown_stage_verbose(self):
        ui, buf = _make_ui(verbose=True)
        ui.stage_log("mystery", "hello")
        output = buf.getvalue()
        assert "mystery" in output
        assert "hello" in output


# ---------------------------------------------------------------------------
# show_artifact()
# ---------------------------------------------------------------------------


class TestShowArtifact:
    def test_show_artifact_displays_title_and_content(self):
        ui, buf = _make_ui()
        ui.show_artifact("My Title", "Some content here")
        output = buf.getvalue()
        assert "My Title" in output
        assert "Some content here" in output

    def test_show_artifact_no_truncation_within_limit(self):
        content = "a" * (MAX_DISPLAY_CHARS - 1)
        ui, buf = _make_ui()
        ui.show_artifact("Title", content)
        output = buf.getvalue()
        assert "truncated" not in output

    def test_show_artifact_exact_limit_no_truncation(self):
        content = "b" * MAX_DISPLAY_CHARS
        ui, buf = _make_ui()
        ui.show_artifact("Title", content)
        output = buf.getvalue()
        assert "truncated" not in output

    def test_show_artifact_truncates_beyond_limit(self):
        content = "c" * (MAX_DISPLAY_CHARS + 100)
        ui, buf = _make_ui()
        ui.show_artifact("Title", content)
        output = buf.getvalue()
        assert "truncated" in output
        # Full content should NOT be present
        assert "c" * (MAX_DISPLAY_CHARS + 100) not in output
        # But the first MAX_DISPLAY_CHARS chars should be
        assert "c" * MAX_DISPLAY_CHARS in output

    def test_show_artifact_empty_content(self):
        ui, buf = _make_ui()
        ui.show_artifact("Empty", "")
        output = buf.getvalue()
        assert "Empty" in output


# ---------------------------------------------------------------------------
# grill_question()
# ---------------------------------------------------------------------------


class TestGrillQuestion:
    def test_grill_question_displays_question_and_recommended(self):
        ui, buf = _make_ui()
        with patch.object(ui.console, "input", return_value="my answer"):
            result = ui.grill_question(
                "What scope?", "Module-level", "architecture", 1
            )
        output = buf.getvalue()
        assert "What scope?" in output
        assert "Module-level" in output
        assert "architecture" in output
        assert "Question 1" in output
        assert result == "my answer"

    def test_grill_question_returns_recommended_on_empty_input(self):
        ui, buf = _make_ui()
        with patch.object(ui.console, "input", return_value=""):
            result = ui.grill_question(
                "Question?", "default answer", "cat", 2
            )
        assert result == "default answer"

    def test_grill_question_strips_whitespace(self):
        ui, _ = _make_ui()
        with patch.object(ui.console, "input", return_value="  spaces  "):
            result = ui.grill_question("Q?", "rec", "cat", 1)
        assert result == "spaces"

    def test_grill_question_returns_recommended_on_whitespace_only(self):
        ui, _ = _make_ui()
        with patch.object(ui.console, "input", return_value="   "):
            result = ui.grill_question("Q?", "recommended", "cat", 1)
        assert result == "recommended"

    def test_grill_question_handles_eof_error(self):
        ui, _ = _make_ui()
        with patch.object(ui.console, "input", side_effect=EOFError):
            result = ui.grill_question("Q?", "fallback", "cat", 1)
        assert result == "fallback"

    def test_grill_question_handles_keyboard_interrupt(self):
        ui, _ = _make_ui()
        with patch.object(ui.console, "input", side_effect=KeyboardInterrupt):
            result = ui.grill_question("Q?", "fallback", "cat", 1)
        assert result == "fallback"

    def test_grill_question_displays_number(self):
        ui, buf = _make_ui()
        with patch.object(ui.console, "input", return_value="answer"):
            ui.grill_question("Q?", "rec", "cat", 42)
        output = buf.getvalue()
        assert "Question 42" in output


# ---------------------------------------------------------------------------
# prompt_plan_review()
# ---------------------------------------------------------------------------


class TestPromptPlanReview:
    def test_auto_approve_returns_true_no_prompt(self):
        ui, buf = _make_ui(auto_approve=True)
        approved, feedback = ui.prompt_plan_review("Build plan text")
        assert approved is True
        assert feedback == ""
        output = buf.getvalue()
        assert "auto-approved" in output

    def test_manual_approve_with_approve_input(self):
        ui, _ = _make_ui(auto_approve=False)
        with patch.object(ui.console, "input", return_value="approve"):
            approved, feedback = ui.prompt_plan_review("Plan text")
        assert approved is True
        assert feedback == ""

    def test_manual_approve_with_yes_input(self):
        ui, _ = _make_ui(auto_approve=False)
        with patch.object(ui.console, "input", return_value="yes"):
            approved, feedback = ui.prompt_plan_review("Plan text")
        assert approved is True
        assert feedback == ""

    def test_manual_approve_with_y_input(self):
        ui, _ = _make_ui(auto_approve=False)
        with patch.object(ui.console, "input", return_value="y"):
            approved, feedback = ui.prompt_plan_review("Plan text")
        assert approved is True
        assert feedback == ""

    def test_manual_approve_with_lgtm_input(self):
        ui, _ = _make_ui(auto_approve=False)
        with patch.object(ui.console, "input", return_value="lgtm"):
            approved, feedback = ui.prompt_plan_review("Plan text")
        assert approved is True
        assert feedback == ""

    def test_manual_approve_with_empty_input(self):
        ui, _ = _make_ui(auto_approve=False)
        with patch.object(ui.console, "input", return_value=""):
            approved, feedback = ui.prompt_plan_review("Plan text")
        assert approved is True
        assert feedback == ""

    def test_manual_reject_returns_feedback(self):
        ui, _ = _make_ui(auto_approve=False)
        with patch.object(ui.console, "input", return_value="Please add error handling"):
            approved, feedback = ui.prompt_plan_review("Plan text")
        assert approved is False
        assert feedback == "Please add error handling"

    def test_eof_error_auto_approves(self):
        ui, buf = _make_ui(auto_approve=False)
        with patch.object(ui.console, "input", side_effect=EOFError):
            approved, feedback = ui.prompt_plan_review("Plan text")
        assert approved is True
        assert feedback == ""
        output = buf.getvalue()
        assert "auto-approving" in output

    def test_keyboard_interrupt_auto_approves(self):
        ui, _ = _make_ui(auto_approve=False)
        with patch.object(ui.console, "input", side_effect=KeyboardInterrupt):
            approved, feedback = ui.prompt_plan_review("Plan text")
        assert approved is True
        assert feedback == ""

    def test_shows_plan_in_artifact(self):
        ui, buf = _make_ui(auto_approve=True)
        ui.prompt_plan_review("The plan details here")
        output = buf.getvalue()
        assert "Build Plan" in output
        assert "The plan details here" in output

    def test_approve_case_insensitive(self):
        ui, _ = _make_ui(auto_approve=False)
        with patch.object(ui.console, "input", return_value="APPROVE"):
            approved, feedback = ui.prompt_plan_review("Plan")
        assert approved is True

    def test_approve_with_leading_trailing_whitespace(self):
        ui, _ = _make_ui(auto_approve=False)
        with patch.object(ui.console, "input", return_value="  approve  "):
            approved, feedback = ui.prompt_plan_review("Plan")
        assert approved is True
        assert feedback == ""


# ---------------------------------------------------------------------------
# unit_start / unit_kept / unit_reverted
# ---------------------------------------------------------------------------


class TestUnitLifecycle:
    def test_unit_start_prints_index_and_title(self):
        ui, buf = _make_ui()
        ui.unit_start("unit-1", "Add auth module", 1, 5)
        output = buf.getvalue()
        assert "(1/5)" in output
        assert "unit-1" in output
        assert "Add auth module" in output

    def test_unit_kept_prints_delta(self):
        ui, buf = _make_ui()
        ui.unit_kept("unit-2", "+30/-5")
        output = buf.getvalue()
        assert "kept" in output
        assert "+30/-5" in output

    def test_unit_reverted_prints_reason(self):
        ui, buf = _make_ui()
        ui.unit_reverted("unit-3", "tests failed")
        output = buf.getvalue()
        assert "reverted" in output
        assert "tests failed" in output

    def test_unit_start_boundary_index(self):
        ui, buf = _make_ui()
        ui.unit_start("u-0", "First unit", 0, 1)
        output = buf.getvalue()
        assert "(0/1)" in output


# ---------------------------------------------------------------------------
# coverage_warning()
# ---------------------------------------------------------------------------


class TestCoverageWarning:
    def test_coverage_warning_shows_modules(self):
        ui, buf = _make_ui()
        warnings = [
            {
                "module": "src/auth.py",
                "coverage_pct": 42,
                "recommendation": "Add integration tests",
            },
        ]
        ui.coverage_warning(warnings)
        output = buf.getvalue()
        assert "src/auth.py" in output
        assert "42" in output
        assert "Add integration tests" in output
        assert "Coverage Warning" in output

    def test_coverage_warning_multiple_modules(self):
        ui, buf = _make_ui()
        warnings = [
            {
                "module": "mod_a.py",
                "coverage_pct": 10,
                "recommendation": "rec A",
            },
            {
                "module": "mod_b.py",
                "coverage_pct": 25,
                "recommendation": "rec B",
            },
        ]
        ui.coverage_warning(warnings)
        output = buf.getvalue()
        assert "mod_a.py" in output
        assert "mod_b.py" in output
        assert "10" in output
        assert "25" in output

    def test_coverage_warning_empty_list(self):
        ui, buf = _make_ui()
        ui.coverage_warning([])
        output = buf.getvalue()
        # Should still render the panel heading
        assert "Coverage Warning" in output


# ---------------------------------------------------------------------------
# _safe_print() error handling
# ---------------------------------------------------------------------------


class TestSafePrint:
    def test_safe_print_handles_broken_pipe(self):
        ui, _ = _make_ui()
        ui.console = MagicMock()
        ui.console.print.side_effect = BrokenPipeError
        # Should NOT raise
        ui._safe_print("test message")

    def test_safe_print_handles_blocking_io_error(self):
        ui, _ = _make_ui()
        ui.console = MagicMock()
        ui.console.print.side_effect = BlockingIOError
        # Should NOT raise
        ui._safe_print("test message")

    def test_safe_print_handles_os_error(self):
        ui, _ = _make_ui()
        ui.console = MagicMock()
        ui.console.print.side_effect = OSError("pipe broken")
        # Should NOT raise
        ui._safe_print("test message")

    def test_safe_print_passes_args_and_kwargs(self):
        ui, _ = _make_ui()
        ui.console = MagicMock()
        ui._safe_print("hello", style="bold", end="\n")
        ui.console.print.assert_called_once_with("hello", style="bold", end="\n")

    def test_safe_print_does_not_suppress_other_exceptions(self):
        ui, _ = _make_ui()
        ui.console = MagicMock()
        ui.console.print.side_effect = ValueError("unexpected")
        with pytest.raises(ValueError, match="unexpected"):
            ui._safe_print("test")


# ---------------------------------------------------------------------------
# auto_approve when stdin is not a tty
# ---------------------------------------------------------------------------


class TestAutoApprove:
    def test_auto_approve_true_when_stdin_not_tty(self):
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            ui = UI(auto_approve=False)
        # Even though auto_approve=False was passed, stdin not being a tty
        # should force auto_approve to True
        assert ui.auto_approve is True

    def test_auto_approve_true_when_explicitly_set(self):
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            ui = UI(auto_approve=True)
        assert ui.auto_approve is True

    def test_auto_approve_false_when_tty_and_not_set(self):
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            ui = UI(auto_approve=False)
        assert ui.auto_approve is False


# ---------------------------------------------------------------------------
# error() / info()
# ---------------------------------------------------------------------------


class TestErrorInfo:
    def test_error_prints_message(self):
        ui, buf = _make_ui()
        ui.error("something broke")
        output = buf.getvalue()
        assert "Error" in output
        assert "something broke" in output

    def test_info_prints_message(self):
        ui, buf = _make_ui()
        ui.info("processing files")
        output = buf.getvalue()
        assert "processing files" in output


# ---------------------------------------------------------------------------
# pr_opened()
# ---------------------------------------------------------------------------


class TestPrOpened:
    def test_pr_opened_displays_url(self):
        ui, buf = _make_ui()
        ui.pr_opened("https://github.com/org/repo/pull/42")
        output = buf.getvalue()
        assert "https://github.com/org/repo/pull/42" in output
        assert "PR Opened" in output


# ---------------------------------------------------------------------------
# show_projects()
# ---------------------------------------------------------------------------


class TestShowProjects:
    def test_show_projects_empty_list(self):
        ui, buf = _make_ui()
        ui.show_projects([])
        output = buf.getvalue()
        assert "No feature sessions found" in output

    def test_show_projects_renders_table(self):
        ui, buf = _make_ui()
        projects = [
            {
                "project_id": "proj-1",
                "repo_path": "/home/user/repo",
                "feature_prompt": "Add auth",
                "status": "complete",
                "stages_completed": ["discover", "plan"],
                "created_at": "2025-01-15T10:30:00Z",
            },
        ]
        ui.show_projects(projects)
        output = buf.getvalue()
        assert "proj-1" in output
        assert "Add auth" in output
        assert "complete" in output

    def test_show_projects_truncates_long_fields(self):
        ui, buf = _make_ui()
        projects = [
            {
                "project_id": "proj-2",
                "repo_path": "x" * 100,
                "feature_prompt": "y" * 100,
                "status": "running",
                "stages_completed": [],
                "created_at": "2025-01-15T10:30:00.123456Z",
            },
        ]
        ui.show_projects(projects)
        output = buf.getvalue()
        assert "proj-2" in output
        # repo_path and feature_prompt are truncated to 40 chars
        assert "x" * 100 not in output
        assert "y" * 100 not in output


# ---------------------------------------------------------------------------
# stage_start handles pipe errors gracefully
# ---------------------------------------------------------------------------


class TestStageStartPipeError:
    def test_stage_start_handles_broken_pipe(self):
        ui, _ = _make_ui()
        ui.console = MagicMock()
        ui.console.rule.side_effect = BrokenPipeError
        # Should NOT raise
        ui.stage_start("discover")
        assert ui._current_stage == "discover"

    def test_stage_start_handles_blocking_io_error(self):
        ui, _ = _make_ui()
        ui.console = MagicMock()
        ui.console.rule.side_effect = BlockingIOError
        ui.stage_start("plan")
        assert ui._current_stage == "plan"

    def test_stage_start_handles_os_error(self):
        ui, _ = _make_ui()
        ui.console = MagicMock()
        ui.console.rule.side_effect = OSError
        ui.stage_start("execute")
        assert ui._current_stage == "execute"
