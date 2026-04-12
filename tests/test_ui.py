"""Tests for graft.ui."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from graft.ui import MAX_DISPLAY_CHARS, STAGE_LABELS, STAGE_ORDER, UI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ui(**kwargs) -> UI:
    """Create a UI with a string-backed Console so output is capturable."""
    ui = UI(**kwargs)
    ui.console = Console(file=None, force_terminal=False, no_color=True, width=120)
    return ui


def _capture_ui(**kwargs) -> tuple[UI, Console]:
    """Return a UI whose console writes to an internal buffer we can read."""
    from io import StringIO

    buf = StringIO()
    ui = UI(**kwargs)
    ui.console = Console(file=buf, force_terminal=False, no_color=True, width=120)
    return ui, buf


# ---------------------------------------------------------------------------
# Existing tests (preserved)
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
# Constants / Init
# ---------------------------------------------------------------------------


def test_stage_order_matches_labels():
    """STAGE_ORDER is derived from STAGE_LABELS keys."""
    assert STAGE_ORDER == list(STAGE_LABELS.keys())


def test_max_display_chars_value():
    """MAX_DISPLAY_CHARS has a sensible positive value."""
    assert MAX_DISPLAY_CHARS == 3000


def test_auto_approve_flag_true():
    """When auto_approve=True is passed it is set regardless of tty."""
    ui = UI(auto_approve=True)
    assert ui.auto_approve is True


def test_verbose_flag_true():
    """Verbose flag is stored."""
    ui = UI(verbose=True)
    assert ui.verbose is True


def test_auto_approve_when_not_tty():
    """auto_approve is True when stdin is not a tty (non-interactive)."""
    with patch("sys.stdin") as mock_stdin:
        mock_stdin.isatty.return_value = False
        ui = UI(auto_approve=False)
        assert ui.auto_approve is True


def test_auto_approve_false_when_tty():
    """auto_approve is False when stdin is a tty and not explicitly set."""
    with patch("sys.stdin") as mock_stdin:
        mock_stdin.isatty.return_value = True
        ui = UI(auto_approve=False)
        assert ui.auto_approve is False


# ---------------------------------------------------------------------------
# _safe_print
# ---------------------------------------------------------------------------


def test_safe_print_handles_broken_pipe():
    """_safe_print swallows BrokenPipeError."""
    ui = UI()
    ui.console = MagicMock()
    ui.console.print.side_effect = BrokenPipeError
    # Should not raise
    ui._safe_print("anything")


def test_safe_print_handles_blocking_io():
    """_safe_print swallows BlockingIOError."""
    ui = UI()
    ui.console = MagicMock()
    ui.console.print.side_effect = BlockingIOError
    ui._safe_print("anything")


def test_safe_print_handles_os_error():
    """_safe_print swallows generic OSError."""
    ui = UI()
    ui.console = MagicMock()
    ui.console.print.side_effect = OSError("write failed")
    ui._safe_print("anything")


def test_safe_print_propagates_other_exceptions():
    """_safe_print does not swallow unrelated exceptions."""
    ui = UI()
    ui.console = MagicMock()
    ui.console.print.side_effect = ValueError("unexpected")
    with pytest.raises(ValueError, match="unexpected"):
        ui._safe_print("anything")


# ---------------------------------------------------------------------------
# banner()
# ---------------------------------------------------------------------------


def test_banner_contains_repo_and_project(capsys):
    """banner() outputs repo path and project id."""
    ui, buf = _capture_ui()
    ui.banner("/home/user/repo", "feat_abc123", "Add dark mode support")
    output = buf.getvalue()
    assert "/home/user/repo" in output
    assert "feat_abc123" in output
    assert "Add dark mode support" in output
    assert "Graft" in output


def test_banner_truncates_long_prompt():
    """Feature prompts longer than 120 chars are truncated with ellipsis."""
    ui, buf = _capture_ui()
    long_prompt = "x" * 200
    ui.banner("/repo", "proj1", long_prompt)
    output = buf.getvalue()
    # The first 120 chars should be present
    assert "x" * 120 in output
    # The full 200-char string should NOT appear (it was truncated)
    assert "x" * 200 not in output
    # Ellipsis character should appear
    assert "…" in output


def test_banner_no_truncation_for_short_prompt():
    """Feature prompts <= 120 chars are not truncated."""
    ui, buf = _capture_ui()
    short_prompt = "a" * 120
    ui.banner("/repo", "proj1", short_prompt)
    output = buf.getvalue()
    assert "a" * 120 in output
    assert "…" not in output


# ---------------------------------------------------------------------------
# stage_start()
# ---------------------------------------------------------------------------


def test_stage_start_sets_current_stage():
    """stage_start records the current stage."""
    ui = UI()
    ui.console = MagicMock()
    ui.stage_start("discover")
    assert ui._current_stage == "discover"


def test_stage_start_uses_label():
    """stage_start displays the human-readable label."""
    ui = UI()
    ui.console = MagicMock()
    ui.stage_start("discover")
    call_args = ui.console.rule.call_args
    assert "Discover" in str(call_args)


def test_stage_start_unknown_stage_uses_raw_name():
    """stage_start falls back to raw stage name if not in STAGE_LABELS."""
    ui = UI()
    ui.console = MagicMock()
    ui.stage_start("custom_stage")
    assert ui._current_stage == "custom_stage"
    call_args = ui.console.rule.call_args
    assert "custom_stage" in str(call_args)


def test_stage_start_handles_broken_pipe():
    """stage_start swallows BrokenPipeError from console.rule."""
    ui = UI()
    ui.console = MagicMock()
    ui.console.rule.side_effect = BrokenPipeError
    # Should not raise
    ui.stage_start("discover")
    assert ui._current_stage == "discover"


def test_stage_start_handles_blocking_io():
    """stage_start swallows BlockingIOError from console.rule."""
    ui = UI()
    ui.console = MagicMock()
    ui.console.rule.side_effect = BlockingIOError
    ui.stage_start("discover")


def test_stage_start_handles_os_error():
    """stage_start swallows OSError from console.rule."""
    ui = UI()
    ui.console = MagicMock()
    ui.console.rule.side_effect = OSError
    ui.stage_start("discover")


# ---------------------------------------------------------------------------
# stage_done()
# ---------------------------------------------------------------------------


def test_stage_done_outputs_complete_message():
    """stage_done prints a completion message with the stage label."""
    ui, buf = _capture_ui()
    ui.stage_done("discover")
    output = buf.getvalue()
    assert "Discover" in output
    assert "complete" in output


def test_stage_done_unknown_stage():
    """stage_done uses raw stage name when not in STAGE_LABELS."""
    ui, buf = _capture_ui()
    ui.stage_done("custom_stage")
    output = buf.getvalue()
    assert "custom_stage" in output
    assert "complete" in output


# ---------------------------------------------------------------------------
# stage_log()
# ---------------------------------------------------------------------------


def test_stage_log_verbose_outputs_message():
    """stage_log prints when verbose=True."""
    ui, buf = _capture_ui(verbose=True)
    ui.stage_log("discover", "found 3 files")
    output = buf.getvalue()
    assert "found 3 files" in output
    assert "Discover" in output


def test_stage_log_not_verbose_suppresses():
    """stage_log is silent when verbose=False."""
    ui, buf = _capture_ui(verbose=False)
    ui.stage_log("discover", "should not appear")
    output = buf.getvalue()
    assert "should not appear" not in output


def test_stage_log_unknown_stage_verbose():
    """stage_log with unknown stage uses raw name when verbose."""
    ui, buf = _capture_ui(verbose=True)
    ui.stage_log("custom", "msg")
    output = buf.getvalue()
    assert "custom" in output
    assert "msg" in output


# ---------------------------------------------------------------------------
# show_artifact()
# ---------------------------------------------------------------------------


def test_show_artifact_displays_title_and_content():
    """show_artifact renders a panel with title and content."""
    ui, buf = _capture_ui()
    ui.show_artifact("My Artifact", "Hello world")
    output = buf.getvalue()
    assert "My Artifact" in output
    assert "Hello world" in output


def test_show_artifact_truncates_long_content():
    """Content longer than MAX_DISPLAY_CHARS is truncated."""
    ui, buf = _capture_ui()
    long_content = "z" * (MAX_DISPLAY_CHARS + 500)
    ui.show_artifact("Big", long_content)
    output = buf.getvalue()
    assert "truncated" in output
    # Full content should NOT appear
    assert "z" * (MAX_DISPLAY_CHARS + 500) not in output


def test_show_artifact_no_truncation_short_content():
    """Content within MAX_DISPLAY_CHARS is not truncated."""
    ui, buf = _capture_ui()
    content = "a" * (MAX_DISPLAY_CHARS - 10)
    ui.show_artifact("Small", content)
    output = buf.getvalue()
    assert "truncated" not in output


def test_show_artifact_exactly_max_chars():
    """Content exactly at MAX_DISPLAY_CHARS is not truncated."""
    ui, buf = _capture_ui()
    content = "b" * MAX_DISPLAY_CHARS
    ui.show_artifact("Exact", content)
    output = buf.getvalue()
    assert "truncated" not in output


# ---------------------------------------------------------------------------
# grill_question()
# ---------------------------------------------------------------------------


def test_grill_question_returns_user_input():
    """grill_question returns user response when non-empty."""
    ui, buf = _capture_ui()
    ui.console.input = MagicMock(return_value="My custom answer")
    result = ui.grill_question("What color?", "blue", "aesthetics", 1)
    assert result == "My custom answer"


def test_grill_question_returns_recommended_on_empty():
    """grill_question returns recommended answer on empty input."""
    ui, buf = _capture_ui()
    ui.console.input = MagicMock(return_value="")
    result = ui.grill_question("What color?", "blue", "aesthetics", 1)
    assert result == "blue"


def test_grill_question_strips_whitespace():
    """grill_question strips leading/trailing whitespace."""
    ui, buf = _capture_ui()
    ui.console.input = MagicMock(return_value="  spaced answer  ")
    result = ui.grill_question("Q?", "default", "cat", 1)
    assert result == "spaced answer"


def test_grill_question_whitespace_only_returns_recommended():
    """grill_question returns recommended when only whitespace entered."""
    ui, buf = _capture_ui()
    ui.console.input = MagicMock(return_value="   ")
    result = ui.grill_question("Q?", "default", "cat", 1)
    assert result == "default"


def test_grill_question_eof_returns_recommended():
    """grill_question returns recommended on EOFError."""
    ui, buf = _capture_ui()
    ui.console.input = MagicMock(side_effect=EOFError)
    result = ui.grill_question("Q?", "fallback", "cat", 1)
    assert result == "fallback"


def test_grill_question_keyboard_interrupt_returns_recommended():
    """grill_question returns recommended on KeyboardInterrupt."""
    ui, buf = _capture_ui()
    ui.console.input = MagicMock(side_effect=KeyboardInterrupt)
    result = ui.grill_question("Q?", "fallback", "cat", 1)
    assert result == "fallback"


def test_grill_question_displays_question_and_metadata():
    """grill_question displays question text, category, and number."""
    ui, buf = _capture_ui()
    ui.console.input = MagicMock(return_value="ans")
    ui.grill_question("What framework?", "React", "tech", 3)
    output = buf.getvalue()
    assert "What framework?" in output
    assert "tech" in output
    assert "React" in output
    assert "Question 3" in output


# ---------------------------------------------------------------------------
# prompt_plan_review()
# ---------------------------------------------------------------------------


def test_prompt_plan_review_auto_approve():
    """prompt_plan_review returns (True, '') when auto_approve is set."""
    ui, buf = _capture_ui(auto_approve=True)
    approved, feedback = ui.prompt_plan_review("Here is the plan")
    assert approved is True
    assert feedback == ""
    output = buf.getvalue()
    assert "auto-approved" in output


def test_prompt_plan_review_approve_responses():
    """prompt_plan_review accepts various approval words."""
    for word in ("approve", "yes", "y", "lgtm", ""):
        ui, buf = _capture_ui()
        # Force auto_approve off
        ui.auto_approve = False
        ui.console.input = MagicMock(return_value=word)
        approved, feedback = ui.prompt_plan_review("Plan content")
        assert approved is True, f"Expected approval for input '{word}'"
        assert feedback == ""


def test_prompt_plan_review_approve_case_insensitive():
    """prompt_plan_review approval words are case-insensitive."""
    for word in ("APPROVE", "Yes", "Y", "LGTM", "Approve"):
        ui, buf = _capture_ui()
        ui.auto_approve = False
        ui.console.input = MagicMock(return_value=word)
        approved, feedback = ui.prompt_plan_review("Plan")
        assert approved is True, f"Expected approval for input '{word}'"


def test_prompt_plan_review_reject_with_feedback():
    """prompt_plan_review returns (False, feedback) on non-approval input."""
    ui, buf = _capture_ui()
    ui.auto_approve = False
    ui.console.input = MagicMock(return_value="Please add more tests")
    approved, feedback = ui.prompt_plan_review("Plan content")
    assert approved is False
    assert feedback == "Please add more tests"


def test_prompt_plan_review_eof_auto_approves():
    """prompt_plan_review auto-approves on EOFError."""
    ui, buf = _capture_ui()
    ui.auto_approve = False
    ui.console.input = MagicMock(side_effect=EOFError)
    approved, feedback = ui.prompt_plan_review("Plan")
    assert approved is True
    assert feedback == ""
    output = buf.getvalue()
    assert "auto-approving" in output


def test_prompt_plan_review_keyboard_interrupt_auto_approves():
    """prompt_plan_review auto-approves on KeyboardInterrupt."""
    ui, buf = _capture_ui()
    ui.auto_approve = False
    ui.console.input = MagicMock(side_effect=KeyboardInterrupt)
    approved, feedback = ui.prompt_plan_review("Plan")
    assert approved is True
    assert feedback == ""


def test_prompt_plan_review_shows_plan_artifact():
    """prompt_plan_review displays the plan via show_artifact."""
    ui, buf = _capture_ui(auto_approve=True)
    ui.prompt_plan_review("Step 1: do stuff\nStep 2: do more")
    output = buf.getvalue()
    assert "Build Plan" in output
    assert "Step 1" in output


# ---------------------------------------------------------------------------
# unit_start()
# ---------------------------------------------------------------------------


def test_unit_start_displays_progress():
    """unit_start shows unit id, title, and progress indicator."""
    ui, buf = _capture_ui()
    ui.unit_start("unit_01", "Add login form", 1, 5)
    output = buf.getvalue()
    assert "unit_01" in output
    assert "Add login form" in output
    assert "(1/5)" in output


def test_unit_start_different_indices():
    """unit_start correctly displays various index/total combinations."""
    ui, buf = _capture_ui()
    ui.unit_start("u2", "Second unit", 2, 10)
    output = buf.getvalue()
    assert "(2/10)" in output


# ---------------------------------------------------------------------------
# unit_kept()
# ---------------------------------------------------------------------------


def test_unit_kept_shows_success():
    """unit_kept shows kept status and delta."""
    ui, buf = _capture_ui()
    ui.unit_kept("unit_01", "+50 -10")
    output = buf.getvalue()
    assert "kept" in output
    assert "+50 -10" in output


# ---------------------------------------------------------------------------
# unit_reverted()
# ---------------------------------------------------------------------------


def test_unit_reverted_shows_failure():
    """unit_reverted shows reverted status and reason."""
    ui, buf = _capture_ui()
    ui.unit_reverted("unit_01", "tests failed")
    output = buf.getvalue()
    assert "reverted" in output
    assert "tests failed" in output


# ---------------------------------------------------------------------------
# pr_opened()
# ---------------------------------------------------------------------------


def test_pr_opened_shows_url():
    """pr_opened displays the PR URL in a panel."""
    ui, buf = _capture_ui()
    ui.pr_opened("https://github.com/org/repo/pull/42")
    output = buf.getvalue()
    assert "https://github.com/org/repo/pull/42" in output
    assert "PR Opened" in output


# ---------------------------------------------------------------------------
# coverage_warning()
# ---------------------------------------------------------------------------


def test_coverage_warning_displays_warnings():
    """coverage_warning displays module names, percentages, and recommendations."""
    ui, buf = _capture_ui()
    warnings = [
        {
            "module": "graft.executor",
            "coverage_pct": 45,
            "recommendation": "Add integration tests",
        },
        {
            "module": "graft.planner",
            "coverage_pct": 30,
            "recommendation": "Mock LLM calls",
        },
    ]
    ui.coverage_warning(warnings)
    output = buf.getvalue()
    assert "graft.executor" in output
    assert "45%" in output
    assert "Add integration tests" in output
    assert "graft.planner" in output
    assert "30%" in output
    assert "Mock LLM calls" in output
    assert "Coverage Warning" in output


def test_coverage_warning_single_entry():
    """coverage_warning works with a single warning."""
    ui, buf = _capture_ui()
    warnings = [
        {
            "module": "graft.ui",
            "coverage_pct": 31,
            "recommendation": "Write more tests",
        },
    ]
    ui.coverage_warning(warnings)
    output = buf.getvalue()
    assert "graft.ui" in output
    assert "31%" in output


def test_coverage_warning_empty_list():
    """coverage_warning with empty list still renders panel header."""
    ui, buf = _capture_ui()
    ui.coverage_warning([])
    output = buf.getvalue()
    assert "Coverage Warning" in output


# ---------------------------------------------------------------------------
# error()
# ---------------------------------------------------------------------------


def test_error_displays_message():
    """error() outputs the error message."""
    ui, buf = _capture_ui()
    ui.error("Something went wrong")
    output = buf.getvalue()
    assert "Error:" in output
    assert "Something went wrong" in output


def test_error_special_characters():
    """error() handles special characters in message."""
    ui, buf = _capture_ui()
    ui.error("File 'foo.py' not found <dir>")
    output = buf.getvalue()
    assert "foo.py" in output


# ---------------------------------------------------------------------------
# info()
# ---------------------------------------------------------------------------


def test_info_displays_message():
    """info() outputs the info message."""
    ui, buf = _capture_ui()
    ui.info("Processing files...")
    output = buf.getvalue()
    assert "Processing files..." in output


# ---------------------------------------------------------------------------
# show_projects()
# ---------------------------------------------------------------------------


def test_show_projects_empty():
    """show_projects with empty list shows 'no sessions' message."""
    ui, buf = _capture_ui()
    ui.show_projects([])
    output = buf.getvalue()
    assert "No feature sessions found" in output


def test_show_projects_with_data():
    """show_projects renders a table with project data."""
    ui, buf = _capture_ui()
    projects = [
        {
            "project_id": "feat_abc",
            "repo_path": "/home/user/repo",
            "feature_prompt": "Add dark mode",
            "status": "in_progress",
            "stages_completed": ["discover", "research"],
            "created_at": "2025-01-15T10:30:00Z",
        },
    ]
    ui.show_projects(projects)
    output = buf.getvalue()
    assert "feat_abc" in output
    assert "Add dark mode" in output
    assert "in_progress" in output
    assert "discover" in output
    assert "research" in output
    assert "2025-01-15T10:30:00" in output


def test_show_projects_multiple():
    """show_projects renders multiple rows."""
    ui, buf = _capture_ui()
    projects = [
        {
            "project_id": "feat_1",
            "repo_path": "/repo1",
            "feature_prompt": "Feature one",
            "status": "done",
            "stages_completed": ["discover"],
            "created_at": "2025-01-01T00:00:00Z",
        },
        {
            "project_id": "feat_2",
            "repo_path": "/repo2",
            "feature_prompt": "Feature two",
            "status": "in_progress",
            "stages_completed": [],
            "created_at": "2025-02-01T00:00:00Z",
        },
    ]
    ui.show_projects(projects)
    output = buf.getvalue()
    assert "feat_1" in output
    assert "feat_2" in output
    assert "Feature one" in output
    assert "Feature two" in output


def test_show_projects_missing_optional_fields():
    """show_projects handles projects with missing optional fields."""
    ui, buf = _capture_ui()
    projects = [
        {
            "project_id": "feat_minimal",
        },
    ]
    ui.show_projects(projects)
    output = buf.getvalue()
    assert "feat_minimal" in output


def test_show_projects_long_values_truncated():
    """show_projects truncates long repo_path and feature_prompt at 40 chars."""
    ui, buf = _capture_ui()
    projects = [
        {
            "project_id": "feat_long",
            "repo_path": "A" * 80,
            "feature_prompt": "B" * 80,
            "status": "done",
            "stages_completed": [],
            "created_at": "2025-01-01T00:00:00.123456Z",
        },
    ]
    ui.show_projects(projects)
    output = buf.getvalue()
    # The 80-char strings should be truncated to 40
    assert "A" * 80 not in output
    assert "B" * 80 not in output
    # created_at truncated to first 19 chars
    assert "2025-01-01T00:00:00" in output


def test_show_projects_empty_stages_shows_dash():
    """show_projects shows em-dash for projects with no completed stages."""
    ui, buf = _capture_ui()
    projects = [
        {
            "project_id": "feat_new",
            "status": "in_progress",
            "stages_completed": [],
            "created_at": "2025-01-01T00:00:00Z",
        },
    ]
    ui.show_projects(projects)
    output = buf.getvalue()
    assert "—" in output


# ---------------------------------------------------------------------------
# show_projects uses console.print directly (not _safe_print)
# ---------------------------------------------------------------------------


def test_show_projects_uses_console_print_directly():
    """show_projects calls console.print, not _safe_print, so errors propagate."""
    ui = UI()
    ui.console = MagicMock()
    ui.show_projects([])
    ui.console.print.assert_called_once()
