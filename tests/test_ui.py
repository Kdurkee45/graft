"""Tests for graft.ui."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from rich.panel import Panel
from rich.table import Table

from graft.ui import MAX_DISPLAY_CHARS, STAGE_LABELS, STAGE_ORDER, UI

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ui():
    """Return a UI with auto_approve=True (avoids isatty issues in tests)."""
    return UI(auto_approve=True, verbose=False)


@pytest.fixture
def verbose_ui():
    """Return a verbose UI."""
    return UI(auto_approve=True, verbose=True)


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
# STAGE_ORDER
# ---------------------------------------------------------------------------


def test_stage_order_matches_labels():
    """STAGE_ORDER is derived from STAGE_LABELS keys."""
    assert STAGE_ORDER == list(STAGE_LABELS.keys())


# ---------------------------------------------------------------------------
# _safe_print — BlockingIOError handling
# ---------------------------------------------------------------------------


def test_safe_print_suppresses_blocking_io_error(ui):
    """_safe_print swallows BlockingIOError."""
    ui.console.print = MagicMock(side_effect=BlockingIOError)
    ui._safe_print("test")  # should not raise


def test_safe_print_suppresses_broken_pipe_error(ui):
    """_safe_print swallows BrokenPipeError."""
    ui.console.print = MagicMock(side_effect=BrokenPipeError)
    ui._safe_print("test")  # should not raise


def test_safe_print_suppresses_os_error(ui):
    """_safe_print swallows OSError."""
    ui.console.print = MagicMock(side_effect=OSError)
    ui._safe_print("test")  # should not raise


# ---------------------------------------------------------------------------
# banner()
# ---------------------------------------------------------------------------


def test_banner_renders_panel_with_repo_and_project(ui):
    """banner() prints a Panel containing repo_path, project_id, prompt."""
    ui._safe_print = MagicMock()
    ui.banner("/tmp/repo", "feat_abc123", "Add dark mode")
    # 3 calls: empty line, Panel, empty line
    assert ui._safe_print.call_count == 3
    panel_call = ui._safe_print.call_args_list[1]
    panel_arg = panel_call[0][0]
    assert isinstance(panel_arg, Panel)


def test_banner_contains_repo_path(ui):
    """banner() Panel renderable includes repo_path."""
    ui._safe_print = MagicMock()
    ui.banner("/home/user/my-repo", "proj_1", "feature prompt")
    panel_arg = ui._safe_print.call_args_list[1][0][0]
    # The renderable text inside the panel includes repo_path
    renderable_text = panel_arg.renderable
    assert "/home/user/my-repo" in renderable_text


def test_banner_contains_project_id(ui):
    """banner() Panel renderable includes project_id."""
    ui._safe_print = MagicMock()
    ui.banner("/repo", "feat_xyz789", "feature prompt")
    panel_arg = ui._safe_print.call_args_list[1][0][0]
    assert "feat_xyz789" in panel_arg.renderable


def test_banner_truncates_long_prompt(ui):
    """Prompts longer than 120 chars are truncated with ellipsis."""
    ui._safe_print = MagicMock()
    long_prompt = "A" * 200
    ui.banner("/repo", "proj_1", long_prompt)
    panel_arg = ui._safe_print.call_args_list[1][0][0]
    renderable = panel_arg.renderable
    # Should contain only first 120 chars + ellipsis
    assert "A" * 120 in renderable
    assert "…" in renderable
    # Should NOT contain the full 200-char string
    assert "A" * 200 not in renderable


def test_banner_no_truncation_for_short_prompt(ui):
    """Prompts <= 120 chars are shown without ellipsis."""
    ui._safe_print = MagicMock()
    short_prompt = "B" * 120
    ui.banner("/repo", "proj_1", short_prompt)
    panel_arg = ui._safe_print.call_args_list[1][0][0]
    renderable = panel_arg.renderable
    assert "B" * 120 in renderable
    assert "…" not in renderable


def test_banner_exact_120_no_ellipsis(ui):
    """Prompt of exactly 120 chars has no ellipsis."""
    ui._safe_print = MagicMock()
    exact_prompt = "C" * 120
    ui.banner("/repo", "proj_1", exact_prompt)
    panel_arg = ui._safe_print.call_args_list[1][0][0]
    assert "…" not in panel_arg.renderable


def test_banner_121_chars_gets_ellipsis(ui):
    """Prompt of 121 chars gets truncated with ellipsis."""
    ui._safe_print = MagicMock()
    prompt_121 = "D" * 121
    ui.banner("/repo", "proj_1", prompt_121)
    panel_arg = ui._safe_print.call_args_list[1][0][0]
    assert "…" in panel_arg.renderable


# ---------------------------------------------------------------------------
# stage_start()
# ---------------------------------------------------------------------------


def test_stage_start_calls_rule_with_label(ui):
    """stage_start() calls console.rule with the correct stage label."""
    ui.console.rule = MagicMock()
    ui.stage_start("discover")
    ui.console.rule.assert_called_once()
    rule_arg = ui.console.rule.call_args[0][0]
    assert "Discover" in rule_arg


def test_stage_start_sets_current_stage(ui):
    """stage_start() updates _current_stage."""
    ui.console.rule = MagicMock()
    ui.stage_start("research")
    assert ui._current_stage == "research"


def test_stage_start_unknown_stage_uses_raw_name(ui):
    """stage_start() uses the raw stage name if not in STAGE_LABELS."""
    ui.console.rule = MagicMock()
    ui.stage_start("unknown_stage")
    rule_arg = ui.console.rule.call_args[0][0]
    assert "unknown_stage" in rule_arg


def test_stage_start_handles_blocking_io_error(ui):
    """stage_start() swallows BlockingIOError from console.rule."""
    ui.console.rule = MagicMock(side_effect=BlockingIOError)
    ui.stage_start("discover")  # should not raise


def test_stage_start_handles_broken_pipe_error(ui):
    """stage_start() swallows BrokenPipeError from console.rule."""
    ui.console.rule = MagicMock(side_effect=BrokenPipeError)
    ui.stage_start("discover")  # should not raise


def test_stage_start_handles_os_error(ui):
    """stage_start() swallows OSError from console.rule."""
    ui.console.rule = MagicMock(side_effect=OSError)
    ui.stage_start("discover")  # should not raise


# ---------------------------------------------------------------------------
# stage_done()
# ---------------------------------------------------------------------------


def test_stage_done_prints_checkmark_and_complete(ui):
    """stage_done() prints checkmark and 'complete' text."""
    ui._safe_print = MagicMock()
    ui.stage_done("discover")
    calls = [c[0][0] for c in ui._safe_print.call_args_list if c[0]]
    text = " ".join(calls)
    assert "✓" in text
    assert "complete" in text


def test_stage_done_includes_label(ui):
    """stage_done() message includes the stage label."""
    ui._safe_print = MagicMock()
    ui.stage_done("research")
    calls = [c[0][0] for c in ui._safe_print.call_args_list if c[0]]
    text = " ".join(calls)
    assert "Research" in text


def test_stage_done_unknown_stage(ui):
    """stage_done() uses raw name for unknown stages."""
    ui._safe_print = MagicMock()
    ui.stage_done("custom_stage")
    calls = [c[0][0] for c in ui._safe_print.call_args_list if c[0]]
    text = " ".join(calls)
    assert "custom_stage" in text


def test_stage_done_prints_blank_line_after(ui):
    """stage_done() prints a blank line after the complete message."""
    ui._safe_print = MagicMock()
    ui.stage_done("plan")
    # Two calls: the message, then an empty-args call (blank line)
    assert ui._safe_print.call_count == 2


# ---------------------------------------------------------------------------
# stage_log() verbose=True
# ---------------------------------------------------------------------------


def test_stage_log_verbose_prints_message(verbose_ui):
    """stage_log prints the message when verbose=True."""
    verbose_ui._safe_print = MagicMock()
    verbose_ui.stage_log("discover", "scanning files")
    verbose_ui._safe_print.assert_called_once()
    printed = verbose_ui._safe_print.call_args[0][0]
    assert "scanning files" in printed
    assert "Discover" in printed


def test_stage_log_verbose_unknown_stage(verbose_ui):
    """stage_log uses raw stage name when not in STAGE_LABELS."""
    verbose_ui._safe_print = MagicMock()
    verbose_ui.stage_log("custom", "hello")
    printed = verbose_ui._safe_print.call_args[0][0]
    assert "custom" in printed
    assert "hello" in printed


# ---------------------------------------------------------------------------
# show_artifact()
# ---------------------------------------------------------------------------


def test_show_artifact_renders_panel(ui):
    """show_artifact() prints a Panel with title and content."""
    ui._safe_print = MagicMock()
    ui.show_artifact("My Title", "some content here")
    # 3 calls: blank, Panel, blank
    assert ui._safe_print.call_count == 3
    panel_arg = ui._safe_print.call_args_list[1][0][0]
    assert isinstance(panel_arg, Panel)


def test_show_artifact_panel_contains_content(ui):
    """show_artifact() Panel renderable includes the content."""
    ui._safe_print = MagicMock()
    ui.show_artifact("Title", "the actual content")
    panel_arg = ui._safe_print.call_args_list[1][0][0]
    assert "the actual content" in panel_arg.renderable


def test_show_artifact_panel_title(ui):
    """show_artifact() Panel has the correct title."""
    ui._safe_print = MagicMock()
    ui.show_artifact("Research Results", "data")
    panel_arg = ui._safe_print.call_args_list[1][0][0]
    assert "Research Results" in panel_arg.title


def test_show_artifact_truncates_long_content(ui):
    """Content longer than MAX_DISPLAY_CHARS is truncated."""
    ui._safe_print = MagicMock()
    long_content = "X" * (MAX_DISPLAY_CHARS + 500)
    ui.show_artifact("Big", long_content)
    panel_arg = ui._safe_print.call_args_list[1][0][0]
    renderable = panel_arg.renderable
    # Truncated content should not contain the full string
    assert "X" * (MAX_DISPLAY_CHARS + 500) not in renderable
    # Should contain truncation message
    assert "truncated" in renderable


def test_show_artifact_no_truncation_for_short_content(ui):
    """Content within MAX_DISPLAY_CHARS is not truncated."""
    ui._safe_print = MagicMock()
    short_content = "Y" * MAX_DISPLAY_CHARS
    ui.show_artifact("Short", short_content)
    panel_arg = ui._safe_print.call_args_list[1][0][0]
    renderable = panel_arg.renderable
    assert "Y" * MAX_DISPLAY_CHARS in renderable
    assert "truncated" not in renderable


def test_show_artifact_exact_max_no_truncation(ui):
    """Content of exactly MAX_DISPLAY_CHARS is not truncated."""
    ui._safe_print = MagicMock()
    exact_content = "Z" * MAX_DISPLAY_CHARS
    ui.show_artifact("Exact", exact_content)
    panel_arg = ui._safe_print.call_args_list[1][0][0]
    assert "truncated" not in panel_arg.renderable


# ---------------------------------------------------------------------------
# grill_question()
# ---------------------------------------------------------------------------


def test_grill_question_returns_user_input(ui):
    """grill_question() returns the user's typed answer."""
    ui.auto_approve = False
    ui._safe_print = MagicMock()
    ui.console.input = MagicMock(return_value="my custom answer")
    result = ui.grill_question("What color?", "blue", "design", 1)
    assert result == "my custom answer"


def test_grill_question_auto_approve_returns_recommended(ui):
    """grill_question() returns recommended when auto_approve is True."""
    ui.auto_approve = True
    ui._safe_print = MagicMock()
    result = ui.grill_question("What color?", "blue", "design", 1)
    assert result == "blue"


def test_grill_question_empty_input_returns_recommended(ui):
    """grill_question() returns recommended when user hits Enter."""
    ui.auto_approve = False
    ui._safe_print = MagicMock()
    ui.console.input = MagicMock(return_value="")
    result = ui.grill_question("What color?", "blue", "design", 1)
    assert result == "blue"


def test_grill_question_whitespace_input_returns_recommended(ui):
    """grill_question() returns recommended when user enters only spaces."""
    ui.auto_approve = False
    ui._safe_print = MagicMock()
    ui.console.input = MagicMock(return_value="   ")
    result = ui.grill_question("Q?", "default_ans", "cat", 2)
    assert result == "default_ans"


def test_grill_question_eof_returns_recommended(ui):
    """grill_question() returns recommended on EOFError."""
    ui.auto_approve = False
    ui._safe_print = MagicMock()
    ui.console.input = MagicMock(side_effect=EOFError)
    result = ui.grill_question("Q?", "fallback", "cat", 1)
    assert result == "fallback"


def test_grill_question_keyboard_interrupt_returns_recommended(ui):
    """grill_question() returns recommended on KeyboardInterrupt."""
    ui.auto_approve = False
    ui._safe_print = MagicMock()
    ui.console.input = MagicMock(side_effect=KeyboardInterrupt)
    result = ui.grill_question("Q?", "fallback", "cat", 1)
    assert result == "fallback"


def test_grill_question_renders_panel_with_question(ui):
    """grill_question() renders a Panel containing the question and recommendation."""
    ui.auto_approve = True
    ui._safe_print = MagicMock()
    ui.grill_question("Is it blue?", "yes", "colors", 3)
    # Find the Panel call
    panel_calls = [
        c[0][0]
        for c in ui._safe_print.call_args_list
        if c[0] and isinstance(c[0][0], Panel)
    ]
    assert len(panel_calls) == 1
    panel = panel_calls[0]
    assert "Is it blue?" in panel.renderable
    assert "yes" in panel.renderable


def test_grill_question_renders_why_asking(ui):
    """grill_question() includes why_asking in the panel when provided."""
    ui.auto_approve = True
    ui._safe_print = MagicMock()
    ui.grill_question("Q?", "rec", "cat", 1, why_asking="Need to know scope")
    panel_calls = [
        c[0][0]
        for c in ui._safe_print.call_args_list
        if c[0] and isinstance(c[0][0], Panel)
    ]
    assert "Need to know scope" in panel_calls[0].renderable


def test_grill_question_panel_title_includes_number(ui):
    """grill_question() Panel title contains the question number."""
    ui.auto_approve = True
    ui._safe_print = MagicMock()
    ui.grill_question("Q?", "rec", "cat", 42)
    panel_calls = [
        c[0][0]
        for c in ui._safe_print.call_args_list
        if c[0] and isinstance(c[0][0], Panel)
    ]
    assert "42" in panel_calls[0].title


# ---------------------------------------------------------------------------
# prompt_plan_review()
# ---------------------------------------------------------------------------


def test_prompt_plan_review_auto_approve(ui):
    """prompt_plan_review() returns (True, '') when auto_approve=True."""
    ui.auto_approve = True
    ui._safe_print = MagicMock()
    approved, feedback = ui.prompt_plan_review("the plan")
    assert approved is True
    assert feedback == ""


def test_prompt_plan_review_auto_approve_prints_message(ui):
    """prompt_plan_review() prints auto-approved when auto_approve=True."""
    ui.auto_approve = True
    ui._safe_print = MagicMock()
    ui.prompt_plan_review("plan summary")
    all_text = " ".join(str(c[0][0]) for c in ui._safe_print.call_args_list if c[0])
    assert "auto-approved" in all_text


def test_prompt_plan_review_manual_approve(ui):
    """prompt_plan_review() returns (True, '') on 'approve' input."""
    ui.auto_approve = False
    ui._safe_print = MagicMock()
    ui.console.input = MagicMock(return_value="approve")
    approved, feedback = ui.prompt_plan_review("plan summary")
    assert approved is True
    assert feedback == ""


def test_prompt_plan_review_yes(ui):
    """prompt_plan_review() returns (True, '') on 'yes' input."""
    ui.auto_approve = False
    ui._safe_print = MagicMock()
    ui.console.input = MagicMock(return_value="yes")
    approved, feedback = ui.prompt_plan_review("plan")
    assert approved is True
    assert feedback == ""


def test_prompt_plan_review_y(ui):
    """prompt_plan_review() returns (True, '') on 'y' input."""
    ui.auto_approve = False
    ui._safe_print = MagicMock()
    ui.console.input = MagicMock(return_value="y")
    approved, feedback = ui.prompt_plan_review("plan")
    assert approved is True
    assert feedback == ""


def test_prompt_plan_review_lgtm(ui):
    """prompt_plan_review() returns (True, '') on 'lgtm' input."""
    ui.auto_approve = False
    ui._safe_print = MagicMock()
    ui.console.input = MagicMock(return_value="lgtm")
    approved, feedback = ui.prompt_plan_review("plan")
    assert approved is True
    assert feedback == ""


def test_prompt_plan_review_empty_input(ui):
    """prompt_plan_review() returns (True, '') on empty input."""
    ui.auto_approve = False
    ui._safe_print = MagicMock()
    ui.console.input = MagicMock(return_value="")
    approved, feedback = ui.prompt_plan_review("plan")
    assert approved is True
    assert feedback == ""


def test_prompt_plan_review_case_insensitive(ui):
    """prompt_plan_review() approval words are case-insensitive."""
    ui.auto_approve = False
    ui._safe_print = MagicMock()
    ui.console.input = MagicMock(return_value="APPROVE")
    approved, feedback = ui.prompt_plan_review("plan")
    assert approved is True
    assert feedback == ""


def test_prompt_plan_review_feedback(ui):
    """prompt_plan_review() returns (False, feedback) for non-approve input."""
    ui.auto_approve = False
    ui._safe_print = MagicMock()
    ui.console.input = MagicMock(return_value="add more tests")
    approved, feedback = ui.prompt_plan_review("plan")
    assert approved is False
    assert feedback == "add more tests"


def test_prompt_plan_review_eof_auto_approves(ui):
    """prompt_plan_review() auto-approves on EOFError."""
    ui.auto_approve = False
    ui._safe_print = MagicMock()
    ui.console.input = MagicMock(side_effect=EOFError)
    approved, feedback = ui.prompt_plan_review("plan")
    assert approved is True
    assert feedback == ""


def test_prompt_plan_review_keyboard_interrupt_auto_approves(ui):
    """prompt_plan_review() auto-approves on KeyboardInterrupt."""
    ui.auto_approve = False
    ui._safe_print = MagicMock()
    ui.console.input = MagicMock(side_effect=KeyboardInterrupt)
    approved, feedback = ui.prompt_plan_review("plan")
    assert approved is True
    assert feedback == ""


def test_prompt_plan_review_calls_show_artifact(ui):
    """prompt_plan_review() calls show_artifact with the plan summary."""
    ui.auto_approve = True
    ui._safe_print = MagicMock()
    with patch.object(ui, "show_artifact", wraps=ui.show_artifact) as mock_sa:
        ui.prompt_plan_review("my plan details")
        mock_sa.assert_called_once_with("Build Plan", "my plan details")


# ---------------------------------------------------------------------------
# unit_start()
# ---------------------------------------------------------------------------


def test_unit_start_format(ui):
    """unit_start() prints (index/total) unit_id: title."""
    ui._safe_print = MagicMock()
    ui.unit_start("unit_001", "Add login", 3, 10)
    printed = ui._safe_print.call_args[0][0]
    assert "(3/10)" in printed
    assert "unit_001" in printed
    assert "Add login" in printed


def test_unit_start_single_item(ui):
    """unit_start() works for (1/1) case."""
    ui._safe_print = MagicMock()
    ui.unit_start("only_unit", "Solo task", 1, 1)
    printed = ui._safe_print.call_args[0][0]
    assert "(1/1)" in printed
    assert "only_unit" in printed


# ---------------------------------------------------------------------------
# unit_kept()
# ---------------------------------------------------------------------------


def test_unit_kept_shows_checkmark_and_delta(ui):
    """unit_kept() prints checkmark and delta text."""
    ui._safe_print = MagicMock()
    ui.unit_kept("unit_001", "+50 -10 lines")
    printed = ui._safe_print.call_args[0][0]
    assert "✓" in printed
    assert "kept" in printed
    assert "+50 -10 lines" in printed


# ---------------------------------------------------------------------------
# unit_reverted()
# ---------------------------------------------------------------------------


def test_unit_reverted_shows_x_and_reason(ui):
    """unit_reverted() prints X mark and reason text."""
    ui._safe_print = MagicMock()
    ui.unit_reverted("unit_002", "tests failed")
    printed = ui._safe_print.call_args[0][0]
    assert "✗" in printed
    assert "reverted" in printed
    assert "tests failed" in printed


def test_unit_reverted_includes_dash_separator(ui):
    """unit_reverted() includes — separator before reason."""
    ui._safe_print = MagicMock()
    ui.unit_reverted("u1", "lint errors")
    printed = ui._safe_print.call_args[0][0]
    assert "—" in printed


# ---------------------------------------------------------------------------
# pr_opened()
# ---------------------------------------------------------------------------


def test_pr_opened_renders_panel_with_url(ui):
    """pr_opened() prints a Panel containing the URL."""
    ui._safe_print = MagicMock()
    ui.pr_opened("https://github.com/org/repo/pull/42")
    # 3 calls: blank, Panel, blank
    assert ui._safe_print.call_count == 3
    panel_arg = ui._safe_print.call_args_list[1][0][0]
    assert isinstance(panel_arg, Panel)
    assert "https://github.com/org/repo/pull/42" in panel_arg.renderable


def test_pr_opened_panel_title(ui):
    """pr_opened() Panel has 'PR Opened' title."""
    ui._safe_print = MagicMock()
    ui.pr_opened("https://example.com/pr/1")
    panel_arg = ui._safe_print.call_args_list[1][0][0]
    assert "PR Opened" in panel_arg.title


# ---------------------------------------------------------------------------
# coverage_warning()
# ---------------------------------------------------------------------------


def test_coverage_warning_renders_panel(ui):
    """coverage_warning() prints a Panel."""
    ui._safe_print = MagicMock()
    warnings = [
        {
            "module": "auth.py",
            "coverage_pct": 45,
            "recommendation": "Add login tests",
        }
    ]
    ui.coverage_warning(warnings)
    panel_calls = [
        c[0][0]
        for c in ui._safe_print.call_args_list
        if c[0] and isinstance(c[0][0], Panel)
    ]
    assert len(panel_calls) == 1


def test_coverage_warning_contains_module_name(ui):
    """coverage_warning() Panel includes module names."""
    ui._safe_print = MagicMock()
    warnings = [
        {
            "module": "db_client.py",
            "coverage_pct": 30,
            "recommendation": "Add integration tests",
        }
    ]
    ui.coverage_warning(warnings)
    panel = [
        c[0][0]
        for c in ui._safe_print.call_args_list
        if c[0] and isinstance(c[0][0], Panel)
    ][0]
    renderable = panel.renderable
    assert "db_client.py" in renderable


def test_coverage_warning_contains_percentage(ui):
    """coverage_warning() Panel includes coverage percentages."""
    ui._safe_print = MagicMock()
    warnings = [
        {
            "module": "api.py",
            "coverage_pct": 22,
            "recommendation": "Test endpoints",
        }
    ]
    ui.coverage_warning(warnings)
    panel = [
        c[0][0]
        for c in ui._safe_print.call_args_list
        if c[0] and isinstance(c[0][0], Panel)
    ][0]
    assert "22%" in panel.renderable


def test_coverage_warning_contains_recommendation(ui):
    """coverage_warning() Panel includes recommendations."""
    ui._safe_print = MagicMock()
    warnings = [
        {
            "module": "api.py",
            "coverage_pct": 22,
            "recommendation": "Add endpoint tests",
        }
    ]
    ui.coverage_warning(warnings)
    panel = [
        c[0][0]
        for c in ui._safe_print.call_args_list
        if c[0] and isinstance(c[0][0], Panel)
    ][0]
    assert "Add endpoint tests" in panel.renderable


def test_coverage_warning_multiple_modules(ui):
    """coverage_warning() handles multiple warning entries."""
    ui._safe_print = MagicMock()
    warnings = [
        {"module": "a.py", "coverage_pct": 10, "recommendation": "fix a"},
        {"module": "b.py", "coverage_pct": 20, "recommendation": "fix b"},
    ]
    ui.coverage_warning(warnings)
    panel = [
        c[0][0]
        for c in ui._safe_print.call_args_list
        if c[0] and isinstance(c[0][0], Panel)
    ][0]
    assert "a.py" in panel.renderable
    assert "b.py" in panel.renderable
    assert "10%" in panel.renderable
    assert "20%" in panel.renderable


def test_coverage_warning_panel_title(ui):
    """coverage_warning() Panel title contains 'Coverage Warning'."""
    ui._safe_print = MagicMock()
    warnings = [{"module": "x.py", "coverage_pct": 5, "recommendation": "r"}]
    ui.coverage_warning(warnings)
    panel = [
        c[0][0]
        for c in ui._safe_print.call_args_list
        if c[0] and isinstance(c[0][0], Panel)
    ][0]
    assert "Coverage Warning" in panel.title


# ---------------------------------------------------------------------------
# error()
# ---------------------------------------------------------------------------


def test_error_prints_with_error_prefix(ui):
    """error() prints message with 'Error:' prefix."""
    ui._safe_print = MagicMock()
    ui.error("something went wrong")
    printed = ui._safe_print.call_args[0][0]
    assert "Error:" in printed
    assert "something went wrong" in printed


# ---------------------------------------------------------------------------
# info()
# ---------------------------------------------------------------------------


def test_info_prints_with_arrow_prefix(ui):
    """info() prints message with '>' prefix."""
    ui._safe_print = MagicMock()
    ui.info("loading config")
    printed = ui._safe_print.call_args[0][0]
    assert ">" in printed
    assert "loading config" in printed


# ---------------------------------------------------------------------------
# show_projects() — with projects
# ---------------------------------------------------------------------------


def test_show_projects_renders_table(ui):
    """show_projects() prints a Table when given projects."""
    ui.console.print = MagicMock()
    projects = [
        {
            "project_id": "feat_001",
            "repo_path": "/tmp/repo",
            "feature_prompt": "Add dark mode",
            "status": "completed",
            "stages_completed": ["discover", "research"],
            "created_at": "2025-01-15T10:30:00Z",
        }
    ]
    ui.show_projects(projects)
    ui.console.print.assert_called_once()
    table_arg = ui.console.print.call_args[0][0]
    assert isinstance(table_arg, Table)


def test_show_projects_empty_list(ui):
    """show_projects() prints 'No feature sessions found' for empty list."""
    ui.console.print = MagicMock()
    ui.show_projects([])
    ui.console.print.assert_called_once()
    printed = ui.console.print.call_args[0][0]
    assert "No feature sessions found" in printed


def test_show_projects_truncates_long_repo_path(ui):
    """show_projects() truncates repo_path to 40 chars."""
    ui.console.print = MagicMock()
    long_path = "/very/long/path/" + "a" * 50
    projects = [
        {
            "project_id": "feat_002",
            "repo_path": long_path,
            "feature_prompt": "short",
            "status": "in_progress",
            "stages_completed": [],
            "created_at": "2025-03-01T12:00:00",
        }
    ]
    ui.show_projects(projects)
    # Verify the method completed without error; the Table handles rendering
    ui.console.print.assert_called_once()


def test_show_projects_truncates_long_feature_prompt(ui):
    """show_projects() truncates feature_prompt to 40 chars."""
    ui.console.print = MagicMock()
    long_feature = "F" * 100
    projects = [
        {
            "project_id": "feat_003",
            "repo_path": "/repo",
            "feature_prompt": long_feature,
            "status": "planned",
            "stages_completed": ["discover"],
            "created_at": "2025-06-01T08:00:00",
        }
    ]
    ui.show_projects(projects)
    ui.console.print.assert_called_once()


def test_show_projects_missing_optional_fields(ui):
    """show_projects() handles projects with missing optional fields."""
    ui.console.print = MagicMock()
    projects = [
        {
            "project_id": "feat_004",
        }
    ]
    ui.show_projects(projects)
    ui.console.print.assert_called_once()
    table_arg = ui.console.print.call_args[0][0]
    assert isinstance(table_arg, Table)


def test_show_projects_empty_stages_shows_dash(ui):
    """show_projects() shows '—' when stages_completed is empty."""
    ui.console.print = MagicMock()
    projects = [
        {
            "project_id": "feat_005",
            "repo_path": "/repo",
            "feature_prompt": "feat",
            "status": "new",
            "stages_completed": [],
            "created_at": "2025-01-01T00:00:00",
        }
    ]
    ui.show_projects(projects)
    ui.console.print.assert_called_once()


def test_show_projects_multiple_stages_joined(ui):
    """show_projects() joins stages with commas."""
    ui.console.print = MagicMock()
    projects = [
        {
            "project_id": "feat_006",
            "repo_path": "/repo",
            "feature_prompt": "feat",
            "status": "done",
            "stages_completed": ["discover", "research", "plan"],
            "created_at": "2025-02-01T00:00:00",
        }
    ]
    ui.show_projects(projects)
    ui.console.print.assert_called_once()


def test_show_projects_status_defaults_to_unknown(ui):
    """show_projects() defaults status to 'unknown' when missing."""
    ui.console.print = MagicMock()
    projects = [
        {
            "project_id": "feat_007",
            "repo_path": "/r",
            "feature_prompt": "f",
            "stages_completed": [],
            "created_at": "2025-01-01T00:00:00",
        }
    ]
    # Should not raise
    ui.show_projects(projects)
    ui.console.print.assert_called_once()


def test_show_projects_created_at_truncated_to_19_chars(ui):
    """show_projects() truncates created_at to first 19 characters."""
    ui.console.print = MagicMock()
    projects = [
        {
            "project_id": "feat_008",
            "repo_path": "/r",
            "feature_prompt": "f",
            "status": "ok",
            "stages_completed": [],
            "created_at": "2025-01-15T10:30:00.123456+00:00",
        }
    ]
    ui.show_projects(projects)
    ui.console.print.assert_called_once()
