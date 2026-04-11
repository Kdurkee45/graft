"""Tests for graft.ui."""

from unittest.mock import MagicMock, call, patch

import pytest
from rich.panel import Panel
from rich.table import Table

from graft.ui import MAX_DISPLAY_CHARS, STAGE_LABELS, STAGE_ORDER, UI


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
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ui():
    """Return a UI instance with a mocked Console."""
    instance = UI(auto_approve=True, verbose=True)
    instance.console = MagicMock()
    return instance


@pytest.fixture
def ui_interactive():
    """Return a UI configured for interactive mode (auto_approve=False)."""
    instance = UI(auto_approve=False, verbose=False)
    # Force auto_approve off (in CI stdin is not a tty, so __init__ may flip it)
    instance.auto_approve = False
    instance.console = MagicMock()
    return instance


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_stage_order_matches_labels():
    """STAGE_ORDER list matches the STAGE_LABELS keys in order."""
    assert STAGE_ORDER == list(STAGE_LABELS.keys())


def test_max_display_chars_value():
    assert MAX_DISPLAY_CHARS == 3000


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_auto_approve_explicit():
    """auto_approve=True forces the flag regardless of tty."""
    ui = UI(auto_approve=True)
    assert ui.auto_approve is True


def test_init_verbose_true():
    ui = UI(verbose=True)
    assert ui.verbose is True


def test_init_auto_approve_non_tty():
    """When stdin is not a tty, auto_approve is set True."""
    with patch("graft.ui.sys") as mock_sys:
        mock_sys.stdin.isatty.return_value = False
        ui = UI(auto_approve=False)
        assert ui.auto_approve is True


def test_init_auto_approve_tty():
    """When stdin IS a tty and auto_approve not requested, stays False."""
    with patch("graft.ui.sys") as mock_sys:
        mock_sys.stdin.isatty.return_value = True
        ui = UI(auto_approve=False)
        assert ui.auto_approve is False


# ---------------------------------------------------------------------------
# _safe_print
# ---------------------------------------------------------------------------


def test_safe_print_delegates_to_console(ui):
    ui._safe_print("hello", style="bold")
    ui.console.print.assert_called_once_with("hello", style="bold")


def test_safe_print_swallows_blocking_io_error(ui):
    ui.console.print.side_effect = BlockingIOError
    ui._safe_print("data")  # should not raise


def test_safe_print_swallows_broken_pipe_error(ui):
    ui.console.print.side_effect = BrokenPipeError
    ui._safe_print("data")  # should not raise


def test_safe_print_swallows_os_error(ui):
    ui.console.print.side_effect = OSError("pipe broke")
    ui._safe_print("data")  # should not raise


def test_safe_print_propagates_other_exceptions(ui):
    ui.console.print.side_effect = ValueError("unexpected")
    with pytest.raises(ValueError, match="unexpected"):
        ui._safe_print("data")


# ---------------------------------------------------------------------------
# banner()
# ---------------------------------------------------------------------------


def test_banner_short_prompt(ui):
    """Banner with a short prompt shows the full text (no truncation)."""
    ui.banner("repo/path", "proj-123", "Add login feature")
    # Three _safe_print calls: blank line, Panel, blank line
    assert ui.console.print.call_count == 3
    panel_call = ui.console.print.call_args_list[1]
    panel_obj = panel_call[0][0]
    assert isinstance(panel_obj, Panel)


def test_banner_long_prompt_truncated(ui):
    """Prompts longer than 120 chars get truncated with ellipsis."""
    long_prompt = "x" * 200
    ui.banner("repo/path", "proj-123", long_prompt)
    panel_call = ui.console.print.call_args_list[1]
    panel_obj = panel_call[0][0]
    # The renderable inside the Panel should contain the truncated text
    rendered_text = panel_obj.renderable
    assert "…" in rendered_text
    # First 120 chars present
    assert "x" * 120 in rendered_text


def test_banner_exactly_120_chars(ui):
    """A 120-char prompt is NOT truncated."""
    prompt = "y" * 120
    ui.banner("repo", "id", prompt)
    panel_call = ui.console.print.call_args_list[1]
    rendered_text = panel_call[0][0].renderable
    assert "…" not in rendered_text


def test_banner_contains_repo_and_project(ui):
    """Banner includes repo path and project ID."""
    ui.banner("/my/repo", "sess-42", "Do a thing")
    panel_call = ui.console.print.call_args_list[1]
    rendered_text = panel_call[0][0].renderable
    assert "/my/repo" in rendered_text
    assert "sess-42" in rendered_text


# ---------------------------------------------------------------------------
# stage_start()
# ---------------------------------------------------------------------------


def test_stage_start_sets_current_stage(ui):
    ui.stage_start("discover")
    assert ui._current_stage == "discover"


def test_stage_start_calls_console_rule(ui):
    ui.stage_start("research")
    ui.console.rule.assert_called_once()
    rule_text = ui.console.rule.call_args[0][0]
    assert "Research" in rule_text


def test_stage_start_unknown_stage_uses_raw_name(ui):
    """An unrecognised stage key falls back to the raw string."""
    ui.stage_start("custom_stage")
    assert ui._current_stage == "custom_stage"
    rule_text = ui.console.rule.call_args[0][0]
    assert "custom_stage" in rule_text


def test_stage_start_swallows_pipe_errors(ui):
    """stage_start catches BlockingIOError/BrokenPipeError from console.rule."""
    ui.console.rule.side_effect = BrokenPipeError
    ui.stage_start("plan")  # should not raise


def test_stage_start_swallows_blocking_io_error(ui):
    ui.console.rule.side_effect = BlockingIOError
    ui.stage_start("plan")  # should not raise


def test_stage_start_swallows_os_error(ui):
    ui.console.rule.side_effect = OSError
    ui.stage_start("plan")  # should not raise


# ---------------------------------------------------------------------------
# stage_done()
# ---------------------------------------------------------------------------


def test_stage_done_prints_complete(ui):
    ui.stage_done("execute")
    calls = [c[0][0] for c in ui.console.print.call_args_list]
    assert any("Execute" in c and "complete" in c for c in calls)


def test_stage_done_unknown_stage(ui):
    ui.stage_done("mystery")
    calls = [c[0][0] for c in ui.console.print.call_args_list]
    assert any("mystery" in c and "complete" in c for c in calls)


# ---------------------------------------------------------------------------
# stage_log()
# ---------------------------------------------------------------------------


def test_stage_log_verbose_true(ui):
    """When verbose=True, stage_log prints the message."""
    ui.verbose = True
    ui.stage_log("discover", "found 5 files")
    ui.console.print.assert_called_once()
    text = ui.console.print.call_args[0][0]
    assert "found 5 files" in text
    assert "Discover" in text


def test_stage_log_verbose_false(ui):
    """When verbose=False, stage_log is a no-op."""
    ui.verbose = False
    ui.stage_log("discover", "should be hidden")
    ui.console.print.assert_not_called()


def test_stage_log_unknown_stage_verbose(ui):
    ui.verbose = True
    ui.stage_log("other", "msg")
    text = ui.console.print.call_args[0][0]
    assert "other" in text


# ---------------------------------------------------------------------------
# show_artifact()
# ---------------------------------------------------------------------------


def test_show_artifact_short_content(ui):
    """Content under MAX_DISPLAY_CHARS is shown in full."""
    content = "Short content"
    ui.show_artifact("My Artifact", content)
    # 3 calls: blank, Panel, blank
    assert ui.console.print.call_count == 3
    panel = ui.console.print.call_args_list[1][0][0]
    assert isinstance(panel, Panel)
    assert panel.renderable == content


def test_show_artifact_truncation(ui):
    """Content over MAX_DISPLAY_CHARS is truncated with a notice."""
    content = "a" * (MAX_DISPLAY_CHARS + 500)
    ui.show_artifact("Big Artifact", content)
    panel = ui.console.print.call_args_list[1][0][0]
    rendered = panel.renderable
    # Only first MAX_DISPLAY_CHARS of content preserved
    assert rendered.startswith("a" * MAX_DISPLAY_CHARS)
    assert "truncated" in rendered
    assert "full artifact on disk" in rendered


def test_show_artifact_exactly_max_chars(ui):
    """Content exactly at MAX_DISPLAY_CHARS is NOT truncated."""
    content = "b" * MAX_DISPLAY_CHARS
    ui.show_artifact("Exact", content)
    panel = ui.console.print.call_args_list[1][0][0]
    assert panel.renderable == content
    assert "truncated" not in panel.renderable


def test_show_artifact_panel_title(ui):
    """Panel title contains the provided title."""
    ui.show_artifact("My Title", "body")
    panel = ui.console.print.call_args_list[1][0][0]
    assert "My Title" in panel.title


# ---------------------------------------------------------------------------
# grill_question()
# ---------------------------------------------------------------------------


def test_grill_question_user_provides_answer(ui):
    """When user types an answer, it is returned."""
    ui.console.input.return_value = "  My answer  "
    result = ui.grill_question("What scope?", "All files", "scope", 1)
    assert result == "My answer"


def test_grill_question_empty_returns_recommended(ui):
    """Pressing Enter (empty string) returns the recommended answer."""
    ui.console.input.return_value = ""
    result = ui.grill_question("What scope?", "All files", "scope", 1)
    assert result == "All files"


def test_grill_question_whitespace_returns_recommended(ui):
    """Whitespace-only input returns the recommended answer."""
    ui.console.input.return_value = "   "
    result = ui.grill_question("Q?", "default", "cat", 2)
    assert result == "default"


def test_grill_question_eof_returns_recommended(ui):
    """EOFError (piped stdin) falls back to recommended."""
    ui.console.input.side_effect = EOFError
    result = ui.grill_question("Q?", "fallback", "cat", 1)
    assert result == "fallback"


def test_grill_question_keyboard_interrupt_returns_recommended(ui):
    """KeyboardInterrupt falls back to recommended."""
    ui.console.input.side_effect = KeyboardInterrupt
    result = ui.grill_question("Q?", "safe default", "cat", 3)
    assert result == "safe default"


def test_grill_question_panel_content(ui):
    """The question panel includes question text, category, and recommended."""
    ui.console.input.return_value = "answer"
    ui.grill_question("Is it good?", "Yes it is", "quality", 5)
    # Panel is the second print call (first is blank line)
    panel = ui.console.print.call_args_list[1][0][0]
    assert isinstance(panel, Panel)
    rendered = panel.renderable
    assert "Is it good?" in rendered
    assert "quality" in rendered
    assert "Yes it is" in rendered


def test_grill_question_panel_title_number(ui):
    """Panel title includes the question number."""
    ui.console.input.return_value = ""
    ui.grill_question("Q?", "rec", "cat", 7)
    panel = ui.console.print.call_args_list[1][0][0]
    assert "7" in panel.title


# ---------------------------------------------------------------------------
# prompt_plan_review()
# ---------------------------------------------------------------------------


def test_prompt_plan_review_auto_approve(ui):
    """With auto_approve=True, plan is approved without input."""
    ui.auto_approve = True
    approved, feedback = ui.prompt_plan_review("The plan summary")
    assert approved is True
    assert feedback == ""
    # console.input should NOT be called
    ui.console.input.assert_not_called()


def test_prompt_plan_review_auto_approve_prints_notice(ui):
    """Auto-approve path prints an auto-approved notice."""
    ui.auto_approve = True
    ui.prompt_plan_review("plan")
    all_text = " ".join(c[0][0] for c in ui.console.print.call_args_list if c[0])
    assert "auto-approved" in all_text


def test_prompt_plan_review_approve_explicit(ui_interactive):
    """Typing 'approve' returns (True, '')."""
    ui_interactive.console.input.return_value = "approve"
    approved, feedback = ui_interactive.prompt_plan_review("plan text")
    assert approved is True
    assert feedback == ""


@pytest.mark.parametrize("response", ["yes", "y", "lgtm", "YES", "Y", "LGTM", ""])
def test_prompt_plan_review_approve_variants(ui_interactive, response):
    """Various affirmative inputs all approve the plan."""
    ui_interactive.console.input.return_value = response
    approved, feedback = ui_interactive.prompt_plan_review("plan")
    assert approved is True
    assert feedback == ""


def test_prompt_plan_review_reject_with_feedback(ui_interactive):
    """Non-approve input returns (False, feedback)."""
    ui_interactive.console.input.return_value = "Please add tests"
    approved, feedback = ui_interactive.prompt_plan_review("plan")
    assert approved is False
    assert feedback == "Please add tests"


def test_prompt_plan_review_eof_auto_approves(ui_interactive):
    """EOFError during input auto-approves."""
    ui_interactive.console.input.side_effect = EOFError
    approved, feedback = ui_interactive.prompt_plan_review("plan")
    assert approved is True
    assert feedback == ""


def test_prompt_plan_review_keyboard_interrupt_auto_approves(ui_interactive):
    """KeyboardInterrupt during input auto-approves."""
    ui_interactive.console.input.side_effect = KeyboardInterrupt
    approved, feedback = ui_interactive.prompt_plan_review("plan")
    assert approved is True
    assert feedback == ""


def test_prompt_plan_review_shows_artifact(ui_interactive):
    """prompt_plan_review shows the plan via show_artifact before asking."""
    ui_interactive.console.input.return_value = "approve"
    ui_interactive.prompt_plan_review("my detailed plan")
    # show_artifact creates a Panel with the plan content
    panels = [
        c[0][0]
        for c in ui_interactive.console.print.call_args_list
        if c[0] and isinstance(c[0][0], Panel)
    ]
    assert len(panels) >= 1
    assert panels[0].renderable == "my detailed plan"


# ---------------------------------------------------------------------------
# unit_start / unit_kept / unit_reverted
# ---------------------------------------------------------------------------


def test_unit_start(ui):
    ui.unit_start("U-001", "Add login", 1, 5)
    text = ui.console.print.call_args[0][0]
    assert "1" in text and "5" in text
    assert "U-001" in text
    assert "Add login" in text


def test_unit_kept(ui):
    ui.unit_kept("U-001", "+20 -3")
    text = ui.console.print.call_args[0][0]
    assert "kept" in text
    assert "+20 -3" in text


def test_unit_reverted(ui):
    ui.unit_reverted("U-002", "tests failed")
    text = ui.console.print.call_args[0][0]
    assert "reverted" in text
    assert "tests failed" in text


# ---------------------------------------------------------------------------
# pr_opened()
# ---------------------------------------------------------------------------


def test_pr_opened(ui):
    url = "https://github.com/org/repo/pull/42"
    ui.pr_opened(url)
    # 3 prints: blank, Panel, blank
    assert ui.console.print.call_count == 3
    panel = ui.console.print.call_args_list[1][0][0]
    assert isinstance(panel, Panel)
    assert url in panel.renderable
    assert "PR Opened" in panel.title


# ---------------------------------------------------------------------------
# coverage_warning()
# ---------------------------------------------------------------------------


def test_coverage_warning_with_warnings(ui):
    warnings = [
        {
            "module": "auth.py",
            "coverage_pct": 12,
            "recommendation": "Add integration tests",
        },
        {
            "module": "db.py",
            "coverage_pct": 5,
            "recommendation": "Mock the connection",
        },
    ]
    ui.coverage_warning(warnings)
    # 3 prints: blank, Panel, blank
    assert ui.console.print.call_count == 3
    panel = ui.console.print.call_args_list[1][0][0]
    assert isinstance(panel, Panel)
    rendered = panel.renderable
    assert "auth.py" in rendered
    assert "12" in rendered
    assert "Add integration tests" in rendered
    assert "db.py" in rendered
    assert "Coverage Warning" in panel.title


def test_coverage_warning_empty_list(ui):
    """An empty warnings list still renders (header only)."""
    ui.coverage_warning([])
    assert ui.console.print.call_count == 3
    panel = ui.console.print.call_args_list[1][0][0]
    assert isinstance(panel, Panel)
    assert "Low Test Coverage" in panel.renderable


# ---------------------------------------------------------------------------
# error() / info()
# ---------------------------------------------------------------------------


def test_error(ui):
    ui.error("Something broke")
    text = ui.console.print.call_args[0][0]
    assert "Error" in text
    assert "Something broke" in text


def test_info(ui):
    ui.info("Processing...")
    text = ui.console.print.call_args[0][0]
    assert "Processing..." in text


# ---------------------------------------------------------------------------
# show_projects()
# ---------------------------------------------------------------------------


def test_show_projects_empty(ui):
    """Empty list prints 'no sessions found' message."""
    ui.show_projects([])
    ui.console.print.assert_called_once()
    text = ui.console.print.call_args[0][0]
    assert "No feature sessions found" in text


def test_show_projects_populated(ui):
    """Populated project list renders a Table."""
    projects = [
        {
            "project_id": "p-1",
            "repo_path": "/repo/one",
            "feature_prompt": "Add widgets",
            "status": "running",
            "stages_completed": ["discover", "research"],
            "created_at": "2025-01-15T10:30:00Z",
        },
        {
            "project_id": "p-2",
            "repo_path": "/repo/two",
            "feature_prompt": "Fix bug",
            "status": "done",
            "stages_completed": [],
            "created_at": "2025-02-20T12:00:00Z",
        },
    ]
    ui.show_projects(projects)
    ui.console.print.assert_called_once()
    table = ui.console.print.call_args[0][0]
    assert isinstance(table, Table)


def test_show_projects_missing_optional_fields(ui):
    """Projects with missing optional fields don't crash."""
    projects = [{"project_id": "p-3"}]
    ui.show_projects(projects)
    table = ui.console.print.call_args[0][0]
    assert isinstance(table, Table)


def test_show_projects_long_fields_truncated(ui):
    """Repo path and feature prompt are truncated to 40 chars in add_row."""
    long_path = "x" * 100
    long_prompt = "y" * 100
    projects = [
        {
            "project_id": "p-4",
            "repo_path": long_path,
            "feature_prompt": long_prompt,
            "status": "running",
            "stages_completed": ["discover"],
            "created_at": "2025-03-01T00:00:00.000000Z",
        },
    ]
    ui.show_projects(projects)
    # No assertion on truncation directly (it's inside Table internals),
    # but this verifies no crash and Table is built
    assert isinstance(ui.console.print.call_args[0][0], Table)


# ---------------------------------------------------------------------------
# Non-interactive / tty path tests
# ---------------------------------------------------------------------------


def test_auto_approve_non_tty_skips_plan_input():
    """When stdin is not a tty, prompt_plan_review auto-approves."""
    with patch("graft.ui.sys") as mock_sys:
        mock_sys.stdin.isatty.return_value = False
        ui = UI(auto_approve=False)
    ui.console = MagicMock()
    approved, feedback = ui.prompt_plan_review("plan")
    assert approved is True
    ui.console.input.assert_not_called()


def test_auto_approve_non_tty_grill_uses_recommended():
    """When stdin is not a tty, grill_question should still work via fallback."""
    with patch("graft.ui.sys") as mock_sys:
        mock_sys.stdin.isatty.return_value = False
        ui = UI()
    ui.console = MagicMock()
    # Simulate EOFError since stdin is not a real tty
    ui.console.input.side_effect = EOFError
    result = ui.grill_question("Q?", "recommended_value", "cat", 1)
    assert result == "recommended_value"
