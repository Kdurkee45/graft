"""Tests for graft.ui."""

from unittest.mock import MagicMock, call, patch

import pytest
from rich.panel import Panel

from graft.ui import MAX_DISPLAY_CHARS, STAGE_LABELS, STAGE_ORDER, UI


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------


def test_stage_labels_match_pipeline():
    """Stage labels cover all pipeline stages."""
    expected = {"discover", "research", "grill", "plan", "plan_review", "execute", "verify"}
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
# Helpers
# ---------------------------------------------------------------------------


def _make_ui(**kwargs) -> UI:
    """Create a UI with a mocked Console so we can inspect calls."""
    ui = UI(**kwargs)
    ui.console = MagicMock()
    return ui


def _printed_strings(ui: UI) -> str:
    """Join all positional args passed to console.print into one string for easy assertions."""
    parts = []
    for c in ui.console.print.call_args_list:
        for arg in c.args:
            parts.append(str(arg))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# banner()
# ---------------------------------------------------------------------------


class TestBanner:
    def test_banner_prints_panel(self):
        """banner() prints a Panel with repo info."""
        ui = _make_ui()
        ui.banner("/repo/path", "proj-123", "Add login feature")
        # console.print is called 3 times: blank, Panel, blank
        assert ui.console.print.call_count == 3
        panel_call = ui.console.print.call_args_list[1]
        panel_arg = panel_call.args[0]
        assert isinstance(panel_arg, Panel)

    def test_banner_truncates_long_prompt(self):
        """Feature prompts longer than 120 chars are truncated with ellipsis."""
        ui = _make_ui()
        long_prompt = "x" * 200
        ui.banner("/repo", "id-1", long_prompt)
        panel_arg = ui.console.print.call_args_list[1].args[0]
        rendered = panel_arg.renderable
        assert "…" in rendered
        # The first 120 chars should be present
        assert "x" * 120 in rendered

    def test_banner_short_prompt_no_ellipsis(self):
        """Feature prompts within 120 chars are not truncated."""
        ui = _make_ui()
        short_prompt = "y" * 50
        ui.banner("/repo", "id-2", short_prompt)
        panel_arg = ui.console.print.call_args_list[1].args[0]
        rendered = panel_arg.renderable
        assert "…" not in rendered
        assert short_prompt in rendered


# ---------------------------------------------------------------------------
# stage_start() / stage_done()
# ---------------------------------------------------------------------------


class TestStageStartDone:
    def test_stage_start_sets_current_stage(self):
        """stage_start() records the current stage."""
        ui = _make_ui()
        ui.stage_start("discover")
        assert ui._current_stage == "discover"

    def test_stage_start_uses_label(self):
        """stage_start() prints the human-readable label via console.rule."""
        ui = _make_ui()
        ui.stage_start("research")
        ui.console.rule.assert_called_once()
        rule_text = ui.console.rule.call_args.args[0]
        assert "Research" in rule_text

    def test_stage_start_unknown_stage_falls_back_to_raw_name(self):
        """stage_start() falls back to the raw stage name if not in STAGE_LABELS."""
        ui = _make_ui()
        ui.stage_start("custom_stage")
        assert ui._current_stage == "custom_stage"
        rule_text = ui.console.rule.call_args.args[0]
        assert "custom_stage" in rule_text

    def test_stage_done_prints_complete_label(self):
        """stage_done() prints '✓ <label> complete'."""
        ui = _make_ui()
        ui.stage_done("plan")
        text = _printed_strings(ui)
        assert "Plan" in text
        assert "complete" in text

    def test_stage_done_unknown_stage(self):
        """stage_done() uses raw name for unknown stages."""
        ui = _make_ui()
        ui.stage_done("unknown_stage")
        text = _printed_strings(ui)
        assert "unknown_stage" in text


# ---------------------------------------------------------------------------
# stage_log()
# ---------------------------------------------------------------------------


class TestStageLog:
    def test_stage_log_visible_when_verbose(self):
        """stage_log() prints when verbose=True."""
        ui = _make_ui(verbose=True)
        ui.stage_log("discover", "scanning files")
        assert ui.console.print.call_count == 1
        text = _printed_strings(ui)
        assert "scanning files" in text

    def test_stage_log_suppressed_when_not_verbose(self):
        """stage_log() does NOT print when verbose=False."""
        ui = _make_ui(verbose=False)
        ui.stage_log("discover", "should be hidden")
        ui.console.print.assert_not_called()


# ---------------------------------------------------------------------------
# show_artifact()
# ---------------------------------------------------------------------------


class TestShowArtifact:
    def test_show_artifact_normal_content(self):
        """show_artifact() displays content in a Panel."""
        ui = _make_ui()
        ui.show_artifact("My Title", "some content here")
        assert ui.console.print.call_count == 3  # blank, Panel, blank
        panel = ui.console.print.call_args_list[1].args[0]
        assert isinstance(panel, Panel)
        assert "some content here" in panel.renderable

    def test_show_artifact_truncates_large_content(self):
        """Content exceeding MAX_DISPLAY_CHARS is truncated."""
        ui = _make_ui()
        big_content = "A" * (MAX_DISPLAY_CHARS + 500)
        ui.show_artifact("Big", big_content)
        panel = ui.console.print.call_args_list[1].args[0]
        rendered = panel.renderable
        assert "truncated" in rendered
        # Only the first MAX_DISPLAY_CHARS characters of the original content
        assert "A" * MAX_DISPLAY_CHARS in rendered

    def test_show_artifact_exact_max_not_truncated(self):
        """Content exactly at MAX_DISPLAY_CHARS is NOT truncated."""
        ui = _make_ui()
        exact_content = "B" * MAX_DISPLAY_CHARS
        ui.show_artifact("Exact", exact_content)
        panel = ui.console.print.call_args_list[1].args[0]
        rendered = panel.renderable
        assert "truncated" not in rendered


# ---------------------------------------------------------------------------
# grill_question()
# ---------------------------------------------------------------------------


class TestGrillQuestion:
    def test_grill_question_returns_user_input(self):
        """grill_question() returns the user's typed answer."""
        ui = _make_ui()
        ui.console.input.return_value = "my answer"
        result = ui.grill_question("What color?", "blue", "preference", 1)
        assert result == "my answer"

    def test_grill_question_empty_returns_recommended(self):
        """Empty response falls back to the recommended answer."""
        ui = _make_ui()
        ui.console.input.return_value = "  "
        result = ui.grill_question("What color?", "blue", "preference", 1)
        assert result == "blue"

    def test_grill_question_eoferror_returns_recommended(self):
        """EOFError (e.g. piped stdin) falls back to recommended."""
        ui = _make_ui()
        ui.console.input.side_effect = EOFError
        result = ui.grill_question("Q?", "default_answer", "cat", 3)
        assert result == "default_answer"

    def test_grill_question_keyboard_interrupt_returns_recommended(self):
        """KeyboardInterrupt (Ctrl-C) falls back to recommended."""
        ui = _make_ui()
        ui.console.input.side_effect = KeyboardInterrupt
        result = ui.grill_question("Q?", "fallback", "cat", 2)
        assert result == "fallback"

    def test_grill_question_shows_panel_with_metadata(self):
        """grill_question() displays a Panel with the question, category, and recommended."""
        ui = _make_ui()
        ui.console.input.return_value = "answer"
        ui.grill_question("Is it fast?", "yes", "performance", 5)
        # At least one Panel with the question text
        panels = [
            c.args[0]
            for c in ui.console.print.call_args_list
            if c.args and isinstance(c.args[0], Panel)
        ]
        assert len(panels) == 1
        rendered = panels[0].renderable
        assert "Is it fast?" in rendered
        assert "performance" in rendered
        assert "yes" in rendered


# ---------------------------------------------------------------------------
# prompt_plan_review()
# ---------------------------------------------------------------------------


class TestPromptPlanReview:
    def test_auto_approve_skips_input(self):
        """When auto_approve=True, plan is approved without user input."""
        ui = _make_ui(auto_approve=True)
        approved, feedback = ui.prompt_plan_review("the plan text")
        assert approved is True
        assert feedback == ""
        ui.console.input.assert_not_called()

    def test_approve_response(self):
        """Typing 'approve' approves the plan."""
        ui = _make_ui(auto_approve=False)
        ui.console.input.return_value = "approve"
        approved, feedback = ui.prompt_plan_review("plan text")
        assert approved is True
        assert feedback == ""

    def test_yes_response(self):
        """Typing 'yes' approves the plan."""
        ui = _make_ui(auto_approve=False)
        ui.console.input.return_value = "yes"
        approved, feedback = ui.prompt_plan_review("plan")
        assert approved is True

    def test_y_response(self):
        """Typing 'y' approves the plan."""
        ui = _make_ui(auto_approve=False)
        ui.console.input.return_value = "y"
        approved, feedback = ui.prompt_plan_review("plan")
        assert approved is True

    def test_lgtm_response(self):
        """Typing 'lgtm' approves the plan."""
        ui = _make_ui(auto_approve=False)
        ui.console.input.return_value = "lgtm"
        approved, feedback = ui.prompt_plan_review("plan")
        assert approved is True

    def test_empty_response_approves(self):
        """Empty response (just Enter) approves the plan."""
        ui = _make_ui(auto_approve=False)
        ui.console.input.return_value = ""
        approved, feedback = ui.prompt_plan_review("plan")
        assert approved is True
        assert feedback == ""

    def test_feedback_response_rejects(self):
        """Typing anything else rejects and returns feedback."""
        ui = _make_ui(auto_approve=False)
        ui.console.input.return_value = "please add error handling"
        approved, feedback = ui.prompt_plan_review("plan")
        assert approved is False
        assert feedback == "please add error handling"

    def test_eoferror_auto_approves(self):
        """EOFError during input auto-approves the plan."""
        ui = _make_ui(auto_approve=False)
        ui.console.input.side_effect = EOFError
        approved, feedback = ui.prompt_plan_review("plan")
        assert approved is True
        assert feedback == ""

    def test_keyboard_interrupt_auto_approves(self):
        """KeyboardInterrupt during input auto-approves the plan."""
        ui = _make_ui(auto_approve=False)
        ui.console.input.side_effect = KeyboardInterrupt
        approved, feedback = ui.prompt_plan_review("plan")
        assert approved is True
        assert feedback == ""

    def test_approve_case_insensitive(self):
        """Approval keywords are case-insensitive."""
        ui = _make_ui(auto_approve=False)
        ui.console.input.return_value = "APPROVE"
        approved, _ = ui.prompt_plan_review("plan")
        assert approved is True

        ui2 = _make_ui(auto_approve=False)
        ui2.console.input.return_value = "Yes"
        approved2, _ = ui2.prompt_plan_review("plan")
        assert approved2 is True


# ---------------------------------------------------------------------------
# show_projects()
# ---------------------------------------------------------------------------


class TestShowProjects:
    def test_empty_projects_shows_message(self):
        """Empty project list shows a 'no sessions' message."""
        ui = _make_ui()
        ui.show_projects([])
        text = _printed_strings(ui)
        assert "No feature sessions found" in text

    def test_show_projects_renders_table(self):
        """Non-empty project list renders a Rich Table."""
        from rich.table import Table

        ui = _make_ui()
        projects = [
            {
                "project_id": "abc-123",
                "repo_path": "/tmp/repo",
                "feature_prompt": "Add tests",
                "status": "done",
                "stages_completed": ["discover", "plan"],
                "created_at": "2025-01-15T10:30:00Z",
            }
        ]
        ui.show_projects(projects)
        # The table is passed directly to console.print
        table_arg = ui.console.print.call_args.args[0]
        assert isinstance(table_arg, Table)

    def test_show_projects_handles_missing_keys(self):
        """show_projects() handles projects with missing optional keys."""
        ui = _make_ui()
        projects = [
            {
                "project_id": "minimal-1",
            }
        ]
        # Should not raise
        ui.show_projects(projects)
        ui.console.print.assert_called_once()

    def test_show_projects_truncates_long_values(self):
        """Long repo paths and feature prompts are truncated to 40 chars."""
        ui = _make_ui()
        projects = [
            {
                "project_id": "trunc-1",
                "repo_path": "R" * 100,
                "feature_prompt": "F" * 100,
                "status": "active",
                "stages_completed": [],
                "created_at": "2025-01-15T10:30:00.123456Z",
            }
        ]
        ui.show_projects(projects)
        # No assertion on the exact truncation—we just verify it doesn't crash
        # and that console.print was called with a Table.
        from rich.table import Table

        table_arg = ui.console.print.call_args.args[0]
        assert isinstance(table_arg, Table)


# ---------------------------------------------------------------------------
# error() / info()
# ---------------------------------------------------------------------------


class TestErrorInfo:
    def test_error_prints_message(self):
        """error() prints an 'Error:' prefixed message."""
        ui = _make_ui()
        ui.error("something broke")
        text = _printed_strings(ui)
        assert "Error" in text
        assert "something broke" in text

    def test_info_prints_message(self):
        """info() prints an informational message."""
        ui = _make_ui()
        ui.info("status update")
        text = _printed_strings(ui)
        assert "status update" in text


# ---------------------------------------------------------------------------
# _safe_print() — BrokenPipeError handling
# ---------------------------------------------------------------------------


class TestSafePrint:
    def test_broken_pipe_error_suppressed(self):
        """BrokenPipeError from console.print is silently caught."""
        ui = _make_ui()
        ui.console.print.side_effect = BrokenPipeError
        # Should not raise
        ui._safe_print("test")

    def test_blocking_io_error_suppressed(self):
        """BlockingIOError from console.print is silently caught."""
        ui = _make_ui()
        ui.console.print.side_effect = BlockingIOError
        ui._safe_print("test")

    def test_os_error_suppressed(self):
        """OSError from console.print is silently caught."""
        ui = _make_ui()
        ui.console.print.side_effect = OSError
        ui._safe_print("test")

    def test_other_exceptions_propagate(self):
        """Non-IO exceptions from console.print still propagate."""
        ui = _make_ui()
        ui.console.print.side_effect = ValueError("unexpected")
        with pytest.raises(ValueError, match="unexpected"):
            ui._safe_print("test")


# ---------------------------------------------------------------------------
# stage_start() — BrokenPipeError in console.rule
# ---------------------------------------------------------------------------


class TestStageStartErrorHandling:
    def test_broken_pipe_in_rule_suppressed(self):
        """BrokenPipeError from console.rule in stage_start is caught."""
        ui = _make_ui()
        ui.console.rule.side_effect = BrokenPipeError
        # Should not raise
        ui.stage_start("discover")
        assert ui._current_stage == "discover"

    def test_os_error_in_rule_suppressed(self):
        """OSError from console.rule in stage_start is caught."""
        ui = _make_ui()
        ui.console.rule.side_effect = OSError
        ui.stage_start("plan")
        assert ui._current_stage == "plan"


# ---------------------------------------------------------------------------
# TTY vs non-TTY — auto_approve logic
# ---------------------------------------------------------------------------


class TestAutoApprove:
    def test_auto_approve_true_flag(self):
        """Passing auto_approve=True forces auto-approval."""
        ui = UI(auto_approve=True)
        assert ui.auto_approve is True

    @patch("sys.stdin")
    def test_auto_approve_when_stdin_not_tty(self, mock_stdin):
        """auto_approve is True when stdin is not a TTY."""
        mock_stdin.isatty.return_value = False
        ui = UI(auto_approve=False)
        assert ui.auto_approve is True

    @patch("sys.stdin")
    def test_no_auto_approve_when_stdin_is_tty(self, mock_stdin):
        """auto_approve is False when stdin IS a TTY and flag is False."""
        mock_stdin.isatty.return_value = True
        ui = UI(auto_approve=False)
        assert ui.auto_approve is False


# ---------------------------------------------------------------------------
# unit_start(), unit_kept(), unit_reverted()
# ---------------------------------------------------------------------------


class TestUnitTracking:
    def test_unit_start_prints_index_and_title(self):
        """unit_start() shows index/total, id, and title."""
        ui = _make_ui()
        ui.unit_start("unit-1", "Create user model", 1, 3)
        text = _printed_strings(ui)
        assert "1/3" in text
        assert "unit-1" in text
        assert "Create user model" in text

    def test_unit_kept_prints_delta(self):
        """unit_kept() shows a green checkmark and the delta."""
        ui = _make_ui()
        ui.unit_kept("unit-1", "+50 -3")
        text = _printed_strings(ui)
        assert "kept" in text
        assert "+50 -3" in text

    def test_unit_reverted_prints_reason(self):
        """unit_reverted() shows a red X and the reason."""
        ui = _make_ui()
        ui.unit_reverted("unit-1", "tests failed")
        text = _printed_strings(ui)
        assert "reverted" in text
        assert "tests failed" in text


# ---------------------------------------------------------------------------
# pr_opened()
# ---------------------------------------------------------------------------


class TestPrOpened:
    def test_pr_opened_prints_url_in_panel(self):
        """pr_opened() displays the URL inside a Panel."""
        ui = _make_ui()
        ui.pr_opened("https://github.com/org/repo/pull/42")
        panels = [
            c.args[0]
            for c in ui.console.print.call_args_list
            if c.args and isinstance(c.args[0], Panel)
        ]
        assert len(panels) == 1
        assert "https://github.com/org/repo/pull/42" in panels[0].renderable


# ---------------------------------------------------------------------------
# coverage_warning()
# ---------------------------------------------------------------------------


class TestCoverageWarning:
    def test_coverage_warning_prints_panel(self):
        """coverage_warning() displays warnings inside a Panel."""
        ui = _make_ui()
        warnings = [
            {
                "module": "src/auth.py",
                "coverage_pct": 23,
                "recommendation": "Add integration tests for OAuth flow",
            },
            {
                "module": "src/db.py",
                "coverage_pct": 15,
                "recommendation": "Add tests for migrations",
            },
        ]
        ui.coverage_warning(warnings)
        panels = [
            c.args[0]
            for c in ui.console.print.call_args_list
            if c.args and isinstance(c.args[0], Panel)
        ]
        assert len(panels) == 1
        rendered = panels[0].renderable
        assert "src/auth.py" in rendered
        assert "src/db.py" in rendered
        assert "23%" in rendered

    def test_coverage_warning_empty_list(self):
        """coverage_warning() still prints a panel even with no warnings."""
        ui = _make_ui()
        ui.coverage_warning([])
        panels = [
            c.args[0]
            for c in ui.console.print.call_args_list
            if c.args and isinstance(c.args[0], Panel)
        ]
        assert len(panels) == 1


# ---------------------------------------------------------------------------
# STAGE_ORDER
# ---------------------------------------------------------------------------


class TestStageOrder:
    def test_stage_order_matches_labels_keys(self):
        """STAGE_ORDER preserves the order of STAGE_LABELS keys."""
        assert STAGE_ORDER == list(STAGE_LABELS.keys())

    def test_stage_order_length(self):
        """STAGE_ORDER has exactly 7 stages."""
        assert len(STAGE_ORDER) == 7
