"""Tests for graft.ui."""

import io
from unittest.mock import MagicMock, patch

from rich.console import Console

from graft.ui import MAX_DISPLAY_CHARS, STAGE_LABELS, STAGE_ORDER, UI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ui(**kwargs) -> tuple[UI, io.StringIO]:
    """Create a UI whose console writes to a StringIO buffer for assertions.

    Returns (ui, buffer).  Use ``buffer.getvalue()`` to inspect plain-text
    output (Rich markup is stripped because we set ``no_color=True``).
    """
    buf = io.StringIO()
    ui = UI(**kwargs)
    ui.console = Console(file=buf, no_color=True, width=120)
    return ui, buf


# ===========================================================================
# Existing tests (preserved verbatim)
# ===========================================================================


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


# ===========================================================================
# New tests
# ===========================================================================


class TestBanner:
    """Tests for UI.banner()."""

    def test_banner_renders_repo_path_session_and_prompt(self):
        ui, buf = _make_ui()
        ui.banner("/home/user/repo", "sess-123", "Add login page")
        output = buf.getvalue()
        assert "/home/user/repo" in output
        assert "sess-123" in output
        assert "Add login page" in output

    def test_banner_truncates_long_feature_prompt(self):
        ui, buf = _make_ui()
        long_prompt = "x" * 200
        ui.banner("/repo", "id-1", long_prompt)
        output = buf.getvalue()
        # Should contain the first 120 chars and the ellipsis character
        assert "x" * 120 in output
        assert "…" in output
        # Should NOT contain the full 200-char string
        assert "x" * 200 not in output

    def test_banner_no_truncation_when_prompt_short(self):
        ui, buf = _make_ui()
        short_prompt = "y" * 120  # exactly 120 — no truncation
        ui.banner("/repo", "id-2", short_prompt)
        output = buf.getvalue()
        assert "y" * 120 in output
        assert "…" not in output


class TestShowArtifact:
    """Tests for UI.show_artifact()."""

    def test_show_artifact_renders_title_and_content(self):
        ui, buf = _make_ui()
        ui.show_artifact("My Title", "Hello world content")
        output = buf.getvalue()
        assert "My Title" in output
        assert "Hello world content" in output

    def test_show_artifact_truncates_long_content(self):
        ui, buf = _make_ui()
        long_content = "z" * (MAX_DISPLAY_CHARS + 500)
        ui.show_artifact("Big", long_content)
        output = buf.getvalue()
        assert "truncated" in output
        # The full content should not be present
        assert "z" * (MAX_DISPLAY_CHARS + 500) not in output

    def test_show_artifact_no_truncation_for_short_content(self):
        ui, buf = _make_ui()
        ui.show_artifact("Small", "tiny")
        output = buf.getvalue()
        assert "tiny" in output
        assert "truncated" not in output


class TestGrillQuestion:
    """Tests for UI.grill_question()."""

    def test_grill_question_returns_user_input(self):
        ui, _buf = _make_ui()
        ui.console.input = MagicMock(return_value="my custom answer")
        result = ui.grill_question("What color?", "blue", "prefs", 1)
        assert result == "my custom answer"

    def test_grill_question_returns_recommended_on_empty_input(self):
        ui, _buf = _make_ui()
        ui.console.input = MagicMock(return_value="   ")
        result = ui.grill_question("What color?", "blue", "prefs", 1)
        assert result == "blue"

    def test_grill_question_returns_recommended_on_eof(self):
        ui, _buf = _make_ui()
        ui.console.input = MagicMock(side_effect=EOFError)
        result = ui.grill_question("Q?", "default_answer", "cat", 3)
        assert result == "default_answer"

    def test_grill_question_returns_recommended_on_keyboard_interrupt(self):
        ui, _buf = _make_ui()
        ui.console.input = MagicMock(side_effect=KeyboardInterrupt)
        result = ui.grill_question("Q?", "fallback", "cat", 2)
        assert result == "fallback"

    def test_grill_question_renders_category_and_recommended(self):
        ui, buf = _make_ui()
        ui.console.input = MagicMock(return_value="ok")
        ui.grill_question("What size?", "large", "sizing", 5)
        output = buf.getvalue()
        assert "sizing" in output
        assert "large" in output
        assert "Question 5" in output


class TestPromptPlanReview:
    """Tests for UI.prompt_plan_review()."""

    def test_prompt_plan_review_auto_approve(self):
        ui, buf = _make_ui(auto_approve=True)
        approved, feedback = ui.prompt_plan_review("Build steps here")
        assert approved is True
        assert feedback == ""
        output = buf.getvalue()
        assert "auto-approved" in output

    def test_prompt_plan_review_user_approves(self):
        ui, buf = _make_ui()
        ui.auto_approve = False
        ui.console.input = MagicMock(return_value="approve")
        approved, feedback = ui.prompt_plan_review("Plan text")
        assert approved is True
        assert feedback == ""

    def test_prompt_plan_review_user_approves_with_yes(self):
        ui, _buf = _make_ui()
        ui.auto_approve = False
        ui.console.input = MagicMock(return_value="y")
        approved, feedback = ui.prompt_plan_review("Plan")
        assert approved is True
        assert feedback == ""

    def test_prompt_plan_review_user_approves_with_lgtm(self):
        ui, _buf = _make_ui()
        ui.auto_approve = False
        ui.console.input = MagicMock(return_value="lgtm")
        approved, feedback = ui.prompt_plan_review("Plan")
        assert approved is True
        assert feedback == ""

    def test_prompt_plan_review_user_approves_with_empty(self):
        ui, _buf = _make_ui()
        ui.auto_approve = False
        ui.console.input = MagicMock(return_value="")
        approved, feedback = ui.prompt_plan_review("Plan")
        assert approved is True
        assert feedback == ""

    def test_prompt_plan_review_user_gives_feedback(self):
        ui, _buf = _make_ui()
        ui.auto_approve = False
        ui.console.input = MagicMock(return_value="Add more tests please")
        approved, feedback = ui.prompt_plan_review("Plan text")
        assert approved is False
        assert feedback == "Add more tests please"

    def test_prompt_plan_review_eof_auto_approves(self):
        ui, buf = _make_ui()
        ui.auto_approve = False
        ui.console.input = MagicMock(side_effect=EOFError)
        approved, feedback = ui.prompt_plan_review("Plan")
        assert approved is True
        assert feedback == ""
        assert "auto-approving" in buf.getvalue()

    def test_prompt_plan_review_keyboard_interrupt_auto_approves(self):
        ui, buf = _make_ui()
        ui.auto_approve = False
        ui.console.input = MagicMock(side_effect=KeyboardInterrupt)
        approved, feedback = ui.prompt_plan_review("Plan")
        assert approved is True
        assert feedback == ""


class TestUnitLifecycle:
    """Tests for unit_start, unit_kept, unit_reverted."""

    def test_unit_start_renders_index_total_and_title(self):
        ui, buf = _make_ui()
        ui.unit_start("unit-a", "Create login form", 1, 5)
        output = buf.getvalue()
        assert "(1/5)" in output
        assert "unit-a" in output
        assert "Create login form" in output

    def test_unit_kept_renders_delta(self):
        ui, buf = _make_ui()
        ui.unit_kept("unit-b", "+30 -5 lines")
        output = buf.getvalue()
        assert "kept" in output
        assert "+30 -5 lines" in output

    def test_unit_reverted_renders_reason(self):
        ui, buf = _make_ui()
        ui.unit_reverted("unit-c", "tests failed")
        output = buf.getvalue()
        assert "reverted" in output
        assert "tests failed" in output


class TestPrOpened:
    """Tests for UI.pr_opened()."""

    def test_pr_opened_displays_url(self):
        ui, buf = _make_ui()
        ui.pr_opened("https://github.com/org/repo/pull/42")
        output = buf.getvalue()
        assert "https://github.com/org/repo/pull/42" in output
        assert "PR Opened" in output


class TestCoverageWarning:
    """Tests for UI.coverage_warning()."""

    def test_coverage_warning_displays_modules(self):
        ui, buf = _make_ui()
        warnings = [
            {
                "module": "auth.py",
                "coverage_pct": 35,
                "recommendation": "Add integration tests for OAuth flow",
            },
            {
                "module": "db.py",
                "coverage_pct": 42,
                "recommendation": "Cover migration edge cases",
            },
        ]
        ui.coverage_warning(warnings)
        output = buf.getvalue()
        assert "auth.py" in output
        assert "35%" in output
        assert "Add integration tests for OAuth flow" in output
        assert "db.py" in output
        assert "42%" in output
        assert "Coverage Warning" in output

    def test_coverage_warning_empty_list(self):
        ui, buf = _make_ui()
        ui.coverage_warning([])
        output = buf.getvalue()
        # Should still render the panel, just with no module lines
        assert "Coverage Warning" in output


class TestShowProjects:
    """Tests for UI.show_projects()."""

    def test_show_projects_renders_table_with_data(self):
        ui, buf = _make_ui()
        projects = [
            {
                "project_id": "proj-1",
                "repo_path": "/home/user/repo",
                "feature_prompt": "Add dark mode",
                "status": "completed",
                "stages_completed": ["discover", "plan"],
                "created_at": "2025-01-15T10:30:00Z",
            },
        ]
        ui.show_projects(projects)
        output = buf.getvalue()
        assert "proj-1" in output
        assert "/home/user/repo" in output
        assert "Add dark mode" in output
        assert "completed" in output
        assert "discover" in output
        assert "plan" in output

    def test_show_projects_empty_list(self):
        ui, buf = _make_ui()
        ui.show_projects([])
        output = buf.getvalue()
        assert "No feature sessions found" in output

    def test_show_projects_missing_optional_fields(self):
        ui, buf = _make_ui()
        projects = [
            {
                "project_id": "proj-2",
            },
        ]
        ui.show_projects(projects)
        output = buf.getvalue()
        assert "proj-2" in output
        # The dash for empty stages
        assert "—" in output


class TestSafePrint:
    """Tests for UI._safe_print() error handling."""

    def test_safe_print_catches_broken_pipe_error(self):
        ui, _buf = _make_ui()
        ui.console.print = MagicMock(side_effect=BrokenPipeError)
        # Should not raise
        ui._safe_print("anything")

    def test_safe_print_catches_blocking_io_error(self):
        ui, _buf = _make_ui()
        ui.console.print = MagicMock(side_effect=BlockingIOError)
        ui._safe_print("anything")

    def test_safe_print_catches_os_error(self):
        ui, _buf = _make_ui()
        ui.console.print = MagicMock(side_effect=OSError("pipe gone"))
        ui._safe_print("anything")

    def test_safe_print_propagates_other_exceptions(self):
        ui, _buf = _make_ui()
        ui.console.print = MagicMock(side_effect=ValueError("boom"))
        try:
            ui._safe_print("anything")
            assert False, "Expected ValueError to propagate"
        except ValueError:
            pass


class TestStageLog:
    """Tests for stage_log verbose behaviour."""

    def test_stage_log_shown_when_verbose(self):
        ui, buf = _make_ui(verbose=True)
        ui.stage_log("discover", "found 3 files")
        output = buf.getvalue()
        assert "found 3 files" in output
        assert "Discover" in output

    def test_stage_log_suppressed_when_not_verbose_via_helper(self):
        ui, buf = _make_ui(verbose=False)
        ui.stage_log("research", "reading docs")
        output = buf.getvalue()
        assert "reading docs" not in output


class TestStageDone:
    """Tests for UI.stage_done()."""

    def test_stage_done_renders_complete_message(self):
        ui, buf = _make_ui()
        ui.stage_done("execute")
        output = buf.getvalue()
        assert "Execute" in output
        assert "complete" in output


class TestStageStart:
    """Tests for UI.stage_start()."""

    def test_stage_start_sets_current_stage(self):
        ui, _buf = _make_ui()
        ui.stage_start("grill")
        assert ui._current_stage == "grill"

    def test_stage_start_renders_label(self):
        ui, buf = _make_ui()
        ui.stage_start("verify")
        output = buf.getvalue()
        assert "Verify" in output


class TestErrorAndInfo:
    """Tests for UI.error() and UI.info()."""

    def test_error_renders_message(self):
        ui, buf = _make_ui()
        ui.error("something went wrong")
        output = buf.getvalue()
        assert "Error:" in output
        assert "something went wrong" in output

    def test_info_renders_message(self):
        ui, buf = _make_ui()
        ui.info("processing step 2")
        output = buf.getvalue()
        assert "processing step 2" in output


class TestStageOrder:
    """Tests for STAGE_ORDER constant."""

    def test_stage_order_matches_label_keys(self):
        assert STAGE_ORDER == list(STAGE_LABELS.keys())

    def test_stage_order_length(self):
        assert len(STAGE_ORDER) == 7
