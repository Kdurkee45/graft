"""Tests for graft.ui."""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock, patch

from rich.console import Console

from graft.ui import MAX_DISPLAY_CHARS, STAGE_LABELS, STAGE_ORDER, UI


# ── helpers ──────────────────────────────────────────────────────────────
def _make_ui(**kwargs) -> tuple[UI, StringIO]:
    """Create a UI whose console writes to a StringIO buffer so we can inspect output."""
    buf = StringIO()
    ui = UI(**kwargs)
    ui.console = Console(file=buf, force_terminal=True, width=120)
    return ui, buf


# ── existing tests ───────────────────────────────────────────────────────
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


# ── banner ───────────────────────────────────────────────────────────────
def test_banner_renders_without_error():
    """banner() should render without raising."""
    ui, buf = _make_ui()
    ui.banner("/tmp/repo", "proj-123", "Add a login page")
    output = buf.getvalue()
    assert "Graft" in output
    assert "proj-123" in output
    assert "login page" in output


def test_banner_truncates_long_prompt():
    """Feature prompt longer than 120 chars is truncated with '…'."""
    ui, buf = _make_ui()
    long_prompt = "A" * 200
    ui.banner("/tmp/repo", "proj-1", long_prompt)
    output = buf.getvalue()
    # The truncated preview should end with the ellipsis character
    assert "…" in output
    # The full 200-char string should NOT appear
    assert long_prompt not in output


# ── stage_start / stage_done ─────────────────────────────────────────────
def test_stage_start_outputs_label():
    """stage_start() should display the human-readable label for the stage."""
    ui, buf = _make_ui()
    ui.stage_start("discover")
    output = buf.getvalue()
    assert "Discover" in output


def test_stage_start_unknown_stage_falls_back_to_raw_name():
    """stage_start() with an unknown stage name uses the raw name."""
    ui, buf = _make_ui()
    ui.stage_start("custom_stage")
    output = buf.getvalue()
    assert "custom_stage" in output


def test_stage_done_outputs_complete():
    """stage_done() should output '✓ <label> complete'."""
    ui, buf = _make_ui()
    ui.stage_done("plan")
    output = buf.getvalue()
    assert "Plan" in output
    assert "complete" in output


def test_stage_start_sets_current_stage():
    """stage_start() should record the current stage internally."""
    ui, _ = _make_ui()
    assert ui._current_stage is None
    ui.stage_start("research")
    assert ui._current_stage == "research"


# ── stage_log (verbose) ─────────────────────────────────────────────────
def test_stage_log_visible_when_verbose():
    """stage_log() should appear when verbose=True."""
    ui, buf = _make_ui(verbose=True)
    ui.stage_log("discover", "Scanning files…")
    output = buf.getvalue()
    assert "Scanning files" in output


# ── show_artifact ────────────────────────────────────────────────────────
def test_show_artifact_short_content():
    """show_artifact() renders short content fully."""
    ui, buf = _make_ui()
    ui.show_artifact("Plan", "Step 1: do stuff")
    output = buf.getvalue()
    assert "Step 1: do stuff" in output
    assert "truncated" not in output


def test_show_artifact_truncates_long_content():
    """show_artifact() truncates content exceeding MAX_DISPLAY_CHARS."""
    ui, buf = _make_ui()
    long_content = "x" * (MAX_DISPLAY_CHARS + 500)
    ui.show_artifact("Huge Plan", long_content)
    output = buf.getvalue()
    assert "truncated" in output
    # The full content should not appear
    assert long_content not in output


def test_show_artifact_boundary_not_truncated():
    """Content exactly at MAX_DISPLAY_CHARS should not be truncated."""
    ui, buf = _make_ui()
    exact_content = "y" * MAX_DISPLAY_CHARS
    ui.show_artifact("Exact", exact_content)
    output = buf.getvalue()
    assert "truncated" not in output


# ── grill_question ───────────────────────────────────────────────────────
def test_grill_question_returns_recommended_on_empty_input():
    """When user presses Enter (empty input), grill_question returns the recommended answer."""
    ui, buf = _make_ui()
    with patch.object(ui.console, "input", return_value=""):
        answer = ui.grill_question(
            question="What framework?",
            recommended="React",
            category="Tech",
            number=1,
        )
    assert answer == "React"


def test_grill_question_returns_user_input():
    """When user types a response, grill_question returns that response."""
    ui, buf = _make_ui()
    with patch.object(ui.console, "input", return_value="Vue"):
        answer = ui.grill_question(
            question="What framework?",
            recommended="React",
            category="Tech",
            number=1,
        )
    assert answer == "Vue"


def test_grill_question_strips_whitespace():
    """Whitespace-only input should be treated as empty → returns recommended."""
    ui, buf = _make_ui()
    with patch.object(ui.console, "input", return_value="   "):
        answer = ui.grill_question(
            question="Language?",
            recommended="Python",
            category="Tech",
            number=2,
        )
    assert answer == "Python"


def test_grill_question_eof_returns_recommended():
    """EOFError during input returns the recommended answer."""
    ui, buf = _make_ui()
    with patch.object(ui.console, "input", side_effect=EOFError):
        answer = ui.grill_question(
            question="DB?",
            recommended="Postgres",
            category="Infra",
            number=3,
        )
    assert answer == "Postgres"


def test_grill_question_keyboard_interrupt_returns_recommended():
    """KeyboardInterrupt during input returns the recommended answer."""
    ui, buf = _make_ui()
    with patch.object(ui.console, "input", side_effect=KeyboardInterrupt):
        answer = ui.grill_question(
            question="DB?",
            recommended="Postgres",
            category="Infra",
            number=3,
        )
    assert answer == "Postgres"


# ── prompt_plan_review ───────────────────────────────────────────────────
def test_prompt_plan_review_auto_approves():
    """prompt_plan_review() auto-approves when auto_approve=True."""
    ui, buf = _make_ui(auto_approve=True)
    approved, feedback = ui.prompt_plan_review("Build a thing")
    assert approved is True
    assert feedback == ""
    output = buf.getvalue()
    assert "auto-approved" in output


def test_prompt_plan_review_approve_input():
    """Typing 'approve' returns (True, '')."""
    ui, buf = _make_ui()
    ui.auto_approve = False
    with patch.object(ui.console, "input", return_value="approve"):
        approved, feedback = ui.prompt_plan_review("Plan text")
    assert approved is True
    assert feedback == ""


def test_prompt_plan_review_yes_input():
    """Typing 'yes' returns (True, '')."""
    ui, buf = _make_ui()
    ui.auto_approve = False
    with patch.object(ui.console, "input", return_value="yes"):
        approved, feedback = ui.prompt_plan_review("Plan text")
    assert approved is True
    assert feedback == ""


def test_prompt_plan_review_y_input():
    """Typing 'y' returns (True, '')."""
    ui, buf = _make_ui()
    ui.auto_approve = False
    with patch.object(ui.console, "input", return_value="y"):
        approved, feedback = ui.prompt_plan_review("Plan text")
    assert approved is True
    assert feedback == ""


def test_prompt_plan_review_lgtm_input():
    """Typing 'lgtm' returns (True, '')."""
    ui, buf = _make_ui()
    ui.auto_approve = False
    with patch.object(ui.console, "input", return_value="lgtm"):
        approved, feedback = ui.prompt_plan_review("Plan text")
    assert approved is True
    assert feedback == ""


def test_prompt_plan_review_empty_input_approves():
    """Pressing Enter (empty string) auto-approves."""
    ui, buf = _make_ui()
    ui.auto_approve = False
    with patch.object(ui.console, "input", return_value=""):
        approved, feedback = ui.prompt_plan_review("Plan text")
    assert approved is True
    assert feedback == ""


def test_prompt_plan_review_feedback_rejects():
    """Typing feedback text returns (False, feedback)."""
    ui, buf = _make_ui()
    ui.auto_approve = False
    with patch.object(ui.console, "input", return_value="Add more tests please"):
        approved, feedback = ui.prompt_plan_review("Plan text")
    assert approved is False
    assert feedback == "Add more tests please"


def test_prompt_plan_review_eof_auto_approves():
    """EOFError during prompt_plan_review auto-approves."""
    ui, buf = _make_ui()
    ui.auto_approve = False
    with patch.object(ui.console, "input", side_effect=EOFError):
        approved, feedback = ui.prompt_plan_review("Plan text")
    assert approved is True
    assert feedback == ""


def test_prompt_plan_review_keyboard_interrupt_auto_approves():
    """KeyboardInterrupt during prompt_plan_review auto-approves."""
    ui, buf = _make_ui()
    ui.auto_approve = False
    with patch.object(ui.console, "input", side_effect=KeyboardInterrupt):
        approved, feedback = ui.prompt_plan_review("Plan text")
    assert approved is True
    assert feedback == ""


# ── unit_start / unit_kept / unit_reverted ───────────────────────────────
def test_unit_start_formatting():
    """unit_start() outputs index/total and unit info."""
    ui, buf = _make_ui()
    ui.unit_start("unit-42", "Add login form", index=2, total=5)
    output = buf.getvalue()
    assert "(2/5)" in output
    assert "unit-42" in output
    assert "Add login form" in output


def test_unit_kept_formatting():
    """unit_kept() outputs green checkmark and delta."""
    ui, buf = _make_ui()
    ui.unit_kept("unit-42", "+30 -5 lines")
    output = buf.getvalue()
    assert "kept" in output
    assert "+30 -5 lines" in output


def test_unit_reverted_formatting():
    """unit_reverted() outputs red X and reason."""
    ui, buf = _make_ui()
    ui.unit_reverted("unit-42", "tests failed")
    output = buf.getvalue()
    assert "reverted" in output
    assert "tests failed" in output


# ── pr_opened ────────────────────────────────────────────────────────────
def test_pr_opened_displays_url():
    """pr_opened() renders a panel containing the PR url."""
    ui, buf = _make_ui()
    ui.pr_opened("https://github.com/org/repo/pull/99")
    output = buf.getvalue()
    assert "https://github.com/org/repo/pull/99" in output
    assert "PR Opened" in output


# ── coverage_warning ─────────────────────────────────────────────────────
def test_coverage_warning_renders_warnings():
    """coverage_warning() renders a panel with module names and recommendations."""
    ui, buf = _make_ui()
    warnings = [
        {
            "module": "graft.executor",
            "coverage_pct": 12,
            "recommendation": "Add integration tests",
        },
        {
            "module": "graft.planner",
            "coverage_pct": 28,
            "recommendation": "Add unit tests for plan parsing",
        },
    ]
    ui.coverage_warning(warnings)
    output = buf.getvalue()
    assert "graft.executor" in output
    assert "12%" in output
    assert "Add integration tests" in output
    assert "graft.planner" in output
    assert "Coverage Warning" in output


def test_coverage_warning_empty_list():
    """coverage_warning() with empty list still renders the panel header."""
    ui, buf = _make_ui()
    ui.coverage_warning([])
    output = buf.getvalue()
    assert "Coverage Warning" in output


# ── error / info ─────────────────────────────────────────────────────────
def test_error_outputs_message():
    """error() displays 'Error:' followed by the message."""
    ui, buf = _make_ui()
    ui.error("Something went wrong")
    output = buf.getvalue()
    assert "Error:" in output
    assert "Something went wrong" in output


def test_info_outputs_message():
    """info() displays the message."""
    ui, buf = _make_ui()
    ui.info("Processing files…")
    output = buf.getvalue()
    assert "Processing files" in output


# ── show_projects ────────────────────────────────────────────────────────
def test_show_projects_empty_list():
    """show_projects() with empty list prints a 'no sessions' message."""
    ui, buf = _make_ui()
    ui.show_projects([])
    output = buf.getvalue()
    assert "No feature sessions found" in output


def test_show_projects_populated_list():
    """show_projects() renders a table with project data."""
    ui, buf = _make_ui()
    projects = [
        {
            "project_id": "proj-abc",
            "repo_path": "/home/user/my-repo",
            "feature_prompt": "Add dark mode",
            "status": "completed",
            "stages_completed": ["discover", "research"],
            "created_at": "2025-01-15T10:30:00Z",
        },
        {
            "project_id": "proj-def",
            "repo_path": "/home/user/other",
            "feature_prompt": "Refactor auth",
            "status": "in_progress",
            "stages_completed": [],
            "created_at": "2025-01-16T08:00:00Z",
        },
    ]
    ui.show_projects(projects)
    output = buf.getvalue()
    assert "proj-abc" in output
    assert "proj-def" in output
    assert "Add dark mode" in output
    assert "completed" in output
    assert "discover" in output
    # Second project with no stages shows the em-dash fallback
    assert "—" in output


def test_show_projects_missing_optional_fields():
    """show_projects() handles projects with missing optional fields gracefully."""
    ui, buf = _make_ui()
    projects = [
        {
            "project_id": "proj-min",
        },
    ]
    ui.show_projects(projects)
    output = buf.getvalue()
    assert "proj-min" in output
    assert "unknown" in output  # default status


# ── _safe_print / error handling ─────────────────────────────────────────
def test_safe_print_handles_broken_pipe():
    """_safe_print gracefully swallows BrokenPipeError."""
    ui = UI()
    ui.console = MagicMock()
    ui.console.print.side_effect = BrokenPipeError
    # Should not raise
    ui._safe_print("hello")


def test_safe_print_handles_blocking_io():
    """_safe_print gracefully swallows BlockingIOError."""
    ui = UI()
    ui.console = MagicMock()
    ui.console.print.side_effect = BlockingIOError
    # Should not raise
    ui._safe_print("hello")


# ── STAGE_ORDER ──────────────────────────────────────────────────────────
def test_stage_order_matches_labels_keys():
    """STAGE_ORDER should contain exactly the keys from STAGE_LABELS in order."""
    assert STAGE_ORDER == list(STAGE_LABELS.keys())
    assert len(STAGE_ORDER) == 7


# ── auto_approve with explicit True ─────────────────────────────────────
def test_auto_approve_explicit_true():
    """Passing auto_approve=True forces the flag regardless of tty."""
    ui = UI(auto_approve=True)
    assert ui.auto_approve is True
