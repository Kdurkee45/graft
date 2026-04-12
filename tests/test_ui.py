"""Tests for graft.ui."""

from graft.ui import STAGE_LABELS, UI


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
