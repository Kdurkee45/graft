"""Tests for graft.graph."""

from graft.graph import ORDERED_STAGES, build_graph
from graft.ui import UI


def test_ordered_stages():
    """Pipeline has 6 stages in correct order."""
    assert ORDERED_STAGES == [
        "discover",
        "research",
        "grill",
        "plan",
        "execute",
        "verify",
    ]


def test_build_graph_compiles():
    """build_graph returns a compiled graph without error."""
    ui = UI(auto_approve=True)
    compiled = build_graph(ui)
    assert compiled is not None


def test_build_graph_with_entry_stage():
    """build_graph respects entry_stage parameter."""
    ui = UI(auto_approve=True)
    compiled = build_graph(ui, entry_stage="execute")
    assert compiled is not None


def test_build_graph_invalid_entry_defaults_to_discover():
    """Invalid entry_stage falls back to discover."""
    ui = UI(auto_approve=True)
    compiled = build_graph(ui, entry_stage="nonexistent")
    assert compiled is not None
