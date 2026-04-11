"""Tests for graft.stages.execute."""

from graft.stages.execute import _order_by_dependencies


def test_order_by_dependencies_no_deps():
    """Units without dependencies maintain order."""
    plan = [
        {"unit_id": "a"},
        {"unit_id": "b"},
        {"unit_id": "c"},
    ]
    result = _order_by_dependencies(plan)
    assert [u["unit_id"] for u in result] == ["a", "b", "c"]


def test_order_by_dependencies_simple_chain():
    """Units with linear dependencies are ordered correctly."""
    plan = [
        {"unit_id": "c", "depends_on": ["b"]},
        {"unit_id": "a", "depends_on": []},
        {"unit_id": "b", "depends_on": ["a"]},
    ]
    result = _order_by_dependencies(plan)
    ids = [u["unit_id"] for u in result]
    assert ids.index("a") < ids.index("b") < ids.index("c")


def test_order_by_dependencies_circular():
    """Circular dependencies don't infinite loop — remaining units appended."""
    plan = [
        {"unit_id": "a", "depends_on": ["b"]},
        {"unit_id": "b", "depends_on": ["a"]},
    ]
    result = _order_by_dependencies(plan)
    assert len(result) == 2


def test_order_by_dependencies_multiple_deps():
    """Unit with multiple dependencies waits for all."""
    plan = [
        {"unit_id": "c", "depends_on": ["a", "b"]},
        {"unit_id": "a", "depends_on": []},
        {"unit_id": "b", "depends_on": []},
    ]
    result = _order_by_dependencies(plan)
    ids = [u["unit_id"] for u in result]
    assert ids.index("c") > ids.index("a")
    assert ids.index("c") > ids.index("b")
