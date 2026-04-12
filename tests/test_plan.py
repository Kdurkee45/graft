"""Tests for graft.stages.plan."""

from graft.stages.plan import estimate_cost, plan_review_router


def test_estimate_cost_empty_plan():
    """Empty plan still has pipeline overhead."""
    low, high = estimate_cost([])
    assert low > 0
    assert high > low


def test_estimate_cost_with_units():
    """Cost scales with number and risk of units."""
    plan = [
        {"risk": "low"},
        {"risk": "medium"},
        {"risk": "high"},
    ]
    low, high = estimate_cost(plan)
    assert low > 3.0  # More than just overhead
    assert high > low


def test_plan_review_router_approved():
    assert plan_review_router({"plan_approved": True}) == "execute"


def test_plan_review_router_not_approved():
    assert plan_review_router({"plan_approved": False}) == "plan"
