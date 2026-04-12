"""LangGraph state machine — orchestrates the 6-stage feature pipeline.

    Discover → Research → Grill → [Grill↔Research loop]
    → Plan → [Plan Review] → Execute → Verify

The Grill→Research loop-back is conditional: only triggers if Grill reveals
that a fundamental technical assumption from Research was wrong.
"""

from __future__ import annotations

import functools
from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from graft.stages.discover import discover_node
from graft.stages.execute import execute_node
from graft.stages.grill import grill_node, grill_router
from graft.stages.plan import plan_node, plan_review_node, plan_review_router
from graft.stages.research import research_node
from graft.stages.verify import verify_node
from graft.state import FeatureState
from graft.ui import UI


def _wrap(fn, ui: UI):
    """Wrap a node function so it receives the shared UI instance."""

    @functools.wraps(fn)
    async def wrapper(state: FeatureState) -> dict[str, Any]:
        return cast(dict[str, Any], await fn(state, ui))

    return wrapper


ORDERED_STAGES = ["discover", "research", "grill", "plan", "execute", "verify"]

_NODE_FACTORIES = {
    "discover": lambda ui: [("discover", discover_node)],
    "research": lambda ui: [("research", research_node)],
    "grill": lambda ui: [("grill", grill_node)],
    "plan": lambda ui: [("plan", plan_node), ("plan_review", plan_review_node)],
    "execute": lambda ui: [("execute", execute_node)],
    "verify": lambda ui: [("verify", verify_node)],
}

# Edge configuration: (source, target, required_stages)
_SIMPLE_EDGES: list[tuple[str, str, frozenset[str]]] = [
    ("discover", "research", frozenset({"discover", "research"})),
    ("research", "grill", frozenset({"research", "grill"})),
    ("plan", "plan_review", frozenset({"plan"})),
    ("execute", "verify", frozenset({"execute", "verify"})),
    ("verify", END, frozenset({"verify"})),
]

# Conditional edge configuration: (source, router, mapping, required_stages)
_CONDITIONAL_EDGES: list[tuple[str, Any, dict[str, str], frozenset[str]]] = [
    (
        "grill",
        grill_router,
        {"plan": "plan", "research": "research"},
        frozenset({"grill", "plan"}),
    ),
    (
        "plan_review",
        plan_review_router,
        {"execute": "execute", "plan": "plan"},
        frozenset({"plan", "execute"}),
    ),
]


def build_graph(ui: UI, *, entry_stage: str = "discover") -> CompiledStateGraph:
    """Construct and compile the feature pipeline graph.

    ``entry_stage`` controls where the graph starts (default: discover).
    Used by ``resume`` to skip completed stages.
    """
    if entry_stage not in ORDERED_STAGES:
        entry_stage = "discover"

    entry_idx = ORDERED_STAGES.index(entry_stage)
    active = frozenset(ORDERED_STAGES[entry_idx:])

    graph = StateGraph(FeatureState)

    # Add nodes for active stages
    for stage in ORDERED_STAGES:
        if stage in active:
            for name, fn in _NODE_FACTORIES[stage](ui):
                graph.add_node(name, _wrap(fn, ui))

    # Wire entry edge
    graph.add_edge(START, entry_stage)

    # Wire simple edges
    for source, target, required in _SIMPLE_EDGES:
        if required <= active:
            graph.add_edge(source, target)

    # Wire conditional edges
    for source, router, mapping, required in _CONDITIONAL_EDGES:
        if required <= active:
            graph.add_conditional_edges(source, router, mapping)

    return graph.compile()
