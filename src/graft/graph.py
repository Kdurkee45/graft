"""LangGraph state machine — orchestrates the 6-stage feature pipeline.

    Discover → Research → Grill → [Grill↔Research loop] → Plan → [Plan Review] → Execute → Verify

The Grill→Research loop-back is conditional: only triggers if Grill reveals
that a fundamental technical assumption from Research was wrong.
"""

from __future__ import annotations

import functools
from typing import Any

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
        return await fn(state, ui)

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

# Declarative edge definitions: (required_active_stages, source, target_or_router[, route_map])
# Tuples of length 3 are simple edges; length 4 are conditional edges.
_EDGE_DEFS: list[tuple] = [
    # Discover → Research
    ({"discover", "research"}, "discover", "research"),
    # Research → Grill
    ({"research", "grill"}, "research", "grill"),
    # Grill → conditional: Plan or loop back to Research
    (
        {"grill", "plan"},
        "grill",
        grill_router,
        {"plan": "plan", "research": "research"},
    ),
    # Plan → Plan Review
    ({"plan"}, "plan", "plan_review"),
    # Plan Review → conditional: Execute or revise Plan
    (
        {"plan", "execute"},
        "plan_review",
        plan_review_router,
        {"execute": "execute", "plan": "plan"},
    ),
    # Execute → Verify
    ({"execute", "verify"}, "execute", "verify"),
    # Verify → END
    ({"verify"}, "verify", END),
]


def _wire_edges(graph: StateGraph, active: set[str]) -> None:
    """Add edges to *graph* for all stages present in *active*."""
    for defn in _EDGE_DEFS:
        required = defn[0]
        if not required.issubset(active):
            continue
        if len(defn) == 3:
            # Simple edge: (required, source, target)
            graph.add_edge(defn[1], defn[2])
        else:
            # Conditional edge: (required, source, router, route_map)
            graph.add_conditional_edges(defn[1], defn[2], defn[3])


def build_graph(ui: UI, *, entry_stage: str = "discover") -> CompiledStateGraph:
    """Construct and compile the feature pipeline graph.

    ``entry_stage`` controls where the graph starts (default: discover).
    Used by ``resume`` to skip completed stages.
    """
    if entry_stage not in ORDERED_STAGES:
        entry_stage = "discover"

    entry_idx = ORDERED_STAGES.index(entry_stage)
    active = set(ORDERED_STAGES[entry_idx:])

    graph = StateGraph(FeatureState)

    # Add nodes for active stages
    for stage in ORDERED_STAGES:
        if stage in active:
            for name, fn in _NODE_FACTORIES[stage](ui):
                graph.add_node(name, _wrap(fn, ui))

    # Wire edges
    graph.add_edge(START, entry_stage)
    _wire_edges(graph, active)

    return graph.compile()
