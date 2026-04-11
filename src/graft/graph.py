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
        result: dict[str, Any] = await fn(state, ui)
        return result

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

    # Discover → Research
    if "discover" in active and "research" in active:
        graph.add_edge("discover", "research")

    # Research → Grill
    if "research" in active and "grill" in active:
        graph.add_edge("research", "grill")

    # Grill → conditional: Plan or loop back to Research
    if "grill" in active and "plan" in active:
        graph.add_conditional_edges(
            "grill",
            grill_router,
            {"plan": "plan", "research": "research"},
        )

    # Plan → Plan Review → Execute (with conditional routing)
    if "plan" in active:
        graph.add_edge("plan", "plan_review")
        if "execute" in active:
            graph.add_conditional_edges(
                "plan_review",
                plan_review_router,
                {"execute": "execute", "plan": "plan"},
            )

    # Execute → Verify
    if "execute" in active and "verify" in active:
        graph.add_edge("execute", "verify")

    # Verify → END
    if "verify" in active:
        graph.add_edge("verify", END)

    return graph.compile()
