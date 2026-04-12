"""Plan stage — turns the feature spec into an ordered list of atomic build units.

Each unit has dependencies, pattern references, acceptance criteria, and
co-located tests. Includes cost estimation and an optional human gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from graft.agent import run_agent
from graft.artifacts import mark_stage_complete, save_artifact
from graft.state import FeatureState
from graft.ui import UI

# Cost estimation constants per unit risk level
_COST_PER_UNIT = {
    "low": (0.30, 0.80),
    "medium": (0.60, 1.50),
    "high": (1.00, 3.00),
}
_PIPELINE_OVERHEAD = (3.00, 8.00)  # Discovery + Research + Grill + Verify


def estimate_cost(plan: list[dict]) -> tuple[float, float]:
    """Estimate the token cost range for executing a plan.

    Returns (low, high) in USD.
    """
    low = _PIPELINE_OVERHEAD[0]
    high = _PIPELINE_OVERHEAD[1]
    for unit in plan:
        risk = unit.get("risk", "medium")
        unit_low, unit_high = _COST_PER_UNIT.get(risk, _COST_PER_UNIT["medium"])
        low += unit_low
        high += unit_high
    return round(low, 2), round(high, 2)


SYSTEM_PROMPT = """\
You are a Staff Software Architect specializing in implementation planning.

You have full context:
- Codebase profile (architecture, patterns, conventions)
- Technical assessment (reuse analysis, gaps, edge cases)
- Feature specification (every decision documented with Q&A)

Your job: produce an ordered list of atomic build units that implement
the feature completely.

## Rules

1. Each build unit must be atomic — one focused implementation chunk.
2. Order by dependency — migrations before API, API before components,
   components before pages, shared before specific.
3. Co-locate tests with the code they test. If a unit creates code,
   it should also create its tests (set tests_included: true).
4. Every unit MUST have a pattern_reference — an existing file in the
   codebase that demonstrates the convention to follow. The agent
   implementing this unit will read this file first.
5. Tag each unit with service, risk, blast_radius, depends_on,
   acceptance_criteria, and tests_included.
6. Respect user constraints.
7. Keep units small enough that reverting one doesn't cascade.

## Output

Write `build_plan.json` to the working directory with this shape:

```json
{
  "plan_id": "feat_XXXXX",
  "feature_name": "...",
  "total_units": 12,
  "estimated_cost": "$8-15",
  "units": [
    {
      "unit_id": "feat_01",
      "title": "Create trades table migration",
      "description": "Detailed description of what to implement...",
      "category": "database|types|api|component|integration|page|tests",
      "service": "packages/db",
      "risk": "low|medium|high",
      "blast_radius": "1 file (new migration)",
      "depends_on": [],
      "acceptance_criteria": ["migration runs cleanly", "table created"],
      "pattern_reference": "supabase/migrations/20260301_create_draft.sql",
      "tests_included": false
    }
  ]
}
```

Prefer safe, small units over large risky ones.
Write the file to the current working directory.
"""


async def plan_node(state: FeatureState, ui: UI) -> dict[str, Any]:
    """LangGraph node: generate the build plan."""
    ui.stage_start("plan")
    repo_path = state["repo_path"]
    project_dir = state["project_dir"]
    feature_prompt = state.get("feature_prompt", "")
    codebase_profile = state.get("codebase_profile", {})
    technical_assessment = state.get("technical_assessment", {})
    feature_spec = state.get("feature_spec", {})
    constraints = state.get("constraints", [])
    max_units = state.get("max_units", 0)

    prompt_parts = [
        f"Create a build plan for this feature in the codebase at: {repo_path}",
        f"\nFEATURE: {feature_prompt}",
        f"\nCODEBASE PROFILE:\n{json.dumps(codebase_profile, indent=2)}",
        f"\nTECHNICAL ASSESSMENT:\n{json.dumps(technical_assessment, indent=2)}",
        f"\nFEATURE SPEC:\n{json.dumps(feature_spec, indent=2)}",
    ]
    if constraints:
        prompt_parts.append(f"\nCONSTRAINTS: {'; '.join(constraints)}")
    if max_units > 0:
        prompt_parts.append(f"\nMaximum build units: {max_units}")
    prompt_parts.append(
        "\nExplore the codebase to find the best pattern_reference for each unit. "
        "Read actual files to ensure references are valid."
    )

    await run_agent(
        persona="Staff Software Architect (Implementation Planner)",
        system_prompt=SYSTEM_PROMPT,
        user_prompt="\n".join(prompt_parts),
        cwd=repo_path,
        project_dir=project_dir,
        stage="plan",
        ui=ui,
        model=state.get("model"),
        max_turns=25,
        allowed_tools=["Read", "Bash", "Glob", "Grep"],
    )

    plan_path = Path(repo_path) / "build_plan.json"
    build_plan: list[dict] = []
    plan_raw: dict = {}
    if plan_path.exists():
        try:
            plan_raw = json.loads(plan_path.read_text())
            build_plan = plan_raw.get("units", [])
        except json.JSONDecodeError:
            ui.error("Failed to parse build_plan.json — using empty plan.")
    else:
        ui.error("Agent did not produce build_plan.json.")

    save_artifact(project_dir, "build_plan.json", json.dumps(plan_raw, indent=2))
    if plan_path.exists():
        plan_path.unlink()

    mark_stage_complete(project_dir, "plan")
    ui.stage_done("plan")

    return {
        "build_plan": build_plan,
        "current_stage": "plan",
    }


async def plan_review_node(state: FeatureState, ui: UI) -> dict[str, Any]:
    """Human gate: review and optionally adjust the build plan."""
    plan = state.get("build_plan", [])
    auto_approve = state.get("auto_approve", False)
    cost_low, cost_high = estimate_cost(plan)

    lines = [
        f"Total build units: {len(plan)}",
        f"Estimated cost: ${cost_low:.2f} – ${cost_high:.2f}",
        "",
    ]
    for i, unit in enumerate(plan, 1):
        risk_color = {"low": "green", "medium": "yellow", "high": "red"}.get(
            unit.get("risk", ""), "white"
        )
        tests_tag = " [dim]+tests[/dim]" if unit.get("tests_included") else ""
        lines.append(
            f"  {i}. [{risk_color}]{unit.get('risk', '?'):>6}[/{risk_color}] "
            f"{unit.get('unit_id', '?')}: {unit.get('title', 'Untitled')}{tests_tag}"
        )
        lines.append(
            f"         [{unit.get('category', '?')}] {unit.get('blast_radius', '?')}"
        )
        if unit.get("pattern_reference"):
            lines.append(f"         [dim]pattern: {unit['pattern_reference']}[/dim]")

    summary = "\n".join(lines)

    if auto_approve:
        ui.show_artifact("Build Plan", summary)
        ui.info("Plan auto-approved.")
        return {"plan_approved": True}

    approved, feedback = ui.prompt_plan_review(summary)

    if approved:
        ui.info("Plan approved — proceeding to execute.")
        return {"plan_approved": True}
    else:
        ui.info(f"Plan rejected — re-planning with feedback: {feedback}")
        return {"plan_approved": False, "plan_feedback": feedback}


def plan_review_router(state: FeatureState) -> str:
    """Route after plan review: execute if approved."""
    return "execute" if state.get("plan_approved") else "plan"
