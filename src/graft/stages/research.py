"""Research stage — figures out what the feature needs given what exists.

Takes the codebase profile from Discover and the feature prompt,
produces a technical assessment with reuse analysis, gap analysis,
pattern matching, edge cases, and open questions for the Grill phase.
"""

from __future__ import annotations

import json
from typing import Any

from graft.agent import run_agent
from graft.artifacts import save_artifact
from graft.stages._helpers import (
    async_read_text,
    cleanup_artifacts,
    find_artifact,
    resolve_stage_cwd,
    stage_node,
)
from graft.state import FeatureState
from graft.ui import UI

SYSTEM_PROMPT = """You are a Staff Software Architect specializing in feature design for
existing applications.

You have two inputs:
1. A codebase profile (architecture, patterns, data model, conventions)
2. A feature description from a human

Your job: figure out what this feature actually needs — technically and
architecturally. Bridge "what the human wants" and "what the codebase needs."

## What You Must Produce

1. **Reuse Analysis**
   - What existing tables, models, or types can the feature leverage?
   - What existing components or UI patterns can be reused?
   - What existing API endpoints or services can be extended?
   - What existing utility functions or helpers apply?

2. **Gap Analysis**
   - What new database tables, columns, or migrations are needed?
   - What new API endpoints or server actions are needed?
   - What new components need to be built?
   - What new types or interfaces need to be defined?

3. **Pattern Matching**
   - Identify the closest existing feature in the codebase
   - Document which patterns from that feature should be followed
   - Flag any cases where existing patterns won't work and alternatives
     are needed

4. **Edge Case Identification**
   - Concurrency issues (what if two users do this simultaneously?)
   - Permission boundaries (who can do what?)
   - Data validation requirements
   - Error states and recovery
   - Performance implications (will this feature add N+1 queries?)

5. **Integration Points**
   - Which existing features does this interact with?
   - Does it need real-time updates?
   - Does it need notifications?
   - Does it affect existing navigation, routing, or layout?

6. **Open Questions**
   - Questions that CANNOT be answered by reading the codebase
   - Questions that require human INTENT (behavior, UX, scope)
   - These feed directly into the Grill stage

## Output

Write two files to the working directory:

### 1. research_report.md
A narrative document explaining your findings.

### 2. technical_assessment.json
Structured data with this shape:

```json
{
  "feature_prompt": "...",
  "reusable_components": [
    {"path": "src/...", "reason": "..."}
  ],
  "new_artifacts_needed": [
    {"type": "table|component|api|type|migration", "name": "...", "description": "..."}
  ],
  "pattern_to_follow": "src/features/draft/ (closest existing feature)",
  "edge_cases": [...],
  "integration_points": [...],
  "open_questions": [
    {"question": "...", "category": "intent|edge_case|preference|prioritization",
     "recommended_answer": "..."}
  ]
}
```

CRITICAL: Every open_question MUST have a recommended_answer based on
codebase patterns, industry norms, or technical reasoning. The human
should be able to say "yes" to most of them.

Write both files to the current working directory.
"""


def _build_research_prompt(
    repo_path: str,
    feature_prompt: str,
    codebase_profile: dict,
    constraints: list[str],
) -> str:
    """Assemble the user prompt for the research stage."""
    prompt_parts = [
        "Research what is needed to build this feature"
        f" into the codebase at: {repo_path}",
        f"\nFEATURE: {feature_prompt}",
        f"\nCODEBASE PROFILE:\n{json.dumps(codebase_profile, indent=2)}",
    ]
    if constraints:
        prompt_parts.append(f"\nCONSTRAINTS: {'; '.join(constraints)}")
    prompt_parts.append(
        "\nExplore the actual codebase to validate and extend the profile. "
        "Read key files. Understand the real patterns, not just the metadata."
    )
    return "\n".join(prompt_parts)


@stage_node("research")
async def research_node(state: FeatureState, ui: UI) -> dict[str, Any]:
    """LangGraph node: research what the feature needs given the codebase."""
    repo_path = state["repo_path"]
    project_dir = state["project_dir"]
    feature_prompt = state.get("feature_prompt", "")
    codebase_profile = state.get("codebase_profile", {})
    scope_path = state.get("scope_path", "")
    constraints = state.get("constraints", [])

    research_cwd = resolve_stage_cwd(repo_path, scope_path)
    user_prompt = _build_research_prompt(
        repo_path, feature_prompt, codebase_profile, constraints
    )

    result = await run_agent(
        persona="Staff Software Architect (Feature Specialist)",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        cwd=research_cwd,
        project_dir=project_dir,
        stage="research",
        ui=ui,
        model=state.get("model"),
        max_turns=30,
        allowed_tools=["Bash", "Read", "Glob", "Grep"],
    )

    # Read agent outputs
    report_path = find_artifact("research_report.md", research_cwd, repo_path)
    assessment_path = find_artifact(
        "technical_assessment.json", research_cwd, repo_path
    )

    research_report = (
        (await async_read_text(report_path)) if report_path.exists() else result.text
    )
    save_artifact(project_dir, "research_report.md", research_report)

    technical_assessment: dict = {}
    if assessment_path.exists():
        try:
            technical_assessment = json.loads(await async_read_text(assessment_path))
        except json.JSONDecodeError:
            ui.error("Failed to parse technical_assessment.json from agent output.")

    save_artifact(
        project_dir,
        "technical_assessment.json",
        json.dumps(technical_assessment, indent=2),
    )

    open_questions = technical_assessment.get("open_questions", [])
    if open_questions:
        ui.info(
            f"Research identified {len(open_questions)}"
            " open question(s) for the Grill phase."
        )

    # Clean up
    cleanup_artifacts(
        research_cwd,
        repo_path,
        ["research_report.md", "technical_assessment.json"],
    )

    return {
        "technical_assessment": technical_assessment,
        "research_report": research_report,
    }
