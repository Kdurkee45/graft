"""Grill stage — structured interrogation to extract human intent.

Walks through open questions from Research one at a time, with recommended
answers. Compiles all decisions into a feature_spec.json that serves as
the complete, unambiguous specification.

This is the primary human touchpoint — efficient Q&A, not document writing.
"""

from __future__ import annotations

import json

from graft.agent import run_agent
from graft.artifacts import load_artifact, mark_stage_complete, save_artifact
from graft.state import FeatureState
from graft.ui import UI

COMPILE_SYSTEM_PROMPT = """You are a Principal Product Architect. You have just completed a structured
interrogation with a human about a feature they want to build.

You have:
1. The original feature prompt
2. A codebase profile (architecture, patterns, conventions)
3. A technical assessment (reuse analysis, gaps, edge cases)
4. A complete transcript of the Q&A session with all decisions

Your job: compile everything into a definitive feature specification.

## Output

Write `feature_spec.json` to the working directory with this shape:

```json
{
  "feature_name": "Trade System",
  "feature_prompt": "original prompt...",
  "decisions": [
    {
      "question": "...",
      "recommended": "...",
      "answer": "...",
      "category": "intent|edge_case|preference|prioritization",
      "implications": ["needs timer UI", "needs cron job"]
    }
  ],
  "scope": {
    "mvp": ["item1", "item2"],
    "follow_up": ["item3", "item4"]
  },
  "constraints": [
    "Must follow existing Supabase RLS patterns",
    "Must work on both web and mobile"
  ],
  "technical_notes": [
    "Closest pattern: src/features/draft/",
    "Reuse PlayerCard component for trade UI"
  ]
}
```

Be precise. Every decision must be documented. Every implication must be
noted. This spec drives the entire Plan and Execute stages — ambiguity
here means bugs later.

Write the file to the current working directory.
"""


async def grill_node(state: FeatureState, ui: UI) -> dict:
    """LangGraph node: interrogate the human for intent, preferences, edge cases."""
    ui.stage_start("grill")
    repo_path = state["repo_path"]
    project_dir = state["project_dir"]
    feature_prompt = state.get("feature_prompt", "")
    codebase_profile = state.get("codebase_profile", {})
    technical_assessment = state.get("technical_assessment", {})
    constraints = state.get("constraints", [])

    # Get open questions from Research
    open_questions = technical_assessment.get("open_questions", [])

    # If Research didn't produce questions, generate them via agent
    if not open_questions:
        ui.info(
            "No open questions from Research — generating questions from context..."
        )
        open_questions = await _generate_questions(
            repo_path,
            project_dir,
            feature_prompt,
            codebase_profile,
            technical_assessment,
            ui,
            state.get("model"),
        )

    # Walk through questions one at a time
    transcript_lines: list[str] = []
    decisions: list[dict] = []

    for i, q in enumerate(open_questions, 1):
        question = q.get("question", q) if isinstance(q, dict) else str(q)
        recommended = (
            q.get("recommended_answer", "No recommendation")
            if isinstance(q, dict)
            else "No recommendation"
        )
        category = q.get("category", "intent") if isinstance(q, dict) else "intent"

        answer = ui.grill_question(question, recommended, category, i)

        transcript_lines.append(f"Q{i} [{category}]: {question}")
        transcript_lines.append(f"  Recommended: {recommended}")
        transcript_lines.append(f"  Answer: {answer}")
        transcript_lines.append("")

        decisions.append(
            {
                "question": question,
                "recommended": recommended,
                "answer": answer,
                "category": category,
            }
        )

    grill_transcript = "\n".join(transcript_lines)
    save_artifact(project_dir, "grill_transcript.md", grill_transcript)

    # Compile decisions into feature spec using an agent
    ui.info("Compiling feature specification from decisions...")

    compile_prompt = (
        f"Compile the feature specification.\n\n"
        f"FEATURE PROMPT: {feature_prompt}\n\n"
        f"CODEBASE PROFILE:\n{json.dumps(codebase_profile, indent=2)}\n\n"
        f"TECHNICAL ASSESSMENT:\n{json.dumps(technical_assessment, indent=2)}\n\n"
        f"GRILL TRANSCRIPT:\n{grill_transcript}\n\n"
        f"CONSTRAINTS: {'; '.join(constraints) if constraints else 'None'}\n\n"
        f"Write feature_spec.json to the working directory."
    )

    from pathlib import Path

    await run_agent(
        persona="Principal Product Architect",
        system_prompt=COMPILE_SYSTEM_PROMPT,
        user_prompt=compile_prompt,
        cwd=repo_path,
        project_dir=project_dir,
        stage="grill_compile",
        ui=ui,
        model=state.get("model"),
        max_turns=10,
        allowed_tools=["Read", "Write", "Bash"],
    )

    # Read compiled spec
    spec_path = Path(repo_path) / "feature_spec.json"
    feature_spec: dict = {}
    if spec_path.exists():
        try:
            feature_spec = json.loads(spec_path.read_text())
        except json.JSONDecodeError:
            ui.error("Failed to parse feature_spec.json.")

    save_artifact(project_dir, "feature_spec.json", json.dumps(feature_spec, indent=2))
    if spec_path.exists():
        spec_path.unlink()

    # Check if Research needs a redo (rare — only if a fundamental assumption was wrong)
    research_redo = feature_spec.get("research_redo_needed", False)
    if research_redo:
        ui.info(
            "Grill revealed a fundamental assumption change — looping back to Research."
        )

    mark_stage_complete(project_dir, "grill")
    ui.stage_done("grill")

    return {
        "feature_spec": feature_spec,
        "grill_transcript": grill_transcript,
        "grill_complete": True,
        "research_redo_needed": research_redo,
        "current_stage": "grill",
    }


def grill_router(state: FeatureState) -> str:
    """Route after Grill: loop back to Research if needed, otherwise proceed to Plan."""
    if state.get("research_redo_needed", False):
        return "research"
    return "plan"


async def _generate_questions(
    repo_path: str,
    project_dir: str,
    feature_prompt: str,
    codebase_profile: dict,
    technical_assessment: dict,
    ui: UI,
    model: str | None,
) -> list[dict]:
    """Generate open questions when Research didn't produce them."""
    from pathlib import Path

    gen_prompt = (
        f"Generate focused questions for building this feature.\n\n"
        f"FEATURE: {feature_prompt}\n\n"
        f"CODEBASE PROFILE:\n{json.dumps(codebase_profile, indent=2)}\n\n"
        f"TECHNICAL ASSESSMENT:\n{json.dumps(technical_assessment, indent=2)}\n\n"
        f"Write a JSON file called `open_questions.json` with an array of objects, "
        f"each having: question, category (intent/edge_case/preference/prioritization), "
        f"and recommended_answer.\n\n"
        f"Focus on questions that REQUIRE human intent — things the code can't answer. "
        f"Provide a recommended answer for each. 5-15 questions is typical."
    )

    generate_system = (
        "You are a Principal Product Interrogator. Generate focused questions "
        "about a feature that require human intent to answer. Each question must "
        "have a recommended answer based on codebase patterns and industry norms."
    )

    await run_agent(
        persona="Principal Product Interrogator",
        system_prompt=generate_system,
        user_prompt=gen_prompt,
        cwd=repo_path,
        project_dir=project_dir,
        stage="grill_generate",
        ui=ui,
        model=model,
        max_turns=10,
        allowed_tools=["Read", "Write", "Bash", "Glob", "Grep"],
    )

    questions_path = Path(repo_path) / "open_questions.json"
    if questions_path.exists():
        try:
            questions = json.loads(questions_path.read_text())
            questions_path.unlink()
            return questions if isinstance(questions, list) else []
        except json.JSONDecodeError:
            pass
    return []
