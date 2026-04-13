"""Grill stage — adaptive conversational interrogation for feature building.

A real-time conversation loop where the agent drives the interrogation,
adapting each question based on the user's answers AND the codebase context
from the Discover and Research stages. The agent asks one question at a time,
digs deeper on vague answers, skips questions already answered by codebase
analysis, and stops when it has enough context to plan the feature.

In auto-approve mode, the agent's recommended answer is used for every
question. The conversation still adapts — recommendations are informed by
codebase patterns and earlier answers.
"""

from __future__ import annotations

import json
from typing import Any

from graft.agent import run_agent
from graft.artifacts import mark_stage_complete, save_artifact
from graft.stages._helpers import async_read_text, cleanup_artifacts, find_artifact
from graft.state import FeatureState
from graft.ui import UI

MAX_QUESTIONS = 25

CONVERSATION_SYSTEM_PROMPT = """\
You are a Principal Product Interrogator conducting a structured discovery
conversation about a feature to add to an existing codebase. You have the
codebase profile and technical assessment from prior analysis.

Your goal: understand exactly what the human wants to build, how it should
integrate with the existing codebase, and what the edge cases are.

RULES:
1. Ask ONE question at a time. Wait for the answer before asking the next.
2. Every question MUST include a recommended_answer — your best guess
   based on the codebase patterns, technical assessment, and what you've
   heard so far.
3. Adapt your questions to the user's answers AND the codebase context.
   If the codebase uses Supabase, recommend Supabase patterns. If they
   say "keep it simple," don't ask about complex integrations.
4. Dig deeper when answers are vague. "It should notify users" → ask
   about notification channels, triggers, frequency, opt-out.
5. Skip questions when the codebase analysis already provides the answer.
   If Discover found the auth pattern, don't ask "how should auth work?"
6. Track what you know and what you don't. Stop when you have enough
   to plan and build the feature confidently.
7. Think in layers:
   - Layer 1: Core intent, user-facing behavior, scope boundaries
   - Layer 2: Data model changes, API surface, integration points
   - Layer 3: Edge cases, error handling, UI specifics
   You don't need Layer 3 for every aspect — only critical paths.
8. Be conversational, not robotic. Acknowledge the user's answers.
   Reference the codebase context when relevant ("Since the codebase
   already uses X, I'd recommend...").

RESPONSE FORMAT:
You MUST respond with a single JSON object and nothing else.

When you need to ask a question:
{
  "status": "question",
  "question": "Your question here",
  "category": "intent|scope|data_model|integration|edge_case|preference|prioritization|ui|api|auth|workflow",
  "recommended_answer": "Your recommendation based on codebase patterns and conversation",
  "why_asking": "Brief explanation of why this matters for the feature"
}

When you have enough context:
{
  "status": "done",
  "summary": "Here's what I understand about the feature: ...",
  "assumptions": ["Assumption 1 — filling a gap", "..."],
  "confidence": "high|medium|low"
}

Do NOT ask more than 25 questions. Do NOT include text outside the JSON.
"""

COMPILE_SYSTEM_PROMPT = """\
You are a Principal Product Architect. You have just completed a structured
interrogation with a human about a feature they want to build on an existing
codebase.

You have:
1. The original feature prompt
2. A codebase profile (architecture, patterns, conventions)
3. A technical assessment (reuse analysis, gaps, edge cases)
4. A complete transcript of the adaptive Q&A session with all decisions

Your job: compile everything into a definitive feature specification.

## Output

Write `feature_spec.json` to the working directory with this shape:

```json
{
  "feature_name": "Trade System",
  "feature_prompt": "original prompt...",
  "summary": "one-paragraph description of what we're building",
  "decisions": [
    {
      "question": "...",
      "answer": "...",
      "category": "intent|edge_case|preference|prioritization",
      "implications": ["needs timer UI", "needs cron job"]
    }
  ],
  "assumptions": [
    "Assumed X since user didn't specify",
    "..."
  ],
  "confidence": "high|medium|low",
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

Be precise. Every decision must be documented. Every assumption must be
noted. This spec drives the entire Plan and Execute stages — ambiguity
here means bugs later.

Write the file to the current working directory.
"""


def _parse_agent_response(text: str) -> dict:
    """Extract JSON from the agent's response text."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for marker in ["```json", "```"]:
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.index("```", start) if "```" in text[start:] else len(text)
            try:
                return json.loads(text[start:end].strip())
            except json.JSONDecodeError:
                pass

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace : last_brace + 1])
        except json.JSONDecodeError:
            pass

    return {"status": "error", "message": "Failed to parse agent response"}


def _build_history_prompt(
    feature_prompt: str,
    codebase_profile: dict,
    technical_assessment: dict,
    constraints: list[str],
    history: list[dict],
) -> str:
    """Build the prompt for the next conversation turn with full context."""
    parts = [
        f"FEATURE: {feature_prompt}",
        f"\nCODEBASE PROFILE:\n{json.dumps(codebase_profile, indent=2)}",
        f"\nTECHNICAL ASSESSMENT:\n{json.dumps(technical_assessment, indent=2)}",
    ]
    if constraints:
        parts.append(f"\nCONSTRAINTS: {'; '.join(constraints)}")

    if history:
        parts.append("\nCONVERSATION SO FAR:")
        for entry in history:
            if entry["role"] == "agent":
                data = entry["data"]
                parts.append(
                    f"\nQ{entry['turn']} [{data.get('category', '?')}]: {data.get('question', '?')}"
                )
                parts.append(f"  Recommended: {data.get('recommended_answer', '?')}")
                parts.append(f"  Why asked: {data.get('why_asking', '?')}")
            elif entry["role"] == "user":
                parts.append(f"  User answered: {entry['answer']}")

    parts.append("\nAsk your next question, or respond with status 'done' if you have enough context.")
    return "\n".join(parts)


async def _ask_one_question(
    feature_prompt: str,
    codebase_profile: dict,
    technical_assessment: dict,
    constraints: list[str],
    history: list[dict],
    repo_path: str,
    project_dir: str,
    ui: UI,
    model: str | None,
    turn: int,
) -> dict:
    """Run one conversation turn — agent produces the next question or done signal."""
    prompt = _build_history_prompt(
        feature_prompt, codebase_profile, technical_assessment, constraints, history,
    )

    result = await run_agent(
        persona="Principal Product Interrogator",
        system_prompt=CONVERSATION_SYSTEM_PROMPT,
        user_prompt=prompt,
        cwd=repo_path,
        project_dir=project_dir,
        stage=f"grill_q{turn}",
        ui=ui,
        model=model,
        max_turns=3,
        allowed_tools=["Read"],
    )

    return _parse_agent_response(result.text)


async def grill_node(state: FeatureState, ui: UI) -> dict[str, Any]:
    """LangGraph node: adaptive conversational interrogation."""
    ui.stage_start("grill")
    repo_path = state["repo_path"]
    project_dir = state["project_dir"]
    feature_prompt = state.get("feature_prompt", "")
    codebase_profile = state.get("codebase_profile", {})
    technical_assessment = state.get("technical_assessment", {})
    constraints = state.get("constraints", [])
    auto_approve = state.get("auto_approve", False)

    history: list[dict] = []
    transcript_lines: list[str] = [
        "# Grill Transcript",
        "",
        f"**Feature:** {feature_prompt}",
        f"**Mode:** {'auto-approve' if auto_approve else 'interactive'}",
        "",
        "---",
        "",
    ]
    decisions: list[dict] = []
    assumptions: list[str] = []

    for turn in range(1, MAX_QUESTIONS + 1):
        response = await _ask_one_question(
            feature_prompt, codebase_profile, technical_assessment, constraints,
            history, repo_path, project_dir, ui, state.get("model"), turn,
        )

        if response.get("status") == "error":
            ui.error(f"Grill agent returned unparseable response on turn {turn}. Wrapping up.")
            break

        if response.get("status") == "done":
            assumptions = response.get("assumptions", [])
            confidence = response.get("confidence", "medium")
            transcript_lines.append(f"**Agent concluded after {turn - 1} questions.**")
            transcript_lines.append(f"")
            transcript_lines.append(f"**Summary:** {response.get('summary', '')}")
            transcript_lines.append(f"**Confidence:** {confidence}")
            if assumptions:
                transcript_lines.append("")
                transcript_lines.append("**Assumptions:**")
                for a in assumptions:
                    transcript_lines.append(f"- {a}")
            ui.info(f"Grill complete after {turn - 1} questions (confidence: {confidence}).")
            break

        question = response.get("question", "")
        recommended = response.get("recommended_answer", "")
        category = response.get("category", "intent")
        why_asking = response.get("why_asking", "")

        if not question:
            ui.error(f"Grill agent returned empty question on turn {turn}. Wrapping up.")
            break

        answer = ui.grill_question(
            question=question,
            recommended=recommended,
            category=category,
            number=turn,
            why_asking=why_asking,
        )

        if answer.lower() == "done":
            ui.info("User requested early exit — agent will fill remaining gaps with assumptions.")
            history.append({"role": "agent", "data": response, "turn": turn})
            history.append({"role": "user", "answer": "I'm done answering questions. Fill in any remaining gaps with your best judgment and wrap up."})
            wrap_up = await _ask_one_question(
                feature_prompt, codebase_profile, technical_assessment, constraints,
                history, repo_path, project_dir, ui, state.get("model"), turn + 1,
            )
            if wrap_up.get("status") == "done":
                assumptions = wrap_up.get("assumptions", [])
            transcript_lines.append(f"**User ended conversation at Q{turn}. Agent filled gaps.**")
            if assumptions:
                transcript_lines.append("")
                transcript_lines.append("**Assumptions:**")
                for a in assumptions:
                    transcript_lines.append(f"- {a}")
            break

        history.append({"role": "agent", "data": response, "turn": turn})
        history.append({"role": "user", "answer": answer})

        transcript_lines.append(f"### Q{turn} [{category}]")
        transcript_lines.append("")
        transcript_lines.append(f"**{question}**")
        transcript_lines.append("")
        transcript_lines.append(f"*Why I'm asking: {why_asking}*")
        transcript_lines.append("")
        transcript_lines.append(f"Recommended: {recommended}")
        transcript_lines.append("")
        transcript_lines.append(f"**Answer:** {answer}")
        transcript_lines.append("")

        decisions.append({
            "question": question,
            "recommended": recommended,
            "answer": answer,
            "category": category,
            "why_asking": why_asking,
        })

    grill_transcript = "\n".join(transcript_lines)
    save_artifact(project_dir, "grill_transcript.md", grill_transcript)

    # Compile decisions into feature spec
    ui.info("Compiling feature specification from conversation...")

    compile_prompt = (
        f"Compile the feature specification.\n\n"
        f"FEATURE PROMPT: {feature_prompt}\n\n"
        f"CODEBASE PROFILE:\n{json.dumps(codebase_profile, indent=2)}\n\n"
        f"TECHNICAL ASSESSMENT:\n{json.dumps(technical_assessment, indent=2)}\n\n"
        f"CONVERSATION TRANSCRIPT:\n{grill_transcript}\n\n"
        f"ASSUMPTIONS:\n{json.dumps(assumptions, indent=2)}\n\n"
        f"CONSTRAINTS: {'; '.join(constraints) if constraints else 'None'}\n\n"
        f"Write feature_spec.json to the working directory."
    )

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
    spec_path = find_artifact("feature_spec.json", repo_path, repo_path)
    feature_spec: dict = {}
    if spec_path.exists():
        try:
            feature_spec = json.loads(await async_read_text(spec_path))
        except json.JSONDecodeError:
            ui.error("Failed to parse feature_spec.json.")

    save_artifact(project_dir, "feature_spec.json", json.dumps(feature_spec, indent=2))
    cleanup_artifacts(repo_path, repo_path, ["feature_spec.json"])

    research_redo = feature_spec.get("research_redo_needed", False)
    if research_redo:
        ui.info("Grill revealed a fundamental assumption change — looping back to Research.")

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
