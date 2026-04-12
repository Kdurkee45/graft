"""Graft CLI — the user-facing interface to the feature factory."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from graft.artifacts import create_project, list_projects, load_artifact
from graft.config import Settings
from graft.graph import build_graph
from graft.state import FeatureState
from graft.ui import UI

app = typer.Typer(
    name="graft",
    help=(
        "AI-powered feature building — discover, research,"
        " grill, plan, execute, verify, and open a PR."
    ),
    add_completion=False,
)


@app.command()
def build(
    repo_path: str = typer.Argument(..., help="Path to the repository"),
    feature_prompt: str = typer.Argument(
        ..., help="Description of the feature to build"
    ),
    path: str = typer.Option(
        "", "--path", "-p", help="Scope to a subdirectory (monorepo support)"
    ),
    constraint: list[str] = typer.Option(
        [], "--constraint", "-c", help="Constraints (repeatable)"
    ),
    max_units: int = typer.Option(
        0, "--max-units", help="Max build units (0 = unlimited)"
    ),
    auto_approve: bool = typer.Option(
        False, "--auto-approve", help="Skip the plan review gate"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show agent message stream"
    ),
) -> None:
    """Build a feature into an existing codebase."""
    settings = Settings.load()
    ui = UI(auto_approve=auto_approve, verbose=verbose)

    # Resolve repo path
    resolved_path = str(Path(repo_path).expanduser().resolve())
    if not Path(resolved_path).exists():
        ui.error(f"Repository not found: {resolved_path}")
        raise typer.Exit(1)

    # Resolve --path scope
    scope_path = ""
    if path:
        scoped = Path(resolved_path) / path
        if not scoped.exists():
            ui.error(f"Scope path not found: {scoped}")
            raise typer.Exit(1)
        scope_path = path
        ui.info(f"Scoped to: {path}")

    project_id, project_dir = create_project(
        settings.projects_root, resolved_path, feature_prompt
    )
    ui.banner(resolved_path, project_id, feature_prompt)

    initial_state: FeatureState = {
        "repo_path": resolved_path,
        "project_id": project_id,
        "project_dir": str(project_dir),
        "feature_prompt": feature_prompt,
        "scope_path": scope_path,
        "constraints": list(constraint),
        "max_units": max_units,
        "auto_approve": auto_approve,
        "codebase_profile": {},
        "discovery_report": "",
        "technical_assessment": {},
        "research_report": "",
        "feature_spec": {},
        "grill_transcript": "",
        "build_plan": [],
        "feature_report": "",
        "current_unit_index": 0,
        "units_completed": [],
        "units_reverted": [],
        "units_skipped": [],
        "plan_approved": False,
        "grill_complete": False,
        "research_redo_needed": False,
        "feature_branch": f"feature/{project_id}",
        "pr_url": "",
        "model": settings.model,
        "max_agent_turns": settings.max_agent_turns,
        "current_stage": "",
    }

    compiled = build_graph(ui)
    result = asyncio.run(compiled.ainvoke(initial_state))

    pr_url = result.get("pr_url", "")
    if pr_url:
        ui.info(f"PR opened: {pr_url}")
    else:
        ui.info("Feature build complete. Review the results and open a PR manually.")

    ui.info(f"Session artifacts: {project_dir}")


@app.command()
def resume(
    project_path: str = typer.Argument(
        ..., help="Path to the session directory (e.g. ~/.graft/projects/feat_XXXXX)"
    ),
    from_stage: str = typer.Option(
        "execute",
        "--from",
        help="Stage to resume from: discover, research, grill, plan, execute, verify",
    ),
    auto_approve: bool = typer.Option(
        False, "--auto-approve", help="Skip the plan review gate"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show agent message stream"
    ),
) -> None:
    """Resume a feature session from a specific stage."""
    settings = Settings.load()
    ui = UI(auto_approve=auto_approve, verbose=verbose)

    project_dir = Path(project_path).expanduser()
    if not project_dir.exists():
        ui.error(f"Session directory not found: {project_dir}")
        raise typer.Exit(1)

    meta_path = project_dir / "metadata.json"
    if not meta_path.exists():
        ui.error("No metadata.json found — is this a valid Graft session?")
        raise typer.Exit(1)

    meta = json.loads(meta_path.read_text())
    repo_path = meta["repo_path"]
    project_id = meta["project_id"]
    feature_prompt = meta.get("feature_prompt", "")

    ui.banner(repo_path, project_id, feature_prompt)
    ui.info(f"Resuming from stage: {from_stage}")

    # Reload artifacts from previous stages
    profile_raw = load_artifact(str(project_dir), "codebase_profile.json")
    codebase_profile = json.loads(profile_raw) if profile_raw else {}

    assessment_raw = load_artifact(str(project_dir), "technical_assessment.json")
    technical_assessment = json.loads(assessment_raw) if assessment_raw else {}

    spec_raw = load_artifact(str(project_dir), "feature_spec.json")
    feature_spec = json.loads(spec_raw) if spec_raw else {}

    plan_raw = load_artifact(str(project_dir), "build_plan.json")
    plan_data = json.loads(plan_raw) if plan_raw else {}

    state: FeatureState = {
        "repo_path": repo_path,
        "project_id": project_id,
        "project_dir": str(project_dir),
        "feature_prompt": feature_prompt,
        "scope_path": "",
        "constraints": [],
        "max_units": 0,
        "auto_approve": auto_approve,
        "codebase_profile": codebase_profile,
        "discovery_report": load_artifact(str(project_dir), "discovery_report.md")
        or "",
        "technical_assessment": technical_assessment,
        "research_report": load_artifact(str(project_dir), "research_report.md") or "",
        "feature_spec": feature_spec,
        "grill_transcript": load_artifact(str(project_dir), "grill_transcript.md")
        or "",
        "build_plan": plan_data.get("units", []),
        "feature_report": "",
        "current_unit_index": 0,
        "units_completed": [],
        "units_reverted": [],
        "units_skipped": [],
        "plan_approved": True,
        "grill_complete": True,
        "research_redo_needed": False,
        "feature_branch": f"feature/{project_id}",
        "pr_url": "",
        "model": settings.model,
        "max_agent_turns": settings.max_agent_turns,
        "current_stage": "",
    }

    compiled = build_graph(ui, entry_stage=from_stage)
    result = asyncio.run(compiled.ainvoke(state))

    pr_url = result.get("pr_url", "")
    if pr_url:
        ui.info(f"PR opened: {pr_url}")


@app.command(name="list")
def list_cmd() -> None:
    """List all feature sessions."""
    settings = Settings.load()
    ui = UI()
    projects = list_projects(settings.projects_root)
    ui.show_projects(projects)


if __name__ == "__main__":
    app()
