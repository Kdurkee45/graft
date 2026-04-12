"""Discover stage — builds a complete mental model of the existing codebase.

Maps architecture, patterns, data model, conventions, and test infrastructure.
Produces codebase_profile.json and discovery_report.md that every subsequent
stage references.

Also performs module-level test coverage analysis on integration-critical
modules and warns if coverage is thin.
"""

from __future__ import annotations

import json
from typing import Any

from graft.agent import run_agent
from graft.artifacts import mark_stage_complete, save_artifact
from graft.stages._helpers import (
    async_read_text,
    cleanup_artifacts,
    find_artifact,
    resolve_stage_cwd,
)
from graft.state import FeatureState
from graft.ui import UI

SYSTEM_PROMPT = """\
You are a Principal Codebase Archaeologist with deep expertise in understanding
existing software systems. Your job is to build a complete mental model of a
codebase — architecture, patterns, data model, conventions, and test
infrastructure.

## What You Must Discover

1. **Project Structure**
   - Map the full directory tree and file organization
   - Identify framework(s), language(s), package manager(s)
   - Detect monorepo structure (services, packages, shared code)
   - Map entry points, routing, middleware chains

2. **Data Model**
   - Catalog database tables/collections (schema, relationships, indexes)
   - Map ORM models, types, interfaces
   - Identify data access patterns (repositories, services, direct queries)
   - Document existing migrations infrastructure

3. **Architecture Patterns**
   - Component patterns (naming, file structure, prop patterns)
   - State management approach (Redux, Zustand, context, signals)
   - API patterns (REST, GraphQL, tRPC, server actions)
   - Auth/authorization model (how permissions work, middleware)
   - Error handling patterns
   - Logging and observability patterns

4. **Integration Surface**
   - External services and APIs already integrated
   - Environment variables and configuration approach
   - Feature flags system (if any)
   - Real-time capabilities (WebSockets, SSE, polling)

5. **Test Infrastructure**
   - Test framework, runner, assertion style
   - Test file organization and naming conventions
   - Fixture and mock patterns
   - **MODULE-LEVEL coverage analysis**: For each major module/service,
     measure or estimate test coverage. List modules with low or zero
     coverage in the `coverage_warnings` field.
   - E2E test setup (if any)

6. **Conventions**
   - Git workflow (branch naming, commit messages, PR templates)
   - Code style (formatter, linter, documented conventions)
   - Documentation patterns
   - Any CLAUDE.md, .cursorrules, or agent instructions present

## Output

You MUST produce two files in the working directory:

### 1. discovery_report.md
A comprehensive narrative report covering all findings.

### 2. codebase_profile.json
A structured JSON reference with the following shape:

```json
{
  "timestamp": "ISO-8601",
  "project": {
    "name": "project-name",
    "languages": ["typescript"],
    "frameworks": ["nextjs"],
    "package_manager": "npm",
    "monorepo": true
  },
  "services": [...],
  "data_model": {
    "orm": "supabase",
    "tables": [...],
    "key_relationships": [...]
  },
  "patterns": {
    "state_management": "...",
    "api": "...",
    "auth": "...",
    "components": "...",
    "routing": "..."
  },
  "test_infrastructure": {
    "framework": "vitest",
    "runner": "vitest run",
    "coverage": "v8",
    "e2e": null,
    "conventions": "co-located __tests__ folders"
  },
  "conventions": {
    "git_workflow": "...",
    "commit_style": "...",
    "code_style": "..."
  },
  "coverage_warnings": [
    {
      "module": "src/services/roster.ts",
      "coverage_pct": 12,
      "recommendation": "hone optimize --focus tests --path src/services/roster.ts"
    }
  ]
}
```

The coverage_warnings array should list modules with coverage below 40%.
If you cannot measure exact coverage, estimate based on test file presence
and test density. An empty array means coverage looks adequate.

Write both files to the current working directory. Be thorough.
"""


def _build_discover_prompt(repo_path: str, scope_path: str, feature_prompt: str) -> str:
    """Assemble the user prompt for the discover stage."""
    prompt_parts = [
        f"Discover and map the codebase at: {repo_path}",
    ]
    if scope_path:
        prompt_parts.append(
            f"\nSCOPE: Focus primarily on '{scope_path}/'"
            f" but understand the full project context."
        )
    if feature_prompt:
        prompt_parts.append(
            f'\nUPCOMING FEATURE: "{feature_prompt}"\n'
            "Pay special attention to modules and patterns that this feature "
            "is likely to integrate with. Coverage warnings should prioritize "
            "these integration-critical modules."
        )
    prompt_parts.append(
        "\nProduce a comprehensive discovery report and codebase profile."
    )
    return "\n".join(prompt_parts)


async def discover_node(state: FeatureState, ui: UI) -> dict[str, Any]:
    """LangGraph node: discover the codebase architecture and patterns."""
    ui.stage_start("discover")
    repo_path = state["repo_path"]
    project_dir = state["project_dir"]
    scope_path = state.get("scope_path", "")
    feature_prompt = state.get("feature_prompt", "")

    discover_cwd = resolve_stage_cwd(repo_path, scope_path)
    user_prompt = _build_discover_prompt(repo_path, scope_path, feature_prompt)

    result = await run_agent(
        persona="Principal Codebase Archaeologist",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        cwd=discover_cwd,
        project_dir=project_dir,
        stage="discover",
        ui=ui,
        model=state.get("model"),
        max_turns=40,
        allowed_tools=["Bash", "Read", "Glob", "Grep"],
    )

    # Read agent outputs
    report_path = find_artifact("discovery_report.md", discover_cwd, repo_path)
    profile_path = find_artifact("codebase_profile.json", discover_cwd, repo_path)

    discovery_report = (
        (await async_read_text(report_path)) if report_path.exists() else result.text
    )
    save_artifact(project_dir, "discovery_report.md", discovery_report)

    codebase_profile: dict = {}
    if profile_path.exists():
        try:
            codebase_profile = json.loads(await async_read_text(profile_path))
        except json.JSONDecodeError:
            ui.error("Failed to parse codebase_profile.json from agent output.")

    save_artifact(
        project_dir, "codebase_profile.json", json.dumps(codebase_profile, indent=2)
    )

    # Show coverage warnings if any
    coverage_warnings = codebase_profile.get("coverage_warnings", [])
    if coverage_warnings:
        ui.coverage_warning(coverage_warnings)

    # Clean up — don't leave artifacts in the repo
    cleanup_artifacts(
        discover_cwd,
        repo_path,
        ["discovery_report.md", "codebase_profile.json"],
    )

    mark_stage_complete(project_dir, "discover")
    ui.stage_done("discover")

    return {
        "codebase_profile": codebase_profile,
        "discovery_report": discovery_report,
        "current_stage": "discover",
    }
