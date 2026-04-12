"""Execute stage — implements build units one at a time.

Each unit follows: read pattern → implement → commit → test → keep/revert.
Pattern references are critical — the agent reads them first to understand
conventions. Tests are co-located with the code they test.

All work happens on a single feature branch with atomic commits.
"""

from __future__ import annotations

import json
import subprocess

from graft.agent import run_agent
from graft.artifacts import mark_stage_complete, save_artifact
from graft.state import FeatureState
from graft.ui import UI

SYSTEM_PROMPT = """You are a Principal Software Engineer building a feature into an existing codebase.

You will be given ONE specific build task with:
- A description of what to implement
- A pattern_reference file to follow (read this FIRST)
- Acceptance criteria
- Whether to include tests

## Rules

1. **Read the pattern_reference file first.** Understand the convention.
   Your code MUST follow the same patterns — naming, structure, error
   handling, imports, everything. The feature should look like it was
   written by the same team.

2. Make the change described in the task. Be precise and surgical.

3. If tests_included is true, write co-located tests following the
   project's test patterns.

4. Keep changes minimal and focused on this ONE unit.

5. Do NOT modify unrelated files.

6. Do NOT introduce new dependencies unless explicitly required.

7. After making changes, do NOT run tests or linters yourself — the
   pipeline handles verification automatically.

Be surgical. One task. Follow the pattern. Do it well.
"""


VERIFY_SCRIPT = """#!/bin/bash
set -e

# Detect project type and run appropriate tests
if [ -f "package.json" ]; then
    npm test -- --watchAll=false --passWithNoTests 2>&1 || exit 1
elif [ -f "pyproject.toml" ] || [ -f "setup.py" ] || [ -f "requirements.txt" ]; then
    python -m pytest --tb=short -q 2>&1 || exit 1
elif [ -f "Cargo.toml" ]; then
    cargo test 2>&1 || exit 1
elif [ -f "go.mod" ]; then
    go test ./... 2>&1 || exit 1
fi

exit 0
"""


def _git(repo_path: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command in the repo directory."""
    return subprocess.run(
        ["git", *args],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=check,
    )


def _run_tests(repo_path: str) -> tuple[bool, str]:
    """Run the project's test suite. Returns (passed, output)."""
    try:
        result = subprocess.run(
            ["bash", "-c", VERIFY_SCRIPT],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = (result.stdout + result.stderr).strip()
        return result.returncode == 0, output[-2000:]
    except subprocess.TimeoutExpired:
        return False, "Test suite timed out (300s limit)"
    except FileNotFoundError:
        return True, "No test runner found — skipping tests"


def _run_lint(repo_path: str) -> tuple[bool, str]:
    """Run linter/formatter. Returns (passed, output)."""
    # Try common linters
    for cmd in [
        ["npx", "eslint", ".", "--fix"],
        ["python", "-m", "ruff", "check", ".", "--fix"],
        ["npx", "prettier", "--write", "."],
    ]:
        try:
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                return True, "Lint passed"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return True, "No linter found — skipping"


def _order_by_dependencies(plan: list[dict]) -> list[dict]:
    """Topological sort of build units by depends_on."""
    completed_ids: set[str] = set()
    ordered: list[dict] = []
    remaining = list(plan)

    max_passes = len(remaining) + 1
    for _ in range(max_passes):
        if not remaining:
            break
        next_remaining = []
        for unit in remaining:
            deps = unit.get("depends_on", [])
            if all(d in completed_ids for d in deps):
                ordered.append(unit)
                completed_ids.add(unit.get("unit_id", ""))
            else:
                next_remaining.append(unit)
        if len(next_remaining) == len(remaining):
            # Circular or unresolvable — append remaining as-is
            ordered.extend(next_remaining)
            break
        remaining = next_remaining

    return ordered


async def execute_node(state: FeatureState, ui: UI) -> dict:
    """LangGraph node: execute build units one at a time."""
    ui.stage_start("execute")
    repo_path = state["repo_path"]
    project_dir = state["project_dir"]
    plan = state.get("build_plan", [])
    max_turns = state.get("max_agent_turns", 50)

    if not plan:
        ui.error("No build plan — nothing to execute.")
        mark_stage_complete(project_dir, "execute")
        return {"current_stage": "execute"}

    ordered_plan = _order_by_dependencies(plan)

    # Create feature branch
    branch_name = state.get(
        "feature_branch",
        f"feature/{state.get('project_id', 'session')}",
    )
    result = _git(repo_path, "checkout", "-b", branch_name, check=False)
    if result.returncode != 0:
        _git(repo_path, "checkout", branch_name, check=False)

    units_completed: list[dict] = []
    units_reverted: list[dict] = []
    units_skipped: list[dict] = []
    completed_ids: set[str] = set()

    for i, unit in enumerate(ordered_plan, 1):
        unit_id = unit.get("unit_id", f"feat_{i:02d}")
        title = unit.get("title", "Untitled")
        description = unit.get("description", title)
        pattern_ref = unit.get("pattern_reference", "")
        tests_included = unit.get("tests_included", False)
        acceptance_criteria = unit.get("acceptance_criteria", [])

        # Check dependencies
        deps = unit.get("depends_on", [])
        unmet = [d for d in deps if d not in completed_ids]
        if unmet:
            ui.unit_reverted(unit_id, f"Unmet dependencies: {', '.join(unmet)}")
            units_skipped.append(
                {
                    "unit_id": unit_id,
                    "reason": f"Dependencies not met: {', '.join(unmet)}",
                }
            )
            continue

        ui.unit_start(unit_id, title, i, len(ordered_plan))

        # Build the prompt
        prompt_parts = [
            f"BUILD TASK: {title}",
            f"\nDESCRIPTION:\n{description}",
            "\nACCEPTANCE CRITERIA:",
            *[f"- {c}" for c in acceptance_criteria],
        ]
        if pattern_ref:
            prompt_parts.append(
                f"\nPATTERN REFERENCE: {pattern_ref}\n"
                "Read this file FIRST. Follow its conventions exactly."
            )
        if tests_included:
            prompt_parts.append(
                "\nTESTS: Write co-located tests for this unit. "
                "Follow the project's existing test patterns."
            )
        prompt_parts.append(f"\nWORKING DIRECTORY: {repo_path}")
        prompt_parts.append("\nMake the change now. Be precise and surgical.")

        try:
            await run_agent(
                persona=f"Principal Software Engineer [{unit_id}]",
                system_prompt=SYSTEM_PROMPT,
                user_prompt="\n".join(prompt_parts),
                cwd=repo_path,
                project_dir=project_dir,
                stage=f"execute_{unit_id}",
                ui=ui,
                model=state.get("model"),
                max_turns=max_turns,
            )
        except RuntimeError as exc:
            ui.unit_reverted(unit_id, f"Agent failed: {exc}")
            units_reverted.append({"unit_id": unit_id, "reason": str(exc)})
            continue

        # Commit
        _git(repo_path, "add", "-A")
        commit_result = _git(
            repo_path,
            "commit",
            "-m",
            f"feat: {title}",
            check=False,
        )
        if commit_result.returncode != 0:
            ui.unit_reverted(unit_id, "No changes made")
            units_reverted.append({"unit_id": unit_id, "reason": "No changes produced"})
            continue

        # Test
        tests_passed, test_output = _run_tests(repo_path)
        if not tests_passed:
            _git(repo_path, "revert", "HEAD", "--no-edit")
            ui.unit_reverted(unit_id, "Tests failed")
            units_reverted.append(
                {
                    "unit_id": unit_id,
                    "reason": f"Tests failed: {test_output[:200]}",
                }
            )
            continue

        # Lint (auto-fix and amend if needed)
        lint_passed, lint_output = _run_lint(repo_path)
        if lint_passed:
            # Amend commit with any lint fixes
            _git(repo_path, "add", "-A")
            _git(repo_path, "commit", "--amend", "--no-edit", check=False)

        ui.unit_kept(unit_id, "Implemented and passing")
        completed_ids.add(unit_id)
        units_completed.append(
            {
                "unit_id": unit_id,
                "title": title,
                "category": unit.get("category", ""),
                "tests_included": tests_included,
            }
        )

    # Save execution log
    log = {
        "units_completed": len(units_completed),
        "units_reverted": len(units_reverted),
        "units_skipped": len(units_skipped),
        "total_planned": len(plan),
        "completed": units_completed,
        "reverted": units_reverted,
        "skipped": units_skipped,
    }
    save_artifact(project_dir, "execution_log.json", json.dumps(log, indent=2))
    mark_stage_complete(project_dir, "execute")
    ui.stage_done("execute")

    return {
        "units_completed": units_completed,
        "units_reverted": units_reverted,
        "units_skipped": units_skipped,
        "feature_branch": branch_name,
        "current_stage": "execute",
    }
