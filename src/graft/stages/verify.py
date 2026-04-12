"""Verify stage — full regression check, feature validation, and PR opening.

Combines validation (does the feature work?), regression (did anything break?),
and delivery (open the PR). The PR is always the final human gate.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

from graft.agent import run_agent
from graft.artifacts import mark_project_done, mark_stage_complete, save_artifact
from graft.stages._helpers import cleanup_artifacts, read_text_artifact
from graft.state import FeatureState
from graft.ui import UI

SYSTEM_PROMPT = """\
You are a Principal Quality Engineer and Technical Program Manager performing
a final verification of a newly built feature.

You have:
- The codebase profile (architecture, patterns)
- The feature specification (every decision documented)
- The build plan (what was supposed to be built)
- The execution log (what was actually built vs. reverted)

## Your Job

### 1. Regression Check
- Run the full test suite (all services/packages)
- Run linters and type checkers
- Confirm zero existing test failures

### 2. Feature Validation
- Walk through the feature spec decisions
- Verify each MVP scope item is implemented
- Check acceptance criteria for every completed build unit
- Verify edge cases identified in the Grill phase are handled

### 3. Integration Check
- Verify routing/navigation works (if applicable)
- Verify auth/permissions are correctly applied (if applicable)
- Verify the feature follows the patterns documented in Discovery

### 4. Coverage Delta
- Measure test coverage on the new feature code
- Report what percentage of new code is covered by tests

## Output

Write `feature_report.md` to the working directory with:

- **Feature Summary** — what was built
- **Build Results** — units completed vs. reverted vs. skipped
- **Regression Status** — all tests pass? lint clean? type-safe?
- **Feature Validation** — each spec item checked off or flagged
- **Coverage Delta** — test coverage on new code
- **Files Changed** — list of all files created and modified
- **Follow-Up Items** — remaining scope from the feature spec (if any)
- **Hone Suggestion** — exact `hone optimize` command scoped to the
  new feature's files (always include this)

Write the file to the current working directory. Be thorough and factual.
"""


def _open_pr(repo_path: str, branch: str, title: str, body: str) -> str | None:
    """Open a PR using the gh CLI. Returns the PR URL or None."""
    try:
        subprocess.run(
            ["git", "push", "-u", "origin", branch],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        result = subprocess.run(
            ["gh", "pr", "create", "--title", title, "--body", body, "--head", branch],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
    ):
        return None


async def verify_node(state: FeatureState, ui: UI) -> dict[str, Any]:
    """LangGraph node: validate the feature, run regression, and open the PR."""
    ui.stage_start("verify")
    repo_path = state["repo_path"]
    project_dir = state["project_dir"]
    feature_prompt = state.get("feature_prompt", "")
    codebase_profile = state.get("codebase_profile", {})
    feature_spec = state.get("feature_spec", {})
    build_plan = state.get("build_plan", [])
    units_completed = state.get("units_completed", [])
    units_reverted = state.get("units_reverted", [])
    units_skipped = state.get("units_skipped", [])

    prompt = (
        f"Verify the feature built into the codebase at: {repo_path}\n\n"
        f"FEATURE: {feature_prompt}\n\n"
        f"CODEBASE PROFILE:\n{json.dumps(codebase_profile, indent=2)}\n\n"
        f"FEATURE SPEC:\n{json.dumps(feature_spec, indent=2)}\n\n"
        f"BUILD PLAN ({len(build_plan)} units):\n{json.dumps(build_plan, indent=2)}\n\n"
        f"UNITS COMPLETED ({len(units_completed)}):\n"
        f"{json.dumps(units_completed, indent=2)}\n\n"
        f"UNITS REVERTED ({len(units_reverted)}):\n"
        f"{json.dumps(units_reverted, indent=2)}\n\n"
        f"UNITS SKIPPED ({len(units_skipped)}):\n"
        f"{json.dumps(units_skipped, indent=2)}\n\n"
        f"Run all checks and produce feature_report.md."
    )

    result = await run_agent(
        persona="Principal Quality Engineer",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
        cwd=repo_path,
        project_dir=project_dir,
        stage="verify",
        ui=ui,
        model=state.get("model"),
        max_turns=30,
        allowed_tools=["Bash", "Read", "Glob", "Grep"],
    )

    # Read report
    feature_report = await read_text_artifact(
        "feature_report.md", repo_path, repo_path, fallback=result.text
    )
    save_artifact(project_dir, "feature_report.md", feature_report)
    cleanup_artifacts(repo_path, repo_path, ["feature_report.md"])

    # Open PR
    branch = state.get("feature_branch", "")
    pr_url = ""
    feature_name = feature_spec.get("feature_name", "Feature")

    if branch:
        # Stage any verification artifacts cleanup
        subprocess.run(
            ["git", "add", "-A"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        # Only commit if there are changes
        subprocess.run(
            [
                "git",
                "commit",
                "-m",
                "chore: cleanup verification artifacts",
                "--allow-empty",
            ],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )

        pr_title = f"Feature: {feature_name}"
        url = _open_pr(repo_path, branch, pr_title, feature_report)
        if url:
            pr_url = url
            ui.pr_opened(pr_url)
        else:
            ui.info(
                f"Could not open PR automatically. Push branch '{branch}' "
                f"and open a PR manually."
            )
            ui.info(f"Report saved to: {project_dir}/artifacts/feature_report.md")

    mark_stage_complete(project_dir, "verify")
    if pr_url:
        mark_project_done(project_dir, pr_url)

    ui.stage_done("verify")

    return {
        "feature_report": feature_report,
        "pr_url": pr_url,
        "current_stage": "verify",
    }
