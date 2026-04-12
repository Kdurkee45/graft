"""Pipeline state — typed dict that flows through every LangGraph node."""

from __future__ import annotations

from typing import Any, Annotated, TypedDict


def _replace(a: str, b: str) -> str:
    """Reducer that always takes the latest value."""
    return b


def _replace_bool(a: bool, b: bool) -> bool:
    return b


def _replace_int(a: int, b: int) -> int:
    return b


def _replace_list(a: list[Any], b: list[Any]) -> list[Any]:
    return b


def _replace_dict(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return b


class FeatureState(TypedDict, total=False):
    # -- Inputs
    repo_path: Annotated[str, _replace]
    project_id: Annotated[str, _replace]
    project_dir: Annotated[str, _replace]
    feature_prompt: Annotated[str, _replace]

    # -- User inputs (from CLI flags)
    scope_path: Annotated[str, _replace]
    constraints: Annotated[list[str], _replace_list]
    max_units: Annotated[int, _replace_int]
    auto_approve: Annotated[bool, _replace_bool]

    # -- Stage artifacts
    codebase_profile: Annotated[dict[str, Any], _replace_dict]
    discovery_report: Annotated[str, _replace]
    technical_assessment: Annotated[dict[str, Any], _replace_dict]
    research_report: Annotated[str, _replace]
    feature_spec: Annotated[dict[str, Any], _replace_dict]
    grill_transcript: Annotated[str, _replace]
    build_plan: Annotated[list[dict[str, Any]], _replace_list]
    feature_report: Annotated[str, _replace]

    # -- Execution tracking
    current_unit_index: Annotated[int, _replace_int]
    units_completed: Annotated[list[dict[str, Any]], _replace_list]
    units_reverted: Annotated[list[dict[str, Any]], _replace_list]
    units_skipped: Annotated[list[dict[str, Any]], _replace_list]

    # -- Gates
    plan_approved: Annotated[bool, _replace_bool]
    grill_complete: Annotated[bool, _replace_bool]
    research_redo_needed: Annotated[bool, _replace_bool]

    # -- Git
    feature_branch: Annotated[str, _replace]
    pr_url: Annotated[str, _replace]

    # -- Settings
    model: Annotated[str, _replace]
    max_agent_turns: Annotated[int, _replace_int]

    # -- Pipeline state
    current_stage: Annotated[str, _replace]
