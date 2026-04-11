"""Tests for graft.artifacts."""

import json

import pytest

from graft.artifacts import (
    create_project,
    list_projects,
    load_artifact,
    mark_project_done,
    mark_stage_complete,
    save_artifact,
    save_log,
)


@pytest.fixture
def projects_root(tmp_path):
    root = tmp_path / "projects"
    root.mkdir()
    return root


def test_create_project(projects_root):
    """create_project creates directory structure and metadata."""
    project_id, project_dir = create_project(
        projects_root, "/tmp/repo", "Add dark mode"
    )
    assert project_id.startswith("feat_")
    assert (project_dir / "artifacts").is_dir()
    assert (project_dir / "logs").is_dir()
    assert (project_dir / "metadata.json").exists()

    meta = json.loads((project_dir / "metadata.json").read_text())
    assert meta["project_id"] == project_id
    assert meta["repo_path"] == "/tmp/repo"
    assert meta["feature_prompt"] == "Add dark mode"
    assert meta["status"] == "in_progress"


def test_save_and_load_artifact(projects_root):
    project_id, project_dir = create_project(projects_root, "/tmp/repo", "test")
    save_artifact(str(project_dir), "test.md", "# Hello")
    content = load_artifact(str(project_dir), "test.md")
    assert content == "# Hello"


def test_load_artifact_missing(projects_root):
    project_id, project_dir = create_project(projects_root, "/tmp/repo", "test")
    assert load_artifact(str(project_dir), "nonexistent.md") is None


def test_save_log(projects_root):
    project_id, project_dir = create_project(projects_root, "/tmp/repo", "test")
    save_log(str(project_dir), "audit", "Log entry 1")
    save_log(str(project_dir), "audit", "Log entry 2")
    log_content = (project_dir / "logs" / "audit.log").read_text()
    assert "Log entry 1" in log_content
    assert "Log entry 2" in log_content


def test_mark_stage_complete(projects_root):
    project_id, project_dir = create_project(projects_root, "/tmp/repo", "test")
    mark_stage_complete(str(project_dir), "discover")
    mark_stage_complete(str(project_dir), "research")
    meta = json.loads((project_dir / "metadata.json").read_text())
    assert meta["stages_completed"] == ["discover", "research"]


def test_mark_stage_complete_idempotent(projects_root):
    project_id, project_dir = create_project(projects_root, "/tmp/repo", "test")
    mark_stage_complete(str(project_dir), "discover")
    mark_stage_complete(str(project_dir), "discover")
    meta = json.loads((project_dir / "metadata.json").read_text())
    assert meta["stages_completed"] == ["discover"]


def test_mark_project_done(projects_root):
    project_id, project_dir = create_project(projects_root, "/tmp/repo", "test")
    mark_project_done(str(project_dir), "https://github.com/org/repo/pull/1")
    meta = json.loads((project_dir / "metadata.json").read_text())
    assert meta["status"] == "completed"
    assert meta["pr_url"] == "https://github.com/org/repo/pull/1"


def test_list_projects(projects_root):
    create_project(projects_root, "/tmp/repo1", "Feature A")
    create_project(projects_root, "/tmp/repo2", "Feature B")
    projects = list_projects(projects_root)
    assert len(projects) == 2
    prompts = {p["feature_prompt"] for p in projects}
    assert prompts == {"Feature A", "Feature B"}


def test_list_projects_empty(tmp_path):
    empty_root = tmp_path / "empty"
    assert list_projects(empty_root) == []
