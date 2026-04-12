"""Tests for graft.stages.grill."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from graft.stages.grill import _generate_questions, grill_node, grill_router


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_repo(tmp_path):
    """Create a temporary repo directory and project directory."""
    repo = tmp_path / "repo"
    repo.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    return repo, project


@pytest.fixture
def mock_ui():
    """Create a mock UI with all methods grill uses."""
    ui = MagicMock()
    ui.grill_question = MagicMock(return_value="Accepted recommendation")
    ui.stage_start = MagicMock()
    ui.stage_done = MagicMock()
    ui.info = MagicMock()
    ui.error = MagicMock()
    return ui


@pytest.fixture
def sample_open_questions():
    """Open questions as produced by the Research stage."""
    return [
        {
            "question": "Should dark mode persist across sessions?",
            "recommended_answer": "Yes, store in localStorage",
            "category": "intent",
        },
        {
            "question": "What about OS-level preference?",
            "recommended_answer": "Respect prefers-color-scheme as default",
            "category": "preference",
        },
        {
            "question": "Handle images in dark mode?",
            "recommended_answer": "Invert diagrams, dim photos",
            "category": "edge_case",
        },
    ]


@pytest.fixture
def base_state(tmp_repo, sample_open_questions):
    """Minimal FeatureState dict for grill_node."""
    repo, project = tmp_repo
    return {
        "repo_path": str(repo),
        "project_dir": str(project),
        "feature_prompt": "Add dark mode support",
        "codebase_profile": {"framework": "react", "patterns": ["component-based"]},
        "technical_assessment": {"open_questions": sample_open_questions},
        "constraints": ["Must support IE11"],
        "model": "claude-sonnet-4-20250514",
    }


# ---------------------------------------------------------------------------
# grill_router tests (preserved)
# ---------------------------------------------------------------------------


def test_grill_router_no_redo():
    """Normal flow: Grill -> Plan."""
    assert grill_router({"research_redo_needed": False}) == "plan"


def test_grill_router_redo():
    """Loop-back: Grill -> Research when redo needed."""
    assert grill_router({"research_redo_needed": True}) == "research"


def test_grill_router_default():
    """Default (no flag): proceed to Plan."""
    assert grill_router({}) == "plan"


# ---------------------------------------------------------------------------
# grill_node: walks open_questions and builds transcript
# ---------------------------------------------------------------------------


async def test_grill_node_walks_open_questions(base_state, mock_ui, tmp_repo):
    """grill_node iterates over each open_question, calls ui.grill_question, and
    compiles the transcript."""
    repo, project = tmp_repo
    answers = ["Yes, persist it", "Yes, respect OS", "Invert only diagrams"]
    mock_ui.grill_question = MagicMock(side_effect=answers)

    feature_spec = {"feature_name": "Dark Mode", "decisions": []}
    spec_path = Path(repo) / "feature_spec.json"

    async def _fake_run_agent(**kwargs):
        if kwargs["stage"] == "grill_compile":
            spec_path.write_text(json.dumps(feature_spec))

    with (
        patch(
            "graft.stages.grill.run_agent", new=AsyncMock(side_effect=_fake_run_agent)
        ),
        patch("graft.stages.grill.save_artifact") as mock_save,
        patch("graft.stages.grill.mark_stage_complete") as mock_mark,
    ):
        result = await grill_node(base_state, mock_ui)

    # ui.grill_question called once per question
    assert mock_ui.grill_question.call_count == 3

    # Verify call args match each question
    calls = mock_ui.grill_question.call_args_list
    assert calls[0] == call(
        "Should dark mode persist across sessions?",
        "Yes, store in localStorage",
        "intent",
        1,
    )
    assert calls[1] == call(
        "What about OS-level preference?",
        "Respect prefers-color-scheme as default",
        "preference",
        2,
    )
    assert calls[2] == call(
        "Handle images in dark mode?",
        "Invert diagrams, dim photos",
        "edge_case",
        3,
    )

    # Result contains transcript and is marked complete
    assert result["grill_complete"] is True
    assert result["current_stage"] == "grill"
    assert "grill_transcript" in result


async def test_grill_node_transcript_format(base_state, mock_ui, tmp_repo):
    """Verify the exact Q&A transcript format produced by grill_node."""
    repo, project = tmp_repo
    answers = ["Answer A", "Answer B", "Answer C"]
    mock_ui.grill_question = MagicMock(side_effect=answers)

    feature_spec = {"feature_name": "Dark Mode"}
    spec_path = Path(repo) / "feature_spec.json"

    async def _fake_run_agent(**kwargs):
        if kwargs["stage"] == "grill_compile":
            spec_path.write_text(json.dumps(feature_spec))

    with (
        patch(
            "graft.stages.grill.run_agent", new=AsyncMock(side_effect=_fake_run_agent)
        ),
        patch("graft.stages.grill.save_artifact"),
        patch("graft.stages.grill.mark_stage_complete"),
    ):
        result = await grill_node(base_state, mock_ui)

    transcript = result["grill_transcript"]
    lines = transcript.split("\n")

    # Each Q block is: question line, recommended line, answer line, blank line
    assert lines[0] == "Q1 [intent]: Should dark mode persist across sessions?"
    assert lines[1] == "  Recommended: Yes, store in localStorage"
    assert lines[2] == "  Answer: Answer A"
    assert lines[3] == ""  # blank separator

    assert lines[4] == "Q2 [preference]: What about OS-level preference?"
    assert lines[5] == "  Recommended: Respect prefers-color-scheme as default"
    assert lines[6] == "  Answer: Answer B"
    assert lines[7] == ""

    assert lines[8] == "Q3 [edge_case]: Handle images in dark mode?"
    assert lines[9] == "  Recommended: Invert diagrams, dim photos"
    assert lines[10] == "  Answer: Answer C"


# ---------------------------------------------------------------------------
# grill_node: fallback to _generate_questions when no open_questions
# ---------------------------------------------------------------------------


async def test_grill_node_falls_back_to_generate_questions(mock_ui, tmp_repo):
    """When technical_assessment has no open_questions, grill_node generates them
    via _generate_questions."""
    repo, project = tmp_repo
    state = {
        "repo_path": str(repo),
        "project_dir": str(project),
        "feature_prompt": "Add search",
        "codebase_profile": {},
        "technical_assessment": {},  # no open_questions
        "constraints": [],
        "model": None,
    }

    generated_questions = [
        {
            "question": "Full-text or fuzzy?",
            "recommended_answer": "Full-text",
            "category": "intent",
        },
    ]
    questions_path = Path(repo) / "open_questions.json"
    feature_spec_path = Path(repo) / "feature_spec.json"

    call_count = {"n": 0}

    async def _fake_run_agent(**kwargs):
        if kwargs["stage"] == "grill_generate":
            questions_path.write_text(json.dumps(generated_questions))
        elif kwargs["stage"] == "grill_compile":
            feature_spec_path.write_text(json.dumps({"feature_name": "Search"}))
        call_count["n"] += 1

    mock_ui.grill_question = MagicMock(return_value="Full-text search")

    with (
        patch(
            "graft.stages.grill.run_agent", new=AsyncMock(side_effect=_fake_run_agent)
        ),
        patch("graft.stages.grill.save_artifact"),
        patch("graft.stages.grill.mark_stage_complete"),
    ):
        result = await grill_node(state, mock_ui)

    # Should have called info about generating questions
    mock_ui.info.assert_any_call(
        "No open questions from Research — generating questions from context..."
    )

    # Agent called twice: once for generation, once for compile
    assert call_count["n"] == 2

    # The generated question should have been asked
    assert mock_ui.grill_question.call_count == 1
    mock_ui.grill_question.assert_called_once_with(
        "Full-text or fuzzy?", "Full-text", "intent", 1
    )

    assert result["grill_complete"] is True


async def test_grill_node_empty_open_questions_list_triggers_fallback(
    mock_ui, tmp_repo
):
    """An explicit empty list for open_questions also triggers fallback."""
    repo, project = tmp_repo
    state = {
        "repo_path": str(repo),
        "project_dir": str(project),
        "feature_prompt": "Add logging",
        "codebase_profile": {},
        "technical_assessment": {"open_questions": []},  # empty list
        "constraints": [],
        "model": None,
    }

    feature_spec_path = Path(repo) / "feature_spec.json"

    async def _fake_run_agent(**kwargs):
        if kwargs["stage"] == "grill_generate":
            # Agent produces no questions file — edge case
            pass
        elif kwargs["stage"] == "grill_compile":
            feature_spec_path.write_text(json.dumps({}))

    with (
        patch(
            "graft.stages.grill.run_agent", new=AsyncMock(side_effect=_fake_run_agent)
        ),
        patch("graft.stages.grill.save_artifact"),
        patch("graft.stages.grill.mark_stage_complete"),
    ):
        result = await grill_node(state, mock_ui)

    mock_ui.info.assert_any_call(
        "No open questions from Research — generating questions from context..."
    )
    # No questions generated, so grill_question not called
    mock_ui.grill_question.assert_not_called()
    # Transcript is empty
    assert result["grill_transcript"] == ""


# ---------------------------------------------------------------------------
# _generate_questions: agent spawning and JSON parsing
# ---------------------------------------------------------------------------


async def test_generate_questions_success(tmp_repo, mock_ui):
    """_generate_questions spawns an agent, reads open_questions.json, returns list."""
    repo, project = tmp_repo
    questions = [
        {"question": "Q1?", "recommended_answer": "A1", "category": "intent"},
        {"question": "Q2?", "recommended_answer": "A2", "category": "edge_case"},
    ]
    questions_path = Path(repo) / "open_questions.json"

    async def _fake_run_agent(**kwargs):
        assert kwargs["stage"] == "grill_generate"
        assert kwargs["persona"] == "Principal Product Interrogator"
        questions_path.write_text(json.dumps(questions))

    with patch(
        "graft.stages.grill.run_agent", new=AsyncMock(side_effect=_fake_run_agent)
    ):
        result = await _generate_questions(
            repo_path=str(repo),
            project_dir=str(project),
            feature_prompt="Add feature X",
            codebase_profile={"lang": "python"},
            technical_assessment={"gaps": ["auth"]},
            ui=mock_ui,
            model="claude-sonnet-4-20250514",
        )

    assert result == questions
    # File should be cleaned up
    assert not questions_path.exists()


async def test_generate_questions_missing_json(tmp_repo, mock_ui):
    """_generate_questions returns empty list when agent doesn't produce the file."""
    repo, project = tmp_repo

    async def _fake_run_agent(**kwargs):
        pass  # Agent doesn't create the file

    with patch(
        "graft.stages.grill.run_agent", new=AsyncMock(side_effect=_fake_run_agent)
    ):
        result = await _generate_questions(
            repo_path=str(repo),
            project_dir=str(project),
            feature_prompt="Add feature Y",
            codebase_profile={},
            technical_assessment={},
            ui=mock_ui,
            model=None,
        )

    assert result == []


async def test_generate_questions_malformed_json(tmp_repo, mock_ui):
    """_generate_questions returns empty list on malformed JSON."""
    repo, project = tmp_repo
    questions_path = Path(repo) / "open_questions.json"

    async def _fake_run_agent(**kwargs):
        questions_path.write_text("this is not valid json {{{")

    with patch(
        "graft.stages.grill.run_agent", new=AsyncMock(side_effect=_fake_run_agent)
    ):
        result = await _generate_questions(
            repo_path=str(repo),
            project_dir=str(project),
            feature_prompt="Add feature Z",
            codebase_profile={},
            technical_assessment={},
            ui=mock_ui,
            model=None,
        )

    assert result == []


async def test_generate_questions_non_list_json(tmp_repo, mock_ui):
    """_generate_questions returns empty list when JSON is valid but not a list."""
    repo, project = tmp_repo
    questions_path = Path(repo) / "open_questions.json"

    async def _fake_run_agent(**kwargs):
        questions_path.write_text(json.dumps({"questions": ["q1", "q2"]}))

    with patch(
        "graft.stages.grill.run_agent", new=AsyncMock(side_effect=_fake_run_agent)
    ):
        result = await _generate_questions(
            repo_path=str(repo),
            project_dir=str(project),
            feature_prompt="Some feature",
            codebase_profile={},
            technical_assessment={},
            ui=mock_ui,
            model=None,
        )

    assert result == []


# ---------------------------------------------------------------------------
# grill_node: compile agent and feature_spec.json handling
# ---------------------------------------------------------------------------


async def test_grill_node_reads_and_saves_feature_spec(base_state, mock_ui, tmp_repo):
    """grill_node invokes compile agent, reads feature_spec.json, and saves it
    as an artifact."""
    repo, project = tmp_repo
    expected_spec = {
        "feature_name": "Dark Mode",
        "decisions": [{"question": "Q1", "answer": "A1"}],
        "scope": {"mvp": ["toggle"], "follow_up": ["scheduled"]},
    }
    spec_path = Path(repo) / "feature_spec.json"

    async def _fake_run_agent(**kwargs):
        if kwargs["stage"] == "grill_compile":
            # Verify compile agent gets correct params
            assert kwargs["cwd"] == str(repo)
            assert kwargs["persona"] == "Principal Product Architect"
            assert kwargs["max_turns"] == 10
            assert "Read" in kwargs["allowed_tools"]
            assert "Write" in kwargs["allowed_tools"]
            assert "Bash" in kwargs["allowed_tools"]
            spec_path.write_text(json.dumps(expected_spec))

    with (
        patch(
            "graft.stages.grill.run_agent", new=AsyncMock(side_effect=_fake_run_agent)
        ),
        patch("graft.stages.grill.save_artifact") as mock_save,
        patch("graft.stages.grill.mark_stage_complete"),
    ):
        result = await grill_node(base_state, mock_ui)

    assert result["feature_spec"] == expected_spec

    # save_artifact called for both transcript and spec
    save_calls = mock_save.call_args_list
    artifact_names = [c[0][1] for c in save_calls]
    assert "grill_transcript.md" in artifact_names
    assert "feature_spec.json" in artifact_names

    # The spec artifact content should be the JSON string
    spec_save_call = [c for c in save_calls if c[0][1] == "feature_spec.json"][0]
    saved_content = json.loads(spec_save_call[0][2])
    assert saved_content == expected_spec


async def test_grill_node_handles_missing_feature_spec(mock_ui, tmp_repo):
    """When compile agent fails to produce feature_spec.json, grill_node
    gracefully returns an empty spec."""
    repo, project = tmp_repo
    state = {
        "repo_path": str(repo),
        "project_dir": str(project),
        "feature_prompt": "Add caching",
        "codebase_profile": {},
        "technical_assessment": {
            "open_questions": [
                {"question": "TTL?", "recommended_answer": "1h", "category": "intent"}
            ]
        },
        "constraints": [],
        "model": None,
    }

    async def _fake_run_agent(**kwargs):
        pass  # Agent doesn't create the file

    with (
        patch(
            "graft.stages.grill.run_agent", new=AsyncMock(side_effect=_fake_run_agent)
        ),
        patch("graft.stages.grill.save_artifact"),
        patch("graft.stages.grill.mark_stage_complete"),
    ):
        result = await grill_node(state, mock_ui)

    assert result["feature_spec"] == {}
    assert result["grill_complete"] is True


async def test_grill_node_handles_malformed_feature_spec(mock_ui, tmp_repo):
    """When feature_spec.json is malformed, grill_node logs error and returns
    empty spec."""
    repo, project = tmp_repo
    state = {
        "repo_path": str(repo),
        "project_dir": str(project),
        "feature_prompt": "Add auth",
        "codebase_profile": {},
        "technical_assessment": {
            "open_questions": [
                {
                    "question": "OAuth?",
                    "recommended_answer": "Yes",
                    "category": "intent",
                }
            ]
        },
        "constraints": [],
        "model": None,
    }
    spec_path = Path(repo) / "feature_spec.json"

    async def _fake_run_agent(**kwargs):
        if kwargs["stage"] == "grill_compile":
            spec_path.write_text("not valid json {{{")

    with (
        patch(
            "graft.stages.grill.run_agent", new=AsyncMock(side_effect=_fake_run_agent)
        ),
        patch("graft.stages.grill.save_artifact"),
        patch("graft.stages.grill.mark_stage_complete"),
    ):
        result = await grill_node(state, mock_ui)

    assert result["feature_spec"] == {}
    mock_ui.error.assert_called_once_with("Failed to parse feature_spec.json.")


# ---------------------------------------------------------------------------
# grill_node: cleanup removes feature_spec.json from repo
# ---------------------------------------------------------------------------


async def test_grill_node_cleans_up_feature_spec_from_repo(
    base_state, mock_ui, tmp_repo
):
    """After reading feature_spec.json, grill_node deletes it from the repo dir."""
    repo, project = tmp_repo
    spec_path = Path(repo) / "feature_spec.json"

    async def _fake_run_agent(**kwargs):
        if kwargs["stage"] == "grill_compile":
            spec_path.write_text(json.dumps({"feature_name": "Test"}))

    with (
        patch(
            "graft.stages.grill.run_agent", new=AsyncMock(side_effect=_fake_run_agent)
        ),
        patch("graft.stages.grill.save_artifact"),
        patch("graft.stages.grill.mark_stage_complete"),
    ):
        await grill_node(base_state, mock_ui)

    # The spec file should have been removed from the repo
    assert not spec_path.exists()


async def test_grill_node_no_cleanup_error_when_spec_missing(mock_ui, tmp_repo):
    """When feature_spec.json was never created, cleanup doesn't raise."""
    repo, project = tmp_repo
    state = {
        "repo_path": str(repo),
        "project_dir": str(project),
        "feature_prompt": "Add X",
        "codebase_profile": {},
        "technical_assessment": {"open_questions": []},
        "constraints": [],
        "model": None,
    }

    async def _fake_run_agent(**kwargs):
        pass  # No files created

    with (
        patch(
            "graft.stages.grill.run_agent", new=AsyncMock(side_effect=_fake_run_agent)
        ),
        patch("graft.stages.grill.save_artifact"),
        patch("graft.stages.grill.mark_stage_complete"),
    ):
        # Should not raise
        result = await grill_node(state, mock_ui)

    assert result["feature_spec"] == {}


# ---------------------------------------------------------------------------
# grill_node: research_redo_needed flag
# ---------------------------------------------------------------------------


async def test_grill_node_research_redo_needed_true(base_state, mock_ui, tmp_repo):
    """When feature_spec contains research_redo_needed=True, grill_node propagates it."""
    repo, project = tmp_repo
    spec_with_redo = {
        "feature_name": "Dark Mode",
        "research_redo_needed": True,
    }
    spec_path = Path(repo) / "feature_spec.json"

    async def _fake_run_agent(**kwargs):
        if kwargs["stage"] == "grill_compile":
            spec_path.write_text(json.dumps(spec_with_redo))

    with (
        patch(
            "graft.stages.grill.run_agent", new=AsyncMock(side_effect=_fake_run_agent)
        ),
        patch("graft.stages.grill.save_artifact"),
        patch("graft.stages.grill.mark_stage_complete"),
    ):
        result = await grill_node(base_state, mock_ui)

    assert result["research_redo_needed"] is True
    mock_ui.info.assert_any_call(
        "Grill revealed a fundamental assumption change — looping back to Research."
    )


async def test_grill_node_research_redo_needed_false(base_state, mock_ui, tmp_repo):
    """When feature_spec has no research_redo_needed, it defaults to False."""
    repo, project = tmp_repo
    spec_path = Path(repo) / "feature_spec.json"

    async def _fake_run_agent(**kwargs):
        if kwargs["stage"] == "grill_compile":
            spec_path.write_text(json.dumps({"feature_name": "Dark Mode"}))

    with (
        patch(
            "graft.stages.grill.run_agent", new=AsyncMock(side_effect=_fake_run_agent)
        ),
        patch("graft.stages.grill.save_artifact"),
        patch("graft.stages.grill.mark_stage_complete"),
    ):
        result = await grill_node(base_state, mock_ui)

    assert result["research_redo_needed"] is False


# ---------------------------------------------------------------------------
# grill_node: stage lifecycle and artifact saving
# ---------------------------------------------------------------------------


async def test_grill_node_stage_lifecycle(base_state, mock_ui, tmp_repo):
    """grill_node calls stage_start, mark_stage_complete, and stage_done."""
    repo, project = tmp_repo
    spec_path = Path(repo) / "feature_spec.json"

    async def _fake_run_agent(**kwargs):
        if kwargs["stage"] == "grill_compile":
            spec_path.write_text(json.dumps({}))

    with (
        patch(
            "graft.stages.grill.run_agent", new=AsyncMock(side_effect=_fake_run_agent)
        ),
        patch("graft.stages.grill.save_artifact"),
        patch("graft.stages.grill.mark_stage_complete") as mock_mark,
    ):
        await grill_node(base_state, mock_ui)

    mock_ui.stage_start.assert_called_once_with("grill")
    mock_ui.stage_done.assert_called_once_with("grill")
    mock_mark.assert_called_once_with(str(project), "grill")


# ---------------------------------------------------------------------------
# grill_node: questions as plain strings (non-dict)
# ---------------------------------------------------------------------------


async def test_grill_node_handles_string_questions(mock_ui, tmp_repo):
    """When open_questions are plain strings instead of dicts, grill_node
    handles them gracefully with default category/recommended."""
    repo, project = tmp_repo
    state = {
        "repo_path": str(repo),
        "project_dir": str(project),
        "feature_prompt": "Add notifications",
        "codebase_profile": {},
        "technical_assessment": {
            "open_questions": [
                "Should we use push notifications?",
                "Email fallback needed?",
            ]
        },
        "constraints": [],
        "model": None,
    }
    spec_path = Path(repo) / "feature_spec.json"

    async def _fake_run_agent(**kwargs):
        if kwargs["stage"] == "grill_compile":
            spec_path.write_text(json.dumps({"feature_name": "Notifications"}))

    mock_ui.grill_question = MagicMock(return_value="Yes")

    with (
        patch(
            "graft.stages.grill.run_agent", new=AsyncMock(side_effect=_fake_run_agent)
        ),
        patch("graft.stages.grill.save_artifact"),
        patch("graft.stages.grill.mark_stage_complete"),
    ):
        result = await grill_node(state, mock_ui)

    assert mock_ui.grill_question.call_count == 2

    # String questions should use defaults
    calls = mock_ui.grill_question.call_args_list
    assert calls[0] == call(
        "Should we use push notifications?",
        "No recommendation",
        "intent",
        1,
    )
    assert calls[1] == call(
        "Email fallback needed?",
        "No recommendation",
        "intent",
        2,
    )

    # Transcript should reflect string questions
    transcript = result["grill_transcript"]
    assert "Q1 [intent]: Should we use push notifications?" in transcript
    assert "Recommended: No recommendation" in transcript


async def test_grill_node_handles_mixed_question_types(mock_ui, tmp_repo):
    """When open_questions mix dicts and strings, both are handled."""
    repo, project = tmp_repo
    state = {
        "repo_path": str(repo),
        "project_dir": str(project),
        "feature_prompt": "Add exports",
        "codebase_profile": {},
        "technical_assessment": {
            "open_questions": [
                {
                    "question": "CSV or Excel?",
                    "recommended_answer": "CSV",
                    "category": "preference",
                },
                "Include headers?",
            ]
        },
        "constraints": [],
        "model": None,
    }
    spec_path = Path(repo) / "feature_spec.json"

    async def _fake_run_agent(**kwargs):
        if kwargs["stage"] == "grill_compile":
            spec_path.write_text(json.dumps({}))

    mock_ui.grill_question = MagicMock(return_value="Both")

    with (
        patch(
            "graft.stages.grill.run_agent", new=AsyncMock(side_effect=_fake_run_agent)
        ),
        patch("graft.stages.grill.save_artifact"),
        patch("graft.stages.grill.mark_stage_complete"),
    ):
        result = await grill_node(state, mock_ui)

    calls = mock_ui.grill_question.call_args_list
    assert calls[0] == call("CSV or Excel?", "CSV", "preference", 1)
    assert calls[1] == call("Include headers?", "No recommendation", "intent", 2)


# ---------------------------------------------------------------------------
# grill_node: compile prompt includes constraints
# ---------------------------------------------------------------------------


async def test_grill_node_compile_prompt_includes_constraints(
    base_state, mock_ui, tmp_repo
):
    """Compile agent receives constraints in its prompt."""
    repo, project = tmp_repo
    spec_path = Path(repo) / "feature_spec.json"
    captured_prompts = {}

    async def _fake_run_agent(**kwargs):
        if kwargs["stage"] == "grill_compile":
            captured_prompts["user_prompt"] = kwargs["user_prompt"]
            spec_path.write_text(json.dumps({}))

    with (
        patch(
            "graft.stages.grill.run_agent", new=AsyncMock(side_effect=_fake_run_agent)
        ),
        patch("graft.stages.grill.save_artifact"),
        patch("graft.stages.grill.mark_stage_complete"),
    ):
        await grill_node(base_state, mock_ui)

    assert "Must support IE11" in captured_prompts["user_prompt"]


async def test_grill_node_compile_prompt_no_constraints(mock_ui, tmp_repo):
    """Compile agent prompt handles empty constraints gracefully."""
    repo, project = tmp_repo
    state = {
        "repo_path": str(repo),
        "project_dir": str(project),
        "feature_prompt": "Add feature",
        "codebase_profile": {},
        "technical_assessment": {
            "open_questions": [
                {"question": "Q?", "recommended_answer": "A", "category": "intent"}
            ]
        },
        "constraints": [],
        "model": None,
    }
    spec_path = Path(repo) / "feature_spec.json"
    captured_prompts = {}

    async def _fake_run_agent(**kwargs):
        if kwargs["stage"] == "grill_compile":
            captured_prompts["user_prompt"] = kwargs["user_prompt"]
            spec_path.write_text(json.dumps({}))

    with (
        patch(
            "graft.stages.grill.run_agent", new=AsyncMock(side_effect=_fake_run_agent)
        ),
        patch("graft.stages.grill.save_artifact"),
        patch("graft.stages.grill.mark_stage_complete"),
    ):
        await grill_node(state, mock_ui)

    assert "CONSTRAINTS: None" in captured_prompts["user_prompt"]


# ---------------------------------------------------------------------------
# _generate_questions: agent parameters verification
# ---------------------------------------------------------------------------


async def test_generate_questions_agent_params(tmp_repo, mock_ui):
    """_generate_questions passes correct tools and stage to run_agent."""
    repo, project = tmp_repo
    captured_kwargs = {}

    async def _fake_run_agent(**kwargs):
        captured_kwargs.update(kwargs)

    with patch(
        "graft.stages.grill.run_agent", new=AsyncMock(side_effect=_fake_run_agent)
    ):
        await _generate_questions(
            repo_path=str(repo),
            project_dir=str(project),
            feature_prompt="Feature",
            codebase_profile={},
            technical_assessment={},
            ui=mock_ui,
            model="claude-sonnet-4-20250514",
        )

    assert captured_kwargs["stage"] == "grill_generate"
    assert captured_kwargs["model"] == "claude-sonnet-4-20250514"
    assert captured_kwargs["max_turns"] == 10
    assert set(captured_kwargs["allowed_tools"]) == {
        "Read",
        "Write",
        "Bash",
        "Glob",
        "Grep",
    }


# ---------------------------------------------------------------------------
# grill_node: missing state fields default gracefully
# ---------------------------------------------------------------------------


async def test_grill_node_minimal_state(mock_ui, tmp_repo):
    """grill_node handles a minimal state with missing optional fields."""
    repo, project = tmp_repo
    state = {
        "repo_path": str(repo),
        "project_dir": str(project),
        # no feature_prompt, codebase_profile, technical_assessment, constraints, model
    }
    spec_path = Path(repo) / "feature_spec.json"

    async def _fake_run_agent(**kwargs):
        if kwargs["stage"] == "grill_compile":
            spec_path.write_text(json.dumps({}))

    with (
        patch(
            "graft.stages.grill.run_agent", new=AsyncMock(side_effect=_fake_run_agent)
        ),
        patch("graft.stages.grill.save_artifact"),
        patch("graft.stages.grill.mark_stage_complete"),
    ):
        result = await grill_node(state, mock_ui)

    assert result["grill_complete"] is True
    assert result["feature_spec"] == {}
