"""Tests for graft.stages.grill."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graft.stages.grill import _generate_questions, grill_node, grill_router


# ---------------------------------------------------------------------------
# Router tests (existing)
# ---------------------------------------------------------------------------


def test_grill_router_no_redo():
    """Normal flow: Grill → Plan."""
    assert grill_router({"research_redo_needed": False}) == "plan"


def test_grill_router_redo():
    """Loop-back: Grill → Research when redo needed."""
    assert grill_router({"research_redo_needed": True}) == "research"


def test_grill_router_default():
    """Default (no flag): proceed to Plan."""
    assert grill_router({}) == "plan"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ui(**overrides) -> MagicMock:
    """Create a mock UI with sensible defaults."""
    ui = MagicMock()
    ui.stage_start = MagicMock()
    ui.stage_done = MagicMock()
    ui.info = MagicMock()
    ui.error = MagicMock()
    ui.grill_question = MagicMock(return_value="accepted recommendation")
    for k, v in overrides.items():
        setattr(ui, k, v)
    return ui


def _make_state(tmp_path: Path, **overrides) -> dict:
    """Build a minimal FeatureState dict pointing at *tmp_path*."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(exist_ok=True)
    project_dir = tmp_path / "project"
    (project_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (project_dir / "logs").mkdir(parents=True, exist_ok=True)

    state = {
        "repo_path": str(repo_dir),
        "project_dir": str(project_dir),
        "feature_prompt": "Add a leaderboard feature",
        "codebase_profile": {"language": "python"},
        "technical_assessment": {},
        "constraints": ["must be fast"],
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# grill_node – pre-populated open_questions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grill_node_with_open_questions(tmp_path):
    """When technical_assessment has open_questions, grill_node walks through
    them, calls ui.grill_question for each, and compiles a feature spec."""
    questions = [
        {
            "question": "Should the leaderboard be real-time?",
            "recommended_answer": "Yes, use websockets",
            "category": "intent",
        },
        {
            "question": "How many entries to show?",
            "recommended_answer": "Top 100",
            "category": "preference",
        },
    ]
    state = _make_state(
        tmp_path,
        technical_assessment={"open_questions": questions},
    )

    ui = _make_ui()
    # Return distinct answers for the two questions
    ui.grill_question = MagicMock(side_effect=["Yes, websockets", "50 entries"])

    # The agent writes feature_spec.json into repo_path during compilation
    spec_payload = {"feature_name": "Leaderboard", "decisions": []}

    async def fake_run_agent(**kwargs):
        spec_path = Path(kwargs["cwd"]) / "feature_spec.json"
        spec_path.write_text(json.dumps(spec_payload))

    with patch("graft.stages.grill.run_agent", new=fake_run_agent), \
         patch("graft.stages.grill.save_artifact") as mock_save, \
         patch("graft.stages.grill.mark_stage_complete") as mock_mark:

        result = await grill_node(state, ui)

    # Questions were presented
    assert ui.grill_question.call_count == 2

    # Returned dict has the compiled spec
    assert result["feature_spec"] == spec_payload
    assert result["grill_complete"] is True
    assert result["current_stage"] == "grill"

    # Transcript includes both Q&A pairs
    transcript = result["grill_transcript"]
    assert "Q1 [intent]" in transcript
    assert "Q2 [preference]" in transcript
    assert "Yes, websockets" in transcript
    assert "50 entries" in transcript

    # save_artifact called for transcript and spec
    assert mock_save.call_count == 2
    mock_mark.assert_called_once_with(str(tmp_path / "project"), "grill")


# ---------------------------------------------------------------------------
# grill_node – no open_questions triggers _generate_questions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grill_node_no_questions_triggers_generate(tmp_path):
    """When technical_assessment has no open_questions, grill_node calls
    _generate_questions via agent to produce them."""
    state = _make_state(tmp_path, technical_assessment={})

    ui = _make_ui()
    # _generate_questions will produce one question, grill_question answers it
    ui.grill_question = MagicMock(return_value="auto answer")

    generated_qs = [
        {
            "question": "What DB should we use?",
            "recommended_answer": "Postgres",
            "category": "intent",
        }
    ]

    call_count = {"generate": 0, "compile": 0}

    async def fake_run_agent(**kwargs):
        if kwargs["stage"] == "grill_generate":
            call_count["generate"] += 1
            # Agent writes open_questions.json
            q_path = Path(kwargs["cwd"]) / "open_questions.json"
            q_path.write_text(json.dumps(generated_qs))
        elif kwargs["stage"] == "grill_compile":
            call_count["compile"] += 1
            spec_path = Path(kwargs["cwd"]) / "feature_spec.json"
            spec_path.write_text(json.dumps({"feature_name": "Leaderboard"}))

    with patch("graft.stages.grill.run_agent", new=fake_run_agent), \
         patch("graft.stages.grill.save_artifact"), \
         patch("graft.stages.grill.mark_stage_complete"):

        result = await grill_node(state, ui)

    # _generate_questions was invoked (generate stage)
    assert call_count["generate"] == 1
    # Compile stage was also invoked
    assert call_count["compile"] == 1
    # The generated question was presented
    ui.grill_question.assert_called_once()
    assert "What DB should we use?" in ui.grill_question.call_args[0][0]
    # Info message about generating questions was shown
    ui.info.assert_any_call("No open questions from Research — generating questions from context...")


# ---------------------------------------------------------------------------
# _generate_questions – happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_questions_parses_json(tmp_path):
    """_generate_questions reads questions from the JSON file the agent writes."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = str(tmp_path / "project")
    ui = _make_ui()

    expected = [
        {"question": "Q1?", "recommended_answer": "A1", "category": "intent"},
        {"question": "Q2?", "recommended_answer": "A2", "category": "edge_case"},
    ]

    async def fake_run_agent(**kwargs):
        q_path = Path(kwargs["cwd"]) / "open_questions.json"
        q_path.write_text(json.dumps(expected))

    with patch("graft.stages.grill.run_agent", new=fake_run_agent):
        result = await _generate_questions(
            str(repo_dir), project_dir, "prompt", {}, {}, ui, None,
        )

    assert result == expected
    # Agent should clean up the temp file
    assert not (repo_dir / "open_questions.json").exists()


# ---------------------------------------------------------------------------
# _generate_questions – invalid JSON → empty list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_questions_invalid_json_returns_empty(tmp_path):
    """If the agent writes malformed JSON, _generate_questions returns []."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = str(tmp_path / "project")
    ui = _make_ui()

    async def fake_run_agent(**kwargs):
        q_path = Path(kwargs["cwd"]) / "open_questions.json"
        q_path.write_text("NOT VALID JSON {{{{")

    with patch("graft.stages.grill.run_agent", new=fake_run_agent):
        result = await _generate_questions(
            str(repo_dir), project_dir, "prompt", {}, {}, ui, None,
        )

    assert result == []


# ---------------------------------------------------------------------------
# _generate_questions – no file written → empty list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_questions_no_file_returns_empty(tmp_path):
    """If the agent doesn't write a file at all, _generate_questions returns []."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = str(tmp_path / "project")
    ui = _make_ui()

    async def fake_run_agent(**kwargs):
        pass  # agent does nothing

    with patch("graft.stages.grill.run_agent", new=fake_run_agent):
        result = await _generate_questions(
            str(repo_dir), project_dir, "prompt", {}, {}, ui, None,
        )

    assert result == []


# ---------------------------------------------------------------------------
# _generate_questions – non-list JSON → empty list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_questions_non_list_json_returns_empty(tmp_path):
    """If the agent writes a JSON object instead of a list, return []."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    project_dir = str(tmp_path / "project")
    ui = _make_ui()

    async def fake_run_agent(**kwargs):
        q_path = Path(kwargs["cwd"]) / "open_questions.json"
        q_path.write_text(json.dumps({"not": "a list"}))

    with patch("graft.stages.grill.run_agent", new=fake_run_agent):
        result = await _generate_questions(
            str(repo_dir), project_dir, "prompt", {}, {}, ui, None,
        )

    assert result == []


# ---------------------------------------------------------------------------
# grill_node – research_redo_needed flag propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grill_node_sets_research_redo_when_spec_says_so(tmp_path):
    """If the compiled feature_spec contains research_redo_needed=True the
    result dict propagates it so the router can loop back to Research."""
    questions = [{"question": "Keep going?", "recommended_answer": "Yes", "category": "intent"}]
    state = _make_state(
        tmp_path,
        technical_assessment={"open_questions": questions},
    )
    ui = _make_ui()

    async def fake_run_agent(**kwargs):
        if kwargs["stage"] == "grill_compile":
            spec_path = Path(kwargs["cwd"]) / "feature_spec.json"
            spec_path.write_text(json.dumps({
                "feature_name": "X",
                "research_redo_needed": True,
            }))

    with patch("graft.stages.grill.run_agent", new=fake_run_agent), \
         patch("graft.stages.grill.save_artifact"), \
         patch("graft.stages.grill.mark_stage_complete"):

        result = await grill_node(state, ui)

    assert result["research_redo_needed"] is True
    # UI informed the user about the redo
    ui.info.assert_any_call(
        "Grill revealed a fundamental assumption change — looping back to Research."
    )


@pytest.mark.asyncio
async def test_grill_node_no_research_redo_by_default(tmp_path):
    """When feature_spec has no research_redo_needed flag, default is False."""
    questions = [{"question": "Color?", "recommended_answer": "Blue", "category": "preference"}]
    state = _make_state(
        tmp_path,
        technical_assessment={"open_questions": questions},
    )
    ui = _make_ui()

    async def fake_run_agent(**kwargs):
        if kwargs["stage"] == "grill_compile":
            spec_path = Path(kwargs["cwd"]) / "feature_spec.json"
            spec_path.write_text(json.dumps({"feature_name": "Y"}))

    with patch("graft.stages.grill.run_agent", new=fake_run_agent), \
         patch("graft.stages.grill.save_artifact"), \
         patch("graft.stages.grill.mark_stage_complete"):

        result = await grill_node(state, ui)

    assert result["research_redo_needed"] is False


# ---------------------------------------------------------------------------
# Transcript formatting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grill_transcript_formatting(tmp_path):
    """Verify transcript lines follow the expected format per question."""
    questions = [
        {
            "question": "Use caching?",
            "recommended_answer": "Redis",
            "category": "edge_case",
        },
    ]
    state = _make_state(
        tmp_path,
        technical_assessment={"open_questions": questions},
    )
    ui = _make_ui()
    ui.grill_question = MagicMock(return_value="Memcached instead")

    async def fake_run_agent(**kwargs):
        if kwargs["stage"] == "grill_compile":
            spec_path = Path(kwargs["cwd"]) / "feature_spec.json"
            spec_path.write_text(json.dumps({}))

    with patch("graft.stages.grill.run_agent", new=fake_run_agent), \
         patch("graft.stages.grill.save_artifact"), \
         patch("graft.stages.grill.mark_stage_complete"):

        result = await grill_node(state, ui)

    lines = result["grill_transcript"].split("\n")
    assert lines[0] == "Q1 [edge_case]: Use caching?"
    assert lines[1] == "  Recommended: Redis"
    assert lines[2] == "  Answer: Memcached instead"
    # Trailing blank line after the Q&A block
    assert lines[3] == ""


# ---------------------------------------------------------------------------
# grill_node – malformed feature_spec.json
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grill_node_bad_spec_json_logs_error(tmp_path):
    """If the compiled feature_spec.json is invalid JSON, ui.error is called
    and the result contains an empty dict for feature_spec."""
    questions = [{"question": "OK?", "recommended_answer": "Yes", "category": "intent"}]
    state = _make_state(
        tmp_path,
        technical_assessment={"open_questions": questions},
    )
    ui = _make_ui()

    async def fake_run_agent(**kwargs):
        if kwargs["stage"] == "grill_compile":
            spec_path = Path(kwargs["cwd"]) / "feature_spec.json"
            spec_path.write_text("BROKEN JSON {{{")

    with patch("graft.stages.grill.run_agent", new=fake_run_agent), \
         patch("graft.stages.grill.save_artifact"), \
         patch("graft.stages.grill.mark_stage_complete"):

        result = await grill_node(state, ui)

    assert result["feature_spec"] == {}
    ui.error.assert_called_once_with("Failed to parse feature_spec.json.")


# ---------------------------------------------------------------------------
# grill_node – string questions (non-dict) in open_questions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grill_node_string_questions(tmp_path):
    """open_questions can be plain strings — grill_node handles them
    by falling back to defaults for recommended and category."""
    questions = ["Is this a string question?"]
    state = _make_state(
        tmp_path,
        technical_assessment={"open_questions": questions},
    )
    ui = _make_ui()
    ui.grill_question = MagicMock(return_value="Yes it is")

    async def fake_run_agent(**kwargs):
        if kwargs["stage"] == "grill_compile":
            spec_path = Path(kwargs["cwd"]) / "feature_spec.json"
            spec_path.write_text(json.dumps({"feature_name": "Str"}))

    with patch("graft.stages.grill.run_agent", new=fake_run_agent), \
         patch("graft.stages.grill.save_artifact"), \
         patch("graft.stages.grill.mark_stage_complete"):

        result = await grill_node(state, ui)

    # The string was used as the question text
    call_args = ui.grill_question.call_args[0]
    assert call_args[0] == "Is this a string question?"
    assert call_args[1] == "No recommendation"  # default recommended
    assert call_args[2] == "intent"  # default category

    assert "Q1 [intent]: Is this a string question?" in result["grill_transcript"]


# ---------------------------------------------------------------------------
# grill_node – spec_path cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grill_node_cleans_up_spec_file(tmp_path):
    """After reading feature_spec.json from repo_path, grill_node deletes it."""
    questions = [{"question": "Q?", "recommended_answer": "A", "category": "intent"}]
    state = _make_state(
        tmp_path,
        technical_assessment={"open_questions": questions},
    )
    ui = _make_ui()

    async def fake_run_agent(**kwargs):
        if kwargs["stage"] == "grill_compile":
            spec_path = Path(kwargs["cwd"]) / "feature_spec.json"
            spec_path.write_text(json.dumps({"feature_name": "Cleanup"}))

    with patch("graft.stages.grill.run_agent", new=fake_run_agent), \
         patch("graft.stages.grill.save_artifact"), \
         patch("graft.stages.grill.mark_stage_complete"):

        result = await grill_node(state, ui)

    # The temp file in repo_path should have been removed
    spec_path = Path(state["repo_path"]) / "feature_spec.json"
    assert not spec_path.exists()
    # But the spec was captured in the result
    assert result["feature_spec"]["feature_name"] == "Cleanup"
