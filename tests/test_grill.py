"""Tests for graft.stages.grill."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graft.stages.grill import grill_node, grill_router


# ---------------------------------------------------------------------------
# Existing grill_router tests (preserved)
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
# Fixtures for grill_node tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_project(tmp_path):
    """Create a temporary project directory with the artifacts/ and metadata.json
    needed by save_artifact / mark_stage_complete."""
    project_dir = tmp_path / "project"
    (project_dir / "artifacts").mkdir(parents=True)
    metadata = {
        "project_id": "feat_test1234",
        "repo_path": str(tmp_path / "repo"),
        "feature_prompt": "test feature",
        "status": "in_progress",
        "stages_completed": [],
    }
    (project_dir / "metadata.json").write_text(json.dumps(metadata))
    # Also create the repo directory
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True)
    return tmp_path, repo_dir, project_dir


def _make_state(tmp_project, *, open_questions=None, constraints=None, extra=None):
    """Build a minimal FeatureState dict for grill_node."""
    tmp_path, repo_dir, project_dir = tmp_project
    technical_assessment = {}
    if open_questions is not None:
        technical_assessment["open_questions"] = open_questions
    state = {
        "repo_path": str(repo_dir),
        "project_dir": str(project_dir),
        "feature_prompt": "Build a trade system",
        "codebase_profile": {"language": "python"},
        "technical_assessment": technical_assessment,
        "constraints": constraints or [],
    }
    if extra:
        state.update(extra)
    return state


def _make_ui(answers=None):
    """Create a mock UI whose grill_question returns answers sequentially."""
    ui = MagicMock()
    if answers is not None:
        ui.grill_question = MagicMock(side_effect=answers)
    else:
        ui.grill_question = MagicMock(return_value="accepted")
    return ui


def _agent_writes_spec(spec_dict):
    """Return an async side-effect for run_agent that writes feature_spec.json."""

    async def _side_effect(*, cwd, project_dir, **kwargs):
        spec_path = Path(cwd) / "feature_spec.json"
        spec_path.write_text(json.dumps(spec_dict))

    return _side_effect


# ---------------------------------------------------------------------------
# grill_node — question phase
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_question_phase_walks_open_questions(mock_agent, tmp_project):
    """Each open_question triggers a UI.grill_question call with correct args."""
    questions = [
        {
            "question": "Should trades expire?",
            "recommended_answer": "Yes, after 24h",
            "category": "edge_case",
        },
        {
            "question": "Allow counter-offers?",
            "recommended_answer": "No for MVP",
            "category": "intent",
        },
    ]
    state = _make_state(tmp_project, open_questions=questions)
    ui = _make_ui(answers=["yes, 24 hours", "no"])
    mock_agent.side_effect = _agent_writes_spec({"feature_name": "Trade"})

    result = await grill_node(state, ui)

    # grill_question called once per open question
    assert ui.grill_question.call_count == 2

    # Verify first call args: (question, recommended, category, number)
    first_call = ui.grill_question.call_args_list[0]
    assert first_call[0] == ("Should trades expire?", "Yes, after 24h", "edge_case", 1)

    second_call = ui.grill_question.call_args_list[1]
    assert second_call[0] == ("Allow counter-offers?", "No for MVP", "intent", 2)


# ---------------------------------------------------------------------------
# grill_node — decisions list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_collects_decisions(mock_agent, tmp_project):
    """Decisions list in the compile prompt captures question, recommended, answer, category."""
    questions = [
        {
            "question": "Auth method?",
            "recommended_answer": "Supabase RLS",
            "category": "preference",
        },
    ]
    state = _make_state(tmp_project, open_questions=questions)
    ui = _make_ui(answers=["use clerk instead"])

    spec = {"feature_name": "Trade", "decisions": []}
    mock_agent.side_effect = _agent_writes_spec(spec)

    await grill_node(state, ui)

    # The compile agent call is the first (only) run_agent call
    assert mock_agent.call_count == 1
    compile_prompt = mock_agent.call_args[1]["user_prompt"]
    # The prompt must contain the user's answer and the question
    assert "Auth method?" in compile_prompt
    assert "use clerk instead" in compile_prompt
    assert "Supabase RLS" in compile_prompt


# ---------------------------------------------------------------------------
# grill_node — compilation phase
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_compilation_returns_feature_spec(mock_agent, tmp_project):
    """run_agent writes feature_spec.json, grill_node reads it back and returns it."""
    questions = [
        {"question": "Q1?", "recommended_answer": "R1", "category": "intent"},
    ]
    state = _make_state(tmp_project, open_questions=questions)
    ui = _make_ui(answers=["my answer"])

    expected_spec = {
        "feature_name": "Trade System",
        "decisions": [{"question": "Q1?", "answer": "my answer"}],
        "scope": {"mvp": ["item1"], "follow_up": []},
    }
    mock_agent.side_effect = _agent_writes_spec(expected_spec)

    result = await grill_node(state, ui)

    assert result["feature_spec"] == expected_spec
    # Temp spec file in repo should be cleaned up
    assert not (Path(state["repo_path"]) / "feature_spec.json").exists()


# ---------------------------------------------------------------------------
# grill_node — transcript building
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_transcript_content(mock_agent, tmp_project):
    """grill_transcript.md has Q#, category, question, recommended, and user answer."""
    questions = [
        {
            "question": "Mobile support?",
            "recommended_answer": "Web only for MVP",
            "category": "prioritization",
        },
        {
            "question": "Real-time updates?",
            "recommended_answer": "Polling is fine",
            "category": "preference",
        },
    ]
    state = _make_state(tmp_project, open_questions=questions)
    ui = _make_ui(answers=["both web and mobile", "use websockets"])
    mock_agent.side_effect = _agent_writes_spec({})

    result = await grill_node(state, ui)

    transcript = result["grill_transcript"]
    # Question labels
    assert "Q1 [prioritization]: Mobile support?" in transcript
    assert "Q2 [preference]: Real-time updates?" in transcript
    # Recommended answers
    assert "Recommended: Web only for MVP" in transcript
    assert "Recommended: Polling is fine" in transcript
    # User answers
    assert "Answer: both web and mobile" in transcript
    assert "Answer: use websockets" in transcript


# ---------------------------------------------------------------------------
# grill_node — artifact saving
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_saves_artifacts(mock_agent, tmp_project):
    """Both grill_transcript.md and feature_spec.json are saved to project artifacts."""
    _, repo_dir, project_dir = tmp_project
    questions = [
        {"question": "Q?", "recommended_answer": "R", "category": "intent"},
    ]
    state = _make_state(tmp_project, open_questions=questions)
    ui = _make_ui(answers=["A"])

    spec = {"feature_name": "Test"}
    mock_agent.side_effect = _agent_writes_spec(spec)

    await grill_node(state, ui)

    artifacts_dir = project_dir / "artifacts"
    # grill_transcript.md exists and has content
    transcript_path = artifacts_dir / "grill_transcript.md"
    assert transcript_path.exists()
    assert "Q?" in transcript_path.read_text()

    # feature_spec.json exists and is valid JSON matching the spec
    spec_path = artifacts_dir / "feature_spec.json"
    assert spec_path.exists()
    saved_spec = json.loads(spec_path.read_text())
    assert saved_spec == spec


# ---------------------------------------------------------------------------
# grill_node — empty open_questions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_empty_questions_generates_and_produces_spec(
    mock_agent, tmp_project
):
    """When open_questions is empty, _generate_questions is called first, then compile."""
    _, repo_dir, project_dir = tmp_project
    state = _make_state(tmp_project, open_questions=[])
    ui = _make_ui(answers=["generated answer"])

    # First run_agent call: _generate_questions writes open_questions.json
    # Second run_agent call: compile writes feature_spec.json
    generated_questions = [
        {
            "question": "Generated Q?",
            "recommended_answer": "Gen R",
            "category": "intent",
        },
    ]

    call_count = 0

    async def agent_side_effect(*, cwd, project_dir, stage, **kwargs):
        nonlocal call_count
        if stage == "grill_generate":
            # Write generated questions
            q_path = Path(cwd) / "open_questions.json"
            q_path.write_text(json.dumps(generated_questions))
        elif stage == "grill_compile":
            # Write feature spec
            spec_path = Path(cwd) / "feature_spec.json"
            spec_path.write_text(json.dumps({"feature_name": "Generated"}))
        call_count += 1

    mock_agent.side_effect = agent_side_effect

    result = await grill_node(state, ui)

    # _generate_questions agent + compile agent = 2 calls
    assert call_count == 2
    # UI was told no open questions
    ui.info.assert_any_call(
        "No open questions from Research — generating questions from context..."
    )
    # grill_question was called with the generated question
    ui.grill_question.assert_called_once_with("Generated Q?", "Gen R", "intent", 1)
    # Valid spec returned
    assert result["feature_spec"] == {"feature_name": "Generated"}
    assert result["grill_complete"] is True


# ---------------------------------------------------------------------------
# grill_node — grill_complete flag
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_sets_grill_complete(mock_agent, tmp_project):
    """Returned state always has grill_complete=True."""
    questions = [
        {"question": "Q?", "recommended_answer": "R", "category": "intent"},
    ]
    state = _make_state(tmp_project, open_questions=questions)
    ui = _make_ui(answers=["ok"])
    mock_agent.side_effect = _agent_writes_spec({})

    result = await grill_node(state, ui)

    assert result["grill_complete"] is True
    assert result["current_stage"] == "grill"


# ---------------------------------------------------------------------------
# grill_node — research_redo_needed propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_research_redo_needed_true(mock_agent, tmp_project):
    """When feature_spec contains research_redo_needed=True, it propagates to result."""
    questions = [
        {"question": "Q?", "recommended_answer": "R", "category": "intent"},
    ]
    state = _make_state(tmp_project, open_questions=questions)
    ui = _make_ui(answers=["completely different approach"])

    mock_agent.side_effect = _agent_writes_spec({"research_redo_needed": True})

    result = await grill_node(state, ui)

    assert result["research_redo_needed"] is True
    # UI should indicate the loop-back
    ui.info.assert_any_call(
        "Grill revealed a fundamental assumption change — looping back to Research."
    )


@pytest.mark.asyncio
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_research_redo_needed_false(mock_agent, tmp_project):
    """When feature_spec does not flag redo, research_redo_needed is False."""
    questions = [
        {"question": "Q?", "recommended_answer": "R", "category": "intent"},
    ]
    state = _make_state(tmp_project, open_questions=questions)
    ui = _make_ui(answers=["sounds good"])

    mock_agent.side_effect = _agent_writes_spec({"feature_name": "OK"})

    result = await grill_node(state, ui)

    assert result["research_redo_needed"] is False


# ---------------------------------------------------------------------------
# grill_node — plain string questions (not dicts)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_handles_plain_string_questions(mock_agent, tmp_project):
    """open_questions entries that are plain strings instead of dicts are handled."""
    questions = ["What color scheme?", "How many pages?"]
    state = _make_state(tmp_project, open_questions=questions)
    ui = _make_ui(answers=["blue", "five"])
    mock_agent.side_effect = _agent_writes_spec({})

    result = await grill_node(state, ui)

    # Both questions asked with defaults for recommended/category
    assert ui.grill_question.call_count == 2
    first = ui.grill_question.call_args_list[0][0]
    assert first == ("What color scheme?", "No recommendation", "intent", 1)
    second = ui.grill_question.call_args_list[1][0]
    assert second == ("How many pages?", "No recommendation", "intent", 2)

    assert "Answer: blue" in result["grill_transcript"]
    assert "Answer: five" in result["grill_transcript"]


# ---------------------------------------------------------------------------
# grill_node — malformed feature_spec.json from agent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_handles_invalid_spec_json(mock_agent, tmp_project):
    """When agent writes invalid JSON to feature_spec.json, grill_node still completes."""
    _, repo_dir, project_dir = tmp_project
    questions = [
        {"question": "Q?", "recommended_answer": "R", "category": "intent"},
    ]
    state = _make_state(tmp_project, open_questions=questions)
    ui = _make_ui(answers=["ok"])

    async def _write_bad_json(*, cwd, **kwargs):
        (Path(cwd) / "feature_spec.json").write_text("NOT VALID JSON {{{")

    mock_agent.side_effect = _write_bad_json

    result = await grill_node(state, ui)

    # Should return empty dict for spec and report the error
    assert result["feature_spec"] == {}
    ui.error.assert_called_once_with("Failed to parse feature_spec.json.")
    assert result["grill_complete"] is True


# ---------------------------------------------------------------------------
# grill_node — agent doesn't write feature_spec.json at all
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_handles_missing_spec_file(mock_agent, tmp_project):
    """When the compile agent doesn't produce feature_spec.json, node still completes."""
    questions = [
        {"question": "Q?", "recommended_answer": "R", "category": "intent"},
    ]
    state = _make_state(tmp_project, open_questions=questions)
    ui = _make_ui(answers=["ok"])

    # Agent does nothing — no file written
    mock_agent.side_effect = AsyncMock()

    result = await grill_node(state, ui)

    assert result["feature_spec"] == {}
    assert result["grill_complete"] is True
