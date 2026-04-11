"""Tests for graft.stages.grill."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graft.stages.grill import _generate_questions, grill_node, grill_router


# ---------------------------------------------------------------------------
# grill_router (existing tests)
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
# Helpers
# ---------------------------------------------------------------------------


def _make_ui() -> MagicMock:
    """Return a mock UI with all methods used by grill_node."""
    ui = MagicMock()
    ui.stage_start = MagicMock()
    ui.stage_done = MagicMock()
    ui.info = MagicMock()
    ui.error = MagicMock()
    ui.grill_question = MagicMock(return_value="user answer")
    return ui


def _base_state(tmp_path: Path) -> dict:
    """Minimal valid state for grill_node."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "artifacts").mkdir()
    # metadata.json needed by mark_stage_complete
    meta = {
        "project_id": "test",
        "stages_completed": [],
        "status": "in_progress",
    }
    (project_dir / "metadata.json").write_text(json.dumps(meta))
    return {
        "repo_path": str(repo_path),
        "project_dir": str(project_dir),
        "feature_prompt": "Build a trade system",
        "codebase_profile": {"language": "python"},
        "technical_assessment": {
            "open_questions": [
                {
                    "question": "Should trades expire?",
                    "recommended_answer": "Yes, after 24h",
                    "category": "intent",
                },
                {
                    "question": "Max items per trade?",
                    "recommended_answer": "5",
                    "category": "edge_case",
                },
            ],
        },
        "constraints": ["Must use Supabase"],
    }


# ---------------------------------------------------------------------------
# grill_node — core happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grill_node_walks_all_questions(tmp_path):
    """grill_node calls ui.grill_question once per open question."""
    state = _base_state(tmp_path)
    ui = _make_ui()
    spec = {"feature_name": "Trade System", "decisions": []}
    spec_path = Path(state["repo_path"]) / "feature_spec.json"

    async def _write_spec(**kwargs):
        spec_path.write_text(json.dumps(spec))

    with patch("graft.stages.grill.run_agent", new_callable=AsyncMock, side_effect=_write_spec):
        result = await grill_node(state, ui)

    assert ui.grill_question.call_count == 2
    # Verify the question text passed to UI
    first_call = ui.grill_question.call_args_list[0]
    assert first_call[0][0] == "Should trades expire?"
    assert first_call[0][1] == "Yes, after 24h"
    assert first_call[0][2] == "intent"
    assert first_call[0][3] == 1  # question number


@pytest.mark.asyncio
async def test_grill_node_returns_correct_state_keys(tmp_path):
    """grill_node result has all expected state keys."""
    state = _base_state(tmp_path)
    ui = _make_ui()
    spec = {"feature_name": "Trade System"}
    spec_path = Path(state["repo_path"]) / "feature_spec.json"

    async def _write_spec(**kwargs):
        spec_path.write_text(json.dumps(spec))

    with patch("graft.stages.grill.run_agent", new_callable=AsyncMock, side_effect=_write_spec):
        result = await grill_node(state, ui)

    assert result["grill_complete"] is True
    assert result["current_stage"] == "grill"
    assert result["research_redo_needed"] is False
    assert isinstance(result["feature_spec"], dict)
    assert isinstance(result["grill_transcript"], str)


@pytest.mark.asyncio
async def test_grill_node_saves_transcript_artifact(tmp_path):
    """grill_node writes grill_transcript.md to the artifacts directory."""
    state = _base_state(tmp_path)
    ui = _make_ui()
    ui.grill_question.side_effect = ["yes expire", "10 items"]
    spec_path = Path(state["repo_path"]) / "feature_spec.json"

    async def _write_spec(**kwargs):
        spec_path.write_text(json.dumps({"feature_name": "Trade"}))

    with patch("graft.stages.grill.run_agent", new_callable=AsyncMock, side_effect=_write_spec):
        result = await grill_node(state, ui)

    transcript_path = Path(state["project_dir"]) / "artifacts" / "grill_transcript.md"
    assert transcript_path.exists()
    content = transcript_path.read_text()
    assert "Should trades expire?" in content
    assert "yes expire" in content
    assert "Max items per trade?" in content
    assert "10 items" in content


@pytest.mark.asyncio
async def test_grill_node_compiles_feature_spec(tmp_path):
    """grill_node reads the agent-written feature_spec.json and stores it."""
    state = _base_state(tmp_path)
    ui = _make_ui()
    spec = {"feature_name": "Trade System", "scope": {"mvp": ["trades"]}}
    spec_path = Path(state["repo_path"]) / "feature_spec.json"

    async def _write_spec(**kwargs):
        spec_path.write_text(json.dumps(spec))

    with patch("graft.stages.grill.run_agent", new_callable=AsyncMock, side_effect=_write_spec):
        result = await grill_node(state, ui)

    assert result["feature_spec"]["feature_name"] == "Trade System"
    assert result["feature_spec"]["scope"]["mvp"] == ["trades"]
    # The temp file in repo should be cleaned up
    assert not spec_path.exists()
    # The artifact copy should be saved
    artifact_path = Path(state["project_dir"]) / "artifacts" / "feature_spec.json"
    assert artifact_path.exists()


@pytest.mark.asyncio
async def test_grill_node_handles_invalid_spec_json(tmp_path):
    """grill_node handles malformed feature_spec.json gracefully."""
    state = _base_state(tmp_path)
    ui = _make_ui()
    spec_path = Path(state["repo_path"]) / "feature_spec.json"

    async def _write_bad_spec(**kwargs):
        spec_path.write_text("NOT VALID JSON {{{")

    with patch("graft.stages.grill.run_agent", new_callable=AsyncMock, side_effect=_write_bad_spec):
        result = await grill_node(state, ui)

    ui.error.assert_called_once_with("Failed to parse feature_spec.json.")
    assert result["feature_spec"] == {}
    assert result["grill_complete"] is True


@pytest.mark.asyncio
async def test_grill_node_handles_missing_spec_file(tmp_path):
    """grill_node handles agent failing to write feature_spec.json."""
    state = _base_state(tmp_path)
    ui = _make_ui()

    with patch("graft.stages.grill.run_agent", new_callable=AsyncMock):
        result = await grill_node(state, ui)

    # No error called — just an empty spec
    ui.error.assert_not_called()
    assert result["feature_spec"] == {}
    assert result["grill_complete"] is True


@pytest.mark.asyncio
async def test_grill_node_research_redo_flag(tmp_path):
    """grill_node propagates research_redo_needed from feature_spec."""
    state = _base_state(tmp_path)
    ui = _make_ui()
    spec = {"feature_name": "Trade", "research_redo_needed": True}
    spec_path = Path(state["repo_path"]) / "feature_spec.json"

    async def _write_spec(**kwargs):
        spec_path.write_text(json.dumps(spec))

    with patch("graft.stages.grill.run_agent", new_callable=AsyncMock, side_effect=_write_spec):
        result = await grill_node(state, ui)

    assert result["research_redo_needed"] is True
    ui.info.assert_any_call(
        "Grill revealed a fundamental assumption change — looping back to Research."
    )


@pytest.mark.asyncio
async def test_grill_node_marks_stage_complete(tmp_path):
    """grill_node marks the 'grill' stage as complete in metadata."""
    state = _base_state(tmp_path)
    ui = _make_ui()
    spec_path = Path(state["repo_path"]) / "feature_spec.json"

    async def _write_spec(**kwargs):
        spec_path.write_text(json.dumps({}))

    with patch("graft.stages.grill.run_agent", new_callable=AsyncMock, side_effect=_write_spec):
        await grill_node(state, ui)

    meta = json.loads((Path(state["project_dir"]) / "metadata.json").read_text())
    assert "grill" in meta["stages_completed"]
    ui.stage_start.assert_called_once_with("grill")
    ui.stage_done.assert_called_once_with("grill")


# ---------------------------------------------------------------------------
# grill_node — question generation fallback (no open_questions)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grill_node_generates_questions_when_none_from_research(tmp_path):
    """When technical_assessment has no open_questions, agent generates them."""
    state = _base_state(tmp_path)
    # Remove open_questions so the generation path is triggered
    state["technical_assessment"] = {"reuse_analysis": "some analysis"}
    ui = _make_ui()
    ui.grill_question.return_value = "auto answer"

    repo_path = Path(state["repo_path"])
    spec_path = repo_path / "feature_spec.json"
    questions_path = repo_path / "open_questions.json"
    call_count = 0

    generated_questions = [
        {"question": "Generated Q1?", "recommended_answer": "R1", "category": "intent"},
        {"question": "Generated Q2?", "recommended_answer": "R2", "category": "preference"},
    ]

    async def _agent_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: generate questions
            questions_path.write_text(json.dumps(generated_questions))
        else:
            # Second call: compile spec
            spec_path.write_text(json.dumps({"feature_name": "Gen"}))

    with patch("graft.stages.grill.run_agent", new_callable=AsyncMock, side_effect=_agent_side_effect):
        result = await grill_node(state, ui)

    ui.info.assert_any_call("No open questions from Research — generating questions from context...")
    assert ui.grill_question.call_count == 2
    first_call = ui.grill_question.call_args_list[0]
    assert first_call[0][0] == "Generated Q1?"


@pytest.mark.asyncio
async def test_grill_node_empty_questions_from_generation(tmp_path):
    """When question generation produces nothing, grill proceeds with zero questions."""
    state = _base_state(tmp_path)
    state["technical_assessment"] = {}  # No open_questions key at all
    ui = _make_ui()

    spec_path = Path(state["repo_path"]) / "feature_spec.json"
    call_count = 0

    async def _agent_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Generate questions agent runs but produces no file
            pass
        else:
            spec_path.write_text(json.dumps({"feature_name": "Empty"}))

    with patch("graft.stages.grill.run_agent", new_callable=AsyncMock, side_effect=_agent_side_effect):
        result = await grill_node(state, ui)

    # No questions were asked
    ui.grill_question.assert_not_called()
    assert result["grill_transcript"] == ""
    assert result["grill_complete"] is True


# ---------------------------------------------------------------------------
# grill_node — UI interaction edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grill_node_user_accepts_recommended(tmp_path):
    """When user returns recommended answer, it is recorded in transcript."""
    state = _base_state(tmp_path)
    ui = _make_ui()
    # Simulate user pressing Enter (UI returns recommended)
    ui.grill_question.side_effect = ["Yes, after 24h", "5"]

    spec_path = Path(state["repo_path"]) / "feature_spec.json"

    async def _write_spec(**kwargs):
        spec_path.write_text(json.dumps({}))

    with patch("graft.stages.grill.run_agent", new_callable=AsyncMock, side_effect=_write_spec):
        result = await grill_node(state, ui)

    assert "Answer: Yes, after 24h" in result["grill_transcript"]
    assert "Answer: 5" in result["grill_transcript"]


@pytest.mark.asyncio
async def test_grill_node_string_questions(tmp_path):
    """open_questions as plain strings (not dicts) are handled gracefully."""
    state = _base_state(tmp_path)
    state["technical_assessment"] = {
        "open_questions": [
            "What auth system to use?",
            "Should we add rate limiting?",
        ],
    }
    ui = _make_ui()
    ui.grill_question.return_value = "my answer"

    spec_path = Path(state["repo_path"]) / "feature_spec.json"

    async def _write_spec(**kwargs):
        spec_path.write_text(json.dumps({}))

    with patch("graft.stages.grill.run_agent", new_callable=AsyncMock, side_effect=_write_spec):
        result = await grill_node(state, ui)

    assert ui.grill_question.call_count == 2
    first_call = ui.grill_question.call_args_list[0]
    assert first_call[0][0] == "What auth system to use?"
    assert first_call[0][1] == "No recommendation"  # default for plain strings
    assert first_call[0][2] == "intent"  # default category


@pytest.mark.asyncio
async def test_grill_node_no_constraints(tmp_path):
    """grill_node works when constraints list is empty."""
    state = _base_state(tmp_path)
    state["constraints"] = []
    ui = _make_ui()

    spec_path = Path(state["repo_path"]) / "feature_spec.json"

    async def _write_spec(**kwargs):
        spec_path.write_text(json.dumps({"feature_name": "NC"}))

    with patch("graft.stages.grill.run_agent", new_callable=AsyncMock, side_effect=_write_spec) as mock_agent:
        result = await grill_node(state, ui)

    # Compile call should include "CONSTRAINTS: None"
    compile_call = mock_agent.call_args_list[-1]
    assert "CONSTRAINTS: None" in compile_call.kwargs["user_prompt"]


@pytest.mark.asyncio
async def test_grill_node_missing_optional_state_keys(tmp_path):
    """grill_node works when feature_prompt, codebase_profile, etc. are missing."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "artifacts").mkdir()
    meta = {"project_id": "test", "stages_completed": [], "status": "in_progress"}
    (project_dir / "metadata.json").write_text(json.dumps(meta))

    # Minimal state with only required keys, no feature_prompt/codebase_profile/etc.
    state = {
        "repo_path": str(repo_path),
        "project_dir": str(project_dir),
        "technical_assessment": {
            "open_questions": [
                {"question": "Q?", "recommended_answer": "A", "category": "intent"},
            ]
        },
    }
    ui = _make_ui()
    spec_path = repo_path / "feature_spec.json"

    async def _write_spec(**kwargs):
        spec_path.write_text(json.dumps({}))

    with patch("graft.stages.grill.run_agent", new_callable=AsyncMock, side_effect=_write_spec):
        result = await grill_node(state, ui)

    assert result["grill_complete"] is True


# ---------------------------------------------------------------------------
# _generate_questions (private helper, tested directly for coverage)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_questions_returns_list_from_file(tmp_path):
    """_generate_questions reads the agent-written JSON and returns a list."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    questions = [
        {"question": "Q1?", "recommended_answer": "A1", "category": "intent"},
        {"question": "Q2?", "recommended_answer": "A2", "category": "edge_case"},
    ]
    questions_path = repo_path / "open_questions.json"

    async def _write_questions(**kwargs):
        questions_path.write_text(json.dumps(questions))

    ui = _make_ui()
    with patch("graft.stages.grill.run_agent", new_callable=AsyncMock, side_effect=_write_questions):
        result = await _generate_questions(
            str(repo_path), str(project_dir), "feature", {}, {}, ui, None
        )

    assert len(result) == 2
    assert result[0]["question"] == "Q1?"
    # File should be cleaned up
    assert not questions_path.exists()


@pytest.mark.asyncio
async def test_generate_questions_returns_empty_on_missing_file(tmp_path):
    """_generate_questions returns [] when the agent doesn't create the file."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    ui = _make_ui()
    with patch("graft.stages.grill.run_agent", new_callable=AsyncMock):
        result = await _generate_questions(
            str(repo_path), str(project_dir), "feature", {}, {}, ui, None
        )

    assert result == []


@pytest.mark.asyncio
async def test_generate_questions_returns_empty_on_invalid_json(tmp_path):
    """_generate_questions returns [] when the file contains invalid JSON."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    questions_path = repo_path / "open_questions.json"

    async def _write_bad(**kwargs):
        questions_path.write_text("{not valid json")

    ui = _make_ui()
    with patch("graft.stages.grill.run_agent", new_callable=AsyncMock, side_effect=_write_bad):
        result = await _generate_questions(
            str(repo_path), str(project_dir), "feature", {}, {}, ui, None
        )

    assert result == []


@pytest.mark.asyncio
async def test_generate_questions_returns_empty_on_non_list_json(tmp_path):
    """_generate_questions returns [] when JSON is valid but not a list."""
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    questions_path = repo_path / "open_questions.json"

    async def _write_dict(**kwargs):
        questions_path.write_text(json.dumps({"not": "a list"}))

    ui = _make_ui()
    with patch("graft.stages.grill.run_agent", new_callable=AsyncMock, side_effect=_write_dict):
        result = await _generate_questions(
            str(repo_path), str(project_dir), "feature", {}, {}, ui, None
        )

    assert result == []


# ---------------------------------------------------------------------------
# grill_node — compile agent invocation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grill_node_compile_agent_receives_correct_prompt(tmp_path):
    """The compile agent receives the feature prompt, profile, assessment, and transcript."""
    state = _base_state(tmp_path)
    ui = _make_ui()
    ui.grill_question.side_effect = ["yes", "10"]
    spec_path = Path(state["repo_path"]) / "feature_spec.json"

    async def _write_spec(**kwargs):
        spec_path.write_text(json.dumps({}))

    with patch("graft.stages.grill.run_agent", new_callable=AsyncMock, side_effect=_write_spec) as mock_agent:
        await grill_node(state, ui)

    # The compile call is the only run_agent call (questions came from state)
    assert mock_agent.call_count == 1
    call_kwargs = mock_agent.call_args.kwargs
    assert call_kwargs["persona"] == "Principal Product Architect"
    assert call_kwargs["stage"] == "grill_compile"
    assert call_kwargs["max_turns"] == 10
    assert "Read" in call_kwargs["allowed_tools"]
    assert "Write" in call_kwargs["allowed_tools"]
    assert "Build a trade system" in call_kwargs["user_prompt"]
    assert "Should trades expire?" in call_kwargs["user_prompt"]
    assert "Must use Supabase" in call_kwargs["user_prompt"]


# ---------------------------------------------------------------------------
# Full grill -> compile -> route flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_grill_to_route_plan(tmp_path):
    """End-to-end: grill with questions -> compile -> route to plan."""
    state = _base_state(tmp_path)
    ui = _make_ui()
    ui.grill_question.side_effect = ["approve trades", "max 5"]

    spec = {"feature_name": "Trade", "scope": {"mvp": ["trade_ui"]}}
    spec_path = Path(state["repo_path"]) / "feature_spec.json"

    async def _write_spec(**kwargs):
        spec_path.write_text(json.dumps(spec))

    with patch("graft.stages.grill.run_agent", new_callable=AsyncMock, side_effect=_write_spec):
        result = await grill_node(state, ui)

    # Route should go to plan
    assert grill_router(result) == "plan"
    assert result["feature_spec"]["feature_name"] == "Trade"
    assert result["grill_complete"] is True


@pytest.mark.asyncio
async def test_full_grill_to_route_research_redo(tmp_path):
    """End-to-end: grill triggers research redo when spec says so."""
    state = _base_state(tmp_path)
    ui = _make_ui()
    ui.grill_question.return_value = "actually, use a totally different approach"

    spec = {"feature_name": "Trade", "research_redo_needed": True}
    spec_path = Path(state["repo_path"]) / "feature_spec.json"

    async def _write_spec(**kwargs):
        spec_path.write_text(json.dumps(spec))

    with patch("graft.stages.grill.run_agent", new_callable=AsyncMock, side_effect=_write_spec):
        result = await grill_node(state, ui)

    # Route should loop back to research
    assert grill_router(result) == "research"
    assert result["research_redo_needed"] is True
