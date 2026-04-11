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
def mock_ui():
    """A mock UI with all methods used by grill_node."""
    ui = MagicMock()
    ui.stage_start = MagicMock()
    ui.stage_done = MagicMock()
    ui.info = MagicMock()
    ui.error = MagicMock()
    # Default: user accepts recommended answer (empty string -> recommended)
    ui.grill_question = MagicMock(return_value="user answer")
    return ui


@pytest.fixture
def repo_dir(tmp_path):
    """Temporary repo directory."""
    return tmp_path / "repo"


@pytest.fixture
def project_dir(tmp_path):
    """Temporary project directory with artifacts subdir."""
    p = tmp_path / "project"
    (p / "artifacts").mkdir(parents=True)
    (p / "logs").mkdir(parents=True)
    return p


@pytest.fixture
def base_state(repo_dir, project_dir):
    """Minimal valid FeatureState for grill_node."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    return {
        "repo_path": str(repo_dir),
        "project_dir": str(project_dir),
        "feature_prompt": "Add a trade system",
        "codebase_profile": {"lang": "python"},
        "technical_assessment": {
            "open_questions": [
                {
                    "question": "Should trades expire?",
                    "recommended_answer": "Yes, after 24 hours",
                    "category": "edge_case",
                },
                {
                    "question": "Support counter-offers?",
                    "recommended_answer": "Not in MVP",
                    "category": "prioritization",
                },
            ]
        },
        "constraints": ["Must use Supabase RLS"],
    }


# ---------------------------------------------------------------------------
# grill_router tests (existing + edge cases)
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


def test_grill_router_falsy_values():
    """Falsy non-bool values still route to plan."""
    assert grill_router({"research_redo_needed": 0}) == "plan"
    assert grill_router({"research_redo_needed": ""}) == "plan"
    assert grill_router({"research_redo_needed": None}) == "plan"


def test_grill_router_truthy_values():
    """Truthy non-bool values route to research."""
    assert grill_router({"research_redo_needed": 1}) == "research"
    assert grill_router({"research_redo_needed": "yes"}) == "research"


# ---------------------------------------------------------------------------
# grill_node tests — Q&A phase
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("graft.stages.grill.mark_stage_complete")
@patch("graft.stages.grill.save_artifact")
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_basic_qa_flow(
    mock_run_agent, mock_save, mock_mark, base_state, mock_ui, repo_dir
):
    """grill_node walks through open_questions, builds transcript, and compiles."""
    mock_ui.grill_question.side_effect = ["Yes, 24h", "Defer to v2"]

    # run_agent for compile writes feature_spec.json
    async def write_spec(**kwargs):
        spec = {"feature_name": "Trade System", "decisions": []}
        (repo_dir / "feature_spec.json").write_text(json.dumps(spec))

    mock_run_agent.side_effect = write_spec

    result = await grill_node(base_state, mock_ui)

    # UI interactions
    mock_ui.stage_start.assert_called_once_with("grill")
    mock_ui.stage_done.assert_called_once_with("grill")
    assert mock_ui.grill_question.call_count == 2

    # First question
    mock_ui.grill_question.assert_any_call(
        "Should trades expire?", "Yes, after 24 hours", "edge_case", 1
    )
    # Second question
    mock_ui.grill_question.assert_any_call(
        "Support counter-offers?", "Not in MVP", "prioritization", 2
    )

    # Transcript built correctly
    transcript = result["grill_transcript"]
    assert "Q1 [edge_case]: Should trades expire?" in transcript
    assert "Recommended: Yes, after 24 hours" in transcript
    assert "Answer: Yes, 24h" in transcript
    assert "Q2 [prioritization]: Support counter-offers?" in transcript
    assert "Answer: Defer to v2" in transcript

    # Feature spec populated
    assert result["feature_spec"]["feature_name"] == "Trade System"
    assert result["grill_complete"] is True
    assert result["current_stage"] == "grill"
    assert result["research_redo_needed"] is False

    # Artifacts saved: transcript and feature_spec
    save_calls = mock_save.call_args_list
    artifact_names = [c[0][1] for c in save_calls]
    assert "grill_transcript.md" in artifact_names
    assert "feature_spec.json" in artifact_names

    # Stage marked complete
    mock_mark.assert_called_once_with(str(base_state["project_dir"]), "grill")


@pytest.mark.asyncio
@patch("graft.stages.grill.mark_stage_complete")
@patch("graft.stages.grill.save_artifact")
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_spec_file_cleaned_up(
    mock_run_agent, mock_save, mock_mark, base_state, mock_ui, repo_dir
):
    """feature_spec.json is deleted from repo_path after being read."""
    mock_ui.grill_question.return_value = "ok"

    async def write_spec(**kwargs):
        (repo_dir / "feature_spec.json").write_text('{"feature_name": "X"}')

    mock_run_agent.side_effect = write_spec

    await grill_node(base_state, mock_ui)

    assert not (repo_dir / "feature_spec.json").exists()


@pytest.mark.asyncio
@patch("graft.stages.grill.mark_stage_complete")
@patch("graft.stages.grill.save_artifact")
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_spec_json_decode_error(
    mock_run_agent, mock_save, mock_mark, base_state, mock_ui, repo_dir
):
    """If feature_spec.json is invalid JSON, an error is reported and spec is empty."""
    mock_ui.grill_question.return_value = "ok"

    async def write_bad_spec(**kwargs):
        (repo_dir / "feature_spec.json").write_text("NOT VALID JSON {{{")

    mock_run_agent.side_effect = write_bad_spec

    result = await grill_node(base_state, mock_ui)

    mock_ui.error.assert_called_once_with("Failed to parse feature_spec.json.")
    assert result["feature_spec"] == {}


@pytest.mark.asyncio
@patch("graft.stages.grill.mark_stage_complete")
@patch("graft.stages.grill.save_artifact")
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_no_spec_file_produced(
    mock_run_agent, mock_save, mock_mark, base_state, mock_ui, repo_dir
):
    """If run_agent doesn't produce feature_spec.json, spec is empty dict."""
    mock_ui.grill_question.return_value = "ok"
    # run_agent does nothing — no file created
    mock_run_agent.return_value = None

    result = await grill_node(base_state, mock_ui)

    assert result["feature_spec"] == {}
    assert result["grill_complete"] is True


@pytest.mark.asyncio
@patch("graft.stages.grill.mark_stage_complete")
@patch("graft.stages.grill.save_artifact")
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_research_redo_flag(
    mock_run_agent, mock_save, mock_mark, base_state, mock_ui, repo_dir
):
    """When feature_spec has research_redo_needed=True, result propagates it."""
    mock_ui.grill_question.return_value = "ok"

    async def write_redo_spec(**kwargs):
        spec = {"feature_name": "X", "research_redo_needed": True}
        (repo_dir / "feature_spec.json").write_text(json.dumps(spec))

    mock_run_agent.side_effect = write_redo_spec

    result = await grill_node(base_state, mock_ui)

    assert result["research_redo_needed"] is True
    mock_ui.info.assert_any_call(
        "Grill revealed a fundamental assumption change — looping back to Research."
    )


@pytest.mark.asyncio
@patch("graft.stages.grill.mark_stage_complete")
@patch("graft.stages.grill.save_artifact")
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_string_questions(
    mock_run_agent, mock_save, mock_mark, mock_ui, repo_dir, project_dir
):
    """When open_questions are plain strings (not dicts), defaults are used."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "repo_path": str(repo_dir),
        "project_dir": str(project_dir),
        "feature_prompt": "Add feature",
        "codebase_profile": {},
        "technical_assessment": {
            "open_questions": ["What color?", "What size?"]
        },
        "constraints": [],
    }
    mock_ui.grill_question.return_value = "blue"
    mock_run_agent.return_value = None

    result = await grill_node(state, mock_ui)

    # String questions get default recommended="No recommendation", category="intent"
    mock_ui.grill_question.assert_any_call(
        "What color?", "No recommendation", "intent", 1
    )
    mock_ui.grill_question.assert_any_call(
        "What size?", "No recommendation", "intent", 2
    )
    assert "Q1 [intent]: What color?" in result["grill_transcript"]
    assert "Recommended: No recommendation" in result["grill_transcript"]


@pytest.mark.asyncio
@patch("graft.stages.grill.mark_stage_complete")
@patch("graft.stages.grill.save_artifact")
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_empty_questions_triggers_generate(
    mock_run_agent, mock_save, mock_mark, mock_ui, repo_dir, project_dir
):
    """When technical_assessment has no open_questions, _generate_questions is called."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "repo_path": str(repo_dir),
        "project_dir": str(project_dir),
        "feature_prompt": "Add feature",
        "codebase_profile": {},
        "technical_assessment": {},  # No open_questions
        "constraints": [],
    }

    call_count = 0

    async def agent_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if kwargs.get("stage") == "grill_generate":
            # Write questions file for _generate_questions
            questions = [
                {
                    "question": "Generated Q?",
                    "recommended_answer": "Generated A",
                    "category": "intent",
                }
            ]
            (repo_dir / "open_questions.json").write_text(json.dumps(questions))
        elif kwargs.get("stage") == "grill_compile":
            # Write feature_spec for compile phase
            (repo_dir / "feature_spec.json").write_text('{"feature_name": "X"}')

    mock_run_agent.side_effect = agent_side_effect
    mock_ui.grill_question.return_value = "user said this"

    result = await grill_node(state, mock_ui)

    # run_agent called twice: once for generate, once for compile
    assert mock_run_agent.call_count == 2

    # Info message about generating questions
    mock_ui.info.assert_any_call(
        "No open questions from Research — generating questions from context..."
    )

    # The generated question was presented
    mock_ui.grill_question.assert_called_once_with(
        "Generated Q?", "Generated A", "intent", 1
    )
    assert "Q1 [intent]: Generated Q?" in result["grill_transcript"]


@pytest.mark.asyncio
@patch("graft.stages.grill.mark_stage_complete")
@patch("graft.stages.grill.save_artifact")
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_no_constraints(
    mock_run_agent, mock_save, mock_mark, mock_ui, repo_dir, project_dir
):
    """When constraints is empty, compile prompt says 'None'."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "repo_path": str(repo_dir),
        "project_dir": str(project_dir),
        "feature_prompt": "Add feature",
        "codebase_profile": {},
        "technical_assessment": {
            "open_questions": [{"question": "Q?", "recommended_answer": "A", "category": "intent"}]
        },
        "constraints": [],
    }
    mock_ui.grill_question.return_value = "ok"
    mock_run_agent.return_value = None

    await grill_node(state, mock_ui)

    # The compile call's user_prompt should contain "CONSTRAINTS: None"
    compile_call = mock_run_agent.call_args
    assert "CONSTRAINTS: None" in compile_call.kwargs["user_prompt"]


@pytest.mark.asyncio
@patch("graft.stages.grill.mark_stage_complete")
@patch("graft.stages.grill.save_artifact")
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_with_constraints(
    mock_run_agent, mock_save, mock_mark, mock_ui, repo_dir, project_dir
):
    """When constraints are provided, they appear semicolon-separated in compile prompt."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "repo_path": str(repo_dir),
        "project_dir": str(project_dir),
        "feature_prompt": "Add feature",
        "codebase_profile": {},
        "technical_assessment": {
            "open_questions": [{"question": "Q?", "recommended_answer": "A", "category": "intent"}]
        },
        "constraints": ["Use RLS", "No external APIs"],
    }
    mock_ui.grill_question.return_value = "ok"
    mock_run_agent.return_value = None

    await grill_node(state, mock_ui)

    compile_call = mock_run_agent.call_args
    assert "Use RLS; No external APIs" in compile_call.kwargs["user_prompt"]


@pytest.mark.asyncio
@patch("graft.stages.grill.mark_stage_complete")
@patch("graft.stages.grill.save_artifact")
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_compile_agent_params(
    mock_run_agent, mock_save, mock_mark, base_state, mock_ui, repo_dir
):
    """Compile phase passes correct persona, system prompt, stage, and tools."""
    mock_ui.grill_question.return_value = "ok"
    mock_run_agent.return_value = None

    await grill_node(base_state, mock_ui)

    compile_kwargs = mock_run_agent.call_args.kwargs
    assert compile_kwargs["persona"] == "Principal Product Architect"
    assert compile_kwargs["stage"] == "grill_compile"
    assert compile_kwargs["max_turns"] == 10
    assert compile_kwargs["allowed_tools"] == ["Read", "Write", "Bash"]
    assert compile_kwargs["cwd"] == str(repo_dir)


@pytest.mark.asyncio
@patch("graft.stages.grill.mark_stage_complete")
@patch("graft.stages.grill.save_artifact")
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_model_passed_through(
    mock_run_agent, mock_save, mock_mark, base_state, mock_ui, repo_dir
):
    """The model from state is passed to run_agent."""
    base_state["model"] = "claude-sonnet-4-20250514"
    mock_ui.grill_question.return_value = "ok"
    mock_run_agent.return_value = None

    await grill_node(base_state, mock_ui)

    compile_kwargs = mock_run_agent.call_args.kwargs
    assert compile_kwargs["model"] == "claude-sonnet-4-20250514"


@pytest.mark.asyncio
@patch("graft.stages.grill.mark_stage_complete")
@patch("graft.stages.grill.save_artifact")
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_dict_question_missing_fields(
    mock_run_agent, mock_save, mock_mark, mock_ui, repo_dir, project_dir
):
    """Dict questions with missing fields use sensible defaults."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "repo_path": str(repo_dir),
        "project_dir": str(project_dir),
        "feature_prompt": "Add feature",
        "codebase_profile": {},
        "technical_assessment": {
            "open_questions": [
                {"question": "Only question field?"},  # missing recommended_answer and category
            ]
        },
        "constraints": [],
    }
    mock_ui.grill_question.return_value = "my answer"
    mock_run_agent.return_value = None

    result = await grill_node(state, mock_ui)

    mock_ui.grill_question.assert_called_once_with(
        "Only question field?", "No recommendation", "intent", 1
    )
    assert "Recommended: No recommendation" in result["grill_transcript"]
    assert "Q1 [intent]:" in result["grill_transcript"]


@pytest.mark.asyncio
@patch("graft.stages.grill.mark_stage_complete")
@patch("graft.stages.grill.save_artifact")
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_no_feature_prompt(
    mock_run_agent, mock_save, mock_mark, mock_ui, repo_dir, project_dir
):
    """When state lacks feature_prompt, it defaults to empty string."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "repo_path": str(repo_dir),
        "project_dir": str(project_dir),
        # no feature_prompt
        "codebase_profile": {},
        "technical_assessment": {
            "open_questions": [{"question": "Q?", "recommended_answer": "A", "category": "intent"}]
        },
    }
    mock_ui.grill_question.return_value = "ok"
    mock_run_agent.return_value = None

    result = await grill_node(state, mock_ui)

    compile_kwargs = mock_run_agent.call_args.kwargs
    assert "FEATURE PROMPT: \n" in compile_kwargs["user_prompt"]


@pytest.mark.asyncio
@patch("graft.stages.grill.mark_stage_complete")
@patch("graft.stages.grill.save_artifact")
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_empty_open_questions_list(
    mock_run_agent, mock_save, mock_mark, mock_ui, repo_dir, project_dir
):
    """An empty open_questions list triggers _generate_questions."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "repo_path": str(repo_dir),
        "project_dir": str(project_dir),
        "feature_prompt": "X",
        "codebase_profile": {},
        "technical_assessment": {"open_questions": []},  # empty list
        "constraints": [],
    }
    mock_run_agent.return_value = None
    mock_ui.grill_question.return_value = "ok"

    await grill_node(state, mock_ui)

    # Should have called generate (first call) then compile (second call)
    assert mock_run_agent.call_count == 2
    first_call = mock_run_agent.call_args_list[0]
    assert first_call.kwargs["stage"] == "grill_generate"


# ---------------------------------------------------------------------------
# _generate_questions tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_generate_questions_success(mock_run_agent, repo_dir, mock_ui):
    """Successfully generates and reads questions from JSON file."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    questions_data = [
        {"question": "Q1?", "recommended_answer": "A1", "category": "intent"},
        {"question": "Q2?", "recommended_answer": "A2", "category": "edge_case"},
    ]

    async def write_questions(**kwargs):
        (repo_dir / "open_questions.json").write_text(json.dumps(questions_data))

    mock_run_agent.side_effect = write_questions

    result = await _generate_questions(
        repo_path=str(repo_dir),
        project_dir=str(repo_dir),
        feature_prompt="Add feature",
        codebase_profile={"lang": "python"},
        technical_assessment={},
        ui=mock_ui,
        model=None,
    )

    assert result == questions_data
    # File should be cleaned up
    assert not (repo_dir / "open_questions.json").exists()


@pytest.mark.asyncio
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_generate_questions_file_not_found(mock_run_agent, repo_dir, mock_ui):
    """When agent doesn't produce the file, returns empty list."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    mock_run_agent.return_value = None  # No file written

    result = await _generate_questions(
        repo_path=str(repo_dir),
        project_dir=str(repo_dir),
        feature_prompt="Add feature",
        codebase_profile={},
        technical_assessment={},
        ui=mock_ui,
        model=None,
    )

    assert result == []


@pytest.mark.asyncio
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_generate_questions_invalid_json(mock_run_agent, repo_dir, mock_ui):
    """When questions file has invalid JSON, returns empty list."""
    repo_dir.mkdir(parents=True, exist_ok=True)

    async def write_bad_json(**kwargs):
        (repo_dir / "open_questions.json").write_text("NOT JSON !!!")

    mock_run_agent.side_effect = write_bad_json

    result = await _generate_questions(
        repo_path=str(repo_dir),
        project_dir=str(repo_dir),
        feature_prompt="Add feature",
        codebase_profile={},
        technical_assessment={},
        ui=mock_ui,
        model=None,
    )

    assert result == []


@pytest.mark.asyncio
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_generate_questions_non_list_result(mock_run_agent, repo_dir, mock_ui):
    """When questions file contains a dict instead of list, returns empty list."""
    repo_dir.mkdir(parents=True, exist_ok=True)

    async def write_dict(**kwargs):
        (repo_dir / "open_questions.json").write_text('{"not": "a list"}')

    mock_run_agent.side_effect = write_dict

    result = await _generate_questions(
        repo_path=str(repo_dir),
        project_dir=str(repo_dir),
        feature_prompt="Add feature",
        codebase_profile={},
        technical_assessment={},
        ui=mock_ui,
        model=None,
    )

    assert result == []


@pytest.mark.asyncio
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_generate_questions_agent_params(mock_run_agent, repo_dir, mock_ui):
    """Verify correct parameters passed to run_agent during question generation."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    mock_run_agent.return_value = None

    await _generate_questions(
        repo_path=str(repo_dir),
        project_dir="/some/project",
        feature_prompt="Build a chat widget",
        codebase_profile={"framework": "react"},
        technical_assessment={"gaps": []},
        ui=mock_ui,
        model="claude-sonnet-4-20250514",
    )

    kwargs = mock_run_agent.call_args.kwargs
    assert kwargs["persona"] == "Principal Product Interrogator"
    assert kwargs["stage"] == "grill_generate"
    assert kwargs["cwd"] == str(repo_dir)
    assert kwargs["project_dir"] == "/some/project"
    assert kwargs["model"] == "claude-sonnet-4-20250514"
    assert kwargs["max_turns"] == 10
    assert "Read" in kwargs["allowed_tools"]
    assert "Write" in kwargs["allowed_tools"]
    assert "Bash" in kwargs["allowed_tools"]
    assert "Glob" in kwargs["allowed_tools"]
    assert "Grep" in kwargs["allowed_tools"]
    # User prompt should include the feature prompt
    assert "Build a chat widget" in kwargs["user_prompt"]


@pytest.mark.asyncio
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_generate_questions_file_cleaned_up(mock_run_agent, repo_dir, mock_ui):
    """open_questions.json is deleted after successful read."""
    repo_dir.mkdir(parents=True, exist_ok=True)

    async def write_questions(**kwargs):
        (repo_dir / "open_questions.json").write_text('[{"question": "Q?"}]')

    mock_run_agent.side_effect = write_questions

    await _generate_questions(
        repo_path=str(repo_dir),
        project_dir=str(repo_dir),
        feature_prompt="X",
        codebase_profile={},
        technical_assessment={},
        ui=mock_ui,
        model=None,
    )

    assert not (repo_dir / "open_questions.json").exists()


# ---------------------------------------------------------------------------
# grill_node — transcript correctness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("graft.stages.grill.mark_stage_complete")
@patch("graft.stages.grill.save_artifact")
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_transcript_format(
    mock_run_agent, mock_save, mock_mark, mock_ui, repo_dir, project_dir
):
    """Transcript lines follow the exact expected format."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "repo_path": str(repo_dir),
        "project_dir": str(project_dir),
        "feature_prompt": "X",
        "codebase_profile": {},
        "technical_assessment": {
            "open_questions": [
                {
                    "question": "Use REST or GraphQL?",
                    "recommended_answer": "REST",
                    "category": "preference",
                }
            ]
        },
        "constraints": [],
    }
    mock_ui.grill_question.return_value = "GraphQL"
    mock_run_agent.return_value = None

    result = await grill_node(state, mock_ui)

    lines = result["grill_transcript"].split("\n")
    assert lines[0] == "Q1 [preference]: Use REST or GraphQL?"
    assert lines[1] == "  Recommended: REST"
    assert lines[2] == "  Answer: GraphQL"
    assert lines[3] == ""  # Blank line separator


@pytest.mark.asyncio
@patch("graft.stages.grill.mark_stage_complete")
@patch("graft.stages.grill.save_artifact")
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_transcript_saved_as_artifact(
    mock_run_agent, mock_save, mock_mark, base_state, mock_ui, repo_dir
):
    """The grill transcript is saved as an artifact before compilation."""
    mock_ui.grill_question.return_value = "ok"
    mock_run_agent.return_value = None

    await grill_node(base_state, mock_ui)

    # First save_artifact call should be grill_transcript.md
    first_save = mock_save.call_args_list[0]
    assert first_save[0][1] == "grill_transcript.md"
    # Content should be a string
    assert isinstance(first_save[0][2], str)


@pytest.mark.asyncio
@patch("graft.stages.grill.mark_stage_complete")
@patch("graft.stages.grill.save_artifact")
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_zero_questions(
    mock_run_agent, mock_save, mock_mark, mock_ui, repo_dir, project_dir
):
    """When _generate_questions returns empty list, no Q&A happens but compile still runs."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "repo_path": str(repo_dir),
        "project_dir": str(project_dir),
        "feature_prompt": "X",
        "codebase_profile": {},
        "technical_assessment": {},
        "constraints": [],
    }
    # generate phase returns nothing, compile phase also returns nothing
    mock_run_agent.return_value = None
    mock_ui.grill_question.return_value = "ok"

    result = await grill_node(state, mock_ui)

    # No questions presented to user
    mock_ui.grill_question.assert_not_called()

    # Transcript is empty
    assert result["grill_transcript"] == ""

    # Compile still ran (second call to run_agent)
    assert mock_run_agent.call_count == 2


@pytest.mark.asyncio
@patch("graft.stages.grill.mark_stage_complete")
@patch("graft.stages.grill.save_artifact")
@patch("graft.stages.grill.run_agent", new_callable=AsyncMock)
async def test_grill_node_compile_prompt_includes_transcript(
    mock_run_agent, mock_save, mock_mark, base_state, mock_ui, repo_dir
):
    """The compile prompt includes the full grill transcript."""
    mock_ui.grill_question.side_effect = ["answer1", "answer2"]
    mock_run_agent.return_value = None

    await grill_node(base_state, mock_ui)

    compile_call = mock_run_agent.call_args
    prompt = compile_call.kwargs["user_prompt"]
    assert "GRILL TRANSCRIPT:" in prompt
    assert "Answer: answer1" in prompt
    assert "Answer: answer2" in prompt
    assert "FEATURE PROMPT: Add a trade system" in prompt
    assert "CODEBASE PROFILE:" in prompt
    assert "TECHNICAL ASSESSMENT:" in prompt
