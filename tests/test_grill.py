"""Tests for graft.stages.grill."""

import json
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from graft.stages.grill import (
    COMPILE_SYSTEM_PROMPT,
    _generate_questions,
    grill_node,
    grill_router,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeAgentResult:
    """Minimal stand-in for graft.agent.AgentResult."""

    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    raw_messages: list = field(default_factory=list)
    elapsed_seconds: float = 1.0
    turns_used: int = 5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path):
    """Temporary repo directory."""
    d = tmp_path / "repo"
    d.mkdir()
    return d


@pytest.fixture
def project(tmp_path):
    """Temporary project directory with required sub-structure."""
    d = tmp_path / "project"
    d.mkdir()
    (d / "artifacts").mkdir()
    (d / "logs").mkdir()
    meta = {"project_id": "feat_test01", "stages_completed": []}
    (d / "metadata.json").write_text(json.dumps(meta))
    return d


@pytest.fixture
def ui():
    """Mock UI object exposing the methods grill_node calls."""
    m = MagicMock()
    m.stage_start = MagicMock()
    m.stage_done = MagicMock()
    m.error = MagicMock()
    m.info = MagicMock()
    m.grill_question = MagicMock(return_value="user answer")
    return m


SAMPLE_QUESTIONS = [
    {
        "question": "Should dark mode persist across sessions?",
        "category": "intent",
        "recommended_answer": "Yes, store in localStorage",
    },
    {
        "question": "Support system preference detection?",
        "category": "preference",
        "recommended_answer": "Yes, use prefers-color-scheme",
    },
]

SAMPLE_SPEC = {
    "feature_name": "Dark Mode",
    "feature_prompt": "Add dark mode",
    "decisions": [
        {
            "question": "Should dark mode persist?",
            "recommended": "Yes",
            "answer": "Yes",
            "category": "intent",
            "implications": ["needs localStorage"],
        }
    ],
    "scope": {"mvp": ["toggle"], "follow_up": ["auto-detect"]},
    "constraints": ["Must follow existing patterns"],
    "technical_notes": ["Reuse ThemeProvider"],
}


def _state(repo, project, **kw):
    """Build a minimal FeatureState dict."""
    base = {
        "repo_path": str(repo),
        "project_dir": str(project),
        "feature_prompt": "Add dark mode",
        "codebase_profile": {"project": {"name": "acme"}},
        "technical_assessment": {"open_questions": SAMPLE_QUESTIONS},
    }
    base.update(kw)
    return base


def _write_spec(repo, spec=None):
    """Write feature_spec.json into repo so the compile step can read it."""
    (repo / "feature_spec.json").write_text(json.dumps(spec or SAMPLE_SPEC, indent=2))


# ---------------------------------------------------------------------------
# grill_router (existing tests preserved)
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
# grill_node — happy path
# ---------------------------------------------------------------------------


class TestGrillNodeHappyPath:
    """Core happy-path tests for grill_node."""

    async def test_returns_expected_keys(self, repo, project, ui):
        """grill_node returns feature_spec, grill_transcript, etc."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo)
            result = await grill_node(_state(repo, project), ui)

        assert set(result.keys()) == {
            "feature_spec",
            "grill_transcript",
            "grill_complete",
            "research_redo_needed",
            "current_stage",
        }

    async def test_current_stage_is_grill(self, repo, project, ui):
        """current_stage is set to 'grill'."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo)
            result = await grill_node(_state(repo, project), ui)

        assert result["current_stage"] == "grill"

    async def test_grill_complete_is_true(self, repo, project, ui):
        """grill_complete flag is always True on success."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo)
            result = await grill_node(_state(repo, project), ui)

        assert result["grill_complete"] is True

    async def test_feature_spec_parsed(self, repo, project, ui):
        """Valid feature_spec.json is parsed into the returned dict."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo)
            result = await grill_node(_state(repo, project), ui)

        assert result["feature_spec"] == SAMPLE_SPEC
        assert result["feature_spec"]["feature_name"] == "Dark Mode"

    async def test_grill_transcript_built(self, repo, project, ui):
        """Transcript contains Q&A lines for each question."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo)
            result = await grill_node(_state(repo, project), ui)

        transcript = result["grill_transcript"]
        assert "Q1 [intent]:" in transcript
        assert "Q2 [preference]:" in transcript
        assert "Should dark mode persist" in transcript
        assert "user answer" in transcript

    async def test_ui_lifecycle(self, repo, project, ui):
        """stage_start and stage_done are called in order."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo)
            await grill_node(_state(repo, project), ui)

        ui.stage_start.assert_called_once_with("grill")
        ui.stage_done.assert_called_once_with("grill")

    async def test_marks_stage_complete(self, repo, project, ui):
        """Stage 'grill' is recorded in metadata after success."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo)
            await grill_node(_state(repo, project), ui)

        meta = json.loads((project / "metadata.json").read_text())
        assert "grill" in meta["stages_completed"]


# ---------------------------------------------------------------------------
# grill_node — question iteration & UI interaction
# ---------------------------------------------------------------------------


class TestGrillNodeQuestionIteration:
    """Verify question walking and UI calls."""

    async def test_grill_question_called_per_question(self, repo, project, ui):
        """ui.grill_question is called once per open question."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo)
            await grill_node(_state(repo, project), ui)

        assert ui.grill_question.call_count == 2

    async def test_grill_question_args(self, repo, project, ui):
        """ui.grill_question receives correct arguments for each question."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo)
            await grill_node(_state(repo, project), ui)

        first_call = ui.grill_question.call_args_list[0]
        assert first_call == call(
            "Should dark mode persist across sessions?",
            "Yes, store in localStorage",
            "intent",
            1,
        )
        second_call = ui.grill_question.call_args_list[1]
        assert second_call == call(
            "Support system preference detection?",
            "Yes, use prefers-color-scheme",
            "preference",
            2,
        )

    async def test_transcript_line_format(self, repo, project, ui):
        """Transcript lines follow the expected format."""
        ui.grill_question.return_value = "my custom answer"
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo)
            result = await grill_node(_state(repo, project), ui)

        lines = result["grill_transcript"].split("\n")
        assert lines[0] == ("Q1 [intent]: Should dark mode persist across sessions?")
        assert lines[1] == "  Recommended: Yes, store in localStorage"
        assert lines[2] == "  Answer: my custom answer"
        assert lines[3] == ""  # blank separator

    async def test_decisions_list_populated(self, repo, project, ui):
        """Decisions list has one entry per question with correct keys."""
        ui.grill_question.return_value = "accepted"
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo)
            await grill_node(_state(repo, project), ui)

        # The decisions are embedded in the compile prompt; verify via
        # the transcript that all questions were processed
        _, kwargs = mock_run.call_args
        prompt = kwargs["user_prompt"]
        assert "Q1 [intent]:" in prompt
        assert "Q2 [preference]:" in prompt
        assert "accepted" in prompt

    async def test_mixed_dict_and_string_questions(self, repo, project, ui):
        """Questions can be dicts or plain strings."""
        mixed_questions = [
            {
                "question": "Use websockets?",
                "category": "preference",
                "recommended_answer": "Yes",
            },
            "Should we add logging?",
        ]
        state = _state(
            repo,
            project,
            technical_assessment={"open_questions": mixed_questions},
        )

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo)
            result = await grill_node(state, ui)

        # First call: dict question
        first = ui.grill_question.call_args_list[0]
        assert first == call("Use websockets?", "Yes", "preference", 1)

        # Second call: string question → defaults
        second = ui.grill_question.call_args_list[1]
        assert second == call(
            "Should we add logging?", "No recommendation", "intent", 2
        )

        transcript = result["grill_transcript"]
        assert "Q1 [preference]: Use websockets?" in transcript
        assert "Q2 [intent]: Should we add logging?" in transcript

    async def test_empty_questions_list(self, repo, project, ui):
        """When open_questions is empty AND _generate_questions returns [],
        no questions are asked and transcript is empty."""
        state = _state(repo, project, technical_assessment={"open_questions": []})

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            # _generate_questions also returns empty
            _write_spec(repo)
            await grill_node(state, ui)

        # _generate_questions is called (open_questions is empty/falsy)
        # but returns empty → no grill_question calls
        # First call is _generate_questions agent, second is compile agent
        ui.grill_question.assert_not_called()


# ---------------------------------------------------------------------------
# grill_node — compile agent call
# ---------------------------------------------------------------------------


class TestGrillNodeCompileAgent:
    """Verify compile agent invocation."""

    async def test_compile_agent_system_prompt(self, repo, project, ui):
        """run_agent receives COMPILE_SYSTEM_PROMPT."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo)
            await grill_node(_state(repo, project), ui)

        # Last call is the compile agent (or only call when questions exist)
        compile_call = mock_run.call_args
        _, kwargs = compile_call
        assert kwargs["system_prompt"] is COMPILE_SYSTEM_PROMPT

    async def test_compile_prompt_contains_feature(self, repo, project, ui):
        """Compile prompt includes the feature prompt."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo)
            await grill_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert "FEATURE PROMPT: Add dark mode" in kwargs["user_prompt"]

    async def test_compile_prompt_contains_transcript(self, repo, project, ui):
        """Compile prompt includes the grill transcript."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo)
            await grill_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        prompt = kwargs["user_prompt"]
        assert "GRILL TRANSCRIPT:" in prompt
        assert "Q1 [intent]:" in prompt

    async def test_compile_prompt_contains_codebase_profile(self, repo, project, ui):
        """Compile prompt includes serialized codebase profile."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo)
            await grill_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        prompt = kwargs["user_prompt"]
        assert "CODEBASE PROFILE:" in prompt
        assert "acme" in prompt

    async def test_compile_prompt_contains_constraints(self, repo, project, ui):
        """Compile prompt includes constraints when present."""
        state = _state(
            repo,
            project,
            constraints=["no external deps", "must be accessible"],
        )
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo)
            await grill_node(state, ui)

        _, kwargs = mock_run.call_args
        prompt = kwargs["user_prompt"]
        assert "CONSTRAINTS: no external deps; must be accessible" in prompt

    async def test_compile_prompt_no_constraints(self, repo, project, ui):
        """With no constraints, prompt shows 'None'."""
        state = _state(repo, project, constraints=[])
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo)
            await grill_node(state, ui)

        _, kwargs = mock_run.call_args
        assert "CONSTRAINTS: None" in kwargs["user_prompt"]

    async def test_compile_agent_kwargs(self, repo, project, ui):
        """run_agent compile call has correct stage, max_turns, tools."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo)
            await grill_node(_state(repo, project), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["stage"] == "grill_compile"
        assert kwargs["max_turns"] == 10
        assert kwargs["allowed_tools"] == ["Read", "Write", "Bash"]
        assert kwargs["cwd"] == str(repo)

    async def test_model_forwarded_to_compile(self, repo, project, ui):
        """model from state is passed to the compile agent."""
        state = _state(repo, project, model="claude-sonnet-4-20250514")
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo)
            await grill_node(state, ui)

        _, kwargs = mock_run.call_args
        assert kwargs["model"] == "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# grill_node — feature_spec.json reading
# ---------------------------------------------------------------------------


class TestGrillNodeSpecReading:
    """Verify feature_spec.json parsing, error handling, and cleanup."""

    async def test_valid_spec_parsed(self, repo, project, ui):
        """Valid JSON is parsed into the return dict."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo)
            result = await grill_node(_state(repo, project), ui)

        assert result["feature_spec"] == SAMPLE_SPEC

    async def test_invalid_json_yields_empty_dict(self, repo, project, ui):
        """Invalid JSON → ui.error called, feature_spec is empty dict."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            (repo / "feature_spec.json").write_text("{broken json!!!")
            result = await grill_node(_state(repo, project), ui)

        ui.error.assert_called_once()
        assert "feature_spec.json" in ui.error.call_args[0][0]
        assert result["feature_spec"] == {}

    async def test_missing_spec_file_yields_empty_dict(self, repo, project, ui):
        """When agent didn't write feature_spec.json, returns empty dict."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            # Don't write feature_spec.json
            result = await grill_node(_state(repo, project), ui)

        assert result["feature_spec"] == {}

    async def test_spec_file_cleaned_after_reading(self, repo, project, ui):
        """feature_spec.json is unlinked from repo after reading."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo)
            await grill_node(_state(repo, project), ui)

        assert not (repo / "feature_spec.json").exists()

    async def test_invalid_json_file_still_cleaned(self, repo, project, ui):
        """Even with invalid JSON, the file is still cleaned up."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            (repo / "feature_spec.json").write_text("NOT JSON {{{")
            await grill_node(_state(repo, project), ui)

        assert not (repo / "feature_spec.json").exists()

    async def test_missing_spec_file_no_cleanup_error(self, repo, project, ui):
        """When file doesn't exist, no unlink error occurs."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            # Doesn't crash even though no file to clean up
            await grill_node(_state(repo, project), ui)


# ---------------------------------------------------------------------------
# grill_node — research_redo_needed flag
# ---------------------------------------------------------------------------


class TestGrillNodeResearchRedo:
    """Verify research_redo_needed behavior."""

    async def test_redo_true_triggers_info(self, repo, project, ui):
        """When spec has research_redo_needed: True, ui.info is called."""
        spec_with_redo = {**SAMPLE_SPEC, "research_redo_needed": True}
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo, spec_with_redo)
            result = await grill_node(_state(repo, project), ui)

        assert result["research_redo_needed"] is True
        # Check that the redo info message was emitted
        redo_calls = [
            c
            for c in ui.info.call_args_list
            if "looping back" in str(c).lower() or "fundamental" in str(c).lower()
        ]
        assert len(redo_calls) == 1

    async def test_redo_false_no_redo_message(self, repo, project, ui):
        """When spec has research_redo_needed: False, no redo message."""
        spec_no_redo = {**SAMPLE_SPEC, "research_redo_needed": False}
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo, spec_no_redo)
            result = await grill_node(_state(repo, project), ui)

        assert result["research_redo_needed"] is False
        # The only info calls should be "Compiling..." — no redo message
        redo_calls = [
            c for c in ui.info.call_args_list if "looping back" in str(c).lower()
        ]
        assert len(redo_calls) == 0

    async def test_redo_missing_defaults_false(self, repo, project, ui):
        """When spec lacks research_redo_needed key, defaults to False."""
        spec_no_key = {k: v for k, v in SAMPLE_SPEC.items()}
        spec_no_key.pop("research_redo_needed", None)
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo, spec_no_key)
            result = await grill_node(_state(repo, project), ui)

        assert result["research_redo_needed"] is False

    async def test_empty_spec_redo_defaults_false(self, repo, project, ui):
        """Empty spec (missing file) → research_redo_needed is False."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            result = await grill_node(_state(repo, project), ui)

        assert result["research_redo_needed"] is False


# ---------------------------------------------------------------------------
# grill_node — artifact saving
# ---------------------------------------------------------------------------


class TestGrillNodeArtifacts:
    """Verify artifacts are saved correctly."""

    async def test_grill_transcript_saved(self, repo, project, ui):
        """grill_transcript.md is saved to artifacts directory."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo)
            await grill_node(_state(repo, project), ui)

        art = project / "artifacts" / "grill_transcript.md"
        assert art.exists()
        content = art.read_text()
        assert "Q1 [intent]:" in content

    async def test_feature_spec_artifact_saved(self, repo, project, ui):
        """feature_spec.json is saved to artifacts directory."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo)
            await grill_node(_state(repo, project), ui)

        art = project / "artifacts" / "feature_spec.json"
        assert art.exists()
        saved = json.loads(art.read_text())
        assert saved == SAMPLE_SPEC

    async def test_empty_spec_artifact_saved(self, repo, project, ui):
        """When spec file missing, empty dict artifact is saved."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            await grill_node(_state(repo, project), ui)

        art = project / "artifacts" / "feature_spec.json"
        assert art.exists()
        saved = json.loads(art.read_text())
        assert saved == {}

    async def test_invalid_json_spec_saves_empty_artifact(self, repo, project, ui):
        """When spec has invalid JSON, empty dict artifact is saved."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            (repo / "feature_spec.json").write_text("{bad json")
            await grill_node(_state(repo, project), ui)

        art = project / "artifacts" / "feature_spec.json"
        saved = json.loads(art.read_text())
        assert saved == {}


# ---------------------------------------------------------------------------
# grill_node — fallthrough to _generate_questions
# ---------------------------------------------------------------------------


class TestGrillNodeGenerateQuestionsFallthrough:
    """When technical_assessment has no open_questions, generate them."""

    async def test_no_open_questions_triggers_generate(self, repo, project, ui):
        """Empty open_questions triggers _generate_questions via agent."""
        state = _state(repo, project, technical_assessment={"open_questions": []})

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo)
            await grill_node(state, ui)

        # ui.info called with "No open questions" message
        first_info = ui.info.call_args_list[0]
        assert "No open questions" in first_info[0][0]

    async def test_missing_open_questions_key_triggers_generate(
        self, repo, project, ui
    ):
        """Missing open_questions key triggers _generate_questions."""
        state = _state(repo, project, technical_assessment={})

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            _write_spec(repo)
            await grill_node(state, ui)

        first_info = ui.info.call_args_list[0]
        assert "No open questions" in first_info[0][0]

    async def test_generate_questions_agent_called(self, repo, project, ui):
        """When falling through, the generate agent is invoked first."""
        generated = [
            {
                "question": "Generated Q?",
                "category": "intent",
                "recommended_answer": "Yes",
            }
        ]
        state = _state(repo, project, technical_assessment={"open_questions": []})

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()

            def side_effect(**kwargs):
                if kwargs.get("stage") == "grill_generate":
                    # Simulate agent writing open_questions.json
                    (repo / "open_questions.json").write_text(json.dumps(generated))
                return FakeAgentResult()

            mock_run.side_effect = side_effect
            _write_spec(repo)
            result = await grill_node(state, ui)

        # Should have asked the generated question
        ui.grill_question.assert_called_once_with("Generated Q?", "Yes", "intent", 1)
        assert "Generated Q?" in result["grill_transcript"]


# ---------------------------------------------------------------------------
# _generate_questions
# ---------------------------------------------------------------------------


class TestGenerateQuestions:
    """Tests for the _generate_questions helper."""

    async def test_successful_generation(self, repo, project, ui):
        """Agent writes valid open_questions.json → list returned."""
        questions = [
            {
                "question": "Q1?",
                "category": "intent",
                "recommended_answer": "A1",
            },
            {
                "question": "Q2?",
                "category": "edge_case",
                "recommended_answer": "A2",
            },
        ]

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:

            async def write_questions(**kwargs):
                (repo / "open_questions.json").write_text(json.dumps(questions))
                return FakeAgentResult()

            mock_run.side_effect = write_questions
            result = await _generate_questions(
                str(repo), str(project), "Add dark mode", {}, {}, ui, None
            )

        assert result == questions
        assert len(result) == 2

    async def test_invalid_json_returns_empty(self, repo, project, ui):
        """Invalid JSON in open_questions.json → returns empty list."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:

            async def write_bad_json(**kwargs):
                (repo / "open_questions.json").write_text("NOT JSON {{{")
                return FakeAgentResult()

            mock_run.side_effect = write_bad_json
            result = await _generate_questions(
                str(repo), str(project), "Add dark mode", {}, {}, ui, None
            )

        assert result == []

    async def test_missing_file_returns_empty(self, repo, project, ui):
        """When agent doesn't write open_questions.json → empty list."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            result = await _generate_questions(
                str(repo), str(project), "Add dark mode", {}, {}, ui, None
            )

        assert result == []

    async def test_non_list_json_returns_empty(self, repo, project, ui):
        """When JSON is valid but not a list → returns empty list."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:

            async def write_dict(**kwargs):
                (repo / "open_questions.json").write_text(json.dumps({"not": "a list"}))
                return FakeAgentResult()

            mock_run.side_effect = write_dict
            result = await _generate_questions(
                str(repo), str(project), "Add dark mode", {}, {}, ui, None
            )

        assert result == []

    async def test_file_cleaned_after_reading(self, repo, project, ui):
        """open_questions.json is unlinked after successful read."""
        questions = [
            {"question": "Q?", "category": "intent", "recommended_answer": "A"}
        ]

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:

            async def write_file(**kwargs):
                (repo / "open_questions.json").write_text(json.dumps(questions))
                return FakeAgentResult()

            mock_run.side_effect = write_file
            await _generate_questions(
                str(repo), str(project), "Add dark mode", {}, {}, ui, None
            )

        assert not (repo / "open_questions.json").exists()

    async def test_agent_called_with_correct_args(self, repo, project, ui):
        """run_agent is called with expected kwargs."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            await _generate_questions(
                str(repo),
                str(project),
                "Add dark mode",
                {"lang": "python"},
                {"edge_cases": ["timeout"]},
                ui,
                "claude-sonnet-4-20250514",
            )

        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs["stage"] == "grill_generate"
        assert kwargs["cwd"] == str(repo)
        assert kwargs["project_dir"] == str(project)
        assert kwargs["model"] == "claude-sonnet-4-20250514"
        assert kwargs["max_turns"] == 10
        assert "Glob" in kwargs["allowed_tools"]
        assert "Grep" in kwargs["allowed_tools"]

    async def test_prompt_contains_feature_and_context(self, repo, project, ui):
        """Generate prompt includes feature, profile, and assessment."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            await _generate_questions(
                str(repo),
                str(project),
                "Add SSO login",
                {"lang": "typescript"},
                {"edge_cases": ["session timeout"]},
                ui,
                None,
            )

        _, kwargs = mock_run.call_args
        prompt = kwargs["user_prompt"]
        assert "FEATURE: Add SSO login" in prompt
        assert "CODEBASE PROFILE:" in prompt
        assert "typescript" in prompt
        assert "TECHNICAL ASSESSMENT:" in prompt
        assert "session timeout" in prompt
        assert "open_questions.json" in prompt

    async def test_model_none_forwarded(self, repo, project, ui):
        """When model is None, it is forwarded as-is to run_agent."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            await _generate_questions(str(repo), str(project), "feat", {}, {}, ui, None)

        _, kwargs = mock_run.call_args
        assert kwargs["model"] is None
