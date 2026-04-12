"""Tests for graft.stages.grill."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from graft.stages.grill import _generate_questions, grill_node, grill_router

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


SAMPLE_QUESTIONS = [
    {
        "question": "Should dark mode persist across sessions?",
        "category": "intent",
        "recommended_answer": "Yes, store in localStorage",
    },
    {
        "question": "Should we support system preference detection?",
        "category": "preference",
        "recommended_answer": "Yes, use prefers-color-scheme",
    },
    {
        "question": "What is the MVP scope?",
        "category": "prioritization",
        "recommended_answer": "Toggle only, no scheduling",
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
    "scope": {"mvp": ["toggle"], "follow_up": ["scheduling"]},
    "constraints": ["Follow existing theme patterns"],
    "technical_notes": ["Reuse ThemeProvider"],
}


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
    m.stage_log = MagicMock()
    m.grill_question = MagicMock(return_value="user answer")
    return m


def _state(repo, project, **kw):
    """Build a minimal FeatureState dict."""
    base = {
        "repo_path": str(repo),
        "project_dir": str(project),
        "feature_prompt": "Add dark mode",
        "codebase_profile": {"project": {"name": "acme"}},
        "technical_assessment": {"open_questions": SAMPLE_QUESTIONS},
        "constraints": [],
    }
    base.update(kw)
    return base


def _write_spec_side_effect(repo, spec=None):
    """Return a side_effect callable that writes feature_spec.json when run_agent is called."""
    spec_data = spec if spec is not None else SAMPLE_SPEC

    async def side_effect(**kwargs):
        if kwargs.get("stage") == "grill_compile":
            (Path(repo) / "feature_spec.json").write_text(json.dumps(spec_data))
        return FakeAgentResult(text="compiled")

    return side_effect


# ---------------------------------------------------------------------------
# Router tests (preserved from original)
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
# Happy-path grill_node
# ---------------------------------------------------------------------------


class TestGrillNodeHappyPath:
    """Core happy-path tests for grill_node."""

    async def test_returns_expected_keys(self, repo, project, ui):
        """grill_node returns feature_spec, grill_transcript, grill_complete, etc."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            result = await grill_node(_state(repo, project), ui)

        assert set(result.keys()) == {
            "feature_spec",
            "grill_transcript",
            "grill_complete",
            "research_redo_needed",
            "current_stage",
        }

    async def test_grill_complete_is_true(self, repo, project, ui):
        """grill_complete is always True on success."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            result = await grill_node(_state(repo, project), ui)

        assert result["grill_complete"] is True

    async def test_current_stage_is_grill(self, repo, project, ui):
        """current_stage is set to 'grill'."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            result = await grill_node(_state(repo, project), ui)

        assert result["current_stage"] == "grill"

    async def test_feature_spec_parsed_from_json(self, repo, project, ui):
        """feature_spec is read from the JSON file written by run_agent."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            result = await grill_node(_state(repo, project), ui)

        assert result["feature_spec"]["feature_name"] == "Dark Mode"
        assert len(result["feature_spec"]["decisions"]) == 1

    async def test_ui_lifecycle(self, repo, project, ui):
        """stage_start and stage_done are called with 'grill'."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            await grill_node(_state(repo, project), ui)

        ui.stage_start.assert_called_once_with("grill")
        ui.stage_done.assert_called_once_with("grill")

    async def test_marks_stage_complete(self, repo, project, ui):
        """Stage 'grill' is recorded in metadata after success."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            await grill_node(_state(repo, project), ui)

        meta = json.loads((project / "metadata.json").read_text())
        assert "grill" in meta["stages_completed"]


# ---------------------------------------------------------------------------
# Question presentation loop
# ---------------------------------------------------------------------------


class TestQuestionLoop:
    """Verify the Q&A loop calls UI correctly for each question."""

    async def test_grill_question_called_for_each_question(self, repo, project, ui):
        """ui.grill_question is called once per open question."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            await grill_node(_state(repo, project), ui)

        assert ui.grill_question.call_count == len(SAMPLE_QUESTIONS)

    async def test_grill_question_receives_correct_args(self, repo, project, ui):
        """Each grill_question call passes question, recommended, category, number."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            await grill_node(_state(repo, project), ui)

        calls = ui.grill_question.call_args_list
        # First question (1-indexed)
        assert calls[0] == call(
            "Should dark mode persist across sessions?",
            "Yes, store in localStorage",
            "intent",
            1,
        )
        # Second question
        assert calls[1] == call(
            "Should we support system preference detection?",
            "Yes, use prefers-color-scheme",
            "preference",
            2,
        )
        # Third question
        assert calls[2] == call(
            "What is the MVP scope?",
            "Toggle only, no scheduling",
            "prioritization",
            3,
        )

    async def test_recommended_answer_displayed(self, repo, project, ui):
        """Recommended answer from each question dict is passed to grill_question."""
        single_q = [
            {
                "question": "Which DB?",
                "category": "preference",
                "recommended_answer": "PostgreSQL via Supabase",
            }
        ]
        state = _state(repo, project, technical_assessment={"open_questions": single_q})

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            await grill_node(state, ui)

        ui.grill_question.assert_called_once_with(
            "Which DB?", "PostgreSQL via Supabase", "preference", 1
        )

    async def test_string_questions_handled(self, repo, project, ui):
        """Plain string questions (not dicts) are handled gracefully."""
        string_questions = ["What color?", "What size?"]
        state = _state(
            repo, project, technical_assessment={"open_questions": string_questions}
        )

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            await grill_node(state, ui)

        calls = ui.grill_question.call_args_list
        assert calls[0] == call("What color?", "No recommendation", "intent", 1)
        assert calls[1] == call("What size?", "No recommendation", "intent", 2)

    async def test_empty_questions_list_skips_loop(self, repo, project, ui):
        """Empty open_questions triggers _generate_questions fallback (not direct loop)."""
        state = _state(repo, project, technical_assessment={"open_questions": []})

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await grill_node(state, ui)

        # grill_question not called because _generate_questions returns empty list
        ui.grill_question.assert_not_called()

    async def test_different_answers_per_question(self, repo, project, ui):
        """Each question can receive a different user answer."""
        ui.grill_question.side_effect = ["Answer A", "Answer B", "Answer C"]

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            result = await grill_node(_state(repo, project), ui)

        transcript = result["grill_transcript"]
        assert "Answer A" in transcript
        assert "Answer B" in transcript
        assert "Answer C" in transcript


# ---------------------------------------------------------------------------
# Transcript compilation
# ---------------------------------------------------------------------------


class TestTranscript:
    """Verify grill transcript format and content."""

    async def test_transcript_contains_all_questions(self, repo, project, ui):
        """Every question appears in the transcript."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            result = await grill_node(_state(repo, project), ui)

        transcript = result["grill_transcript"]
        for q in SAMPLE_QUESTIONS:
            assert q["question"] in transcript

    async def test_transcript_contains_recommended_answers(self, repo, project, ui):
        """Recommended answers appear in the transcript."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            result = await grill_node(_state(repo, project), ui)

        transcript = result["grill_transcript"]
        for q in SAMPLE_QUESTIONS:
            assert q["recommended_answer"] in transcript

    async def test_transcript_contains_user_answers(self, repo, project, ui):
        """User answers appear in the transcript."""
        ui.grill_question.return_value = "my custom answer"

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            result = await grill_node(_state(repo, project), ui)

        assert "my custom answer" in result["grill_transcript"]

    async def test_transcript_format_numbered(self, repo, project, ui):
        """Transcript lines are numbered Q1, Q2, Q3 with categories."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            result = await grill_node(_state(repo, project), ui)

        transcript = result["grill_transcript"]
        assert "Q1 [intent]:" in transcript
        assert "Q2 [preference]:" in transcript
        assert "Q3 [prioritization]:" in transcript

    async def test_transcript_saved_as_artifact(self, repo, project, ui):
        """grill_transcript.md is saved in the artifacts directory."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            result = await grill_node(_state(repo, project), ui)

        artifact_path = project / "artifacts" / "grill_transcript.md"
        assert artifact_path.exists()
        assert artifact_path.read_text() == result["grill_transcript"]

    async def test_empty_transcript_when_no_questions(self, repo, project, ui):
        """With no questions (and no generated ones), transcript is empty."""
        state = _state(repo, project, technical_assessment={"open_questions": []})

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            result = await grill_node(state, ui)

        assert result["grill_transcript"] == ""


# ---------------------------------------------------------------------------
# Feature spec generation (JSON from Q&A)
# ---------------------------------------------------------------------------


class TestFeatureSpec:
    """Verify feature_spec.json reading and cleanup."""

    async def test_spec_read_into_result(self, repo, project, ui):
        """feature_spec.json is parsed and returned in result."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            result = await grill_node(_state(repo, project), ui)

        assert result["feature_spec"] == SAMPLE_SPEC

    async def test_spec_saved_as_artifact(self, repo, project, ui):
        """feature_spec.json is persisted in project artifacts."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            await grill_node(_state(repo, project), ui)

        artifact_path = project / "artifacts" / "feature_spec.json"
        assert artifact_path.exists()
        saved = json.loads(artifact_path.read_text())
        assert saved == SAMPLE_SPEC

    async def test_spec_cleaned_up_from_repo(self, repo, project, ui):
        """feature_spec.json is removed from repo after reading."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            await grill_node(_state(repo, project), ui)

        assert not (repo / "feature_spec.json").exists()

    async def test_spec_missing_yields_empty_dict(self, repo, project, ui):
        """When run_agent doesn't produce feature_spec.json, result is empty dict."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="no spec written")
            result = await grill_node(_state(repo, project), ui)

        assert result["feature_spec"] == {}

    async def test_malformed_spec_json_yields_empty_dict(self, repo, project, ui):
        """Invalid JSON in feature_spec.json → ui.error, empty dict."""

        async def write_bad_json(**kwargs):
            if kwargs.get("stage") == "grill_compile":
                (Path(str(repo)) / "feature_spec.json").write_text("NOT VALID {{{")
            return FakeAgentResult(text="")

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = write_bad_json
            result = await grill_node(_state(repo, project), ui)

        ui.error.assert_called_once()
        assert "feature_spec.json" in ui.error.call_args[0][0]
        assert result["feature_spec"] == {}

    async def test_malformed_spec_still_completes_stage(self, repo, project, ui):
        """Malformed spec doesn't prevent stage from completing."""

        async def write_bad_json(**kwargs):
            if kwargs.get("stage") == "grill_compile":
                (Path(str(repo)) / "feature_spec.json").write_text("{broken")
            return FakeAgentResult(text="")

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = write_bad_json
            await grill_node(_state(repo, project), ui)

        meta = json.loads((project / "metadata.json").read_text())
        assert "grill" in meta["stages_completed"]

    async def test_malformed_spec_cleaned_up(self, repo, project, ui):
        """Even broken spec file is cleaned from repo."""

        async def write_bad_json(**kwargs):
            if kwargs.get("stage") == "grill_compile":
                (Path(str(repo)) / "feature_spec.json").write_text("oops")
            return FakeAgentResult(text="")

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = write_bad_json
            await grill_node(_state(repo, project), ui)

        assert not (repo / "feature_spec.json").exists()


# ---------------------------------------------------------------------------
# research_redo_needed flag
# ---------------------------------------------------------------------------


class TestResearchRedo:
    """Verify research_redo_needed detection and routing."""

    async def test_redo_flag_false_by_default(self, repo, project, ui):
        """When spec has no research_redo_needed, flag is False."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            result = await grill_node(_state(repo, project), ui)

        assert result["research_redo_needed"] is False

    async def test_redo_flag_true_from_spec(self, repo, project, ui):
        """When spec sets research_redo_needed=True, flag propagates."""
        redo_spec = {**SAMPLE_SPEC, "research_redo_needed": True}

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo, spec=redo_spec)
            result = await grill_node(_state(repo, project), ui)

        assert result["research_redo_needed"] is True

    async def test_redo_flag_triggers_info_message(self, repo, project, ui):
        """When redo is needed, ui.info is called with a redo message."""
        redo_spec = {**SAMPLE_SPEC, "research_redo_needed": True}

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo, spec=redo_spec)
            await grill_node(_state(repo, project), ui)

        info_messages = [c[0][0] for c in ui.info.call_args_list]
        assert any("Research" in msg and "loop" in msg.lower() for msg in info_messages)

    async def test_redo_false_no_redo_message(self, repo, project, ui):
        """When no redo needed, no redo info message is logged."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            await grill_node(_state(repo, project), ui)

        info_messages = [c[0][0] for c in ui.info.call_args_list]
        assert not any("loop" in msg.lower() for msg in info_messages)

    async def test_empty_spec_redo_is_false(self, repo, project, ui):
        """Empty spec (missing file) means no redo needed."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            result = await grill_node(_state(repo, project), ui)

        assert result["research_redo_needed"] is False


# ---------------------------------------------------------------------------
# _generate_questions fallback
# ---------------------------------------------------------------------------


class TestGenerateQuestions:
    """Test _generate_questions when Research produces no open_questions."""

    async def test_generates_questions_from_agent(self, repo, project, ui):
        """Agent writes open_questions.json, which is parsed and returned."""
        generated = [
            {
                "question": "Generated Q1?",
                "category": "intent",
                "recommended_answer": "Yes",
            },
            {
                "question": "Generated Q2?",
                "category": "edge_case",
                "recommended_answer": "Handle gracefully",
            },
        ]

        async def write_questions(**kwargs):
            (Path(str(repo)) / "open_questions.json").write_text(json.dumps(generated))
            return FakeAgentResult(text="")

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = write_questions
            result = await _generate_questions(
                str(repo),
                str(project),
                "Add dark mode",
                {"project": {"name": "acme"}},
                {},
                ui,
                None,
            )

        assert len(result) == 2
        assert result[0]["question"] == "Generated Q1?"
        assert result[1]["category"] == "edge_case"

    async def test_cleans_up_questions_file(self, repo, project, ui):
        """open_questions.json is removed after reading."""
        generated = [
            {"question": "Q?", "category": "intent", "recommended_answer": "Y"}
        ]

        async def write_questions(**kwargs):
            (Path(str(repo)) / "open_questions.json").write_text(json.dumps(generated))
            return FakeAgentResult(text="")

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = write_questions
            await _generate_questions(str(repo), str(project), "feat", {}, {}, ui, None)

        assert not (repo / "open_questions.json").exists()

    async def test_returns_empty_list_when_no_file(self, repo, project, ui):
        """When agent doesn't write open_questions.json, returns []."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="no file")
            result = await _generate_questions(
                str(repo), str(project), "feat", {}, {}, ui, None
            )

        assert result == []

    async def test_returns_empty_list_on_bad_json(self, repo, project, ui):
        """Malformed open_questions.json → returns []."""

        async def write_bad(**kwargs):
            (Path(str(repo)) / "open_questions.json").write_text("NOT JSON!!!")
            return FakeAgentResult(text="")

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = write_bad
            result = await _generate_questions(
                str(repo), str(project), "feat", {}, {}, ui, None
            )

        assert result == []

    async def test_returns_empty_list_when_json_is_not_list(self, repo, project, ui):
        """If open_questions.json contains a dict instead of list, returns []."""

        async def write_dict(**kwargs):
            (Path(str(repo)) / "open_questions.json").write_text(
                json.dumps({"not": "a list"})
            )
            return FakeAgentResult(text="")

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = write_dict
            result = await _generate_questions(
                str(repo), str(project), "feat", {}, {}, ui, None
            )

        assert result == []

    async def test_agent_called_with_correct_stage(self, repo, project, ui):
        """run_agent is invoked with stage='grill_generate'."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await _generate_questions(str(repo), str(project), "feat", {}, {}, ui, None)

        _, kwargs = mock_run.call_args
        assert kwargs["stage"] == "grill_generate"

    async def test_agent_called_with_correct_tools(self, repo, project, ui):
        """run_agent allowed_tools includes Read, Write, Bash, Glob, Grep."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await _generate_questions(str(repo), str(project), "feat", {}, {}, ui, None)

        _, kwargs = mock_run.call_args
        assert kwargs["allowed_tools"] == ["Read", "Write", "Bash", "Glob", "Grep"]

    async def test_model_forwarded_to_agent(self, repo, project, ui):
        """model parameter is passed through to run_agent."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await _generate_questions(
                str(repo), str(project), "feat", {}, {}, ui, "claude-sonnet-4-20250514"
            )

        _, kwargs = mock_run.call_args
        assert kwargs["model"] == "claude-sonnet-4-20250514"

    async def test_prompt_contains_feature(self, repo, project, ui):
        """The user prompt includes the feature prompt."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await _generate_questions(
                str(repo), str(project), "Add SSO login", {}, {}, ui, None
            )

        _, kwargs = mock_run.call_args
        assert "Add SSO login" in kwargs["user_prompt"]


# ---------------------------------------------------------------------------
# Fallback from research to _generate_questions in grill_node
# ---------------------------------------------------------------------------


class TestFallbackGeneration:
    """Verify grill_node invokes _generate_questions when no open_questions."""

    async def test_info_logged_when_no_questions(self, repo, project, ui):
        """ui.info is called when Research provides no open_questions."""
        state = _state(repo, project, technical_assessment={})

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            await grill_node(state, ui)

        info_messages = [c[0][0] for c in ui.info.call_args_list]
        assert any(
            "generating" in m.lower() or "no open questions" in m.lower()
            for m in info_messages
        )

    async def test_generated_questions_used_in_loop(self, repo, project, ui):
        """When _generate_questions returns questions, they're walked through."""
        state = _state(repo, project, technical_assessment={})
        generated = [
            {
                "question": "Generated Q?",
                "category": "intent",
                "recommended_answer": "Gen answer",
            }
        ]

        call_count = 0

        async def agent_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("stage") == "grill_generate":
                (Path(str(repo)) / "open_questions.json").write_text(
                    json.dumps(generated)
                )
            elif kwargs.get("stage") == "grill_compile":
                (Path(str(repo)) / "feature_spec.json").write_text(
                    json.dumps(SAMPLE_SPEC)
                )
            return FakeAgentResult(text="")

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = agent_side_effect
            await grill_node(state, ui)

        ui.grill_question.assert_called_once_with(
            "Generated Q?", "Gen answer", "intent", 1
        )

    async def test_run_agent_called_twice_for_generate_and_compile(
        self, repo, project, ui
    ):
        """When questions are generated, run_agent is called twice: generate + compile."""
        state = _state(repo, project, technical_assessment={})

        async def agent_side_effect(**kwargs):
            if kwargs.get("stage") == "grill_generate":
                (Path(str(repo)) / "open_questions.json").write_text("[]")
            return FakeAgentResult(text="")

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = agent_side_effect
            await grill_node(state, ui)

        assert mock_run.call_count == 2
        stages = [c[1]["stage"] for c in mock_run.call_args_list]
        assert stages == ["grill_generate", "grill_compile"]


# ---------------------------------------------------------------------------
# Auto-approve mode (UI returns recommended answer)
# ---------------------------------------------------------------------------


class TestAutoApproveMode:
    """Test auto_approve mode where UI automatically returns the recommended answer."""

    async def test_auto_approve_uses_recommended_answers(self, repo, project, ui):
        """When grill_question returns the recommended answer (auto_approve), transcript reflects it."""
        # Simulate auto_approve: grill_question returns whatever recommended was passed
        ui.grill_question.side_effect = lambda q, rec, cat, num: rec

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            result = await grill_node(_state(repo, project), ui)

        transcript = result["grill_transcript"]
        # Verify each answer matches the recommended
        assert "Answer: Yes, store in localStorage" in transcript
        assert "Answer: Yes, use prefers-color-scheme" in transcript
        assert "Answer: Toggle only, no scheduling" in transcript

    async def test_auto_approve_decisions_match_recommended(self, repo, project, ui):
        """In auto_approve mode, decisions have answer == recommended."""
        ui.grill_question.side_effect = lambda q, rec, cat, num: rec

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            result = await grill_node(_state(repo, project), ui)

        # The compile prompt receives the transcript where answers match recommended
        compile_call = mock_run.call_args_list[-1]
        compile_prompt = compile_call[1]["user_prompt"]
        assert "Yes, store in localStorage" in compile_prompt

    async def test_auto_approve_no_interactive_block(self, repo, project, ui):
        """Auto-approve completes without blocking on input (all via grill_question mock)."""
        ui.grill_question.side_effect = lambda q, rec, cat, num: rec

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            result = await grill_node(_state(repo, project), ui)

        assert result["grill_complete"] is True
        assert ui.grill_question.call_count == 3


# ---------------------------------------------------------------------------
# Compile agent invocation
# ---------------------------------------------------------------------------


class TestCompileAgent:
    """Verify run_agent is called correctly for the compile step."""

    async def test_compile_stage_name(self, repo, project, ui):
        """Compile step uses stage='grill_compile'."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            await grill_node(_state(repo, project), ui)

        compile_call = mock_run.call_args_list[-1]
        assert compile_call[1]["stage"] == "grill_compile"

    async def test_compile_allowed_tools(self, repo, project, ui):
        """Compile step has Read, Write, Bash tools."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            await grill_node(_state(repo, project), ui)

        compile_call = mock_run.call_args_list[-1]
        assert compile_call[1]["allowed_tools"] == ["Read", "Write", "Bash"]

    async def test_compile_prompt_includes_feature_prompt(self, repo, project, ui):
        """Compile prompt includes the feature prompt."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            await grill_node(_state(repo, project, feature_prompt="Add SSO login"), ui)

        compile_call = mock_run.call_args_list[-1]
        assert "Add SSO login" in compile_call[1]["user_prompt"]

    async def test_compile_prompt_includes_transcript(self, repo, project, ui):
        """Compile prompt includes the grill transcript."""
        ui.grill_question.return_value = "my special answer"

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            await grill_node(_state(repo, project), ui)

        compile_call = mock_run.call_args_list[-1]
        prompt = compile_call[1]["user_prompt"]
        assert "my special answer" in prompt
        assert "GRILL TRANSCRIPT" in prompt

    async def test_compile_prompt_includes_codebase_profile(self, repo, project, ui):
        """Compile prompt includes codebase profile."""
        profile = {"language": "typescript", "framework": "next.js"}

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            await grill_node(_state(repo, project, codebase_profile=profile), ui)

        compile_call = mock_run.call_args_list[-1]
        prompt = compile_call[1]["user_prompt"]
        assert "CODEBASE PROFILE" in prompt
        assert "typescript" in prompt

    async def test_compile_prompt_includes_constraints(self, repo, project, ui):
        """When constraints are set, they appear in the compile prompt."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            await grill_node(
                _state(
                    repo, project, constraints=["no ext deps", "must be accessible"]
                ),
                ui,
            )

        compile_call = mock_run.call_args_list[-1]
        prompt = compile_call[1]["user_prompt"]
        assert "no ext deps" in prompt
        assert "must be accessible" in prompt

    async def test_compile_prompt_none_constraints(self, repo, project, ui):
        """When constraints list is empty, prompt shows 'None'."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            await grill_node(_state(repo, project, constraints=[]), ui)

        compile_call = mock_run.call_args_list[-1]
        prompt = compile_call[1]["user_prompt"]
        assert "None" in prompt

    async def test_compile_max_turns(self, repo, project, ui):
        """Compile step uses max_turns=10."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            await grill_node(_state(repo, project), ui)

        compile_call = mock_run.call_args_list[-1]
        assert compile_call[1]["max_turns"] == 10

    async def test_model_forwarded_to_compile(self, repo, project, ui):
        """model from state is passed through to compile run_agent."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            await grill_node(
                _state(repo, project, model="claude-sonnet-4-20250514"), ui
            )

        compile_call = mock_run.call_args_list[-1]
        assert compile_call[1]["model"] == "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and missing state fields."""

    async def test_missing_feature_prompt_defaults_empty(self, repo, project, ui):
        """Missing feature_prompt defaults to empty string."""
        state = {
            "repo_path": str(repo),
            "project_dir": str(project),
            "technical_assessment": {"open_questions": []},
        }

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult(text="")
            result = await grill_node(state, ui)

        assert result["current_stage"] == "grill"

    async def test_missing_codebase_profile_defaults_empty(self, repo, project, ui):
        """Missing codebase_profile defaults to empty dict."""
        state = {
            "repo_path": str(repo),
            "project_dir": str(project),
            "technical_assessment": {"open_questions": SAMPLE_QUESTIONS},
        }

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            result = await grill_node(state, ui)

        assert result["grill_complete"] is True

    async def test_missing_constraints_defaults_empty(self, repo, project, ui):
        """Missing constraints defaults to empty list."""
        state = {
            "repo_path": str(repo),
            "project_dir": str(project),
            "technical_assessment": {"open_questions": SAMPLE_QUESTIONS},
        }

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            result = await grill_node(state, ui)

        assert result["grill_complete"] is True

    async def test_question_dict_missing_recommended(self, repo, project, ui):
        """Question dict without recommended_answer uses 'No recommendation'."""
        questions = [{"question": "What color?", "category": "preference"}]
        state = _state(
            repo, project, technical_assessment={"open_questions": questions}
        )

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            await grill_node(state, ui)

        ui.grill_question.assert_called_once_with(
            "What color?", "No recommendation", "preference", 1
        )

    async def test_question_dict_missing_category(self, repo, project, ui):
        """Question dict without category defaults to 'intent'."""
        questions = [{"question": "What size?", "recommended_answer": "Large"}]
        state = _state(
            repo, project, technical_assessment={"open_questions": questions}
        )

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            await grill_node(state, ui)

        ui.grill_question.assert_called_once_with("What size?", "Large", "intent", 1)

    async def test_single_question_flow(self, repo, project, ui):
        """A single question flows correctly through the entire pipeline."""
        questions = [
            {
                "question": "Solo question?",
                "category": "edge_case",
                "recommended_answer": "Handle it",
            }
        ]
        state = _state(
            repo, project, technical_assessment={"open_questions": questions}
        )
        ui.grill_question.return_value = "Yes, handle it carefully"

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _write_spec_side_effect(repo)
            result = await grill_node(state, ui)

        assert "Solo question?" in result["grill_transcript"]
        assert "Yes, handle it carefully" in result["grill_transcript"]
        assert "Q1 [edge_case]:" in result["grill_transcript"]
        assert result["grill_complete"] is True
