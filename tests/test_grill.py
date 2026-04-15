"""Tests for graft.stages.grill."""

import json
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from graft.stages.grill import (
    COMPILE_SYSTEM_PROMPT,
    CONVERSATION_SYSTEM_PROMPT,
    _build_history_prompt,
    _parse_agent_response,
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


def _question_json(
    question: str,
    category: str = "intent",
    recommended: str = "Yes",
    why_asking: str = "Need to know",
) -> str:
    """Return a JSON string for a question response."""
    return json.dumps(
        {
            "status": "question",
            "question": question,
            "category": category,
            "recommended_answer": recommended,
            "why_asking": why_asking,
        }
    )


def _done_json(
    summary: str = "Feature is well understood.",
    assumptions: list[str] | None = None,
    confidence: str = "high",
) -> str:
    """Return a JSON string for a done response."""
    return json.dumps(
        {
            "status": "done",
            "summary": summary,
            "assumptions": assumptions or [],
            "confidence": confidence,
        }
    )


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
        "technical_assessment": {"risk": "low"},
    }
    base.update(kw)
    return base


def _write_spec(repo, spec=None):
    """Write feature_spec.json into repo so the compile step can read it."""
    (repo / "feature_spec.json").write_text(json.dumps(spec or SAMPLE_SPEC, indent=2))


def _make_conversation_side_effect(questions, done_text=None, spec_repo=None):
    """Build a side_effect for run_agent that simulates a conversation.

    *questions* is a list of JSON strings for question responses.
    After all questions are asked, the next call returns *done_text*.
    The final call (compile stage) returns a plain FakeAgentResult.
    """
    if done_text is None:
        done_text = _done_json()

    responses = [FakeAgentResult(text=q) for q in questions]
    responses.append(FakeAgentResult(text=done_text))

    call_idx = {"i": 0}

    async def side_effect(**kwargs):
        if kwargs.get("stage") == "grill_compile":
            if spec_repo is not None:
                _write_spec(spec_repo)
            return FakeAgentResult()
        idx = call_idx["i"]
        call_idx["i"] += 1
        if idx < len(responses):
            return responses[idx]
        return FakeAgentResult(text=done_text)

    return side_effect


# ---------------------------------------------------------------------------
# _parse_agent_response
# ---------------------------------------------------------------------------


class TestParseAgentResponse:
    """Tests for the JSON parsing helper."""

    def test_plain_json(self):
        data = _parse_agent_response('{"status": "done", "summary": "ok"}')
        assert data["status"] == "done"
        assert data["summary"] == "ok"

    def test_json_in_code_fence(self):
        text = '```json\n{"status": "question", "question": "Q?"}\n```'
        data = _parse_agent_response(text)
        assert data["status"] == "question"
        assert data["question"] == "Q?"

    def test_json_embedded_in_text(self):
        text = 'Here is my response: {"status": "done", "summary": "all good"} end'
        data = _parse_agent_response(text)
        assert data["status"] == "done"

    def test_garbage_returns_error(self):
        data = _parse_agent_response("not json at all")
        assert data["status"] == "error"

    def test_empty_string_returns_error(self):
        data = _parse_agent_response("")
        assert data["status"] == "error"


# ---------------------------------------------------------------------------
# _build_history_prompt
# ---------------------------------------------------------------------------


class TestBuildHistoryPrompt:
    """Tests for the prompt builder."""

    def test_includes_feature_prompt(self):
        prompt = _build_history_prompt("Add SSO", {}, {}, [], [])
        assert "FEATURE: Add SSO" in prompt

    def test_includes_codebase_profile(self):
        prompt = _build_history_prompt("feat", {"lang": "python"}, {}, [], [])
        assert "CODEBASE PROFILE:" in prompt
        assert "python" in prompt

    def test_includes_technical_assessment(self):
        prompt = _build_history_prompt("feat", {}, {"risk": "high"}, [], [])
        assert "TECHNICAL ASSESSMENT:" in prompt
        assert "high" in prompt

    def test_includes_constraints(self):
        prompt = _build_history_prompt("feat", {}, {}, ["no deps", "must test"], [])
        assert "CONSTRAINTS: no deps; must test" in prompt

    def test_empty_constraints_omitted(self):
        prompt = _build_history_prompt("feat", {}, {}, [], [])
        assert "CONSTRAINTS" not in prompt

    def test_includes_conversation_history(self):
        history = [
            {
                "role": "agent",
                "data": {
                    "question": "Use websockets?",
                    "category": "integration",
                    "recommended_answer": "Yes",
                    "why_asking": "Need realtime",
                },
                "turn": 1,
            },
            {"role": "user", "answer": "No, use polling"},
        ]
        prompt = _build_history_prompt("feat", {}, {}, [], history)
        assert "CONVERSATION SO FAR:" in prompt
        assert "Use websockets?" in prompt
        assert "No, use polling" in prompt

    def test_empty_history_no_conversation_section(self):
        prompt = _build_history_prompt("feat", {}, {}, [], [])
        assert "CONVERSATION SO FAR:" not in prompt

    def test_ends_with_next_question_instruction(self):
        prompt = _build_history_prompt("feat", {}, {}, [], [])
        assert "Ask your next question" in prompt


# ---------------------------------------------------------------------------
# grill_router (preserved)
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
        questions = [
            _question_json(
                "Should dark mode persist?",
                "intent",
                "Yes, store in localStorage",
                "Critical for UX",
            ),
        ]
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect(
                questions,
                spec_repo=repo,
            )
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
            mock_run.side_effect = _make_conversation_side_effect([], spec_repo=repo)
            result = await grill_node(_state(repo, project), ui)

        assert result["current_stage"] == "grill"

    async def test_grill_complete_is_true(self, repo, project, ui):
        """grill_complete flag is always True on success."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect([], spec_repo=repo)
            result = await grill_node(_state(repo, project), ui)

        assert result["grill_complete"] is True

    async def test_feature_spec_parsed(self, repo, project, ui):
        """Valid feature_spec.json is parsed into the returned dict."""
        questions = [
            _question_json("Q1?", "intent", "rec1", "why1"),
        ]
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect(
                questions,
                spec_repo=repo,
            )
            result = await grill_node(_state(repo, project), ui)

        assert result["feature_spec"] == SAMPLE_SPEC
        assert result["feature_spec"]["feature_name"] == "Dark Mode"

    async def test_grill_transcript_built(self, repo, project, ui):
        """Transcript contains Q&A lines for each question."""
        questions = [
            _question_json(
                "Should dark mode persist?",
                "intent",
                "Yes, store in localStorage",
                "Critical",
            ),
            _question_json(
                "Support system preference?",
                "preference",
                "Yes, use prefers-color-scheme",
                "UX decision",
            ),
        ]
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect(
                questions,
                spec_repo=repo,
            )
            result = await grill_node(_state(repo, project), ui)

        transcript = result["grill_transcript"]
        assert "Q1 [intent]" in transcript
        assert "Q2 [preference]" in transcript
        assert "Should dark mode persist" in transcript
        assert "user answer" in transcript

    async def test_ui_lifecycle(self, repo, project, ui):
        """stage_start and stage_done are called in order."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect([], spec_repo=repo)
            await grill_node(_state(repo, project), ui)

        ui.stage_start.assert_called_once_with("grill")
        ui.stage_done.assert_called_once_with("grill")

    async def test_marks_stage_complete(self, repo, project, ui):
        """Stage 'grill' is recorded in metadata after success."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect([], spec_repo=repo)
            await grill_node(_state(repo, project), ui)

        meta = json.loads((project / "metadata.json").read_text())
        assert "grill" in meta["stages_completed"]


# ---------------------------------------------------------------------------
# grill_node — question iteration & UI interaction
# ---------------------------------------------------------------------------


class TestGrillNodeQuestionIteration:
    """Verify question walking and UI calls."""

    async def test_grill_question_called_per_question(self, repo, project, ui):
        """ui.grill_question is called once per question before done."""
        questions = [
            _question_json("Q1?", "intent", "rec1", "why1"),
            _question_json("Q2?", "preference", "rec2", "why2"),
        ]
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect(
                questions,
                spec_repo=repo,
            )
            await grill_node(_state(repo, project), ui)

        assert ui.grill_question.call_count == 2

    async def test_grill_question_args(self, repo, project, ui):
        """ui.grill_question receives correct keyword arguments."""
        questions = [
            _question_json(
                "Should dark mode persist across sessions?",
                "intent",
                "Yes, store in localStorage",
                "Critical for UX",
            ),
            _question_json(
                "Support system preference detection?",
                "preference",
                "Yes, use prefers-color-scheme",
                "User expectations",
            ),
        ]
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect(
                questions,
                spec_repo=repo,
            )
            await grill_node(_state(repo, project), ui)

        first_call = ui.grill_question.call_args_list[0]
        assert first_call == call(
            question="Should dark mode persist across sessions?",
            recommended="Yes, store in localStorage",
            category="intent",
            number=1,
            why_asking="Critical for UX",
        )
        second_call = ui.grill_question.call_args_list[1]
        assert second_call == call(
            question="Support system preference detection?",
            recommended="Yes, use prefers-color-scheme",
            category="preference",
            number=2,
            why_asking="User expectations",
        )

    async def test_transcript_line_format(self, repo, project, ui):
        """Transcript lines follow the expected markdown format."""
        ui.grill_question.return_value = "my custom answer"
        questions = [
            _question_json(
                "Should dark mode persist across sessions?",
                "intent",
                "Yes, store in localStorage",
                "Critical for UX",
            ),
        ]
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect(
                questions,
                spec_repo=repo,
            )
            result = await grill_node(_state(repo, project), ui)

        transcript = result["grill_transcript"]
        assert "### Q1 [intent]" in transcript
        assert "**Should dark mode persist across sessions?**" in transcript
        assert "Recommended: Yes, store in localStorage" in transcript
        assert "**Answer:** my custom answer" in transcript

    async def test_decisions_list_populated(self, repo, project, ui):
        """Decisions are reflected in the compile prompt."""
        ui.grill_question.return_value = "accepted"
        questions = [
            _question_json("Q1?", "intent", "rec1", "why1"),
            _question_json("Q2?", "preference", "rec2", "why2"),
        ]
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect(
                questions,
                spec_repo=repo,
            )
            await grill_node(_state(repo, project), ui)

        # The compile call is the last one; check its prompt contains the Q&A
        compile_call = [
            c for c in mock_run.call_args_list if c[1].get("stage") == "grill_compile"
        ]
        assert len(compile_call) == 1
        prompt = compile_call[0][1]["user_prompt"]
        assert "Q1 [intent]" in prompt
        assert "Q2 [preference]" in prompt
        assert "accepted" in prompt

    async def test_done_signal_stops_conversation(self, repo, project, ui):
        """Agent returning status=done stops the loop (no questions asked)."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect([], spec_repo=repo)
            result = await grill_node(_state(repo, project), ui)

        ui.grill_question.assert_not_called()
        assert "Agent concluded after 0 questions" in result["grill_transcript"]

    async def test_error_response_stops_conversation(self, repo, project, ui):
        """Agent returning unparseable text stops the loop with error."""
        call_idx = {"i": 0}

        async def side_effect(**kwargs):
            if kwargs.get("stage") == "grill_compile":
                _write_spec(repo)
                return FakeAgentResult()
            call_idx["i"] += 1
            if call_idx["i"] == 1:
                return FakeAgentResult(text="not json at all")
            return FakeAgentResult()

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = side_effect
            await grill_node(_state(repo, project), ui)

        ui.grill_question.assert_not_called()
        ui.error.assert_called()

    async def test_empty_question_stops_conversation(self, repo, project, ui):
        """Agent returning a question with empty text stops the loop."""
        empty_q = json.dumps(
            {
                "status": "question",
                "question": "",
                "category": "intent",
                "recommended_answer": "rec",
                "why_asking": "why",
            }
        )
        call_idx = {"i": 0}

        async def side_effect(**kwargs):
            if kwargs.get("stage") == "grill_compile":
                _write_spec(repo)
                return FakeAgentResult()
            call_idx["i"] += 1
            if call_idx["i"] == 1:
                return FakeAgentResult(text=empty_q)
            return FakeAgentResult()

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = side_effect
            await grill_node(_state(repo, project), ui)

        ui.grill_question.assert_not_called()
        ui.error.assert_called()

    async def test_user_done_triggers_early_exit(self, repo, project, ui):
        """User typing 'done' triggers early exit with agent wrap-up."""
        ui.grill_question.return_value = "done"
        questions = [
            _question_json("Q1?", "intent", "rec", "why"),
        ]
        wrap_up = _done_json(
            summary="Wrapped up early",
            assumptions=["User wants simple approach"],
        )

        call_idx = {"i": 0}

        async def side_effect(**kwargs):
            nonlocal call_idx
            if kwargs.get("stage") == "grill_compile":
                _write_spec(repo)
                return FakeAgentResult()
            call_idx["i"] += 1
            if call_idx["i"] == 1:
                return FakeAgentResult(text=questions[0])
            # Second call is the wrap-up after user said "done"
            return FakeAgentResult(text=wrap_up)

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = side_effect
            result = await grill_node(_state(repo, project), ui)

        assert "User ended conversation" in result["grill_transcript"]


# ---------------------------------------------------------------------------
# grill_node — compile agent call
# ---------------------------------------------------------------------------


class TestGrillNodeCompileAgent:
    """Verify compile agent invocation."""

    async def test_compile_agent_system_prompt(self, repo, project, ui):
        """run_agent receives COMPILE_SYSTEM_PROMPT."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect([], spec_repo=repo)
            await grill_node(_state(repo, project), ui)

        compile_calls = [
            c for c in mock_run.call_args_list if c[1].get("stage") == "grill_compile"
        ]
        assert len(compile_calls) == 1
        assert compile_calls[0][1]["system_prompt"] is COMPILE_SYSTEM_PROMPT

    async def test_compile_prompt_contains_feature(self, repo, project, ui):
        """Compile prompt includes the feature prompt."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect([], spec_repo=repo)
            await grill_node(_state(repo, project), ui)

        compile_calls = [
            c for c in mock_run.call_args_list if c[1].get("stage") == "grill_compile"
        ]
        assert "FEATURE PROMPT: Add dark mode" in compile_calls[0][1]["user_prompt"]

    async def test_compile_prompt_contains_transcript(self, repo, project, ui):
        """Compile prompt includes the grill transcript."""
        questions = [
            _question_json("Q1?", "intent", "rec", "why"),
        ]
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect(
                questions,
                spec_repo=repo,
            )
            await grill_node(_state(repo, project), ui)

        compile_calls = [
            c for c in mock_run.call_args_list if c[1].get("stage") == "grill_compile"
        ]
        prompt = compile_calls[0][1]["user_prompt"]
        assert "CONVERSATION TRANSCRIPT:" in prompt
        assert "Q1 [intent]" in prompt

    async def test_compile_prompt_contains_codebase_profile(self, repo, project, ui):
        """Compile prompt includes serialized codebase profile."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect([], spec_repo=repo)
            await grill_node(_state(repo, project), ui)

        compile_calls = [
            c for c in mock_run.call_args_list if c[1].get("stage") == "grill_compile"
        ]
        prompt = compile_calls[0][1]["user_prompt"]
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
            mock_run.side_effect = _make_conversation_side_effect([], spec_repo=repo)
            await grill_node(state, ui)

        compile_calls = [
            c for c in mock_run.call_args_list if c[1].get("stage") == "grill_compile"
        ]
        prompt = compile_calls[0][1]["user_prompt"]
        assert "CONSTRAINTS: no external deps; must be accessible" in prompt

    async def test_compile_prompt_no_constraints(self, repo, project, ui):
        """With no constraints, prompt shows 'None'."""
        state = _state(repo, project, constraints=[])
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect([], spec_repo=repo)
            await grill_node(state, ui)

        compile_calls = [
            c for c in mock_run.call_args_list if c[1].get("stage") == "grill_compile"
        ]
        assert "CONSTRAINTS: None" in compile_calls[0][1]["user_prompt"]

    async def test_compile_agent_kwargs(self, repo, project, ui):
        """run_agent compile call has correct stage, max_turns, tools."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect([], spec_repo=repo)
            await grill_node(_state(repo, project), ui)

        compile_calls = [
            c for c in mock_run.call_args_list if c[1].get("stage") == "grill_compile"
        ]
        kwargs = compile_calls[0][1]
        assert kwargs["stage"] == "grill_compile"
        assert kwargs["max_turns"] == 10
        assert kwargs["allowed_tools"] == ["Read", "Write", "Bash"]
        assert kwargs["cwd"] == str(repo)

    async def test_model_forwarded_to_compile(self, repo, project, ui):
        """model from state is passed to the compile agent."""
        state = _state(repo, project, model="claude-sonnet-4-20250514")
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect([], spec_repo=repo)
            await grill_node(state, ui)

        compile_calls = [
            c for c in mock_run.call_args_list if c[1].get("stage") == "grill_compile"
        ]
        assert compile_calls[0][1]["model"] == "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# grill_node — conversation agent calls
# ---------------------------------------------------------------------------


class TestGrillNodeConversationAgent:
    """Verify conversation agent (_ask_one_question) invocations."""

    async def test_conversation_agent_system_prompt(self, repo, project, ui):
        """Conversation agent receives CONVERSATION_SYSTEM_PROMPT."""
        questions = [_question_json("Q?", "intent", "rec", "why")]
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect(
                questions,
                spec_repo=repo,
            )
            await grill_node(_state(repo, project), ui)

        conv_calls = [
            c
            for c in mock_run.call_args_list
            if c[1].get("stage", "").startswith("grill_q")
        ]
        assert len(conv_calls) >= 1
        assert conv_calls[0][1]["system_prompt"] is CONVERSATION_SYSTEM_PROMPT

    async def test_conversation_stage_numbered(self, repo, project, ui):
        """Each conversation turn has stage grill_q{N}."""
        questions = [
            _question_json("Q1?", "intent", "r1", "w1"),
            _question_json("Q2?", "scope", "r2", "w2"),
        ]
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect(
                questions,
                spec_repo=repo,
            )
            await grill_node(_state(repo, project), ui)

        conv_calls = [
            c
            for c in mock_run.call_args_list
            if c[1].get("stage", "").startswith("grill_q")
        ]
        stages = [c[1]["stage"] for c in conv_calls]
        assert "grill_q1" in stages
        assert "grill_q2" in stages

    async def test_conversation_uses_read_only_tools(self, repo, project, ui):
        """Conversation agent only has Read tool access."""
        questions = [_question_json("Q?", "intent", "rec", "why")]
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect(
                questions,
                spec_repo=repo,
            )
            await grill_node(_state(repo, project), ui)

        conv_calls = [
            c
            for c in mock_run.call_args_list
            if c[1].get("stage", "").startswith("grill_q")
        ]
        assert conv_calls[0][1]["allowed_tools"] == ["Read"]

    async def test_model_forwarded_to_conversation(self, repo, project, ui):
        """model from state is forwarded to conversation agent."""
        state = _state(repo, project, model="claude-sonnet-4-20250514")
        questions = [_question_json("Q?", "intent", "rec", "why")]
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect(
                questions,
                spec_repo=repo,
            )
            await grill_node(state, ui)

        conv_calls = [
            c
            for c in mock_run.call_args_list
            if c[1].get("stage", "").startswith("grill_q")
        ]
        assert conv_calls[0][1]["model"] == "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# grill_node — feature_spec.json reading
# ---------------------------------------------------------------------------


class TestGrillNodeSpecReading:
    """Verify feature_spec.json parsing, error handling, and cleanup."""

    async def test_valid_spec_parsed(self, repo, project, ui):
        """Valid JSON is parsed into the return dict."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect([], spec_repo=repo)
            result = await grill_node(_state(repo, project), ui)

        assert result["feature_spec"] == SAMPLE_SPEC

    async def test_invalid_json_yields_empty_dict(self, repo, project, ui):
        """Invalid JSON → ui.error called, feature_spec is empty dict."""

        async def side_effect(**kwargs):
            if kwargs.get("stage") == "grill_compile":
                (repo / "feature_spec.json").write_text("{broken json!!!")
                return FakeAgentResult()
            return FakeAgentResult(text=_done_json())

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = side_effect
            result = await grill_node(_state(repo, project), ui)

        ui.error.assert_called_once()
        assert "feature_spec.json" in ui.error.call_args[0][0]
        assert result["feature_spec"] == {}

    async def test_missing_spec_file_yields_empty_dict(self, repo, project, ui):
        """When agent didn't write feature_spec.json, returns empty dict."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect([])
            # Don't write feature_spec.json (no spec_repo)
            result = await grill_node(_state(repo, project), ui)

        assert result["feature_spec"] == {}

    async def test_spec_file_cleaned_after_reading(self, repo, project, ui):
        """feature_spec.json is unlinked from repo after reading."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect([], spec_repo=repo)
            await grill_node(_state(repo, project), ui)

        assert not (repo / "feature_spec.json").exists()

    async def test_invalid_json_file_still_cleaned(self, repo, project, ui):
        """Even with invalid JSON, the file is still cleaned up."""

        async def side_effect(**kwargs):
            if kwargs.get("stage") == "grill_compile":
                (repo / "feature_spec.json").write_text("NOT JSON {{{")
                return FakeAgentResult()
            return FakeAgentResult(text=_done_json())

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = side_effect
            await grill_node(_state(repo, project), ui)

        assert not (repo / "feature_spec.json").exists()

    async def test_missing_spec_file_no_cleanup_error(self, repo, project, ui):
        """When file doesn't exist, no unlink error occurs."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect([])
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

        async def side_effect(**kwargs):
            if kwargs.get("stage") == "grill_compile":
                _write_spec(repo, spec_with_redo)
                return FakeAgentResult()
            return FakeAgentResult(text=_done_json())

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = side_effect
            result = await grill_node(_state(repo, project), ui)

        assert result["research_redo_needed"] is True
        redo_calls = [
            c
            for c in ui.info.call_args_list
            if "looping back" in str(c).lower() or "fundamental" in str(c).lower()
        ]
        assert len(redo_calls) == 1

    async def test_redo_false_no_redo_message(self, repo, project, ui):
        """When spec has research_redo_needed: False, no redo message."""
        spec_no_redo = {**SAMPLE_SPEC, "research_redo_needed": False}

        async def side_effect(**kwargs):
            if kwargs.get("stage") == "grill_compile":
                _write_spec(repo, spec_no_redo)
                return FakeAgentResult()
            return FakeAgentResult(text=_done_json())

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = side_effect
            result = await grill_node(_state(repo, project), ui)

        assert result["research_redo_needed"] is False
        redo_calls = [
            c for c in ui.info.call_args_list if "looping back" in str(c).lower()
        ]
        assert len(redo_calls) == 0

    async def test_redo_missing_defaults_false(self, repo, project, ui):
        """When spec lacks research_redo_needed key, defaults to False."""
        spec_no_key = {k: v for k, v in SAMPLE_SPEC.items()}
        spec_no_key.pop("research_redo_needed", None)

        async def side_effect(**kwargs):
            if kwargs.get("stage") == "grill_compile":
                _write_spec(repo, spec_no_key)
                return FakeAgentResult()
            return FakeAgentResult(text=_done_json())

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = side_effect
            result = await grill_node(_state(repo, project), ui)

        assert result["research_redo_needed"] is False

    async def test_empty_spec_redo_defaults_false(self, repo, project, ui):
        """Empty spec (missing file) → research_redo_needed is False."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect([])
            result = await grill_node(_state(repo, project), ui)

        assert result["research_redo_needed"] is False


# ---------------------------------------------------------------------------
# grill_node — artifact saving
# ---------------------------------------------------------------------------


class TestGrillNodeArtifacts:
    """Verify artifacts are saved correctly."""

    async def test_grill_transcript_saved(self, repo, project, ui):
        """grill_transcript.md is saved to artifacts directory."""
        questions = [_question_json("Q1?", "intent", "rec", "why")]
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect(
                questions,
                spec_repo=repo,
            )
            await grill_node(_state(repo, project), ui)

        art = project / "artifacts" / "grill_transcript.md"
        assert art.exists()
        content = art.read_text()
        assert "Q1 [intent]" in content

    async def test_feature_spec_artifact_saved(self, repo, project, ui):
        """feature_spec.json is saved to artifacts directory."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect([], spec_repo=repo)
            await grill_node(_state(repo, project), ui)

        art = project / "artifacts" / "feature_spec.json"
        assert art.exists()
        saved = json.loads(art.read_text())
        assert saved == SAMPLE_SPEC

    async def test_empty_spec_artifact_saved(self, repo, project, ui):
        """When spec file missing, empty dict artifact is saved."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect([])
            await grill_node(_state(repo, project), ui)

        art = project / "artifacts" / "feature_spec.json"
        assert art.exists()
        saved = json.loads(art.read_text())
        assert saved == {}

    async def test_invalid_json_spec_saves_empty_artifact(self, repo, project, ui):
        """When spec has invalid JSON, empty dict artifact is saved."""

        async def side_effect(**kwargs):
            if kwargs.get("stage") == "grill_compile":
                (repo / "feature_spec.json").write_text("{bad json")
                return FakeAgentResult()
            return FakeAgentResult(text=_done_json())

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = side_effect
            await grill_node(_state(repo, project), ui)

        art = project / "artifacts" / "feature_spec.json"
        saved = json.loads(art.read_text())
        assert saved == {}


# ---------------------------------------------------------------------------
# grill_node — auto_approve mode
# ---------------------------------------------------------------------------


class TestGrillNodeAutoApprove:
    """Verify auto_approve passes through to UI and conversation works."""

    async def test_auto_approve_in_state(self, repo, project, ui):
        """auto_approve flag from state is used by the node."""
        questions = [_question_json("Q?", "intent", "rec", "why")]
        state = _state(repo, project, auto_approve=True)
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = _make_conversation_side_effect(
                questions,
                spec_repo=repo,
            )
            result = await grill_node(state, ui)

        assert result["grill_complete"] is True
        ui.grill_question.assert_called_once()


# ---------------------------------------------------------------------------
# MAX_QUESTIONS boundary
# ---------------------------------------------------------------------------


class TestMaxQuestionsBoundary:
    """Tests for behavior when the agent asks all MAX_QUESTIONS."""

    @pytest.mark.asyncio
    async def test_max_questions_exhausted(self, repo, project, ui):
        """When agent never says 'done', loop exits after MAX_QUESTIONS."""
        from graft.stages.grill import MAX_QUESTIONS

        ui.grill_question.return_value = "ok"
        endless_q = _question_json("Tell me more?", "scope")

        call_idx = {"i": 0}

        async def side_effect(**kwargs):
            if kwargs.get("stage") == "grill_compile":
                _write_spec(repo, {})
                return FakeAgentResult()
            idx = call_idx["i"]
            call_idx["i"] += 1
            if idx < MAX_QUESTIONS:
                return FakeAgentResult(text=endless_q)
            return FakeAgentResult(text=_done_json())

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = side_effect
            result = await grill_node(_state(repo, project), ui)

        assert "feature_spec" in result
        assert ui.grill_question.call_count == MAX_QUESTIONS
        assert (
            f"Reached maximum {MAX_QUESTIONS} questions" in result["grill_transcript"]
        )
        ui.info.assert_any_call(
            f"Reached maximum {MAX_QUESTIONS} questions. Compiling spec."
        )


# ---------------------------------------------------------------------------
# Early exit edge cases
# ---------------------------------------------------------------------------


class TestEarlyExitEdgeCases:
    """Edge cases for user typing 'done'."""

    @pytest.mark.asyncio
    async def test_wrap_up_returns_non_done(self, repo, project, ui):
        """Wrap-up returning a question instead of done doesn't crash."""
        ui.grill_question.return_value = "done"
        q1 = _question_json("What platform?", "platform")
        non_done_wrap = _question_json("One more thing?", "scope")

        call_idx = {"i": 0}

        async def side_effect(**kwargs):
            if kwargs.get("stage") == "grill_compile":
                _write_spec(repo, {})
                return FakeAgentResult()
            idx = call_idx["i"]
            call_idx["i"] += 1
            if idx == 0:
                return FakeAgentResult(text=q1)
            # Wrap-up returns a question, not done
            return FakeAgentResult(text=non_done_wrap)

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = side_effect
            result = await grill_node(_state(repo, project), ui)

        assert "feature_spec" in result
        assert "User ended conversation at Q1" in result["grill_transcript"]

    @pytest.mark.asyncio
    async def test_early_exit_records_triggering_question(self, repo, project, ui):
        """When user says 'done', triggering question appears in transcript."""
        ui.grill_question.return_value = "done"
        q1 = _question_json("What platform?", "platform")
        wrap = _done_json(
            summary="Filled gaps",
            assumptions=["Assumed web"],
        )

        call_idx = {"i": 0}

        async def side_effect(**kwargs):
            if kwargs.get("stage") == "grill_compile":
                _write_spec(repo, {})
                return FakeAgentResult()
            idx = call_idx["i"]
            call_idx["i"] += 1
            if idx == 0:
                return FakeAgentResult(text=q1)
            return FakeAgentResult(text=wrap)

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = side_effect
            result = await grill_node(_state(repo, project), ui)

        transcript = result["grill_transcript"]
        assert "What platform?" in transcript
        assert "(user ended early)" in transcript
        assert "User ended conversation at Q1" in transcript
