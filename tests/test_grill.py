"""Tests for graft.stages.grill."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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


def _state(repo, project, **kw):
    """Build a minimal FeatureState dict."""
    base = {
        "repo_path": str(repo),
        "project_dir": str(project),
    }
    base.update(kw)
    return base


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
# Happy-path grill_node — open_questions from Research
# ---------------------------------------------------------------------------


class TestGrillNodeHappyPath:
    """Core happy-path: Research provides open_questions in technical_assessment."""

    async def test_returns_expected_keys(self, repo, project, ui):
        """grill_node returns the correct set of state keys."""
        assessment = {
            "open_questions": [
                {
                    "question": "Auth method?",
                    "recommended_answer": "OAuth",
                    "category": "intent",
                }
            ]
        }

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            result = await grill_node(
                _state(repo, project, technical_assessment=assessment), ui
            )

        assert set(result.keys()) == {
            "feature_spec",
            "grill_transcript",
            "grill_complete",
            "research_redo_needed",
            "current_stage",
        }
        assert result["current_stage"] == "grill"
        assert result["grill_complete"] is True
        assert result["research_redo_needed"] is False

    async def test_processes_all_questions(self, repo, project, ui):
        """Each open_question triggers a ui.grill_question call."""
        questions = [
            {
                "question": "Q1?",
                "recommended_answer": "A1",
                "category": "intent",
            },
            {
                "question": "Q2?",
                "recommended_answer": "A2",
                "category": "edge_case",
            },
            {
                "question": "Q3?",
                "recommended_answer": "A3",
                "category": "preference",
            },
        ]
        assessment = {"open_questions": questions}

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            await grill_node(_state(repo, project, technical_assessment=assessment), ui)

        assert ui.grill_question.call_count == 3
        # Verify question text, recommended, category, and number
        calls = ui.grill_question.call_args_list
        assert calls[0].args == ("Q1?", "A1", "intent", 1)
        assert calls[1].args == ("Q2?", "A2", "edge_case", 2)
        assert calls[2].args == ("Q3?", "A3", "preference", 3)

    async def test_transcript_content(self, repo, project, ui):
        """Transcript captures questions, recommendations, and user answers."""
        ui.grill_question.return_value = "my answer"
        assessment = {
            "open_questions": [
                {
                    "question": "Scope?",
                    "recommended_answer": "MVP only",
                    "category": "prioritization",
                }
            ]
        }

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            result = await grill_node(
                _state(repo, project, technical_assessment=assessment), ui
            )

        transcript = result["grill_transcript"]
        assert "Q1 [prioritization]: Scope?" in transcript
        assert "Recommended: MVP only" in transcript
        assert "Answer: my answer" in transcript

    async def test_ui_lifecycle(self, repo, project, ui):
        """stage_start and stage_done are called in order."""
        assessment = {"open_questions": [{"question": "Q?"}]}

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            await grill_node(_state(repo, project, technical_assessment=assessment), ui)

        ui.stage_start.assert_called_once_with("grill")
        ui.stage_done.assert_called_once_with("grill")

    async def test_no_generate_questions_call_when_questions_present(
        self, repo, project, ui
    ):
        """When open_questions exist, _generate_questions is NOT called."""
        assessment = {
            "open_questions": [
                {"question": "Q?", "recommended_answer": "A", "category": "intent"}
            ]
        }

        with (
            patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run,
            patch(
                "graft.stages.grill._generate_questions", new_callable=AsyncMock
            ) as mock_gen,
        ):
            mock_run.return_value = FakeAgentResult()
            await grill_node(_state(repo, project, technical_assessment=assessment), ui)

        mock_gen.assert_not_called()


# ---------------------------------------------------------------------------
# Fallback: _generate_questions when no open_questions
# ---------------------------------------------------------------------------


class TestGrillNodeFallbackGenerate:
    """When technical_assessment has no open_questions, _generate_questions is invoked."""

    async def test_generate_called_when_no_questions(self, repo, project, ui):
        """grill_node calls _generate_questions when open_questions is empty."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            # First call is _generate_questions agent, second is compile agent
            mock_run.return_value = FakeAgentResult()
            await grill_node(_state(repo, project, technical_assessment={}), ui)

        # ui.info should be called with the fallback message
        info_calls = [c.args[0] for c in ui.info.call_args_list]
        assert any("No open questions" in msg for msg in info_calls)

    async def test_generate_called_when_no_assessment(self, repo, project, ui):
        """grill_node calls _generate_questions when technical_assessment is absent."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            await grill_node(_state(repo, project), ui)

        info_calls = [c.args[0] for c in ui.info.call_args_list]
        assert any("No open questions" in msg for msg in info_calls)

    async def test_generated_questions_are_used(self, repo, project, ui):
        """Questions generated by _generate_questions are walked through by grill_node."""
        generated_qs = [
            {
                "question": "Generated Q?",
                "recommended_answer": "Gen A",
                "category": "edge_case",
            }
        ]

        # Write the questions file that _generate_questions will read
        def write_questions_file(*args, **kwargs):
            qs_path = Path(str(repo)) / "open_questions.json"
            qs_path.write_text(json.dumps(generated_qs))
            return FakeAgentResult()

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            # First call (_generate_questions) writes file; second (compile) is no-op
            mock_run.side_effect = [
                write_questions_file(),  # generate call — we need the side-effect though
                FakeAgentResult(),  # compile call
            ]
            # Manually write the file before the node runs since side_effect
            # doesn't execute the function — do it via a separate mechanism
            qs_path = repo / "open_questions.json"
            qs_path.write_text(json.dumps(generated_qs))

            result = await grill_node(
                _state(repo, project, technical_assessment={}), ui
            )

        # grill_question was called with the generated question
        ui.grill_question.assert_called_once_with(
            "Generated Q?", "Gen A", "edge_case", 1
        )

    async def test_empty_questions_means_no_grill(self, repo, project, ui):
        """If _generate_questions returns empty list, no grill_question calls are made."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            # No open_questions.json file will exist → returns []
            result = await grill_node(
                _state(repo, project, technical_assessment={}), ui
            )

        ui.grill_question.assert_not_called()
        # Transcript should be empty
        assert result["grill_transcript"] == ""


# ---------------------------------------------------------------------------
# Compile agent invocation
# ---------------------------------------------------------------------------


class TestCompileAgent:
    """Verify run_agent is invoked correctly for the compile step."""

    async def test_compile_agent_args(self, repo, project, ui):
        """run_agent for compile step receives correct kwargs."""
        assessment = {
            "open_questions": [
                {"question": "Q?", "recommended_answer": "A", "category": "intent"}
            ]
        }

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            await grill_node(
                _state(
                    repo,
                    project,
                    technical_assessment=assessment,
                    feature_prompt="Add dark mode",
                    codebase_profile={"name": "acme"},
                    model="claude-sonnet-4-20250514",
                ),
                ui,
            )

        _, kwargs = mock_run.call_args
        assert kwargs["system_prompt"] is COMPILE_SYSTEM_PROMPT
        assert kwargs["stage"] == "grill_compile"
        assert kwargs["cwd"] == str(repo)
        assert kwargs["project_dir"] == str(project)
        assert kwargs["max_turns"] == 10
        assert kwargs["allowed_tools"] == ["Read", "Write", "Bash"]
        assert kwargs["model"] == "claude-sonnet-4-20250514"
        assert "Add dark mode" in kwargs["user_prompt"]
        assert "acme" in kwargs["user_prompt"]

    async def test_compile_prompt_includes_transcript(self, repo, project, ui):
        """The compile user_prompt includes the grill transcript."""
        ui.grill_question.return_value = "sure thing"
        assessment = {
            "open_questions": [
                {
                    "question": "Use REST?",
                    "recommended_answer": "Yes",
                    "category": "intent",
                }
            ]
        }

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            await grill_node(_state(repo, project, technical_assessment=assessment), ui)

        _, kwargs = mock_run.call_args
        assert "Use REST?" in kwargs["user_prompt"]
        assert "sure thing" in kwargs["user_prompt"]

    async def test_compile_prompt_includes_constraints(self, repo, project, ui):
        """The compile user_prompt includes constraints from state."""
        assessment = {"open_questions": [{"question": "Q?"}]}

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            await grill_node(
                _state(
                    repo,
                    project,
                    technical_assessment=assessment,
                    constraints=["Must use PostgreSQL", "No external deps"],
                ),
                ui,
            )

        _, kwargs = mock_run.call_args
        assert "Must use PostgreSQL" in kwargs["user_prompt"]
        assert "No external deps" in kwargs["user_prompt"]

    async def test_compile_prompt_no_constraints(self, repo, project, ui):
        """When constraints is empty, prompt shows 'None'."""
        assessment = {"open_questions": [{"question": "Q?"}]}

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            await grill_node(_state(repo, project, technical_assessment=assessment), ui)

        _, kwargs = mock_run.call_args
        assert "CONSTRAINTS: None" in kwargs["user_prompt"]


# ---------------------------------------------------------------------------
# Artifact saving: grill_transcript.md and feature_spec.json
# ---------------------------------------------------------------------------


class TestArtifacts:
    """Verify artifacts are saved to the project directory."""

    async def test_saves_grill_transcript(self, repo, project, ui):
        """grill_transcript.md is persisted under project_dir/artifacts."""
        ui.grill_question.return_value = "go with recommended"
        assessment = {
            "open_questions": [
                {
                    "question": "Framework?",
                    "recommended_answer": "React",
                    "category": "preference",
                }
            ]
        }

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            await grill_node(_state(repo, project, technical_assessment=assessment), ui)

        transcript_path = project / "artifacts" / "grill_transcript.md"
        assert transcript_path.exists()
        content = transcript_path.read_text()
        assert "Framework?" in content
        assert "React" in content
        assert "go with recommended" in content

    async def test_saves_feature_spec_artifact(self, repo, project, ui):
        """feature_spec.json is saved under project_dir/artifacts."""
        assessment = {"open_questions": [{"question": "Q?"}]}
        spec = {"feature_name": "Trade System", "decisions": []}

        def write_spec(*args, **kwargs):
            (Path(str(repo)) / "feature_spec.json").write_text(json.dumps(spec))
            return FakeAgentResult()

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = write_spec
            await grill_node(_state(repo, project, technical_assessment=assessment), ui)

        artifact_path = project / "artifacts" / "feature_spec.json"
        assert artifact_path.exists()
        saved = json.loads(artifact_path.read_text())
        assert saved["feature_name"] == "Trade System"

    async def test_feature_spec_cleaned_from_repo(self, repo, project, ui):
        """feature_spec.json is removed from repo_path after being read."""
        assessment = {"open_questions": [{"question": "Q?"}]}
        spec = {"feature_name": "Clean"}

        def write_spec(*args, **kwargs):
            (Path(str(repo)) / "feature_spec.json").write_text(json.dumps(spec))
            return FakeAgentResult()

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = write_spec
            await grill_node(_state(repo, project, technical_assessment=assessment), ui)

        assert not (repo / "feature_spec.json").exists()

    async def test_empty_spec_when_no_file_produced(self, repo, project, ui):
        """When agent doesn't produce feature_spec.json, spec is empty dict."""
        assessment = {"open_questions": [{"question": "Q?"}]}

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            result = await grill_node(
                _state(repo, project, technical_assessment=assessment), ui
            )

        assert result["feature_spec"] == {}

    async def test_marks_stage_complete(self, repo, project, ui):
        """Stage 'grill' is recorded in metadata after success."""
        assessment = {"open_questions": [{"question": "Q?"}]}

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            await grill_node(_state(repo, project, technical_assessment=assessment), ui)

        meta = json.loads((project / "metadata.json").read_text())
        assert "grill" in meta["stages_completed"]


# ---------------------------------------------------------------------------
# Feature spec parsing
# ---------------------------------------------------------------------------


class TestFeatureSpecParsing:
    """Verify feature_spec.json reading and error handling."""

    async def test_malformed_json_spec(self, repo, project, ui):
        """Invalid JSON in feature_spec.json → ui.error, empty spec returned."""
        assessment = {"open_questions": [{"question": "Q?"}]}

        def write_bad_spec(*args, **kwargs):
            (Path(str(repo)) / "feature_spec.json").write_text("{broken json")
            return FakeAgentResult()

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = write_bad_spec
            result = await grill_node(
                _state(repo, project, technical_assessment=assessment), ui
            )

        ui.error.assert_called_once_with("Failed to parse feature_spec.json.")
        assert result["feature_spec"] == {}

    async def test_malformed_json_still_cleans_up(self, repo, project, ui):
        """Even with bad JSON, feature_spec.json is cleaned from repo."""
        assessment = {"open_questions": [{"question": "Q?"}]}

        def write_bad_spec(*args, **kwargs):
            (Path(str(repo)) / "feature_spec.json").write_text("NOT JSON")
            return FakeAgentResult()

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = write_bad_spec
            await grill_node(_state(repo, project, technical_assessment=assessment), ui)

        assert not (repo / "feature_spec.json").exists()

    async def test_malformed_json_still_completes_stage(self, repo, project, ui):
        """Malformed spec doesn't prevent stage from completing."""
        assessment = {"open_questions": [{"question": "Q?"}]}

        def write_bad_spec(*args, **kwargs):
            (Path(str(repo)) / "feature_spec.json").write_text(">>>")
            return FakeAgentResult()

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = write_bad_spec
            await grill_node(_state(repo, project, technical_assessment=assessment), ui)

        meta = json.loads((project / "metadata.json").read_text())
        assert "grill" in meta["stages_completed"]


# ---------------------------------------------------------------------------
# Research redo detection
# ---------------------------------------------------------------------------


class TestResearchRedo:
    """Detect when the feature spec signals a research_redo_needed."""

    async def test_research_redo_true(self, repo, project, ui):
        """When feature_spec has research_redo_needed=True, result propagates."""
        assessment = {"open_questions": [{"question": "Q?"}]}
        spec = {"feature_name": "X", "research_redo_needed": True}

        def write_spec(*args, **kwargs):
            (Path(str(repo)) / "feature_spec.json").write_text(json.dumps(spec))
            return FakeAgentResult()

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = write_spec
            result = await grill_node(
                _state(repo, project, technical_assessment=assessment), ui
            )

        assert result["research_redo_needed"] is True
        info_calls = [c.args[0] for c in ui.info.call_args_list]
        assert any("looping back to Research" in msg for msg in info_calls)

    async def test_research_redo_false(self, repo, project, ui):
        """When feature_spec has research_redo_needed=False, result is False."""
        assessment = {"open_questions": [{"question": "Q?"}]}
        spec = {"feature_name": "X", "research_redo_needed": False}

        def write_spec(*args, **kwargs):
            (Path(str(repo)) / "feature_spec.json").write_text(json.dumps(spec))
            return FakeAgentResult()

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = write_spec
            result = await grill_node(
                _state(repo, project, technical_assessment=assessment), ui
            )

        assert result["research_redo_needed"] is False

    async def test_research_redo_absent(self, repo, project, ui):
        """When feature_spec lacks research_redo_needed, defaults to False."""
        assessment = {"open_questions": [{"question": "Q?"}]}
        spec = {"feature_name": "X"}

        def write_spec(*args, **kwargs):
            (Path(str(repo)) / "feature_spec.json").write_text(json.dumps(spec))
            return FakeAgentResult()

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = write_spec
            result = await grill_node(
                _state(repo, project, technical_assessment=assessment), ui
            )

        assert result["research_redo_needed"] is False


# ---------------------------------------------------------------------------
# Question format handling
# ---------------------------------------------------------------------------


class TestQuestionFormats:
    """Verify the node handles different question formats (dict, str, partial dict)."""

    async def test_string_questions(self, repo, project, ui):
        """Plain string questions use defaults for recommended and category."""
        assessment = {"open_questions": ["What about auth?", "What about storage?"]}

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            result = await grill_node(
                _state(repo, project, technical_assessment=assessment), ui
            )

        calls = ui.grill_question.call_args_list
        assert calls[0].args == ("What about auth?", "No recommendation", "intent", 1)
        assert calls[1].args == (
            "What about storage?",
            "No recommendation",
            "intent",
            2,
        )

    async def test_dict_question_missing_recommended(self, repo, project, ui):
        """Dict question without recommended_answer uses 'No recommendation'."""
        assessment = {
            "open_questions": [{"question": "Design?", "category": "preference"}]
        }

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            await grill_node(_state(repo, project, technical_assessment=assessment), ui)

        calls = ui.grill_question.call_args_list
        assert calls[0].args == ("Design?", "No recommendation", "preference", 1)

    async def test_dict_question_missing_category(self, repo, project, ui):
        """Dict question without category defaults to 'intent'."""
        assessment = {
            "open_questions": [{"question": "Protocol?", "recommended_answer": "gRPC"}]
        }

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            await grill_node(_state(repo, project, technical_assessment=assessment), ui)

        calls = ui.grill_question.call_args_list
        assert calls[0].args == ("Protocol?", "gRPC", "intent", 1)

    async def test_dict_question_missing_question_key(self, repo, project, ui):
        """Dict without 'question' key uses the dict itself as fallback."""
        assessment = {
            "open_questions": [
                {"recommended_answer": "Something", "category": "edge_case"}
            ]
        }

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            await grill_node(_state(repo, project, technical_assessment=assessment), ui)

        # q.get("question", q) returns q (the dict) when no "question" key
        # This becomes the dict itself as the question text
        calls = ui.grill_question.call_args_list
        assert calls[0].args[0] == {
            "recommended_answer": "Something",
            "category": "edge_case",
        }


# ---------------------------------------------------------------------------
# _generate_questions unit tests
# ---------------------------------------------------------------------------


class TestGenerateQuestions:
    """Direct tests for the _generate_questions helper."""

    async def test_returns_questions_from_file(self, repo, project, ui):
        """When agent writes open_questions.json, the questions are returned."""
        questions = [
            {"question": "Gen Q1?", "recommended_answer": "A1", "category": "intent"},
            {
                "question": "Gen Q2?",
                "recommended_answer": "A2",
                "category": "edge_case",
            },
        ]

        def write_qs(*args, **kwargs):
            (Path(str(repo)) / "open_questions.json").write_text(json.dumps(questions))
            return FakeAgentResult()

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = write_qs
            result = await _generate_questions(
                str(repo), str(project), "Add feature", {}, {}, ui, None
            )

        assert len(result) == 2
        assert result[0]["question"] == "Gen Q1?"
        assert result[1]["category"] == "edge_case"

    async def test_returns_empty_when_no_file(self, repo, project, ui):
        """When agent doesn't write open_questions.json, returns empty list."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            result = await _generate_questions(
                str(repo), str(project), "Add feature", {}, {}, ui, None
            )

        assert result == []

    async def test_returns_empty_on_malformed_json(self, repo, project, ui):
        """When open_questions.json has invalid JSON, returns empty list."""

        def write_bad(*args, **kwargs):
            (Path(str(repo)) / "open_questions.json").write_text("not json!")
            return FakeAgentResult()

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = write_bad
            result = await _generate_questions(
                str(repo), str(project), "Add feature", {}, {}, ui, None
            )

        assert result == []

    async def test_returns_empty_when_json_is_not_list(self, repo, project, ui):
        """When open_questions.json contains a dict (not list), returns empty list."""

        def write_dict(*args, **kwargs):
            (Path(str(repo)) / "open_questions.json").write_text(
                json.dumps({"not": "a list"})
            )
            return FakeAgentResult()

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = write_dict
            result = await _generate_questions(
                str(repo), str(project), "Add feature", {}, {}, ui, None
            )

        assert result == []

    async def test_cleans_up_questions_file(self, repo, project, ui):
        """open_questions.json is removed from repo after being read."""
        qs = [{"question": "Q?", "recommended_answer": "A", "category": "intent"}]

        def write_qs(*args, **kwargs):
            (Path(str(repo)) / "open_questions.json").write_text(json.dumps(qs))
            return FakeAgentResult()

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = write_qs
            await _generate_questions(
                str(repo), str(project), "Add feature", {}, {}, ui, None
            )

        assert not (repo / "open_questions.json").exists()

    async def test_agent_called_with_correct_args(self, repo, project, ui):
        """run_agent is called with the right stage, tools, and prompt content."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            await _generate_questions(
                str(repo),
                str(project),
                "Build a chat feature",
                {"name": "acme"},
                {"risk": "low"},
                ui,
                "claude-sonnet-4-20250514",
            )

        _, kwargs = mock_run.call_args
        assert kwargs["stage"] == "grill_generate"
        assert kwargs["max_turns"] == 10
        assert kwargs["allowed_tools"] == ["Read", "Write", "Bash", "Glob", "Grep"]
        assert kwargs["model"] == "claude-sonnet-4-20250514"
        assert kwargs["cwd"] == str(repo)
        assert kwargs["project_dir"] == str(project)
        assert "Build a chat feature" in kwargs["user_prompt"]
        assert "acme" in kwargs["user_prompt"]
        assert "risk" in kwargs["user_prompt"]

    async def test_agent_receives_generate_system_prompt(self, repo, project, ui):
        """The system prompt mentions 'Principal Product Interrogator'."""
        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            await _generate_questions(
                str(repo), str(project), "Add feature", {}, {}, ui, None
            )

        _, kwargs = mock_run.call_args
        assert "Principal Product Interrogator" in kwargs["system_prompt"]


# ---------------------------------------------------------------------------
# Decision tracking
# ---------------------------------------------------------------------------


class TestDecisions:
    """Verify the decisions list built during Q&A is correct."""

    async def test_decisions_capture_answers(self, repo, project, ui):
        """Each decision records question, recommended, answer, category."""
        ui.grill_question.side_effect = ["yes", "no"]
        assessment = {
            "open_questions": [
                {
                    "question": "Use caching?",
                    "recommended_answer": "Yes",
                    "category": "preference",
                },
                {
                    "question": "Deploy to prod?",
                    "recommended_answer": "Staging first",
                    "category": "prioritization",
                },
            ]
        }

        spec_with_decisions = {
            "feature_name": "X",
            "decisions": [],
        }

        def write_spec(*args, **kwargs):
            (Path(str(repo)) / "feature_spec.json").write_text(
                json.dumps(spec_with_decisions)
            )
            return FakeAgentResult()

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = write_spec
            result = await grill_node(
                _state(repo, project, technical_assessment=assessment), ui
            )

        transcript = result["grill_transcript"]
        assert "Answer: yes" in transcript
        assert "Answer: no" in transcript
        assert "Use caching?" in transcript
        assert "Deploy to prod?" in transcript


# ---------------------------------------------------------------------------
# State defaults
# ---------------------------------------------------------------------------


class TestStateDefaults:
    """Verify defaults when optional state keys are absent."""

    async def test_missing_feature_prompt(self, repo, project, ui):
        """feature_prompt defaults to empty string."""
        assessment = {"open_questions": [{"question": "Q?"}]}

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            await grill_node(_state(repo, project, technical_assessment=assessment), ui)

        _, kwargs = mock_run.call_args
        assert "FEATURE PROMPT: \n" in kwargs["user_prompt"]

    async def test_missing_codebase_profile(self, repo, project, ui):
        """codebase_profile defaults to empty dict."""
        assessment = {"open_questions": [{"question": "Q?"}]}

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            await grill_node(_state(repo, project, technical_assessment=assessment), ui)

        _, kwargs = mock_run.call_args
        assert "{}" in kwargs["user_prompt"]

    async def test_missing_constraints(self, repo, project, ui):
        """constraints defaults to empty list, prompt shows 'None'."""
        assessment = {"open_questions": [{"question": "Q?"}]}

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            await grill_node(_state(repo, project, technical_assessment=assessment), ui)

        _, kwargs = mock_run.call_args
        assert "CONSTRAINTS: None" in kwargs["user_prompt"]

    async def test_model_defaults_to_none(self, repo, project, ui):
        """model defaults to None when not in state."""
        assessment = {"open_questions": [{"question": "Q?"}]}

        with patch("graft.stages.grill.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = FakeAgentResult()
            await grill_node(_state(repo, project, technical_assessment=assessment), ui)

        _, kwargs = mock_run.call_args
        assert kwargs["model"] is None
