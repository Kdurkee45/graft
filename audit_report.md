# Graft (Feature Factory) — Comprehensive Audit Report

**Audit Date:** 2026-04-11
**Project:** graft v0.1.0
**Description:** AI-powered feature building for existing codebases — discover, research, grill, plan, execute, verify, and open a PR.

---

## Phase 1 — Discovery

### Project Overview

Graft is a Python CLI tool that implements a 6-stage AI pipeline for building features into existing codebases. It uses LangGraph for state machine orchestration, the Claude Agent SDK for AI-powered code generation, Typer for the CLI interface, and Rich for terminal UI.

**Architecture:** Linear pipeline with one conditional loop-back:
```
Discover → Research → Grill → [Grill↔Research loop] → Plan → [Plan Review] → Execute → Verify
```

### Service Detection

**Single-service repository.** One `pyproject.toml` at the root with the `hatchling` build backend.

| Attribute         | Value                          |
|-------------------|--------------------------------|
| Language          | Python 3.11                    |
| Framework         | LangGraph + Typer + Claude SDK |
| Package Manager   | uv / hatchling                 |
| Test Runner       | pytest 8.0+                    |
| Linter            | ruff 0.15+                     |
| Type Checker      | mypy 1.20+                     |
| Formatter         | (none configured)              |

### Codebase Size

| Metric     | Count |
|------------|-------|
| Source LOC | 2,164 |
| Test LOC   | 344   |
| Total LOC  | 2,508 |
| Source Files | 15  |
| Test Files  | 8    |
| Total Files | 24   |
| Test:Source Ratio | 0.16:1 |

### Directory Structure

```
src/graft/
├── __init__.py          (1 LOC)   — Package root
├── config.py           (52 LOC)   — Settings from env/.env
├── state.py            (72 LOC)   — LangGraph FeatureState TypedDict
├── artifacts.py        (83 LOC)   — Persistent artifact layer (disk I/O)
├── agent.py           (128 LOC)   — Claude Agent SDK wrapper
├── ui.py              (216 LOC)   — Rich terminal UI
├── graph.py           (105 LOC)   — LangGraph state machine wiring
├── cli.py             (197 LOC)   — Typer CLI (build, resume, list)
└── stages/
    ├── __init__.py      (1 LOC)
    ├── discover.py    (215 LOC)   — Codebase discovery agent
    ├── research.py    (187 LOC)   — Feature research agent
    ├── grill.py       (233 LOC)   — Human interrogation + spec compilation
    ├── plan.py        (210 LOC)   — Build plan generation + review
    ├── execute.py     (293 LOC)   — Unit-by-unit implementation
    └── verify.py      (171 LOC)   — Regression + PR opening
```

### Dependencies

**Runtime:**
- `typer[all]>=0.15.0` — CLI framework
- `rich>=13.0.0` — Terminal UI
- `langgraph>=0.4.0` — State machine orchestration
- `claude-agent-sdk>=0.1.0` — AI agent interface
- `python-dotenv>=1.0.0` — Env file loading

**Dev:**
- `mypy>=1.20.0` — Type checking
- `ruff>=0.15.0` — Linting
- `pytest>=8.0.0` — Testing
- `pytest-cov>=6.0.0` — Coverage

### CI/CD Configuration

**No CI/CD configuration found.** The repository has no `.github/workflows/`, `.circleci/`, `Jenkinsfile`, or `.gitlab-ci.yml`.

---

## Phase 2 — Quality Baseline

### 1. Code Quality (Priority Focus)

#### Ruff Linter Results: 56 violations

| Rule  | Count | Description                      | Severity |
|-------|-------|----------------------------------|----------|
| E501  | 38    | Line too long (>88 chars)        | Style    |
| C901  | 4     | Cyclomatic complexity >10        | Quality  |
| F401  | 5     | Unused imports                   | Quality  |
| F841  | 2     | Unused variable assignments      | Quality  |
| F541  | 3     | f-string without placeholders    | Quality  |
| I001  | 1     | Import block unsorted            | Style    |

**Complexity Violations (C901):**
- `build_graph` in `graph.py` — complexity 12 (max 10)
- `discover_node` in `stages/discover.py` — complexity 12
- `execute_node` in `stages/execute.py` — complexity 11
- `research_node` in `stages/research.py` — complexity 11

These are the main orchestration functions. They could benefit from extracting helper functions for prompt building, file cleanup, and artifact parsing.

**Unused Imports (F401):**
- `pathlib.Path` in `stages/execute.py` (imported but all Path usage is via strings)
- `load_artifact` in `stages/grill.py` (imported but not used)
- `os` and `Path` in `tests/test_config.py` (test file cleanup)
- `STAGE_ORDER` in `tests/test_ui.py` (imported but not asserted)

**Unused Variables (F841):**
- `codebase_profile` and `feature_spec` in `execute_node` — assigned from state but never read. These may be intended for future use in prompt building.
- `result` in `plan_node` — agent result captured but never used.

**Dead Code:**
- 10 lines of dead imports/assignments across 5 files
- No dead functions or classes detected

**Duplication Patterns:**
- The artifact read → parse JSON → save → clean up pattern repeats across all 6 stage files (`discover.py`, `research.py`, `grill.py`, `plan.py`, `verify.py`). This is ~15 lines duplicated 5 times. Could be extracted into a shared helper.
- The `_cwd` resolution logic (check scope_path, resolve directory) repeats in `discover.py` and `research.py`.
- The prompt building pattern (list of parts, conditional appends, join) is repeated in every stage.

#### Late Imports
- `grill.py` imports `pathlib.Path` inside function bodies (lines 130 and 192) instead of at module top level. This is a code smell — the import should be at the module level.

### 2. Type Safety (Priority Focus)

#### Mypy Results: 26 errors in 2 files

**`src/graft/agent.py` — 25 errors (all on line 61):**
All errors stem from a single pattern: building `ClaudeAgentOptions` via `**dict` unpacking:
```python
opts = {
    "system_prompt": system_prompt,
    "cwd": cwd,
    ...
}
options = ClaudeAgentOptions(**opts)  # 25 type errors here
```
The dict is typed as `dict[str, object]` which is incompatible with the specific parameter types expected by `ClaudeAgentOptions`. **Fix:** Construct `ClaudeAgentOptions` directly with keyword arguments instead of dict unpacking, or use a `TypedDict` for the options.

**`src/graft/graph.py` — 1 error (line 31):**
```
Returning Any from function declared to return "dict[Any, Any]"
```
The `_wrap` function's inner `wrapper` returns `Any` from the awaited node function. This needs a proper return type annotation.

#### Type Safety Configuration Concerns

- `ignore_missing_imports = true` in `[tool.mypy]` — This suppresses errors for all third-party libraries without type stubs. While practical for rapid development, it means type errors in interactions with `langgraph`, `claude_agent_sdk`, `typer`, and `rich` are invisible.
- `warn_return_any = true` is correctly enabled, which caught the `graph.py` issue.
- No `strict` mode — missing `disallow_untyped_defs`, `disallow_any_generics`, etc.
- Heavy use of bare `dict` and `list` types in `FeatureState` — e.g., `codebase_profile: dict`, `build_plan: list[dict]`. These would benefit from proper TypedDict or dataclass definitions.

#### Untyped Patterns

- `state.py`: All state fields use bare `dict` and `list` — no generic parameters. The `_replace_list` and `_replace_dict` reducers use unparameterized `list` and `dict`.
- `agent.py`: `_process_message` takes untyped `message` parameter. `AgentResult.raw_messages` is `list` (unparameterized).
- All stage functions return `dict` instead of a typed return structure.

### 3. Security

| Check                    | Result                                   |
|--------------------------|------------------------------------------|
| Hardcoded secrets        | None found                               |
| .env in git              | No (.env is in .gitignore)               |
| Dependency vulnerabilities | Unable to check (pip-audit not installed) |
| subprocess usage         | `execute.py` and `verify.py` use `subprocess.run` with list args (safe from injection) |
| Shell injection risk     | `VERIFY_SCRIPT` in execute.py runs via `bash -c` with a static string — safe |
| Permission mode          | `bypassPermissions` is hardcoded in agent.py — by design for autonomous execution |

**Notable:** The `execute.py` stage uses `git add -A` which could stage sensitive files. However, this operates on a feature branch within the user's repo, which is the intended behavior.

### 4. Performance

- **No bundle size concerns** — CLI tool, not a web app.
- **Subprocess calls** in `execute.py` have appropriate timeouts (60s for git, 300s for tests, 60s for lint).
- **Async design** is correct — uses `asyncio.run` at the CLI entry point and `async/await` throughout the pipeline stages.
- **No N+1 patterns** — file I/O in `list_projects` iterates directories sequentially, which is appropriate for the expected volume.
- The `_order_by_dependencies` topological sort has a safety bound (`max_passes`) to prevent infinite loops on circular deps.

### 5. Test Health

#### Test Results: 39 passed, 0 failed, 0 skipped

All tests pass. Test suite runs in ~0.19s.

#### Coverage: 38% overall

| Module                    | Stmts | Miss | Cover | Assessment        |
|---------------------------|-------|------|-------|-------------------|
| `__init__.py`             | 0     | 0    | 100%  | Trivial           |
| `stages/__init__.py`      | 0     | 0    | 100%  | Trivial           |
| `artifacts.py`            | 51    | 0    | 100%  | Excellent         |
| `state.py`                | 41    | 0    | 100%  | Excellent         |
| `graph.py`                | 45    | 1    | 98%   | Excellent         |
| `config.py`               | 27    | 1    | 96%   | Excellent         |
| `agent.py`                | 56    | 38   | 32%   | **Insufficient**  |
| `ui.py`                   | 107   | 74   | 31%   | **Insufficient**  |
| `stages/plan.py`          | 78    | 55   | 29%   | **Insufficient**  |
| `stages/execute.py`       | 122   | 88   | 28%   | **Insufficient**  |
| `stages/verify.py`        | 54    | 43   | 20%   | **Critical**      |
| `stages/discover.py`      | 50    | 41   | 18%   | **Critical**      |
| `stages/research.py`      | 50    | 41   | 18%   | **Critical**      |
| `stages/grill.py`         | 71    | 58   | 18%   | **Critical**      |
| `cli.py`                  | 77    | 77   | **0%**| **No coverage**   |

#### Test Quality Assessment

**What's tested well:**
- Pure data functions: `artifacts.py` (100%), `state.py` (100%), `config.py` (96%)
- Graph construction: `graph.py` (98%)
- Pure logic functions: `_order_by_dependencies`, `estimate_cost`, routers

**What's not tested:**
- **No async tests at all.** All 6 stage node functions (`discover_node`, `research_node`, `grill_node`, `plan_node`, `execute_node`, `verify_node`) are async and untested. These are the core business logic.
- **CLI commands** (`build`, `resume`, `list`) have 0% coverage.
- **UI methods** — Only attribute checks tested. No tests for `banner()`, `stage_start()`, `show_artifact()`, `grill_question()`, `prompt_plan_review()`, etc.
- **Agent wrapper** — `run_agent()` and `_process_message()` are untested.
- **Subprocess helpers** — `_git()`, `_run_tests()`, `_run_lint()`, `_open_pr()` are untested.

**Git history note:** The commit history shows multiple "test: Write tests" commits followed by immediate reverts, suggesting previous test generation attempts failed (likely due to difficulty mocking the Claude Agent SDK).

### 6. CI/CD Health

**Critical gap: No CI configuration exists.**

| Expected CI Capability        | Status      |
|-------------------------------|-------------|
| CI configuration file         | Missing     |
| Run tests on PRs              | Missing     |
| Run linter (ruff) on PRs      | Missing     |
| Run type checker (mypy) on PRs | Missing     |
| Coverage reporting             | Missing     |
| Dependency vulnerability scan  | Missing     |

### 7. Architecture

#### Strengths
- **Clean separation of concerns:** Config, state, artifacts, agent, UI, graph, and stages are well-separated modules.
- **Immutable settings:** `Settings` is a frozen dataclass.
- **Safety loops:** Execute stage has keep/revert logic with test verification.
- **Resumability:** The `resume` command can restart from any stage with artifact persistence.
- **Graceful error handling:** UI methods catch I/O errors for non-interactive environments.

#### Issues

**Circular dependency risk:**
- `grill.py` has a late import of `pathlib.Path` inside function bodies, suggesting the module may have been refactored to break a circular import — but `Path` is standard library and doesn't need late importing.

**Tight coupling to file system:**
- Every stage writes files to `repo_path`, reads them back, then deletes them. This is the only interface between the agent (which writes files via tools) and the pipeline (which reads results). No abstraction layer exists — if the agent writes to a different location, the stage silently falls back to `result.text`.

**`plan_review_node` always approves:**
- When the user provides feedback (doesn't approve), the function logs "Re-planning is not yet implemented" and approves anyway (line 205). This makes the feedback loop non-functional.

**Hardcoded model name:**
- `claude-opus-4-20250514` is hardcoded in `config.py` as the default. This will need updating as new models are released.

**No error recovery in `resume`:**
- The `resume` command assumes all prior artifacts are valid JSON. If a previous stage wrote malformed JSON, the resume will either crash or silently use empty dicts.

---

## Summary of Priority Issues

### Critical (Must Fix)
1. **No CI/CD** — Tests, linting, and type checking are not automated.
2. **26 mypy errors** — Type safety violations in the agent wrapper and graph module.
3. **0% test coverage on CLI** — The primary user interface is completely untested.
4. **38% overall test coverage** — 7 out of 15 source modules have <30% coverage.

### High (Should Fix)
5. **56 ruff violations** — 38 line-length issues, 4 complexity violations, 7 code quality issues.
6. **Unused code** — Dead imports and unused variables indicate incomplete refactoring.
7. **No async test infrastructure** — Core business logic (all stage nodes) cannot be tested without async test support.
8. **Bare `dict`/`list` types in state** — Lose type safety for the most important data structures.

### Medium (Improvement)
9. **Duplicated artifact parsing** pattern across all stages.
10. **Plan review feedback loop** is non-functional (always approves).
11. **No formatter configured** — Only ruff lint rules, no ruff format or black.
12. **`ignore_missing_imports = true`** hides type errors in third-party interactions.

### Low (Nice to Have)
13. **pip-audit** not available for dependency vulnerability scanning.
14. **Late imports** in grill.py (cosmetic).
15. **Hardcoded model name** in config defaults.
