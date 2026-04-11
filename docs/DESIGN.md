# Feature Factory — Design Document

## Overview

An AI-powered pipeline for building significant new features into existing
applications. Sits between the greenfield Software Factory (zero-to-one app
creation) and the Optimization Factory (improving existing code quality).

The Feature Factory takes a short human prompt — "I need a transfers system
where managers can propose trades" — and turns it into a shipped, tested,
integrated PR through a structured multi-stage pipeline. The human's job is
to describe what they want and answer questions. The machine does the
research, architecture, implementation, and verification.

Core philosophy: **Understand → Ask → Plan → Build → Verify → Ship.**
The key insight is that the machine does the hard work of understanding the
codebase and asking the right questions, so the human never has to write a
PRD or spec from scratch.

**The output is always a PR.** Nothing touches main until a human reviews
and merges. This is the same fundamental safety guarantee as the
Optimization Factory.

---

## Why a Separate Factory

The three factories solve fundamentally different problems:

| Factory        | Starting point  | Challenge                                  |
|----------------|-----------------|-------------------------------------------|
| Software       | Nothing         | Build something that works                 |
| Feature        | Working app     | Add something new that fits what exists    |
| Optimization   | Working app     | Make what exists measurably better          |

The Feature Factory is the hardest. Zero-to-one is paradoxically easier
because there are no constraints — blank canvas. Optimization is mechanical —
measurable score, go make it better. Features in an existing app require
deep codebase understanding AND creative design AND seamless integration.

A feature is too big for a single Jira ticket but too constrained for
greenfield. It needs to:
- Understand existing architecture, patterns, and conventions
- Design something new that follows those patterns
- Touch multiple files, services, or layers
- Integrate with existing auth, state management, routing, etc.
- Not break any existing functionality
- Include tests that fit the existing test infrastructure

Shared infrastructure (reusable from Optimization Factory):
- LangGraph state machine pattern
- Claude Agent SDK wrapper (agent.py)
- Artifact persistence layer
- Rich terminal UI
- Git branch + PR output model
- Keep/revert safety loop (for Execute stage)

---

## Pipeline Stages

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  DISCOVER ──► RESEARCH ──► GRILL ──► PLAN ──► EXECUTE ──► VERIFY    │
│                                │       │                     │       │
│                                └───────┘                 [open PR]   │
│                             [loop-back if                            │
│                              gaps found]                             │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

Discover:  deep audit of the existing codebase architecture
Research:  figure out what the feature needs given what exists
Grill:     interrogate the human for intent, preferences, edge cases
Plan:      produce an ordered execution plan with atomic build units
Execute:   implement changes one at a time, test after each
Verify:    full regression check + feature validation, open PR
```

Six stages. Six agents. Linear with one conditional loop-back.

---

### Stage 1: DISCOVER

**Agent persona:** Principal Codebase Archaeologist
**Purpose:** Build a complete mental model of the existing application —
architecture, patterns, conventions, data model, integration points.

This is NOT the same as the Optimization Factory's audit. That audit
measures quality. This discovery maps architecture. The goal is to
understand the codebase well enough to design a feature that fits
seamlessly into it.

**What it does:**

1. **Project Structure**
   - Map the full directory tree and file organization
   - Identify framework(s), language(s), package manager(s)
   - Detect monorepo structure (services, packages, shared code)
   - Map entry points, routing, middleware chains

2. **Data Model**
   - Catalog database tables/collections (schema, relationships, indexes)
   - Map ORM models, types, interfaces
   - Identify data access patterns (repositories, services, direct queries)
   - Document existing migrations infrastructure

3. **Architecture Patterns**
   - Component patterns (naming, file structure, prop patterns)
   - State management approach (Redux, Zustand, context, signals)
   - API patterns (REST, GraphQL, tRPC, server actions)
   - Auth/authorization model (how permissions work, middleware)
   - Error handling patterns
   - Logging and observability patterns

4. **Integration Surface**
   - External services and APIs already integrated
   - Environment variables and configuration approach
   - Feature flags system (if any)
   - Real-time capabilities (WebSockets, SSE, polling)

5. **Test Infrastructure**
   - Test framework, runner, assertion style
   - Test file organization and naming conventions
   - Fixture and mock patterns
   - Coverage tooling and thresholds
   - E2E test setup (if any)

6. **Conventions**
   - Git workflow (branch naming, commit messages, PR templates)
   - Code style (formatter, linter, documented conventions)
   - Documentation patterns
   - Any CLAUDE.md, .cursorrules, or agent instructions present

Output: `codebase_profile.json` + `discovery_report.md`

The codebase_profile.json is a structured reference that every subsequent
stage can query. It answers: "How does this app do X?" for any X.

```json
{
  "timestamp": "2026-04-10T23:00:00Z",
  "project": {
    "name": "touchline-nextjs",
    "languages": ["typescript"],
    "frameworks": ["nextjs", "react-native", "expo"],
    "package_manager": "npm",
    "monorepo": true
  },
  "services": [...],
  "data_model": {
    "orm": "supabase",
    "tables": ["leagues", "teams", "players", "draft_picks", ...],
    "key_relationships": [...]
  },
  "patterns": {
    "state_management": "react-query + zustand",
    "api": "supabase client + server actions",
    "auth": "supabase auth with RLS",
    "components": "feature-based folders, shared UI components",
    "routing": "next.js app router"
  },
  "test_infrastructure": {
    "framework": "vitest",
    "runner": "vitest run",
    "coverage": "v8",
    "e2e": null,
    "conventions": "co-located __tests__ folders"
  },
  "conventions": {
    "git_workflow": "feature branches, squash merge",
    "commit_style": "conventional commits",
    "code_style": "prettier + eslint"
  }
}
```

**Human involvement:** None. Pure discovery and comprehension.

---

### Stage 2: RESEARCH

**Agent persona:** Staff Software Architect (Feature Specialist)
**Purpose:** Given the feature prompt and codebase understanding, figure
out what this feature actually needs — technically and architecturally.

This stage bridges "what the human wants" and "what the codebase needs."
It answers: what can we reuse? What do we need to create? What are the
edge cases? What similar patterns already exist that we should follow?

**What it does:**

1. **Reuse Analysis**
   - What existing tables, models, or types can the feature leverage?
   - What existing components or UI patterns can be reused?
   - What existing API endpoints or services can be extended?
   - What existing utility functions or helpers apply?

2. **Gap Analysis**
   - What new database tables, columns, or migrations are needed?
   - What new API endpoints or server actions are needed?
   - What new components need to be built?
   - What new types or interfaces need to be defined?

3. **Pattern Matching**
   - Identify the closest existing feature in the codebase
   - Document which patterns from that feature should be followed
   - Flag any cases where existing patterns won't work and alternatives
     are needed

4. **Edge Case Identification**
   - Concurrency issues (what if two users do this simultaneously?)
   - Permission boundaries (who can do what?)
   - Data validation requirements
   - Error states and recovery
   - Performance implications (will this feature add N+1 queries?)

5. **Integration Points**
   - Which existing features does this interact with?
   - Does it need real-time updates?
   - Does it need notifications?
   - Does it affect existing navigation, routing, or layout?

Output: `research_report.md` + `technical_assessment.json`

The research report is a narrative document. The technical assessment is
structured data that feeds into the Plan stage.

```json
{
  "feature_prompt": "transfers system where managers can propose trades...",
  "reusable_components": [
    {"path": "src/components/PlayerCard.tsx", "reason": "player display in trade UI"},
    {"path": "src/hooks/useLeague.ts", "reason": "league context already available"}
  ],
  "new_artifacts_needed": [
    {"type": "table", "name": "trades", "description": "..."},
    {"type": "component", "name": "TradeProposal", "description": "..."},
    {"type": "api", "name": "POST /api/trades", "description": "..."}
  ],
  "pattern_to_follow": "src/features/draft/ (closest existing feature)",
  "edge_cases": [...],
  "integration_points": [...],
  "open_questions": [
    "Should trades have an expiration timer?",
    "Can a manager propose multiple trades simultaneously?",
    "What happens to pending trades when the trade deadline passes?"
  ]
}
```

The `open_questions` array is critical — these are questions the agent
couldn't answer by reading the codebase. They require human intent.
These feed directly into the Grill stage.

**Human involvement:** None. But the open_questions output is the
bridge to the Grill stage.

---

### Stage 3: GRILL

**Agent persona:** Principal Product Interrogator
**Purpose:** Extract every decision and preference from the human that
the codebase can't answer. Walk down each branch of the design tree,
resolving dependencies between decisions one by one.

This stage is inspired by [grill-me](https://github.com/mfpaddock/skills):
relentless, structured interrogation with recommended answers for each
question. The key difference is that Discovery and Research have already
run — so the agent is deeply informed about the codebase. It only asks
questions that require human INTENT.

**Rules:**

1. **Ask one question at a time.** Not a list. One question, wait for
   the answer, then the next.

2. **Provide a recommended answer for each question.** Based on codebase
   patterns, industry norms, or technical reasoning. The human can say
   "yes" or correct it. This makes the interrogation fast — most answers
   are "yeah, that."

3. **If a question can be answered by exploring the codebase, explore
   the codebase instead.** Don't ask the human things the code already
   answers. This is the cardinal rule.

4. **Start with open_questions from Research.** These are the known
   unknowns. Then branch into derived questions as answers reveal new
   decision points.

5. **Walk the decision tree depth-first.** When an answer opens up
   sub-decisions, resolve them before moving on. Don't leave loose
   threads.

6. **Categorize questions:**
   - **Intent:** What does the user actually want? (behavior, UX, scope)
   - **Edge cases:** What happens when X? (error states, limits, timing)
   - **Preferences:** How should this look/feel/work? (UI, naming, flow)
   - **Prioritization:** What's MVP vs. follow-up? (scope control)

**Output:** `feature_spec.json` + `grill_transcript.md`

The feature_spec.json is the "PRD that wrote itself" — a complete,
unambiguous specification derived from the interrogation. Every decision
is documented with the question that prompted it and the answer given.

```json
{
  "feature_name": "Trade System",
  "decisions": [
    {
      "question": "Should trades have an expiration timer?",
      "recommended": "Yes, 48 hours — matches common fantasy platform conventions",
      "answer": "Yes, but 24 hours. Our league weeks are short.",
      "category": "intent",
      "implications": ["needs timer UI component", "needs cron/scheduled check"]
    },
    ...
  ],
  "scope": {
    "mvp": ["propose trade", "accept/reject", "roster validation", "deadline enforcement"],
    "follow_up": ["trade history page", "trade notifications", "counter-offers"]
  },
  "constraints": [
    "Must follow existing Supabase RLS patterns for authorization",
    "Must work on both web and mobile (React Native)"
  ]
}
```

**Conditional loop-back to Research:** If the Grill phase reveals that a
technical assumption from Research was wrong (e.g., "actually we don't
want to use Supabase for this, we want a separate service"), the pipeline
can loop back to Research for a targeted re-assessment. This is the
safety valve — but it should be rare if Research was thorough.

**Human involvement:** HIGH. This is the primary human touchpoint. But
it's efficient — the human is answering focused questions with recommended
answers, not writing a document from scratch.

---

### Stage 4: PLAN

**Agent persona:** Staff Software Architect (Implementation Planner)
**Purpose:** Turn the feature spec + codebase understanding into an
ordered list of atomic build units.

This stage has full context: the codebase profile (from Discover), the
technical assessment (from Research), and the feature spec (from Grill).
It knows what exists, what's needed, and exactly what the human wants.

**What it does:**

1. **Generate build units** — atomic, implementable chunks of work:
   - Database migrations
   - Type/interface definitions
   - API endpoints or server actions
   - Components (starting with lowest-level, shared components first)
   - Integration wiring (routing, navigation, state)
   - Tests (co-located with the units they test)

2. **Order by dependency** — migrations before API, API before components,
   components before pages, shared before specific.

3. **Tag each unit** with:
   - `service` (which part of the monorepo)
   - `risk` (low/medium/high)
   - `blast_radius` (files created or modified)
   - `depends_on` (which units must land first)
   - `acceptance_criteria` (what "done" looks like)
   - `pattern_reference` (existing file to follow as a model)
   - `tests_included` (boolean — does this unit include its own tests?)

4. **Estimate cost** — per-unit token cost estimates, total pipeline cost

5. **Tests-with-feature strategy** — Unlike the Optimization Factory which
   writes tests first as a safety net, the Feature Factory co-locates
   tests with the code they test. Each build unit that creates code also
   creates its tests. This is natural for feature development — you're
   building something new, so the tests are part of the build, not a
   prerequisite.

Output: `build_plan.json`

```json
{
  "plan_id": "feat_2026-04-10",
  "feature_name": "Trade System",
  "total_units": 12,
  "estimated_cost": "$8-15",
  "units": [
    {
      "unit_id": "feat_01",
      "title": "Create trades table migration",
      "category": "database",
      "service": "packages/db",
      "risk": "low",
      "blast_radius": "1 file (new migration)",
      "depends_on": [],
      "acceptance_criteria": ["migration runs cleanly", "table created with correct schema"],
      "pattern_reference": "supabase/migrations/20260301_create_draft_picks.sql",
      "tests_included": false
    },
    {
      "unit_id": "feat_02",
      "title": "Define Trade types and interfaces",
      "category": "types",
      "service": "packages/shared",
      "risk": "low",
      "blast_radius": "1-2 files",
      "depends_on": ["feat_01"],
      "acceptance_criteria": ["types compile", "consistent with existing type patterns"],
      "pattern_reference": "packages/shared/types/draft.ts",
      "tests_included": false
    },
    {
      "unit_id": "feat_03",
      "title": "Build trade proposal API + validation logic",
      "category": "api",
      "service": "apps/web",
      "risk": "medium",
      "blast_radius": "3-4 files",
      "depends_on": ["feat_01", "feat_02"],
      "acceptance_criteria": ["endpoint works", "validates roster limits", "respects deadline"],
      "pattern_reference": "apps/web/app/api/draft/route.ts",
      "tests_included": true
    }
  ]
}
```

**Human involvement:** Optional gate (same as Optimization Factory).

The plan is presented for review. The human can:
- Approve the full plan
- Remove or reorder units
- Adjust scope ("skip the notification unit, that's follow-up")
- Add constraints
- Set a unit limit

For `--auto-approve`, the gate is skipped. The PR is still the final gate.

---

### Stage 5: EXECUTE

**Agent persona:** Principal Software Engineer
**Purpose:** Implement build units one at a time. Each unit is a focused
implementation task with clear acceptance criteria and a pattern to follow.

This stage borrows the Optimization Factory's keep/revert safety loop but
adapts it for feature building:

```
FOR EACH build_unit IN approved_plan:

  1. Read the pattern_reference file(s) — understand the convention
  2. Read all files in the blast radius
  3. Implement the change (following the pattern)
  4. If tests_included: write co-located tests
  5. Commit: git commit -m "feat: <unit title>"
  6. Run the test suite (existing + new tests)
  7. Run linter/formatter (match project conventions)
  8. Evaluate:
     - All existing tests pass → continue
     - New tests pass → continue
     - Any existing test FAILS → REVERT (git revert HEAD)
     - Lint/format fails → auto-fix and amend commit
  9. Log result to build_log.tsv
```

**Key differences from Optimization Factory Execute:**

- **Pattern reference is critical.** Every unit has a `pattern_reference`
  pointing to an existing file that demonstrates the convention. The agent
  reads this first and follows it. This is how features "fit" into an
  existing codebase — they look like they were written by the same team.

- **Tests are co-located, not pre-built.** Tests ship with the code they
  cover. The safety net for existing functionality is the existing test
  suite — if any existing test breaks, the change is reverted.

- **No metric scoring.** Unlike optimization (where "better" is
  measurable), feature building doesn't have a scalar score. Success is:
  does it work + does it not break anything + does it follow conventions.

- **Cumulative integration.** Each unit builds on previous units. Unit 3
  (API) depends on units 1-2 (migration + types) already being committed.
  The topological sort ensures dependencies are satisfied.

All work happens on a single feature branch: `feature/<plan_id>`.

**Human involvement:** None during execution. Fully autonomous.

---

### Stage 6: VERIFY

**Agent persona:** Principal Quality Engineer + Technical Program Manager
**Purpose:** Validate the complete feature, run full regression, and
open the PR.

This stage combines the Optimization Factory's Verify and Report stages —
for features, they're naturally one step.

**What it does:**

1. **Regression Check**
   - Run the full test suite (all services)
   - Run linters and type checkers
   - Confirm zero existing test failures

2. **Feature Validation**
   - Walk through the feature spec decisions
   - Verify each MVP scope item is implemented
   - Check acceptance criteria for every build unit
   - Verify edge cases identified in the Grill phase are handled

3. **Integration Check**
   - Verify routing/navigation works
   - Verify auth/permissions are correctly applied
   - Verify the feature follows the patterns documented in Discovery

4. **Generate Report**
   - Feature summary (what was built)
   - Build units completed vs. reverted
   - Test coverage for new code
   - Files created and modified
   - Remaining follow-up items (from the feature spec scope)

5. **Open PR**
   - Branch: `feature/<plan_id>`
   - Title: "Feature: <feature_name>"
   - Body: the full feature report
   - Each build unit is a separate atomic commit
   - Follow-up items listed as checkboxes for future work

Output: `feature_report.md` + PR opened

**Human involvement:** The PR. The human reviews the complete feature
diff, checks the report, and merges if satisfied.

---

## Human-in-the-Loop Summary

### Design Principle: Ask, Don't Assume

The Feature Factory's primary human interaction is the Grill stage — a
structured conversation that extracts decisions efficiently. The human
never writes a document. They answer questions. The machine compiles
the answers into a complete spec.

| Stage    | Human Role                    | Gate Type        | Can Skip?              |
|----------|-------------------------------|------------------|------------------------|
| Discover | None                          | —                | —                      |
| Research | None                          | —                | —                      |
| Grill    | Answer questions              | Always (primary) | No — this IS the input |
| Plan     | Approve/scope build plan      | Optional         | Yes (--auto-approve)   |
| Execute  | None (autonomous)             | —                | —                      |
| Verify   | Review PR, merge/reject       | Always           | PR never auto-merges   |

Two human touchpoints:
1. **Grill** — the creative/intent input (required)
2. **PR review** — the safety gate (required)

Everything else is autonomous.

---

## State Shape

```python
class FeatureState(TypedDict, total=False):
    # Inputs
    repo_path: str                    # Path to the repo
    project_id: str                   # Unique session ID
    project_dir: str                  # Working directory for artifacts
    feature_prompt: str               # The initial human description

    # User inputs (from CLI flags)
    scope_path: str                   # Monorepo scoping (--path)
    constraints: list[str]            # e.g. ["must work on mobile", "no new deps"]
    max_units: int                    # Max build units to execute
    auto_approve: bool                # Skip the Plan gate

    # Stage artifacts
    codebase_profile: dict            # From Discover
    discovery_report: str             # From Discover
    technical_assessment: dict        # From Research
    research_report: str              # From Research
    feature_spec: dict                # From Grill
    grill_transcript: str             # From Grill
    build_plan: list[dict]            # From Plan
    feature_report: str               # From Verify

    # Execution tracking
    current_unit_index: int
    units_completed: list[dict]       # {unit_id, status, files_created, files_modified}
    units_reverted: list[dict]        # {unit_id, reason}
    units_skipped: list[dict]         # {unit_id, reason}

    # Gates
    plan_approved: bool
    grill_complete: bool
    research_redo_needed: bool        # Loop-back flag

    # Git
    feature_branch: str               # e.g. "feature/feat_2026-04-10"
    pr_url: str                       # Final PR URL

    # Settings
    model: str
    max_agent_turns: int

    # Pipeline state
    current_stage: str
```

---

## CLI Interface

```bash
# Basic usage — build a feature into a local repo
graft build ~/projects/my-app "Add a transfers system where managers can
propose trades between teams, both sides accept/reject, respects roster
limits and trade deadlines"

# With monorepo scoping
graft build ~/projects/my-company --path apps/web "Add dark mode toggle
to the settings page"

# With constraints
graft build ~/projects/my-app "Add Stripe billing" --constraint "no new
dependencies" --constraint "must use existing auth middleware"

# Limit scope
graft build ~/projects/my-app "Add notifications" --max-units 8

# Auto-approve plan (still requires Grill + PR review)
graft build ~/projects/my-app "Add search" --auto-approve

# Resume a previous session
graft resume ~/.graft/projects/feat_XXXXX --from execute

# List past sessions
graft list
```

---

## Key Differences: Feature Factory vs. Others

| Aspect              | Software Factory     | Feature Factory          | Optimization Factory    |
|---------------------|----------------------|--------------------------|------------------------|
| Starting point      | Nothing              | Working app + prompt     | Working app            |
| Primary risk        | Doesn't work         | Breaks what exists       | Breaks what exists     |
| Human input         | Spec/PRD (upfront)   | Answers to questions     | None (or CLI flags)    |
| Codebase knowledge  | N/A                  | Deep (Discover stage)    | Quality-focused        |
| New code            | All of it            | Feature-scoped addition  | Minimal (refactoring)  |
| Test strategy       | Write after build    | Co-located with feature  | Write first (safety)   |
| Success metric      | "It works"           | "It fits + doesn't break"| "Measurably better"    |
| Pattern adherence   | Creates patterns     | Follows existing ones    | Improves patterns      |
| Human gates         | Spec (blocking)      | Grill (required) + PR   | Plan (optional) + PR   |
| Output              | Deployed app         | PR with atomic commits   | PR with atomic commits |

---

## The PRD Problem (and How Grill Solves It)

Traditional feature development requires a PRD before any work starts.
This creates a bottleneck: someone (usually a PM or tech lead) has to
write a detailed spec. For a solo builder or small team, this is often
the highest-friction step — not because writing is hard, but because
writing a GOOD spec requires understanding the codebase deeply enough
to know what's feasible.

The Feature Factory inverts this. Instead of:

```
Human writes PRD → Machine builds
```

It's:

```
Human gives prompt → Machine understands codebase → Machine asks questions
→ Human answers → Machine compiles spec → Machine builds
```

The machine does the hard work of:
1. Understanding the codebase (Discover)
2. Figuring out what's technically needed (Research)
3. Asking the RIGHT questions — informed by the codebase (Grill)
4. Compiling answers into a complete spec (feature_spec.json)

The human's job is reduced to the highest-leverage activity: making
decisions about what they want. Not documenting architecture. Not
mapping file paths. Not writing acceptance criteria. Just: "yes,"
"no," or "actually, I want it like this."

---

## Open Questions

1. ~~**CLI name.**~~ **RESOLVED.** `graft` — the horticultural term for
   attaching a new branch onto an existing tree. Perfect metaphor for
   building new features into existing codebases. Short, memorable,
   not taken by any major CLI tool.

2. ~~**Grill depth control.**~~ **RESOLVED.** No `--quick` flag. The Grill
   agent self-regulates depth — it evaluates completeness after each
   answer and wraps up when all decision branches are resolved. Simple
   features naturally get fewer questions. Complex ones get more. When
   in doubt, the agent goes deeper. You can always skip a question but
   you can't recover from a missing one mid-build.

3. ~~**Multi-feature sessions.**~~ **RESOLVED.** One feature per run.
   Mixing features tangles the Grill phase, creates dependency chaos in
   Execute, and produces unreviable PRs. One feature = one pipeline =
   one PR = one clean review. Run `graft build` twice for two features.

4. ~~**Optimization Factory integration.**~~ **RESOLVED.** Two-tier
   approach. V1: Verify stage suggests the exact `hone optimize` command
   scoped to the new feature's files at the end of the report. Explicit,
   no friction, separate PR. V2: `--optimize` flag chains Graft into
   Hone automatically — build the feature, merge the PR, then run an
   optimization pass on the new code. Graft builds it right, Hone makes
   it better. Complementary tools, clean separation.

5. ~~**Jira/project management output.**~~ **RESOLVED.** Yes, optional.
   `--issues` generates GitHub issues, `--jira` creates Jira tickets.
   Each build unit becomes a ticket with acceptance criteria, pattern
   reference, and dependency info pre-filled. The PR links back to them.
   Tickets come OUT of the process as a byproduct of planning, not into
   it as a prerequisite. Solo builders skip it. Teams get full
   traceability for free.

6. ~~**Existing test suite quality.**~~ **RESOLVED.** Discover measures
   test coverage at the MODULE level, not project level — specifically
   targeting the modules the feature will integrate with (identified via
   the integration surface mapping). A blanket coverage percentage is
   misleading; what matters is whether the specific code paths this
   feature touches have meaningful tests.

   If coverage on integration-critical modules is thin, Graft warns with
   targeted context: "The roster validation module (src/services/roster.ts)
   has 12% test coverage and your feature depends heavily on it. Changes
   here have a high risk of undetected regressions." The suggestion is
   equally scoped: `hone optimize --focus tests --path src/services/roster.ts`.

   This is a warning, not a hard block — the human decides whether to
   proceed with a thin safety net or shore it up first. The tools stay
   composable: Graft builds features, Hone builds test coverage. Run
   them separately, in whatever order makes sense.

   The Verify stage also reports coverage delta on new feature code so
   you can see whether the new code is well-tested even if legacy code
   isn't. V2 consideration: mutation testing on new code during Verify
   to check test quality, not just coverage — catches the "agent writes
   tests for its own code and only tests the happy path" blind spot.
