# Graft

AI-powered feature building for existing codebases. Describe the feature you
want, answer a few focused questions, and get a tested, integrated PR — through
structured discovery, interrogation, planning, and execution.

## How It Works

```
Discover → Research → Grill → Plan → Execute → Verify → PR
```

**6 stages, 6 AI agents, 1 PR.**

1. **Discover** — Maps your codebase: architecture, patterns, data model,
   conventions, test infrastructure. Understands what exists before touching
   anything.

2. **Research** — Investigates what the feature needs: reuse opportunities,
   integration points, edge cases, relevant libraries and patterns.

3. **Grill** — Asks you focused questions one at a time, each with a recommended
   answer. Your spec emerges from the conversation, not from a blank page.
   Type "done" at any point to proceed with defaults.

4. **Plan** — Produces an ordered list of atomic build units with dependencies,
   pattern references, and acceptance criteria.

5. **Execute** — Implements each unit one at a time. Tests after every change,
   auto-reverts on regression against existing tests. Each kept change is an
   atomic git commit.

6. **Verify** — Full regression suite + feature validation. Opens a PR with
   atomic commits and a detailed report.

**You never write a PRD.** You answer questions. The machine does the rest.

## Installation

```bash
# Clone and install
git clone https://github.com/Kdurkee45/graft.git
cd graft
uv sync

# Set up API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

Requires:
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [gh CLI](https://cli.github.com/) (for automatic PR creation)
- Git

## Usage

```bash
# Basic — describe the feature you want
graft build ~/projects/my-app "add a transfers system where managers can propose trades"

# With constraints
graft build ~/projects/my-app "user notifications" --constraint "don't touch the auth module"

# Scope to a subdirectory
graft build ~/projects/my-app "API rate limiting" --path src/api

# Fully autonomous (auto-answer Grill questions with defaults)
graft build ~/projects/my-app "dark mode support" --auto-approve

# Limit build scope
graft build ~/projects/my-app "search feature" --max-units 10

# Resume a previous session
graft resume ~/.graft/projects/feat_XXXXX --from execute

# List past sessions
graft list
```

## Safety Model

1. **Existing tests are the safety net** — every change runs the full existing
   test suite. If any existing test breaks, the change is reverted.
2. **Atomic commits** — each build unit is a separate commit, easy to review
   and revert individually.
3. **All work on a branch** — main is never touched.
4. **PR never auto-merges** — human reviews the complete feature PR.
5. **The Grill is your control** — you decide scope, priorities, and constraints
   through focused Q&A.

## Configuration

Settings via environment variables or `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Required. Claude API key. | — |
| `GITHUB_TOKEN` | For PR creation via gh CLI. | — |
| `GRAFT_MODEL` | Claude model to use. | `claude-opus-4-20250514` |
| `GRAFT_MAX_TURNS` | Max agent turns per stage. | `50` |

## Project Structure

```
src/graft/
├── cli.py              # Typer CLI (build, resume, list)
├── config.py           # Settings from env/dotenv
├── state.py            # LangGraph typed state
├── graph.py            # Pipeline state machine
├── agent.py            # Claude Agent SDK wrapper
├── artifacts.py        # Persistent artifact storage
├── ui.py               # Rich terminal UI
└── stages/
    ├── discover.py     # Stage 1: codebase discovery
    ├── research.py     # Stage 2: feature research
    ├── grill.py        # Stage 3: focused Q&A
    ├── plan.py         # Stage 4: build planning
    ├── execute.py      # Stage 5: implementation
    └── verify.py       # Stage 6: validation + PR
```

## Related Projects

The trilogy: **Kindle** (ignite) → **Graft** (grow) → **Hone** (sharpen)

- **[Kindle](https://github.com/Kdurkee45/kindle)** — Build complete applications from a prompt
- **[Hone](https://github.com/Kdurkee45/hone)** — Optimize existing code quality

## License

TBD
