# Feature Factory

An AI-powered pipeline for building significant new features into existing applications.

**The middle ground:** Too big for a Jira ticket, too constrained for greenfield. Takes a short human prompt and turns it into a shipped, tested, integrated PR — through structured discovery, interrogation, planning, and execution.

## The Idea

You say: _"I need a transfers system where managers can propose trades between teams."_

The machine:
1. **Discovers** your codebase — architecture, patterns, data model, conventions
2. **Researches** what the feature needs — reuse opportunities, gaps, edge cases
3. **Grills you** — focused questions with recommended answers, one at a time
4. **Plans** the build — atomic units with dependencies, pattern references, tests
5. **Executes** — implements each unit, tests after every change, auto-reverts on regression
6. **Verifies** — full regression + feature validation, opens a PR

You never write a PRD. You answer questions. The machine does the rest.

## Status

**Design phase.** See [docs/DESIGN.md](docs/DESIGN.md) for the full design document.

## Related Projects

- **[Software Optimization Factory](https://github.com/Kdurkee45/software-optimization-factory)** — Takes existing code and makes it measurably better (code quality, security, performance)
- **[gstack](https://github.com/garrytan/gstack)** — AI engineering workflow skills for zero-to-one app building

## License

TBD
