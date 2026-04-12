"""Rich terminal UI — streams live progress.

The user always knows what's happening.
"""

from __future__ import annotations

import sys
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

STAGE_LABELS = {
    "discover": "🔍 Discover",
    "research": "🔬 Research",
    "grill": "🔥 Grill",
    "plan": "📋 Plan",
    "plan_review": "👤 Plan Review",
    "execute": "⚡ Execute",
    "verify": "✅ Verify",
}

STAGE_ORDER = list(STAGE_LABELS.keys())

MAX_DISPLAY_CHARS = 3000


class UI:
    def __init__(self, *, auto_approve: bool = False, verbose: bool = False) -> None:
        self.console = Console()
        self._current_stage: str | None = None
        self.auto_approve = auto_approve or not sys.stdin.isatty()
        self.verbose = verbose

    def _safe_print(self, *args: Any, **kwargs: Any) -> None:
        """Print that gracefully handles non-interactive/pipe scenarios."""
        try:
            self.console.print(*args, **kwargs)
        except (BlockingIOError, BrokenPipeError, OSError):
            pass

    def banner(self, repo_path: str, project_id: str, feature_prompt: str) -> None:
        prompt_preview = feature_prompt[:120] + (
            "…" if len(feature_prompt) > 120 else ""
        )
        self._safe_print()
        self._safe_print(
            Panel(
                f"[bold white]{repo_path}[/bold white]\n"
                f"[dim]Session: {project_id}[/dim]\n"
                f"[italic]{prompt_preview}[/italic]",
                title="[bold green]🌿 Graft — Feature Factory[/bold green]",
                border_style="green",
                box=box.DOUBLE,
                padding=(1, 2),
            )
        )
        self._safe_print()

    def stage_start(self, stage: str) -> None:
        self._current_stage = stage
        label = STAGE_LABELS.get(stage, stage)
        try:
            self.console.rule(f"[bold yellow] {label} ", style="yellow")
        except (BlockingIOError, BrokenPipeError, OSError):
            pass

    def stage_done(self, stage: str) -> None:
        label = STAGE_LABELS.get(stage, stage)
        self._safe_print(f"  [green]✓ {label} complete[/green]")
        self._safe_print()

    def stage_log(self, stage: str, message: str) -> None:
        if not self.verbose:
            return
        label = STAGE_LABELS.get(stage, stage)
        self._safe_print(f"  [dim]{label}:[/dim] {message}")

    def show_artifact(self, title: str, content: str) -> None:
        display = content
        if len(content) > MAX_DISPLAY_CHARS:
            display = (
                content[:MAX_DISPLAY_CHARS]
                + "\n\n[dim]… (truncated — see full artifact on disk)[/dim]"
            )
        self._safe_print()
        self._safe_print(
            Panel(
                display,
                title=f"[bold]{title}[/bold]",
                border_style="blue",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )
        self._safe_print()

    def grill_question(
        self, question: str, recommended: str, category: str, number: int
    ) -> str:
        """Present a Grill question and return the human's answer."""
        self._safe_print()
        self._safe_print(
            Panel(
                f"[bold]{question}[/bold]\n\n"
                f"[dim]Category: {category}[/dim]\n"
                f"[cyan]Recommended:[/cyan] {recommended}",
                title=f"[bold magenta]Question {number}[/bold magenta]",
                border_style="magenta",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )
        try:
            response = self.console.input(
                "  [bold]Your answer[/bold] [dim](Enter to accept recommended)[/dim]: "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            response = ""
        return response if response else recommended

    def prompt_plan_review(self, plan_summary: str) -> tuple[bool, str]:
        """Show the build plan and ask for approval.
        Returns (approved, feedback).
        """
        self.show_artifact("Build Plan", plan_summary)

        if self.auto_approve:
            self._safe_print(
                "[bold magenta]Plan review[/bold magenta] [dim](auto-approved)[/dim]"
            )
            return True, ""

        self._safe_print("[bold magenta]Plan Review[/bold magenta]")
        self._safe_print(
            "  Type [bold]approve[/bold] to proceed, or enter feedback to adjust:"
        )
        try:
            response = self.console.input("  [bold]> [/bold]").strip()
        except (EOFError, KeyboardInterrupt):
            self._safe_print("  [dim](no input — auto-approving)[/dim]")
            return True, ""
        if response.lower() in ("approve", "yes", "y", "lgtm", ""):
            return True, ""
        return False, response

    def unit_start(self, unit_id: str, title: str, index: int, total: int) -> None:
        self._safe_print(f"  [cyan]({index}/{total})[/cyan] {unit_id}: {title}")

    def unit_kept(self, unit_id: str, delta: str) -> None:
        self._safe_print(f"    [green]✓ kept[/green] {delta}")

    def unit_reverted(self, unit_id: str, reason: str) -> None:
        self._safe_print(f"    [red]✗ reverted[/red] — {reason}")

    def pr_opened(self, url: str) -> None:
        self._safe_print()
        self._safe_print(
            Panel(
                f"[bold green]{url}[/bold green]",
                title="[bold green]PR Opened[/bold green]",
                border_style="green",
                box=box.DOUBLE,
                padding=(1, 2),
            )
        )
        self._safe_print()

    def coverage_warning(self, warnings: list[dict]) -> None:
        """Display module-level test coverage warnings."""
        lines = [
            "[bold yellow]⚠ Low Test Coverage on Integration Modules[/bold yellow]\n"
        ]
        for w in warnings:
            lines.append(
                f"  [red]{w['module']}[/red] — {w['coverage_pct']}% coverage\n"
                f"    {w['recommendation']}"
            )
        self._safe_print()
        self._safe_print(
            Panel(
                "\n".join(lines),
                title="[bold yellow]Coverage Warning[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )
        self._safe_print()

    def error(self, message: str) -> None:
        self._safe_print(f"[bold red]Error:[/bold red] {message}")

    def info(self, message: str) -> None:
        self._safe_print(f"  [cyan]>[/cyan] {message}")

    def show_projects(self, projects: list[dict]) -> None:
        if not projects:
            self.console.print("[dim]No feature sessions found.[/dim]")
            return
        table = Table(title="Graft — Feature Sessions", box=box.SIMPLE_HEAVY)
        table.add_column("ID", style="cyan")
        table.add_column("Repo", max_width=40)
        table.add_column("Feature", max_width=40)
        table.add_column("Status", style="green")
        table.add_column("Stages")
        table.add_column("Created")
        for p in projects:
            stages = ", ".join(p.get("stages_completed", []))
            table.add_row(
                p["project_id"],
                p.get("repo_path", "")[:40],
                p.get("feature_prompt", "")[:40],
                p.get("status", "unknown"),
                stages or "—",
                p.get("created_at", "")[:19],
            )
        self.console.print(table)
