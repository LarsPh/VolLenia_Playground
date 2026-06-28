from __future__ import annotations

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

class RatePairColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        unit = str(task.fields.get("unit", "it"))
        speed = task.speed
        if speed is None or speed <= 0.0:
            return Text(f"-- {unit}/s -- s/{unit}")
        return Text(f"{speed:.2f} {unit}/s {1.0 / speed:.2f} s/{unit}")

class StatusColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        return Text(str(task.fields.get("status", "")))

class SearchProgress:
    def __init__(self) -> None:
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            RatePairColumn(),
            StatusColumn(),
            console=self.console,
        )

    def __enter__(self) -> "SearchProgress":
        if self.progress is not None:
            self.progress.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.progress is not None:
            self.progress.stop()

    def add_task(self, description: str, *, total: int, unit: str, status: str = "") -> int | None:
        return self.progress.add_task(description, total=max(int(total), 0), unit=unit, status=status)

    def update(self, task_id: int | None, *, advance: int = 0, status: str | None = None) -> None:
        if task_id is None:
            return
        fields = {}
        if status is not None:
            fields["status"] = status
        self.progress.update(task_id, advance=advance, refresh=True, **fields)

    def remove(self, task_id: int | None) -> None:
        if task_id is None:
            return
        self.progress.refresh()
        self.progress.remove_task(task_id)

    def log(self, message: str) -> None:
        self.progress.console.print(message)

__all__ = ["RatePairColumn", "SearchProgress", "StatusColumn"]
