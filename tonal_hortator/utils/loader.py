#!/usr/bin/env python3
"""
Loader utilities for better visual feedback during long-running operations.
"""

import queue
import sys
import threading
import time
from typing import Any, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)


class ProgressBar:
    """Rich-based progress bar for terminal output."""

    def __init__(self, total: int, description: str = "Progress", width: int = 50):
        self.total = total
        self.description = description
        self.width = width
        self.current = 0
        self.console = Console()

        # Create Rich progress bar
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        )
        self.task_id: Optional[Any] = None

    def update(self, increment: int = 1) -> None:
        """Update the progress bar."""
        if self.task_id is None:
            return
        self.current += increment
        self.progress.update(self.task_id, completed=self.current)

    def start(self) -> None:
        """Start the progress bar."""
        self.progress.start()
        self.task_id = self.progress.add_task(self.description, total=self.total)

    def finish(self) -> None:
        """Finish the progress bar."""
        if self.progress:
            self.progress.stop()


class Spinner:
    """Rich-based simple spinner for indeterminate progress."""

    def __init__(self, description: str = "Processing"):
        self.description = description
        self.running = False
        self.console = Console()

        # Create Rich progress with snazzy spinner
        self.progress = Progress(
            SpinnerColumn(spinner_name="dots", style="bold magenta"),
            TextColumn("[bold cyan]{task.description}"),
            console=self.console,
            transient=True,
        )
        self.task_id: Optional[Any] = None

    def start(self) -> None:
        """Start the spinner."""
        self.running = True
        self.progress.start()
        self.task_id = self.progress.add_task(self.description, total=None)

    def stop(self) -> None:
        """Stop the spinner."""
        self.running = False
        if self.progress:
            self.progress.stop()

    def update(self) -> None:
        """Update the spinner animation."""
        if not self.running or self.task_id is None:
            return
        # Rich spinner updates automatically, no manual update needed


class ProgressSpinner:
    """Rich-based spinner with progress tracking and ETA display."""

    def __init__(
        self, total: int, description: str = "Processing", batch_size: int = 1
    ):
        self.total = total
        self.description = description
        self.batch_size = batch_size
        self.current = 0
        self.running = False
        self.console = Console()

        # Create Rich progress bar with snazzy styling
        self.progress = Progress(
            SpinnerColumn(spinner_name="dots", style="bold blue"),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(
                complete_style="bold green",
                finished_style="bold bright_green",
                pulse_style="bold yellow",
            ),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[bold white]({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
            expand=True,
        )
        self.task_id: Optional[Any] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the spinner."""
        self.running = True
        self.progress.start()
        self.task_id = self.progress.add_task(self.description, total=self.total)

    def stop(self) -> None:
        """Stop the spinner."""
        self.running = False
        if self.progress:
            self.progress.stop()

    def update(self, increment: int = 1) -> None:
        """Update the spinner animation and progress."""
        if not self.running or self.task_id is None:
            return

        with self._lock:
            self.current += increment
            self.progress.update(self.task_id, completed=self.current)

    def update_safe(self, increment: int = 1) -> None:
        """Thread-safe update method."""
        try:
            self.update(increment)
        except Exception:
            pass


class BatchLoader:
    """Loader for batch processing operations with progress bar."""

    def __init__(
        self, total_items: int, batch_size: int, description: str = "Processing"
    ):
        self.total_items = total_items
        self.batch_size = batch_size
        self.description = description
        self.total_batches = (total_items + batch_size - 1) // batch_size
        self.current_batch = 0
        self.processed_items = 0
        self.start_time = time.time()
        self.width = 50

    def start_batch(self, batch_num: int, batch_size: int) -> None:
        """Start processing a new batch."""
        self.current_batch = batch_num
        # Don't display anything on start, just update progress bar

    def finish_batch(self, batch_num: int, processed: int, batch_time: float) -> None:
        """Finish processing a batch."""
        self.processed_items += processed
        self._display_progress()

    def _display_progress(self) -> None:
        """Display the current progress bar."""
        if self.total_items == 0:
            return

        percentage = min(100, (self.processed_items / self.total_items) * 100)
        filled_width = int(self.width * self.processed_items // self.total_items)
        bar = "█" * filled_width + "░" * (self.width - filled_width)

        elapsed = time.time() - self.start_time
        if self.processed_items > 0:
            eta = (elapsed / self.processed_items) * (
                self.total_items - self.processed_items
            )
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = "ETA: --"

        # Clear line and display progress
        sys.stdout.write(
            f"\r{self.description}: [{bar}] {percentage:.1f}% ({self.processed_items}/{self.total_items}) {eta_str}"
        )
        sys.stdout.flush()

    def finish(self) -> None:
        """Finish the entire batch processing."""
        total_time = time.time() - self.start_time
        # Complete the progress bar
        self.processed_items = self.total_items
        self._display_progress()
        sys.stdout.write(f" - Completed in {total_time:.2f}s\n")
        sys.stdout.flush()


def create_progress_bar(total: int, description: str = "Progress") -> ProgressBar:
    """Create a progress bar."""
    progress_bar = ProgressBar(total, description)
    progress_bar.start()
    return progress_bar


def create_spinner(description: str = "Processing") -> Spinner:
    """Create a spinner."""
    spinner = Spinner(description)
    spinner.start()
    return spinner


def create_progress_spinner(
    total: int, description: str = "Processing", batch_size: int = 1
) -> ProgressSpinner:
    """Create a progress spinner with ETA."""
    progress_spinner = ProgressSpinner(total, description, batch_size)
    progress_spinner.start()
    return progress_spinner


def create_batch_loader(
    total_items: int, batch_size: int, description: str = "Processing"
) -> BatchLoader:
    """Create a batch loader."""
    return BatchLoader(total_items, batch_size, description)


class SpinnerManager:
    """Manages a ProgressSpinner in a background thread with thread-safe updates."""

    def __init__(
        self, total: int, description: str = "Processing", update_interval: float = 0.1
    ):
        self.spinner = ProgressSpinner(total, description)
        self.update_queue: queue.Queue[int] = queue.Queue()
        self.track_info_queue: queue.Queue[str] = queue.Queue()
        self.update_interval = update_interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.spinner.start()
        self._thread.start()

    def update(self, increment: int = 1) -> None:
        self.update_queue.put(increment)

    def update_track_info(self, track_info: str) -> None:
        self.track_info_queue.put(track_info)

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join()
        self.spinner.stop()

    def _run(self) -> None:
        while not self._stop_event.is_set() or not self.update_queue.empty():
            try:
                # Consume all updates in the queue
                while not self.update_queue.empty():
                    increment = self.update_queue.get_nowait()
                    self.spinner.update_safe(increment)

                # Track info updates are no longer needed since we removed track info display

            except Exception:
                pass
            time.sleep(self.update_interval)


def create_threadsafe_progress_spinner(
    total: int, description: str = "Processing"
) -> SpinnerManager:
    return SpinnerManager(total, description)


def configure_loguru_for_rich() -> None:
    """Configure Loguru to use RichHandler for pretty logging output."""
    try:
        import logging

        # Remove existing Loguru handlers
        from loguru import logger
        from rich.logging import RichHandler

        logger.remove()

        # Add RichHandler
        logger.add(
            RichHandler(
                rich_tracebacks=True,
                markup=True,
                show_time=True,
                show_level=True,
                show_path=True,
            ),
            format="{message}",
            level="INFO",
        )
    except ImportError:
        pass
