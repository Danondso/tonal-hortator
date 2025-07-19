#!/usr/bin/env python3
"""
Loader utilities for better visual feedback during long-running operations.
"""

import queue
import sys
import threading
import time
from typing import Any, Optional

import psutil
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


def get_optimal_batch_size(
    base_size: int = 500,
    memory_factor: float = 0.1,
    cpu_factor: float = 0.5,
    min_size: int = 50,
    max_size: int = 1000,
) -> int:
    """
    Determine optimal batch size based on system resources.

    Args:
        base_size: Base batch size to start with
        memory_factor: Factor to adjust based on available memory (0.0-1.0)
        cpu_factor: Factor to adjust based on CPU cores (0.0-1.0)
        min_size: Minimum allowed batch size
        max_size: Maximum allowed batch size

    Returns:
        Optimal batch size for the current system
    """
    try:
        # Get system information
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count(logical=False) or 1

        # Calculate available memory in GB
        available_memory_gb = memory.available / (1024**3)

        # Adjust batch size based on available memory
        # Assume each track embedding needs ~1KB of memory
        memory_based_size = int(available_memory_gb * 1024 * 1024 * memory_factor)

        # Adjust based on CPU cores (more cores can handle larger batches)
        cpu_based_size = int(base_size * cpu_count * cpu_factor)

        # Take the minimum of memory and CPU based sizes
        optimal_size = min(memory_based_size, cpu_based_size)

        # Ensure it's within bounds
        optimal_size = max(min_size, min(optimal_size, max_size))

        return optimal_size

    except Exception:
        # Fallback to base size if we can't determine system resources
        return base_size


def get_batch_size_with_fallback(
    user_specified: Optional[int] = None,
    base_size: int = 500,
    min_size: int = 50,
    max_size: int = 1000,
) -> int:
    """
    Get batch size with intelligent fallback to system-optimized size.

    Args:
        user_specified: User-specified batch size (None for auto-detection)
        base_size: Base batch size for auto-detection
        min_size: Minimum allowed batch size
        max_size: Maximum allowed batch size

    Returns:
        Final batch size to use
    """
    if user_specified is not None:
        # User specified a size, validate it
        return max(min_size, min(user_specified, max_size))

    # Auto-detect optimal size
    return get_optimal_batch_size(base_size, min_size=min_size, max_size=max_size)
