#!/usr/bin/env python3
"""
🎵 Tonal Hortator - AI-Powered Local Music Playlist Generator
Beautiful CLI interface with Typer and Rich
"""

import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from tonal_hortator.core.embeddings.track_embedder import LocalTrackEmbedder
from tonal_hortator.core.playlist.playlist_generator import LocalPlaylistGenerator
from tonal_hortator.utils.apple_music import open_in_apple_music
from tonal_hortator.utils.library_parser import LibraryParser

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure Rich console
console = Console()

# Create Typer app
app = typer.Typer(
    name="tonal-hortator",
    help="🎵 AI-Powered Local Music Playlist Generator",
    add_completion=False,
    rich_markup_mode="rich",
)

# Global settings
DEFAULT_DB_PATH = "music_library.db"


def show_banner() -> None:
    """Display the app banner"""
    banner = Text()
    banner.append("🎵 ", style="bold magenta")
    banner.append("Tonal Hortator", style="bold cyan")
    banner.append(" - AI-Powered Local Music Playlist Generator", style="italic")

    console.print(Panel(banner, style="cyan", padding=(1, 2)))


def show_status(generator: LocalPlaylistGenerator) -> None:
    """Show database and embedding status"""
    stats = generator.track_embedder.get_embedding_stats()

    table = Table(
        title="📊 Database Status", show_header=True, header_style="bold magenta"
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Tracks", str(stats["total_tracks"]))
    table.add_row("Tracks with Embeddings", str(stats["tracks_with_embeddings"]))
    table.add_row("Embedding Coverage", f"{stats['coverage_percentage']:.1f}%")

    console.print(table)


@app.command()
def yeet(
    xml_path: str = typer.Argument(..., help="Path to iTunes XML library file"),
    db_path: str = typer.Option(DEFAULT_DB_PATH, "--db", "-d", help="Database path"),
    resume: bool = typer.Option(
        False, "--resume", "-r", help="Resume embedding (skip database deletion)"
    ),
) -> None:
    """🗑️ YEET! Complete reset and rebuild with a grunge playlist"""
    show_banner()

    with console.status("[bold green]YEETing everything...", spinner="dots"):
        try:
            # Step 1: Delete database (unless resuming)
            if not resume:
                if os.path.exists(db_path):
                    console.print(f"🗑️  Deleting database: {db_path}")
                    os.remove(db_path)
                else:
                    console.print("ℹ️  No existing database found")
            else:
                console.print(f"⏩ Resuming with database: {db_path}")

            # Step 2: Import library
            console.print(f"📥 Importing library from: {xml_path}")
            if not import_library(xml_path, db_path):
                console.print("[red]❌ Failed to import library[/red]")
                raise typer.Exit(1)

            # Step 3: Embed tracks
            console.print("🧠 Generating embeddings...")
            if not embed_tracks():
                console.print("[red]❌ Failed to embed tracks[/red]")
                raise typer.Exit(1)

            # Step 4: Generate grunge playlist
            console.print("🎸 Generating grunge playlist...")
            generator = LocalPlaylistGenerator(db_path=db_path)
            tracks = generator.generate_playlist(
                "5 grunge songs", max_tracks=5, min_similarity=0.3
            )

            if not tracks:
                console.print("[red]❌ No tracks found for grunge playlist[/red]")
                raise typer.Exit(1)

            # Show results
            generator.print_playlist_summary(tracks, "5 grunge songs")

            # Save and open
            filepath = generator.save_playlist_m3u(tracks, "5 grunge songs")
            console.print(f"✅ Playlist saved: {filepath}")

            if Confirm.ask("🎵 Open in Apple Music?"):
                open_in_apple_music(filepath)

            console.print(
                "[bold green]🎉 YEET complete! Your grunge playlist is ready![/bold green]"
            )

        except Exception as e:
            console.print(f"[red]❌ YEET failed: {e}[/red]")
            raise typer.Exit(1)


@app.command()
def generate(
    query: Optional[str] = typer.Argument(
        None, help="Playlist query (e.g., 'jazz for studying')"
    ),
    max_tracks: Optional[int] = typer.Option(
        None, "--tracks", "-t", help="Maximum number of tracks"
    ),
    similarity: float = typer.Option(
        0.3, "--similarity", "-s", help="Minimum similarity threshold"
    ),
    breadth: int = typer.Option(
        15, "--breadth", "-b", help="Search breadth factor (5-25)"
    ),
    auto_open: bool = typer.Option(
        False, "--open", "-o", help="Auto-open in Apple Music"
    ),
) -> None:
    """🎵 Generate a playlist from your query"""
    show_banner()

    try:
        generator = LocalPlaylistGenerator()
        show_status(generator)

        # Check if embeddings exist
        stats = generator.track_embedder.get_embedding_stats()
        if stats["tracks_with_embeddings"] == 0:
            console.print("[red]❌ No embeddings found. Run 'yeet' first![/red]")
            raise typer.Exit(1)

        # Interactive mode if no query
        if not query:
            console.print("\n[bold cyan]🎵 Interactive Playlist Generator[/bold cyan]")
            console.print("Type 'quit' to exit\n")

            while True:
                query = Prompt.ask("Enter your playlist query")

                if query.lower() in ["quit", "exit", "q"]:
                    break

                if not query:
                    continue

                _generate_single_playlist(
                    generator, query, max_tracks, similarity, breadth, auto_open
                )
        else:
            # Non-interactive mode
            _generate_single_playlist(
                generator, query, max_tracks, similarity, breadth, auto_open
            )

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


def _generate_single_playlist(
    generator: LocalPlaylistGenerator,
    query: str,
    max_tracks: Optional[int],
    similarity: float,
    breadth: int,
    auto_open: bool,
) -> None:
    """Generate a single playlist"""
    with console.status(
        f"[bold green]Generating playlist: {query}[/bold green]", spinner="dots"
    ):
        tracks = generator.generate_playlist(
            query,
            max_tracks=max_tracks,
            min_similarity=similarity,
            search_breadth_factor=breadth,
        )

    if not tracks:
        console.print(f"[red]❌ No tracks found for: {query}[/red]")
        return

    # Show results
    generator.print_playlist_summary(tracks, query)

    # Save playlist
    filepath = generator.save_playlist_m3u(tracks, query)
    console.print(f"✅ Playlist saved: {filepath}")

    # Open in Apple Music
    if auto_open or Confirm.ask("🎵 Open in Apple Music?"):
        open_in_apple_music(filepath)


@app.command()
def embed(
    batch_size: int = typer.Option(
        500, "--batch", "-b", help="Batch size for embeddings"
    ),
    workers: int = typer.Option(
        4, "--workers", "-w", help="Number of worker processes"
    ),
) -> None:
    """🧠 Generate embeddings for all tracks"""
    show_banner()

    with console.status("[bold green]Generating embeddings...", spinner="dots"):
        if embed_tracks(batch_size, max_workers=workers):
            console.print("[bold green]✅ Embeddings complete![/bold green]")
        else:
            console.print("[red]❌ Embedding failed[/red]")
            raise typer.Exit(1)


@app.command()
def import_lib(
    xml_path: str = typer.Argument(..., help="Path to iTunes XML library file"),
    db_path: str = typer.Option(DEFAULT_DB_PATH, "--db", "-d", help="Database path"),
) -> None:
    """📥 Import music library from iTunes XML"""
    show_banner()

    with console.status(
        f"[bold green]Importing library from: {xml_path}[/bold green]", spinner="dots"
    ):
        if import_library(xml_path, db_path):
            console.print("[bold green]✅ Library import complete![/bold green]")
        else:
            console.print("[red]❌ Import failed[/red]")
            raise typer.Exit(1)


@app.command()
def status() -> None:
    """📊 Show database and embedding status"""
    show_banner()

    try:
        generator = LocalPlaylistGenerator()
        show_status(generator)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def interactive() -> None:
    """🎮 Start interactive mode"""
    show_banner()

    try:
        generator = LocalPlaylistGenerator()
        show_status(generator)

        # Check embeddings
        stats = generator.track_embedder.get_embedding_stats()
        if stats["tracks_with_embeddings"] == 0:
            console.print("[red]❌ No embeddings found. Run 'yeet' first![/red]")
            raise typer.Exit(1)

        console.print("\n[bold cyan]🎮 Interactive Mode[/bold cyan]")
        console.print("Commands: generate, status, quit\n")

        while True:
            command = Prompt.ask(
                "What would you like to do?", choices=["generate", "status", "quit"]
            )

            if command == "quit":
                break
            elif command == "status":
                show_status(generator)
            elif command == "generate":
                query = Prompt.ask("Enter your playlist query")
                if query:
                    _generate_single_playlist(generator, query, None, 0.3, 15, False)

        console.print("[bold green]👋 Thanks for using Tonal Hortator![/bold green]")

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


def yeet_everything(
    xml_path: str, db_path: str = "music_library.db", resume: bool = False
) -> None:
    """Legacy function - use yeet command instead"""
    yeet(xml_path, db_path, resume)


def generate_playlist(
    query: Optional[str] = None,
    max_tracks: Optional[int] = None,
    min_similarity: float = 0.3,
    search_breadth_factor: int = 15,
    auto_open: bool = False,
) -> None:
    """Legacy function - use generate command instead"""
    generate(query, max_tracks, min_similarity, search_breadth_factor, auto_open)


def embed_tracks(batch_size: int = 500, max_workers: int = 4) -> bool:
    """Generate embeddings for all tracks in the database"""
    try:
        embedder = LocalTrackEmbedder()
        embedded_count = embedder.embed_all_tracks(
            batch_size=batch_size, max_workers=max_workers
        )
        return embedded_count > 0
    except Exception as e:
        console.print(f"[red]❌ Embedding error: {e}[/red]")
        return False


def import_library(xml_path: str, db_path: str = "music_library.db") -> bool:
    """Import music library from iTunes XML file"""
    try:
        parser = LibraryParser(db_path)
        inserted_count = parser.parse_library(xml_path)
        return inserted_count > 0
    except Exception as e:
        console.print(f"[red]❌ Import error: {e}[/red]")
        return False


def main() -> None:
    """Main entry point"""
    app()


if __name__ == "__main__":
    main()
