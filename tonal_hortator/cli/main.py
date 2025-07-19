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
from tonal_hortator.core.feedback import FeedbackManager
from tonal_hortator.core.playlist.playlist_generator import LocalPlaylistGenerator
from tonal_hortator.utils.apple_music import open_in_apple_music
from tonal_hortator.utils.csv_ingester import MusicCSVIngester
from tonal_hortator.utils.embedding_updater import EmbeddingUpdater
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

    # Collect user feedback
    _collect_playlist_feedback(generator.feedback_manager, query, tracks)

    # Open in Apple Music
    if auto_open or Confirm.ask("🎵 Open in Apple Music?"):
        open_in_apple_music(filepath)


@app.command()
def embed(
    batch_size: int = typer.Option(
        50, "--batch", "-b", help="Batch size for embeddings"
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
def ingest_csv(
    csv_path: str = typer.Argument(..., help="Path to music.csv file from iPod export"),
    db_path: str = typer.Option(DEFAULT_DB_PATH, "--db", "-d", help="Database path"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview changes without applying them"
    ),
    batch_size: int = typer.Option(
        10, "--batch", "-b", help="Batch size for processing"
    ),
) -> None:
    """Ingest music data from iPod CSV export"""
    show_banner()

    try:
        ingester = MusicCSVIngester(db_path)

        with console.status(
            f"[bold green]Ingesting CSV from: {csv_path}[/bold green]", spinner="dots"
        ):
            stats = ingester.ingest_csv(
                csv_path, dry_run=dry_run, batch_size=batch_size
            )

        # Display results
        table = Table(
            title="📊 CSV Ingestion Results",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Rows", str(stats["total_rows"]))
        table.add_row("Processed", str(stats["processed"]))
        table.add_row("Inserted", str(stats["inserted"]))
        table.add_row("Updated", str(stats["updated"]))
        table.add_row("Skipped", str(stats["skipped"]))
        table.add_row("Errors", str(stats["errors"]))
        table.add_row(
            "Tracks for Embedding Update",
            str(len(stats["tracks_to_update_embeddings"])),
        )

        console.print(table)

        if dry_run:
            console.print(
                "[yellow]🔍 DRY RUN COMPLETE - No changes made to database[/yellow]"
            )
        else:
            console.print("[bold green]✅ CSV ingestion complete![/bold green]")

            # Ask if user wants to update embeddings
            if stats["tracks_to_update_embeddings"] and Confirm.ask(
                f"🔄 Update embeddings for {len(stats['tracks_to_update_embeddings'])} modified tracks?"
            ):
                _update_embeddings_for_tracks(stats["tracks_to_update_embeddings"])

    except Exception as e:
        console.print(f"[red]❌ CSV ingestion failed: {e}[/red]")
        raise typer.Exit(1)
    finally:
        ingester.close()


@app.command()
def update_embeddings(
    track_ids: Optional[str] = typer.Argument(
        None, help="Comma-separated list of track IDs to update"
    ),
    file_path: Optional[str] = typer.Option(
        None, "--file", "-f", help="File containing track IDs (one per line)"
    ),
    db_path: str = typer.Option(DEFAULT_DB_PATH, "--db", "-d", help="Database path"),
    batch_size: int = typer.Option(
        50, "--batch", "-b", help="Batch size for embeddings"
    ),
    workers: int = typer.Option(
        4, "--workers", "-w", help="Number of worker processes"
    ),
) -> None:
    """🔄 Update embeddings for specific tracks"""
    show_banner()

    try:
        # Parse track IDs
        track_id_list = []

        if track_ids:
            track_id_list = [
                int(tid.strip())
                for tid in track_ids.split(",")
                if tid.strip().isdigit()
            ]

        if file_path:
            from tonal_hortator.utils.embedding_updater import parse_ids_from_file

            file_ids = parse_ids_from_file(file_path)
            track_id_list.extend(file_ids)

        if not track_id_list:
            console.print("[red]❌ No track IDs provided. Use --help for usage.[/red]")
            raise typer.Exit(1)

        _update_embeddings_for_tracks(track_id_list, db_path, batch_size, workers)

    except Exception as e:
        console.print(f"[red]❌ Embedding update failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def workflow(
    csv_path: str = typer.Argument(..., help="Path to music.csv file from iPod export"),
    db_path: str = typer.Option(DEFAULT_DB_PATH, "--db", "-d", help="Database path"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview CSV changes without applying them"
    ),
    skip_embeddings: bool = typer.Option(
        False, "--skip-embeddings", help="Skip embedding updates"
    ),
    batch_size: int = typer.Option(
        100, "--batch", "-b", help="CSV processing batch size"
    ),
    embed_batch_size: int = typer.Option(
        50, "--embed-batch", "-e", help="Embedding batch size"
    ),
    workers: int = typer.Option(
        4, "--workers", "-w", help="Number of worker processes"
    ),
) -> None:
    """Complete workflow: Ingest CSV and update embeddings"""
    show_banner()

    console.print("[bold cyan]🔄 Starting Complete Workflow[/bold cyan]")
    console.print("This will:")
    console.print("  1. Ingest music data from CSV")
    console.print("  2. 🔄 Update embeddings for modified tracks")
    console.print("3. Show final statistics")
    if not Confirm.ask("Continue with workflow?"):
        console.print("[yellow]Workflow cancelled[/yellow]")
        return

    try:
        # Step 1: Ingestion
        console.print("\n[bold green]Step 1: Ingestion[/bold green]")
        ingester = MusicCSVIngester(db_path)

        with console.status(
            f"[bold green]Ingesting CSV from: {csv_path}[/bold green]", spinner="dots"
        ):
            stats = ingester.ingest_csv(
                csv_path, dry_run=dry_run, batch_size=batch_size
            )

        # Display CSV results
        table = Table(
            title="📊 CSV Ingestion Results",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Rows", str(stats["total_rows"]))
        table.add_row("Processed", str(stats["processed"]))
        table.add_row("Inserted", str(stats["inserted"]))
        table.add_row("Updated", str(stats["updated"]))
        table.add_row("Skipped", str(stats["skipped"]))
        table.add_row("Errors", str(stats["errors"]))

        console.print(table)

        if dry_run:
            console.print(
                "[yellow]🔍 DRY RUN COMPLETE - No changes made to database[/yellow]"
            )
            return

        # Step 2: Embedding Updates (if not skipped and there are tracks to update)
        if stats["tracks_to_update_embeddings"] and Confirm.ask(
            f"🔄 Update embeddings for {len(stats['tracks_to_update_embeddings'])} modified tracks?"
        ):
            _update_embeddings_for_tracks(
                stats["tracks_to_update_embeddings"], db_path, embed_batch_size, workers
            )
        elif skip_embeddings:
            console.print(
                "\n[yellow]⏩ Skipping embedding updates as requested[/yellow]"
            )
        else:
            console.print("\n[yellow]ℹ️  No tracks need embedding updates[/yellow]")

        # Step 3: Final Status
        console.print("\n[bold green]Step 3: Final Status[/bold green]")
        generator = LocalPlaylistGenerator(db_path=db_path)
        show_status(generator)

        console.print("\n[bold green]🎉 Workflow Complete![/bold green]")

    except Exception as e:
        console.print(f"[red]❌ Workflow failed: {e}[/red]")
        raise typer.Exit(1)
    finally:
        ingester.close()


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
def feedback() -> None:
    """📝 Manage feedback and learning"""
    show_banner()

    try:
        feedback_manager = FeedbackManager()

        while True:
            console.print("\n[bold cyan]📝 Feedback Management[/bold cyan]")
            console.print("1. View feedback stats")
            console.print("2. View user preferences")
            console.print("3. Set preference")
            console.print("4. View track ratings")
            console.print("5. Rate a track")
            console.print("6. View learning data")
            console.print("7. Show recommended settings")
            console.print("8. Back to main menu")

            choice = Prompt.ask(
                "Choose an option", choices=["1", "2", "3", "4", "5", "6", "7", "8"]
            )

            if choice == "1":
                _show_feedback_stats(feedback_manager)
            elif choice == "2":
                _show_user_preferences(feedback_manager)
            elif choice == "3":
                _set_preference(feedback_manager)
            elif choice == "4":
                _show_track_ratings(feedback_manager)
            elif choice == "5":
                _rate_track(feedback_manager)
            elif choice == "6":
                _show_learning_data(feedback_manager)
            elif choice == "7":
                _show_recommended_settings(feedback_manager)
            elif choice == "8":
                break

    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


def _show_feedback_stats(feedback_manager: FeedbackManager) -> None:
    """Show feedback statistics"""
    stats = feedback_manager.get_feedback_stats()

    if not stats:
        console.print("[yellow]No feedback data available[/yellow]")
        return

    table = Table(
        title="📊 Feedback Statistics", show_header=True, header_style="bold magenta"
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Feedback", str(stats.get("total_feedback", 0)))

    avg_rating = stats.get("average_rating")
    if avg_rating is not None:
        table.add_row("Average Rating", f"{avg_rating:.1f}")
    else:
        table.add_row("Average Rating", "N/A")

    if "feedback_by_type" in stats:
        for query_type, count in stats["feedback_by_type"].items():
            table.add_row(f"Feedback ({query_type})", str(count))

    console.print(table)


def _show_user_preferences(feedback_manager: FeedbackManager) -> None:
    """Show user preferences"""
    preferences = feedback_manager.get_user_preferences()

    if not preferences:
        console.print("[yellow]No user preferences set[/yellow]")
        return

    table = Table(
        title="⚙️ User Preferences", show_header=True, header_style="bold magenta"
    )
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Type", style="yellow")
    table.add_column("Description", style="blue")

    for key, value, pref_type, description in preferences:
        table.add_row(key, str(value), pref_type, description or "")

    console.print(table)


def _set_preference(feedback_manager: FeedbackManager) -> None:
    """Set a user preference"""
    key = Prompt.ask("Preference key")
    value = Prompt.ask("Preference value")
    pref_type = Prompt.ask(
        "Preference type", choices=["string", "integer", "float", "boolean", "json"]
    )
    description = Prompt.ask("Description (optional)")

    # Convert value based on type
    if pref_type == "integer":
        value = str(int(value))
    elif pref_type == "float":
        value = str(float(value))
    elif pref_type == "boolean":
        value = str(value.lower() == "true")
    elif pref_type == "json":
        import json

        value = json.loads(value)

    if feedback_manager.set_preference(key, value, pref_type, description):
        console.print(f"[green]✅ Preference '{key}' set successfully[/green]")
    else:
        console.print("[red]❌ Failed to set preference[/red]")


def _show_track_ratings(feedback_manager: FeedbackManager) -> None:
    """Show track ratings"""
    ratings = feedback_manager.get_track_ratings()

    if not ratings:
        console.print("[yellow]No track ratings available[/yellow]")
        return

    table = Table(
        title="⭐ Track Ratings", show_header=True, header_style="bold magenta"
    )
    table.add_column("Rating", style="yellow")
    table.add_column("Context", style="cyan")
    table.add_column("Track", style="green")
    table.add_column("Artist", style="blue")

    for rating, context, track_name, artist in ratings:
        stars = "⭐" * rating
        table.add_row(stars, context or "", track_name, artist)

    console.print(table)


def _rate_track(feedback_manager: FeedbackManager) -> None:
    """Rate a track"""
    # This would need track lookup functionality
    console.print(
        "[yellow]Track rating functionality requires track lookup - coming soon![/yellow]"
    )


def _show_learning_data(feedback_manager: FeedbackManager) -> None:
    """Show learning data"""
    learning_data = feedback_manager.get_learning_data()

    if not learning_data:
        console.print("[yellow]No learning data available[/yellow]")
        return

    table = Table(
        title="🧠 Query Learning Data", show_header=True, header_style="bold magenta"
    )
    table.add_column("Query", style="cyan")
    table.add_column("LLM Result", style="green")
    table.add_column("User Correction", style="yellow")
    table.add_column("Score", style="blue")

    for query, llm_result, user_correction, feedback_score in learning_data:
        llm_str = f"{llm_result.get('query_type', 'unknown')}"
        correction_str = str(user_correction) if user_correction else "None"
        score_str = f"{feedback_score:.2f}" if feedback_score else "N/A"

        table.add_row(
            query[:30] + "...", llm_str, correction_str[:20] + "...", score_str
        )

    console.print(table)


def _show_recommended_settings(feedback_manager: FeedbackManager) -> None:
    """Show recommended settings based on user feedback"""
    query_types = ["artist_specific", "similarity", "general"]

    table = Table(
        title="🎯 Recommended Settings", show_header=True, header_style="bold magenta"
    )
    table.add_column("Query Type", style="cyan")
    table.add_column("Similarity Threshold", style="green")
    table.add_column("Search Breadth", style="yellow")
    table.add_column("Avg Rating", style="blue")
    table.add_column("Feedback Count", style="red")

    for query_type in query_types:
        settings = feedback_manager.get_recommended_settings(query_type)

        if settings:
            table.add_row(
                query_type.replace("_", " ").title(),
                str(settings.get("similarity_threshold", "N/A")),
                str(settings.get("search_breadth", "N/A")),
                str(settings.get("average_rating", "N/A")),
                str(settings.get("feedback_count", "N/A")),
            )
        else:
            table.add_row(
                query_type.replace("_", " ").title(), "N/A", "N/A", "N/A", "0"
            )

    console.print(table)


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


def _collect_playlist_feedback(
    feedback_manager: FeedbackManager, query: str, tracks: list
) -> None:
    """Collect user feedback for a generated playlist."""
    if not Confirm.ask("📝 Would you like to provide feedback for this playlist?"):
        return

    console.print(f"\n[bold cyan]🎵 Playlist Feedback for: {query}[/bold cyan]")

    # Overall playlist rating
    rating = Prompt.ask(
        "Rate this playlist (1-5 stars)", choices=["1", "2", "3", "4", "5"]
    )
    comments = Prompt.ask("Comments (optional)")

    # Record playlist feedback
    feedback_manager.record_playlist_feedback(
        query=query,
        query_type="general",  # This would need to be passed from the generator
        parsed_data={},  # This would need to be passed from the generator
        generated_tracks=tracks,
        user_rating=int(rating),
        user_comments=comments,
        playlist_length=len(tracks),
    )

    console.print(f"[green]✅ Playlist feedback recorded (Rating: {rating})[/green]")

    # Individual track ratings
    if Confirm.ask("Would you like to rate individual tracks?"):
        for i, track in enumerate(tracks):
            console.print(f"\n{i+1}. {track['name']} - {track['artist']}")
            if Confirm.ask("Rate this track?"):
                track_rating = Prompt.ask(
                    "Rating (1-5 stars)", choices=["1", "2", "3", "4", "5"]
                )
                feedback_manager.record_track_rating(
                    track_id=track.get("id", 0),
                    rating=int(track_rating),
                    context=f"Playlist: {query}",
                )
                console.print("[green]✅ Track rating recorded[/green]")


def _update_embeddings_for_tracks(
    track_ids: list,
    db_path: str = DEFAULT_DB_PATH,
    batch_size: int = 50,
    workers: int = 4,
) -> None:
    """Update embeddings for a list of track IDs."""
    try:
        updater = EmbeddingUpdater(db_path)
        stats = updater.update_embeddings_for_tracks(track_ids, batch_size, workers)
        updated_count = stats.get("updated", 0)
        console.print(
            f"[green]✅ Embeddings updated for {updated_count} tracks.[/green]"
        )
    except Exception as e:
        console.print(f"[red]❌ Embedding update error: {e}[/red]")
        raise typer.Exit(1)


def main() -> None:
    """Main entry point"""
    app()


if __name__ == "__main__":
    main()
