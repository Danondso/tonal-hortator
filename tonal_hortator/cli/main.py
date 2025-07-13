#!/usr/bin/env python3
"""
Main CLI interface for Tonal Hortator
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from migrate_schema import DatabaseMigrator
from tonal_hortator.core.embeddings.track_embedder import LocalTrackEmbedder
from tonal_hortator.core.playlist.playlist_generator import LocalPlaylistGenerator
from tonal_hortator.utils.apple_music import open_in_apple_music
from tonal_hortator.utils.library_parser import LibraryParser
from tonal_hortator.utils.loader import configure_loguru_for_rich

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure logging to file
log_file = logs_dir / f"tonal_hortator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Set console logging to WARNING to reduce noise
console_handler = logging.StreamHandler(
    sys.stderr
)  # Use stderr for logging to avoid interfering with loaders
console_handler.setLevel(logging.WARNING)
console_formatter = logging.Formatter("%(levelname)s: %(message)s")
console_handler.setFormatter(console_formatter)

# File handler for detailed logging
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)

# Configure root logger
root_logger = logging.getLogger()
root_logger.handlers.clear()  # Remove default handlers
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)
root_logger.setLevel(logging.DEBUG)

# Suppress noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def yeet_everything(
    xml_path: str, db_path: str = "music_library.db", resume: bool = False
) -> bool:
    """YEET! Delete everything and start fresh with a grunge playlist, or resume embedding if --resume is set"""
    try:
        print(
            "🗑️  YEET! Starting complete reset and rebuild..."
            if not resume
            else "⏩ YEET! Resuming embedding and playlist generation..."
        )
        logger.info(
            "🗑️  YEET! Starting complete reset and rebuild..."
            if not resume
            else "⏩ YEET! Resuming embedding and playlist generation..."
        )

        # Step 1: Delete the database (unless resuming)
        if not resume:
            if os.path.exists(db_path):
                print(f"🗑️  Deleting existing database: {db_path}")
                logger.info(f"🗑️  Deleting existing database: {db_path}")
                os.remove(db_path)
                print("✅ Database deleted")
                logger.info("✅ Database deleted")
            else:
                print("ℹ️  No existing database found, starting fresh")
                logger.info("ℹ️  No existing database found, starting fresh")
        else:
            print(f"⏩ Skipping database deletion for resume mode. Using: {db_path}")
            logger.info(
                f"⏩ Skipping database deletion for resume mode. Using: {db_path}"
            )

        # Step 2: Import the XML library
        print(f"📥 Importing library from: {xml_path}")
        logger.info(f"📥 Importing library from: {xml_path}")
        if not import_library(xml_path, db_path):
            print("❌ Failed to import library")
            logger.error("❌ Failed to import library")
            return False

        # Step 3: Embed all tracks
        print("🧠 Starting embeddings...")
        logger.info("🧠 Starting embeddings...")
        if not embed_tracks():
            print("❌ Failed to embed tracks")
            logger.error("❌ Failed to embed tracks")
            return False

        # Step 4: Generate the grunge playlist
        print("🎸 Generating '5 grunge songs' playlist...")
        logger.info("🎸 Generating '5 grunge songs' playlist...")
        generator = LocalPlaylistGenerator(db_path=db_path)

        # Generate the playlist
        tracks = generator.generate_playlist(
            "5 grunge songs", max_tracks=5, min_similarity=0.3
        )

        if not tracks:
            print("❌ No tracks found for grunge playlist")
            logger.error("❌ No tracks found for grunge playlist")
            return False

        # Print summary
        generator.print_playlist_summary(tracks, "5 grunge songs")

        # Save playlist
        filepath = generator.save_playlist_m3u(tracks, "5 grunge songs")
        print(f"✅ Playlist saved to: {filepath}")
        logger.info(f"✅ Playlist saved to: {filepath}")

        # Open in Apple Music
        print("🎵 Opening in Apple Music...")
        logger.info("🎵 Opening in Apple Music...")
        open_in_apple_music(filepath)

        print("🎉 YEET complete! Your grunge playlist is ready!")
        logger.info("🎉 YEET complete! Your grunge playlist is ready!")
        return True

    except Exception as e:
        print(f"❌ YEET failed: {e}")
        logger.error(f"❌ YEET failed: {e}")
        return False


def generate_playlist(
    query: Optional[str] = None,
    max_tracks: Optional[int] = None,  # Allow None to respect query count
    min_similarity: float = 0.3,
    search_breadth_factor: int = 15,  # Default to 15 based on benchmark results
    auto_open: bool = False,
) -> bool:
    """Generate a playlist with optional Apple Music opening"""
    try:
        logger.info("🚀 Starting playlist generation")

        generator = LocalPlaylistGenerator()

        # Get embedding stats
        stats = generator.track_embedder.get_embedding_stats()
        logger.info(f"📊 Database stats: {stats}")

        if stats["tracks_with_embeddings"] == 0:
            logger.warning("No embeddings found. Please run embed_tracks first.")
            return False

        # Interactive mode if no query provided
        if not query:
            print("\n🎵 Tonal Hortator - Local Playlist Generator")
            print("=" * 50)

            while True:
                query = input(
                    "\nEnter your playlist query (or 'quit' to exit): "
                ).strip()

                if query.lower() in ["quit", "exit", "q"]:
                    break

                if not query:
                    print("Please enter a query.")
                    continue

                _generate_single_playlist(
                    generator,
                    query,
                    max_tracks,
                    min_similarity,
                    search_breadth_factor,
                    auto_open,
                )
        else:
            # Non-interactive mode
            return _generate_single_playlist(
                generator,
                query,
                max_tracks,
                min_similarity,
                search_breadth_factor,
                auto_open,
            )

        logger.info("👋 Thanks for using Tonal Hortator!")
        return True

    except Exception as e:
        logger.error(f"❌ Error in playlist generation: {e}")
        return False


def _generate_single_playlist(
    generator: LocalPlaylistGenerator,
    query: str,
    max_tracks: Optional[int],  # Allow None to respect query count
    min_similarity: float,
    search_breadth_factor: int,
    auto_open: bool,
) -> bool:
    """Generate a single playlist"""
    try:
        # Pass max_tracks (can be None) so generate_playlist uses the query's count if needed
        tracks = generator.generate_playlist(
            query,
            max_tracks,
            min_similarity,
            search_breadth_factor=search_breadth_factor,
        )

        if not tracks:
            logger.warning(
                "❌ No tracks found for your query. Try a different search term."
            )
            return False

        # Print summary
        generator.print_playlist_summary(tracks, query)

        # Save playlist
        filepath = generator.save_playlist_m3u(tracks, query)
        logger.info(f"✅ Playlist saved to: {filepath}")

        # Open in Apple Music if requested
        if auto_open:
            open_in_apple_music(filepath)
        else:
            # Ask user
            open_music = input("Open in Apple Music? (y/n): ").strip().lower()
            if open_music in ["y", "yes"]:
                open_in_apple_music(filepath)

        return True

    except Exception as e:
        logger.error(f"❌ Error generating playlist: {e}")
        return False


def embed_tracks(batch_size: int = 500, max_workers: int = 4) -> bool:
    """Embed all tracks in the database using parallel processing

    Args:
        batch_size: Number of tracks to process in each batch
        max_workers: Maximum number of parallel threads for embedding

    Returns:
        True if embedding completed successfully, False otherwise
    """
    try:
        print("🚀 Starting track embedding process")
        logger.info("🚀 Starting track embedding process")

        embedder = LocalTrackEmbedder()

        # Get current stats
        stats = embedder.get_embedding_stats()
        print(f"📊 Current embedding stats: {stats}")
        logger.info(f"📊 Current embedding stats: {stats}")

        # Embed all tracks
        if stats["tracks_without_embeddings"] > 0:
            print(
                f"🔄 Embedding {stats['tracks_without_embeddings']} tracks with batch size {batch_size} and {max_workers} parallel workers..."
            )
            logger.info(
                f"🔄 Embedding {stats['tracks_without_embeddings']} tracks with batch size {batch_size} and {max_workers} parallel workers..."
            )
            embedded_count = embedder.embed_all_tracks(
                batch_size=batch_size, max_workers=max_workers
            )
            print(f"✅ Successfully embedded {embedded_count} tracks")
            logger.info(f"✅ Successfully embedded {embedded_count} tracks")
        else:
            print("✅ All tracks already have embeddings")
            logger.info("✅ All tracks already have embeddings")

        # Get final stats
        final_stats = embedder.get_embedding_stats()
        print(f"📊 Final embedding stats: {final_stats}")
        logger.info(f"📊 Final embedding stats: {final_stats}")

        return True

    except Exception as e:
        print(f"❌ Error in embedding process: {e}")
        logger.error(f"❌ Error in embedding process: {e}")
        return False


def import_library(xml_path: str, db_path: str = "music_library.db") -> bool:
    """Parse the Apple Music XML library and populate the database."""
    try:
        print(f"🚀 Starting library import from: {xml_path}")
        logger.info(f"🚀 Starting library import from: {xml_path}")

        # Create metadata mappings table if DatabaseMigrator is available
        if DatabaseMigrator is not None:
            print("📋 Creating metadata mappings table...")
            logger.info("📋 Creating metadata mappings table...")
            migrator = DatabaseMigrator(db_path=db_path)
            if migrator.create_metadata_mappings_table():
                print("✅ Metadata mappings table created successfully")
                logger.info("✅ Metadata mappings table created successfully")
            else:
                print("⚠️  Could not create metadata mappings table")
                logger.warning("⚠️  Could not create metadata mappings table")

        parser = LibraryParser(db_path=db_path)
        inserted_count = parser.parse_library(xml_path=xml_path)
        print(f"✅ Library import complete. Added {inserted_count} new tracks.")
        logger.info(f"✅ Library import complete. Added {inserted_count} new tracks.")

        # Run full database migration to ensure all metadata columns exist
        if DatabaseMigrator is not None and inserted_count > 0:
            print("🔄 Running database migration to add metadata columns...")
            logger.info("🔄 Running database migration to add metadata columns...")
            if migrator.run_migration():
                print("✅ Database migration completed successfully")
                logger.info("✅ Database migration completed successfully")
            else:
                print("⚠️  Database migration failed, but continuing...")
                logger.warning("⚠️  Database migration failed, but continuing...")

        # After importing, it's good practice to embed the new tracks
        if inserted_count > 0:
            print("Proceeding to embed new tracks...")
            logger.info("Proceeding to embed new tracks...")
            embedder = LocalTrackEmbedder(db_path=db_path)
            embedded_count = embedder.embed_all_tracks()
            print(f"✅ Successfully embedded {embedded_count} new tracks.")
            logger.info(f"✅ Successfully embedded {embedded_count} new tracks.")

        return True

    except Exception as e:
        print(f"❌ Error during library import: {e}")
        logger.error(f"❌ Error during library import: {e}")
        return False


def main() -> int:
    """Main CLI entry point"""
    # Configure loguru to work well with Rich progress bars
    configure_loguru_for_rich()

    parser = argparse.ArgumentParser(
        description="Tonal Hortator - AI-powered local music playlist generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  tonal-hortator generate "upbeat rock songs"
  tonal-hortator generate --max-tracks 30 --auto-open
  tonal-hortator embed
  tonal-hortator interactive
  tonal-hortator import-library "/path/to/your/library.xml"
  tonal-hortator yeet "/path/to/your/library.xml"

Logs are saved to: {log_file}
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate playlist command
    gen_parser = subparsers.add_parser("generate", help="Generate a playlist")
    gen_parser.add_argument("query", nargs="?", help="Search query for playlist")
    gen_parser.add_argument(
        "--max-tracks",
        type=int,
        help="Maximum tracks in playlist (default: use count from query)",
    )
    gen_parser.add_argument(
        "--min-similarity", type=float, default=0.3, help="Minimum similarity threshold"
    )
    gen_parser.add_argument(
        "--auto-open", action="store_true", help="Automatically open in Apple Music"
    )
    gen_parser.add_argument(
        "--search-breadth-factor",
        type=int,
        default=15,
        help="Search breadth factor for playlist generation (default: 15)",
    )

    # Embed tracks command
    embed_parser = subparsers.add_parser("embed", help="Embed all tracks in database")
    embed_parser.add_argument(
        "--batch-size", type=int, default=1000, help="Batch size for embedding"
    )
    embed_parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Maximum parallel workers for embedding",
    )

    # Import library command
    import_parser = subparsers.add_parser(
        "import-library", help="Import tracks from Apple Music XML library"
    )
    import_parser.add_argument(
        "xml_path", help="Path to the Apple Music XML library file"
    )
    import_parser.add_argument(
        "--db-path", default="music_library.db", help="Path to the SQLite database file"
    )

    # Interactive command
    interactive_parser = subparsers.add_parser(
        "interactive", help="Start interactive playlist generator"
    )
    interactive_parser.add_argument(
        "--max-tracks",
        type=int,
        help="Maximum tracks in playlist (default: use count from query)",
    )
    interactive_parser.add_argument(
        "--min-similarity", type=float, default=0.3, help="Minimum similarity threshold"
    )
    interactive_parser.add_argument(
        "--auto-open", action="store_true", help="Automatically open in Apple Music"
    )
    interactive_parser.add_argument(
        "--search-breadth-factor",
        type=int,
        default=15,
        help="Search breadth factor for playlist generation (default: 15)",
    )

    # Yeet command
    yeet_parser = subparsers.add_parser(
        "yeet",
        help="YEET! Delete everything and start fresh with a grunge playlist, or resume embedding with --resume",
    )
    yeet_parser.add_argument(
        "xml_path", help="Path to the Apple Music XML library file for import"
    )
    yeet_parser.add_argument(
        "--db-path", default="music_library.db", help="Path to the SQLite database file"
    )
    yeet_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume embedding and playlist generation without deleting the database",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "generate":
            success = generate_playlist(
                query=args.query,
                max_tracks=(args.max_tracks if args.max_tracks is not None else None),
                min_similarity=args.min_similarity,
                auto_open=args.auto_open,
                search_breadth_factor=args.search_breadth_factor,
            )
        elif args.command == "embed":
            success = embed_tracks(
                batch_size=args.batch_size, max_workers=args.max_workers
            )
        elif args.command == "import-library":
            success = import_library(xml_path=args.xml_path, db_path=args.db_path)
        elif args.command == "interactive":
            success = generate_playlist(
                max_tracks=(
                    args.max_tracks
                    if hasattr(args, "max_tracks") and args.max_tracks is not None
                    else None
                ),
                min_similarity=args.min_similarity,
                auto_open=args.auto_open,
                search_breadth_factor=args.search_breadth_factor,
            )  # Interactive mode
        elif args.command == "yeet":
            success = yeet_everything(
                xml_path=args.xml_path,
                db_path=args.db_path,
                resume=getattr(args, "resume", False),
            )
        else:
            parser.print_help()
            return 1

        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("👋 Goodbye!")
        return 0
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
