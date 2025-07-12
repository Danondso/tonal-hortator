#!/usr/bin/env python3
"""
Main CLI interface for Tonal Hortator
"""

import argparse
import logging
import sys
from typing import Optional

from migrate_schema import DatabaseMigrator
from tonal_hortator.core.playlist_generator import LocalPlaylistGenerator
from tonal_hortator.core.track_embedder import LocalTrackEmbedder
from tonal_hortator.utils.apple_music import open_in_apple_music
from tonal_hortator.utils.library_parser import LibraryParser

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def generate_playlist(
    query: Optional[str] = None,
    max_tracks: Optional[int] = None,  # Allow None to respect query count
    min_similarity: float = 0.3,
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
                    generator, query, max_tracks, min_similarity, auto_open
                )
        else:
            # Non-interactive mode
            return _generate_single_playlist(
                generator, query, max_tracks, min_similarity, auto_open
            )

        print("\n👋 Thanks for using Tonal Hortator!")
        return True

    except Exception as e:
        logger.error(f"❌ Error in playlist generation: {e}")
        return False


def _generate_single_playlist(
    generator: LocalPlaylistGenerator,
    query: str,
    max_tracks: Optional[int],  # Allow None to respect query count
    min_similarity: float,
    auto_open: bool,
) -> bool:
    """Generate a single playlist"""
    try:
        # Pass max_tracks (can be None) so generate_playlist uses the query's count if needed
        tracks = generator.generate_playlist(query, max_tracks, min_similarity)

        if not tracks:
            print("❌ No tracks found for your query. Try a different search term.")
            return False

        # Print summary
        generator.print_playlist_summary(tracks, query)

        # Save playlist
        filepath = generator.save_playlist_m3u(tracks, query)
        print(f"✅ Playlist saved to: {filepath}")

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
        print(f"❌ Error: {e}")
        return False


def embed_tracks(batch_size: int = 100, max_workers: int = 4) -> bool:
    """Embed all tracks in the database using parallel processing

    Args:
        batch_size: Number of tracks to process in each batch
        max_workers: Maximum number of parallel threads for embedding

    Returns:
        True if embedding completed successfully, False otherwise
    """
    try:
        logger.info("🚀 Starting track embedding process")

        embedder = LocalTrackEmbedder()

        # Get current stats
        stats = embedder.get_embedding_stats()
        logger.info(f"📊 Current embedding stats: {stats}")

        # Embed all tracks
        if stats["tracks_without_embeddings"] > 0:
            logger.info(
                f"🔄 Embedding {stats['tracks_without_embeddings']} tracks with batch size {batch_size} and {max_workers} parallel workers..."
            )
            embedded_count = embedder.embed_all_tracks(
                batch_size=batch_size, max_workers=max_workers
            )
            logger.info(f"✅ Successfully embedded {embedded_count} tracks")
        else:
            logger.info("✅ All tracks already have embeddings")

        # Get final stats
        final_stats = embedder.get_embedding_stats()
        logger.info(f"📊 Final embedding stats: {final_stats}")

        return True

    except Exception as e:
        logger.error(f"❌ Error in embedding process: {e}")
        return False


def import_library(xml_path: str, db_path: str = "music_library.db") -> bool:
    """Parse the Apple Music XML library and populate the database."""
    try:
        logger.info(f"🚀 Starting library import from: {xml_path}")

        # Create metadata mappings table if DatabaseMigrator is available
        if DatabaseMigrator is not None:
            logger.info("📋 Creating metadata mappings table...")
            migrator = DatabaseMigrator(db_path=db_path)
            if migrator.create_metadata_mappings_table():
                logger.info("✅ Metadata mappings table created successfully")
            else:
                logger.warning("⚠️  Could not create metadata mappings table")

        parser = LibraryParser(db_path=db_path)
        inserted_count = parser.parse_library(xml_path=xml_path)
        logger.info(f"✅ Library import complete. Added {inserted_count} new tracks.")

        # Run full database migration to ensure all metadata columns exist
        if DatabaseMigrator is not None and inserted_count > 0:
            logger.info("🔄 Running database migration to add metadata columns...")
            if migrator.run_migration():
                logger.info("✅ Database migration completed successfully")
            else:
                logger.warning("⚠️  Database migration failed, but continuing...")

        # After importing, it's good practice to embed the new tracks
        if inserted_count > 0:
            logger.info("Proceeding to embed new tracks...")
            embedder = LocalTrackEmbedder(db_path=db_path)
            embedded_count = embedder.embed_all_tracks()
            logger.info(f"✅ Successfully embedded {embedded_count} new tracks.")

        return True

    except Exception as e:
        logger.error(f"❌ Error during library import: {e}")
        return False


def main() -> int:
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Tonal Hortator - AI-powered local music playlist generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tonal-hortator generate "upbeat rock songs"
  tonal-hortator generate --max-tracks 30 --auto-open
  tonal-hortator embed
  tonal-hortator interactive
  tonal-hortator import-library "/path/to/your/library.xml"
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

    # Embed tracks command
    embed_parser = subparsers.add_parser("embed", help="Embed all tracks in database")
    embed_parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size for embedding"
    )
    embed_parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
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
            )  # Interactive mode
        else:
            parser.print_help()
            return 1

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
