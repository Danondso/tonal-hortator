#!/usr/bin/env python3
"""
Main CLI interface for Tonal Hortator
"""

import argparse
import logging
import sys
import sqlite3
import numpy as np

from ..core.playlist_generator import LocalPlaylistGenerator
from ..core.track_embedder import LocalTrackEmbedder
from ..utils.apple_music import open_in_apple_music
from ..utils.library_parser import LibraryParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def generate_playlist(query: str = None, max_tracks: int = 20, 
                     min_similarity: float = 0.3, auto_open: bool = False):
    """Generate a playlist with optional Apple Music opening"""
    try:
        logger.info("🚀 Starting playlist generation")
        
        generator = LocalPlaylistGenerator()
        
        # Get embedding stats
        stats = generator.track_embedder.get_embedding_stats()
        logger.info(f"📊 Database stats: {stats}")
        
        if stats['tracks_with_embeddings'] == 0:
            logger.warning("No embeddings found. Please run embed_tracks first.")
            return False
        
        # Interactive mode if no query provided
        if not query:
            print("\n🎵 Tonal Hortator - Local Playlist Generator")
            print("=" * 50)
            
            while True:
                query = input("\nEnter your playlist query (or 'quit' to exit): ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    print("Please enter a query.")
                    continue
                
                _generate_single_playlist(generator, query, max_tracks, min_similarity, auto_open)
        else:
            # Non-interactive mode
            return _generate_single_playlist(generator, query, max_tracks, min_similarity, auto_open)
        
        print("\n👋 Thanks for using Tonal Hortator!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in playlist generation: {e}")
        return False


def _generate_single_playlist(generator, query: str, max_tracks: int, 
                             min_similarity: float, auto_open: bool):
    """Generate a single playlist"""
    try:
        # Generate playlist
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
            if open_music in ['y', 'yes']:
                open_in_apple_music(filepath)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error generating playlist: {e}")
        print(f"❌ Error: {e}")
        return False


def embed_tracks(batch_size: int = 100):
    """Embed all tracks in the database"""
    try:
        logger.info("🚀 Starting track embedding process")
        
        embedder = LocalTrackEmbedder()
        
        # Get current stats
        stats = embedder.get_embedding_stats()
        logger.info(f"📊 Current embedding stats: {stats}")
        
        # Embed all tracks
        if stats['tracks_without_embeddings'] > 0:
            logger.info(f"🔄 Embedding {stats['tracks_without_embeddings']} tracks with batch size {batch_size}...")
            embedded_count = embedder.embed_all_tracks()
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


def import_library(xml_path: str, db_path: str = "music_library.db"):
    """Parse the Apple Music XML library and populate the database."""
    try:
        logger.info(f"🚀 Starting library import from: {xml_path}")
        parser = LibraryParser(db_path=db_path)
        inserted_count = parser.parse_library(xml_path=xml_path)
        logger.info(f"✅ Library import complete. Added {inserted_count} new tracks.")
        
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


def cleanup_embeddings(db_path: str = "music_library.db"):
    """Clean up invalid embeddings from the database"""
    try:
        logger.info(f"🔍 Cleaning up embeddings in: {db_path}")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # Count total embeddings
        cur.execute("SELECT COUNT(*) FROM track_embeddings")
        total = cur.fetchone()[0]

        # Find invalid embeddings (empty or wrong dimension)
        cur.execute("SELECT track_id, embedding FROM track_embeddings")
        to_delete = []
        EMBEDDING_DIM = 768  # For nomic-embed-text
        
        for track_id, blob in cur.fetchall():
            try:
                arr = np.frombuffer(blob, dtype=np.float32)
                if arr.size != EMBEDDING_DIM:
                    to_delete.append(track_id)
            except Exception:
                to_delete.append(track_id)

        # Delete invalid embeddings
        for track_id in to_delete:
            cur.execute("DELETE FROM track_embeddings WHERE track_id = ?", (track_id,))
        conn.commit()

        logger.info(f"✅ Removed {len(to_delete)} invalid embeddings out of {total} total.")
        if to_delete:
            logger.info(f"🗑️  Track IDs removed: {to_delete}")
        else:
            logger.info("🎉 No invalid embeddings found!")
        
        conn.close()
        return True

    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")
        return False


def main():
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
  tonal-hortator cleanup
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate playlist command
    gen_parser = subparsers.add_parser('generate', help='Generate a playlist')
    gen_parser.add_argument('query', nargs='?', help='Search query for playlist')
    gen_parser.add_argument('--max-tracks', type=int, default=20, help='Maximum tracks in playlist')
    gen_parser.add_argument('--min-similarity', type=float, default=0.3, help='Minimum similarity threshold')
    gen_parser.add_argument('--auto-open', action='store_true', help='Automatically open in Apple Music')
    
    # Embed tracks command
    embed_parser = subparsers.add_parser('embed', help='Embed all tracks in database')
    embed_parser.add_argument('--batch-size', type=int, default=100, help='Batch size for embedding')
    
    # Import library command
    import_parser = subparsers.add_parser('import-library', help='Import tracks from Apple Music XML library')
    import_parser.add_argument('xml_path', help='Path to the Apple Music XML library file')
    import_parser.add_argument('--db-path', default='music_library.db', help='Path to the SQLite database file')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Start interactive playlist generator')
    interactive_parser.add_argument('--max-tracks', type=int, default=20, help='Maximum tracks in playlist')
    interactive_parser.add_argument('--min-similarity', type=float, default=0.3, help='Minimum similarity threshold')
    interactive_parser.add_argument('--auto-open', action='store_true', help='Automatically open in Apple Music')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up invalid embeddings from database')
    cleanup_parser.add_argument('--db-path', default='music_library.db', help='Path to the SQLite database file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'generate':
            success = generate_playlist(
                query=args.query,
                max_tracks=args.max_tracks,
                min_similarity=args.min_similarity,
                auto_open=args.auto_open
            )
        elif args.command == 'embed':
            success = embed_tracks(batch_size=args.batch_size)
        elif args.command == 'import-library':
            success = import_library(xml_path=args.xml_path, db_path=args.db_path)
        elif args.command == 'interactive':
            success = generate_playlist(
                max_tracks=args.max_tracks,
                min_similarity=args.min_similarity,
                auto_open=args.auto_open
            )  # Interactive mode
        elif args.command == 'cleanup':
            success = cleanup_embeddings(db_path=args.db_path)
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