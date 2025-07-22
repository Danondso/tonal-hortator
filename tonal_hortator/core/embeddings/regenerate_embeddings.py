#!/usr/bin/env python3
"""
Script to regenerate embeddings with enhanced musical analysis data
"""

import argparse
import logging
import sqlite3

from tonal_hortator.core.database.queries import (
    DELETE_ALL_TRACK_EMBEDDINGS,
    GET_TRACKS_WITH_MUSICAL_ANALYSIS,
)
from tonal_hortator.core.embeddings.track_embedder import LocalTrackEmbedder

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def regenerate_embeddings(db_path: str = "music_library.db") -> bool:
    """Regenerate all embeddings with enhanced musical analysis data"""
    try:
        logger.info("🚀 Starting embedding regeneration with musical analysis data")

        # Initialize embedder
        embedder = LocalTrackEmbedder(db_path)

        # Get current stats
        stats = embedder.get_embedding_stats()
        logger.info(f"📊 Current embedding stats: {stats}")

        # Clear existing embeddings to force regeneration
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(DELETE_ALL_TRACK_EMBEDDINGS)
            deleted_count = cursor.rowcount
            conn.commit()
            logger.info(f"🗑️  Cleared {deleted_count} existing embeddings")

        # Regenerate all embeddings
        embedded_count = embedder.embed_all_tracks()
        logger.info(
            f"✅ Successfully regenerated {embedded_count} embeddings with musical analysis"
        )

        # Get final stats
        final_stats = embedder.get_embedding_stats()
        logger.info(f"📊 Final embedding stats: {final_stats}")

        return True

    except Exception as e:
        logger.error(f"❌ Error regenerating embeddings: {e}")
        return False


def test_enhanced_embeddings(db_path: str = "music_library.db") -> None:
    """Test the enhanced embedding system with a sample track"""
    try:
        logger.info("🧪 Testing enhanced embedding system")

        # Get a sample track with musical analysis data
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(GET_TRACKS_WITH_MUSICAL_ANALYSIS)

            track = cursor.fetchone()
            if track:
                track_dict = {
                    "id": track[0],
                    "name": track[1],
                    "artist": track[2],
                    "album": track[3],
                    "genre": track[4],
                    "bpm": track[5],
                    "musical_key": track[6],
                    "key_scale": track[7],
                    "mood": track[8],
                }

                # Test embedding text creation
                from tonal_hortator.core.embeddings.embeddings import (
                    OllamaEmbeddingService,
                )

                service = OllamaEmbeddingService()
                embedding_text = service.create_track_embedding_text(track_dict)

                logger.info(
                    f"📝 Sample track: {track_dict['name']} by {track_dict['artist']}"
                )
                logger.info(f"🎵 Enhanced embedding text: {embedding_text}")

            else:
                logger.warning("No tracks with musical analysis data found")

    except Exception as e:
        logger.error(f"❌ Error testing enhanced embeddings: {e}")


def main() -> int:
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Regenerate embeddings with musical analysis"
    )
    parser.add_argument(
        "--db-path", default="music_library.db", help="Path to SQLite database"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only test enhanced embeddings, don't regenerate",
    )

    args = parser.parse_args()

    if args.test_only:
        test_enhanced_embeddings(args.db_path)
    else:
        if regenerate_embeddings(args.db_path):
            print("✅ Embedding regeneration completed successfully!")
            print("\n🧪 Testing enhanced embeddings:")
            test_enhanced_embeddings(args.db_path)
        else:
            print("❌ Embedding regeneration failed!")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
