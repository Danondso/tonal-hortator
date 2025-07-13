#!/usr/bin/env python3
"""
Local track embedding service using Ollama.
Embeds music tracks for semantic search without requiring internet.
"""

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from tonal_hortator.core.database import (
    CHECK_TABLE_EXISTS,
    CREATE_TRACK_EMBEDDINGS_TABLE,
    GET_TRACKS_WITHOUT_EMBEDDINGS,
    INSERT_TRACK_EMBEDDING,
)
from tonal_hortator.core.embeddings.embeddings import OllamaEmbeddingService
from tonal_hortator.utils.loader import (
    create_progress_spinner,
    create_threadsafe_progress_spinner,
)

# Configure logging
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)


class LocalTrackEmbedder:
    """Embed tracks using local Ollama service"""

    def __init__(
        self,
        db_path: str = "music_library.db",
        embedding_service: Optional[OllamaEmbeddingService] = None,
        conn: Optional[sqlite3.Connection] = None,
    ):
        """
        Initialize the local track embedder

        Args:
            db_path: Path to SQLite database
            embedding_service: OllamaEmbeddingService instance
            conn: Optional existing database connection
        """
        self.db_path = db_path
        self.conn = conn or sqlite3.connect(db_path)
        # Enable WAL mode for better concurrent performance
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.row_factory = sqlite3.Row

        if embedding_service:
            self.embedding_service = embedding_service
        else:
            self.embedding_service = OllamaEmbeddingService()

        # Ensure database has embeddings table
        self._ensure_embeddings_table()

    def _ensure_embeddings_table(self) -> None:
        """Ensure the track_embeddings table exists"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Check if embeddings table exists using centralized query
                cursor.execute(CHECK_TABLE_EXISTS, ("track_embeddings",))

                if not cursor.fetchone():
                    logger.info("Creating track_embeddings table...")
                    cursor.execute(CREATE_TRACK_EMBEDDINGS_TABLE)
                    self.conn.commit()
                    logger.info("✅ Created track_embeddings table")
                else:
                    logger.info("✅ track_embeddings table already exists")

        except Exception as e:
            logger.error(f"❌ Error ensuring embeddings table: {e}")
            raise

    def get_tracks_without_embeddings(self) -> List[Dict[str, Any]]:
        """Get all tracks from the database that don't have embeddings"""
        try:
            cursor = self.conn.cursor()

            # Get tracks that are not in the track_embeddings table using centralized query
            cursor.execute(GET_TRACKS_WITHOUT_EMBEDDINGS)

            tracks = [dict(row) for row in cursor.fetchall()]
            logger.info(f"🔍 Found {len(tracks)} tracks without embeddings")
            return tracks

        except sqlite3.Error as e:
            logger.error(f"❌ Error getting tracks without embeddings: {e}")
            raise

    def create_track_embedding_text(self, track: Dict[str, Any]) -> str:
        """Create text representation of track for embedding"""
        return self.embedding_service.create_track_embedding_text(track)

    def _get_track_display_info(self, track: Dict[str, Any]) -> str:
        """Create a display string for the current track being processed"""
        name = track.get("name", "Unknown Track")
        artist = track.get("artist", "Unknown Artist")

        # Truncate long names to keep the display clean
        if len(name) > 30:
            name = name[:27] + "..."
        if len(artist) > 25:
            artist = artist[:22] + "..."

        return f"🎵 {name} by {artist}"

    def _process_batch(self, batch_tracks: List[Dict[str, Any]], spinner=None) -> int:
        """Process a single batch of tracks (for parallel execution)"""
        try:
            batch_start = time.time()

            # Create a new database connection for this thread using context manager
            with sqlite3.connect(self.db_path) as thread_conn:
                # Enable WAL mode for better concurrent performance
                thread_conn.execute("PRAGMA journal_mode=WAL;")

                # Create embedding texts for this batch
                embedding_texts = []
                for track in batch_tracks:
                    text = self.create_track_embedding_text(track)
                    embedding_texts.append(text)

                # Get embeddings from Ollama with real-time updates
                embeddings = self.embedding_service.get_embeddings_batch_with_progress(
                    embedding_texts, batch_tracks, spinner, batch_size=2000
                )

                # Store embeddings in database using thread-specific connection
                batch_embedded = self._store_embeddings_batch(
                    batch_tracks, embeddings, embedding_texts, thread_conn
                )
                # Progress updates are now handled in the embedding service

            batch_time = time.time() - batch_start
            # Debug logging removed - progress bar now shows this information

            return batch_embedded

        except Exception as e:
            logger.error(f"❌ Error processing batch: {e}")
            return 0

    def embed_tracks_batch(
        self, tracks: List[Dict[str, Any]], batch_size: int = 500, max_workers: int = 4
    ) -> int:
        """
        Embed a batch of tracks using parallel processing with progress spinner

        Args:
            tracks: List of track dictionaries
            batch_size: Number of tracks to process in each batch
            max_workers: Maximum number of parallel threads

        Returns:
            Number of tracks successfully embedded
        """
        if not tracks:
            logger.info("No tracks to embed")
            return 0

        total_tracks = len(tracks)
        embedded_count = 0
        spinner = create_progress_spinner(total_tracks, "Embedding tracks", 1)
        start_time = time.time()

        # Split tracks into batches
        batches = []
        for i in range(0, total_tracks, batch_size):
            batch_tracks = tracks[i : i + batch_size]
            batches.append(batch_tracks)

        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batch processing tasks
            futures = [
                executor.submit(self._process_batch, batch_tracks, spinner)
                for batch_tracks in batches
            ]

            # Collect results as they complete and update spinner
            for future in as_completed(futures):
                try:
                    batch_embedded = future.result()
                    embedded_count += batch_embedded
                except Exception as e:
                    logger.error(f"❌ Batch failed: {e}")

        spinner.stop()
        total_time = time.time() - start_time
        logger.debug(
            f"✅ Successfully embedded {embedded_count} tracks in {total_time:.2f}s"
        )
        return embedded_count

    def _store_embeddings_batch(
        self,
        tracks: List[Dict[str, Any]],
        embeddings: List[np.ndarray],
        embedding_texts: List[str],
        conn: Optional[sqlite3.Connection] = None,
    ) -> int:
        """Store a batch of embeddings in the database using bulk insert"""
        try:
            # Use provided connection or fall back to instance connection
            db_conn = conn or self.conn
            cursor = db_conn.cursor()

            # Prepare bulk insert data
            insert_data = []
            for track, embedding, text in zip(tracks, embeddings, embedding_texts):
                try:
                    # Convert embedding to bytes for storage
                    embedding_bytes = embedding.tobytes()
                    insert_data.append((track["id"], embedding_bytes, text))
                except Exception as e:
                    logger.error(
                        f"❌ Error preparing embedding for track {track['id']}: {e}"
                    )
                    continue

            if insert_data:
                # Use bulk insert for better performance with centralized query
                cursor.executemany(INSERT_TRACK_EMBEDDING, insert_data)

                db_conn.commit()
                return len(insert_data)
            else:
                return 0

        except Exception as e:
            logger.error(f"❌ Error storing embeddings batch: {e}")
            return 0

    def get_all_embeddings(self) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """Get all embeddings and corresponding track data from the database"""
        try:
            cursor = self.conn.cursor()

            # Join tracks and track_embeddings tables
            cursor.execute(
                """
                SELECT
                    te.embedding,
                    t.id,
                    t.name,
                    t.artist,
                    t.album_artist,
                    t.composer,
                    t.album,
                    t.genre,
                    t.year,
                    t.total_time,
                    t.track_number,
                    t.disc_number,
                    t.play_count,
                    t.location,
                    t.bpm,
                    t.musical_key,
                    t.key_scale,
                    t.mood,
                    t.label,
                    t.producer,
                    t.arranger,
                    t.lyricist,
                    t.original_year,
                    t.original_date,
                    t.chord_changes_rate,
                    t.script,
                    t.replay_gain,
                    t.release_country
                FROM tracks t
                LEFT JOIN track_embeddings te ON t.id = te.track_id
                ORDER BY t.id
            """
            )

            rows = cursor.fetchall()
            embeddings = []
            track_data = []

            for row in rows:
                # Convert row to dict for easier access
                row_dict = dict(row)

                # Handle embedding data
                if row_dict["embedding"] is not None:
                    embedding = np.frombuffer(row_dict["embedding"], dtype=np.float32)
                    embeddings.append(embedding)
                    track_data.append(row_dict)
                else:
                    # Skip tracks without embeddings
                    continue

            logger.info(f"📊 Retrieved {len(embeddings)} embeddings from database")
            return embeddings, track_data

        except sqlite3.Error as e:
            logger.error(f"❌ Error getting embeddings: {e}")
            raise

    def embed_all_tracks(self, batch_size: int = 500, max_workers: int = 4) -> int:
        """Embed all tracks that don't have embeddings yet"""
        tracks = self.get_tracks_without_embeddings()
        return self.embed_tracks_batch(
            tracks, batch_size=batch_size, max_workers=max_workers
        )

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about embeddings in the database"""
        try:
            cursor = self.conn.cursor()

            # Total tracks
            cursor.execute("SELECT COUNT(*) FROM tracks")
            total_tracks = cursor.fetchone()[0]

            # Tracks with embeddings
            cursor.execute("SELECT COUNT(*) FROM track_embeddings")
            tracks_with_embeddings = cursor.fetchone()[0]

            # Tracks without embeddings
            tracks_without_embeddings = total_tracks - tracks_with_embeddings

            # Embedding coverage percentage
            coverage = (
                (tracks_with_embeddings / total_tracks * 100) if total_tracks > 0 else 0
            )

            stats = {
                "total_tracks": total_tracks,
                "tracks_with_embeddings": tracks_with_embeddings,
                "tracks_without_embeddings": tracks_without_embeddings,
                "coverage_percentage": round(coverage, 2),
            }

            return stats

        except Exception as e:
            logger.error(f"❌ Error getting embedding stats: {e}")
            raise


def main() -> None:
    """Main function to embed all tracks"""
    try:
        logger.info("🚀 Starting local track embedding process")

        # Initialize embedder
        embedder = LocalTrackEmbedder()

        # Get current stats
        stats = embedder.get_embedding_stats()
        logger.info(f"📊 Current embedding stats: {stats}")

        # Embed all tracks
        if stats["tracks_without_embeddings"] > 0:
            logger.info(f"🔄 Embedding {stats['tracks_without_embeddings']} tracks...")
            embedded_count = embedder.embed_all_tracks()
            logger.info(f"✅ Successfully embedded {embedded_count} tracks")
        else:
            logger.info("✅ All tracks already have embeddings")

        # Get final stats
        final_stats = embedder.get_embedding_stats()
        logger.info(f"📊 Final embedding stats: {final_stats}")

    except Exception as e:
        logger.error(f"❌ Error in main embedding process: {e}")
        raise


if __name__ == "__main__":
    main()
