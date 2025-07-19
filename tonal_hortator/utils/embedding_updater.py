#!/usr/bin/env python3
"""
Embedding updater for tracks modified during CSV ingestion.
Updates embeddings for tracks with new play count and rating data.
Supports both preservation mode and hybrid mode for handling existing embeddings.
"""

import logging
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from tonal_hortator.core.database import (
    GET_TRACK_EMBEDDING,
    INSERT_OR_REPLACE_TRACK_EMBEDDING,
)
from tonal_hortator.core.database.query_helpers import (
    build_delete_embeddings_by_ids_query,
    build_get_tracks_by_ids_query,
)
from tonal_hortator.core.embeddings.track_embedder import LocalTrackEmbedder

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class EmbeddingUpdater:
    """Update embeddings for tracks modified during CSV ingestion."""

    def __init__(self, db_path: str = "music_library.db"):
        self.db_path = db_path
        self.embedder = LocalTrackEmbedder(db_path)

    def update_embeddings_for_tracks(
        self,
        track_ids: List[int],
        batch_size: int = 500,
        max_workers: int = 4,
        mode: str = "preserve",  # "preserve" or "hybrid"
    ) -> Dict[str, Any]:
        """Update embeddings for tracks with different modes."""
        if not track_ids:
            logger.info("No track IDs provided for embedding updates.")
            return {
                "total_tracks": 0,
                "updated": 0,
                "errors": 0,
                "skipped": 0,
                "preserved": 0,
            }
        logger.info(
            f"Starting embedding updates for {len(track_ids)} tracks in {mode} mode."
        )
        stats = {
            "total_tracks": len(track_ids),
            "updated": 0,
            "preserved": 0,
            "errors": 0,
            "skipped": 0,
        }
        try:
            tracks_to_update = self._get_tracks_by_ids(track_ids)
            if not tracks_to_update:
                logger.warning("No tracks found for the provided IDs.")
                return stats
            logger.info(f"Found {len(tracks_to_update)} tracks to process.")
            if mode == "preserve":
                self._update_embeddings_preserve_mode(
                    tracks_to_update, batch_size, max_workers, stats
                )
            elif mode == "hybrid":
                self._update_embeddings_hybrid_mode(
                    tracks_to_update, batch_size, max_workers, stats
                )
            else:
                raise ValueError(f"Invalid mode: {mode}. Use 'preserve' or 'hybrid'.")
            logger.info(
                f"✅ Successfully processed {stats['updated'] + stats['preserved']} embeddings."
            )
        except Exception as e:
            logger.error(f"❌ Error updating embeddings: {e}")
            stats["updated"] = 0
            stats["preserved"] = 0
            stats["skipped"] = 0
            stats["errors"] = self._calculate_error_count(track_ids, locals().get("tracks_to_update"))
        return stats

    def _update_embeddings_preserve_mode(
        self,
        tracks: List[Dict[str, Any]],
        batch_size: int,
        max_workers: int,
        stats: Dict[str, Any],
    ) -> None:
        """Preserve mode: Only update embeddings if track data has significantly changed."""
        tracks_to_update = []
        tracks_to_preserve = []

        for track in tracks:
            if self._should_update_embedding(track):
                tracks_to_update.append(track)
            else:
                tracks_to_preserve.append(track)

        logger.info(
            f"Preserve mode: {len(tracks_to_update)} tracks need updates, {len(tracks_to_preserve)} preserved"
        )

        # Update tracks that need it
        if tracks_to_update:
            track_ids = [t["id"] for t in tracks_to_update]
            self._clear_embeddings_for_tracks(track_ids)
            updated_count = self.embedder.embed_tracks_batch(
                tracks_to_update, batch_size=batch_size, max_workers=max_workers
            )
            stats["updated"] = updated_count
            stats["skipped"] = len(tracks_to_update) - updated_count

        # Count preserved tracks
        stats["preserved"] = len(tracks_to_preserve)

    def _update_embeddings_hybrid_mode(
        self,
        tracks: List[Dict[str, Any]],
        batch_size: int,
        max_workers: int,
        stats: Dict[str, Any],
    ) -> None:
        """Hybrid mode: Combine existing embeddings with new metadata."""
        tracks_with_embeddings = []
        tracks_without_embeddings = []

        # Separate tracks with and without existing embeddings
        for track in tracks:
            existing_embedding = self._get_existing_embedding(track["id"])
            if existing_embedding is not None:
                tracks_with_embeddings.append((track, existing_embedding))
            else:
                tracks_without_embeddings.append(track)

        logger.info(
            f"Hybrid mode: {len(tracks_with_embeddings)} tracks with existing embeddings, {len(tracks_without_embeddings)} new"
        )

        # Process tracks with existing embeddings (hybrid approach)
        if tracks_with_embeddings:
            hybrid_count = self._create_hybrid_embeddings(
                tracks_with_embeddings, batch_size, max_workers
            )
            stats["updated"] = hybrid_count

        # Process tracks without embeddings (normal approach)
        if tracks_without_embeddings:
            track_ids = [t["id"] for t in tracks_without_embeddings]
            self._clear_embeddings_for_tracks(track_ids)
            new_count = self.embedder.embed_tracks_batch(
                tracks_without_embeddings,
                batch_size=batch_size,
                max_workers=max_workers,
            )
            stats["updated"] += new_count
            stats["skipped"] = len(tracks_without_embeddings) - new_count

    def _should_update_embedding(self, track: Dict[str, Any]) -> bool:
        """Check if a track's embedding should be updated based on data changes."""
        # Get the existing embedding to check if it's recent
        existing_embedding = self._get_existing_embedding(track["id"])
        if existing_embedding is None:
            return True  # No existing embedding, needs one

        # For now, always update if we have an existing embedding
        # This is a simple heuristic - you might want to make this more sophisticated
        # by comparing actual field values with what was used to create the original embedding
        return True

    def _get_existing_embedding(self, track_id: int) -> Optional[np.ndarray]:
        """Get existing embedding for a track."""
        try:
            cursor = self.embedder.conn.cursor()
            cursor.execute(GET_TRACK_EMBEDDING, (track_id,))
            result = cursor.fetchone()
            if result and result[0]:
                return np.frombuffer(result[0], dtype=np.float32)
            return None
        except Exception as e:
            logger.warning(
                f"Error getting existing embedding for track {track_id}: {e}"
            )
            return None

    def _create_hybrid_embeddings(
        self,
        tracks_with_embeddings: List[Tuple[Dict[str, Any], np.ndarray]],
        batch_size: int,
        max_workers: int,
    ) -> int:
        """Create hybrid embeddings by combining existing embeddings with new metadata."""
        successful_count: int = 0

        if not tracks_with_embeddings:
            return successful_count

        for track, existing_embedding in tracks_with_embeddings:
            try:
                # Create new embedding text with updated metadata
                new_embedding_text = self.embedder.create_track_embedding_text(track)

                # Get new embedding from the service
                new_embedding = self.embedder.embedding_service.get_embedding(
                    new_embedding_text
                )

                if new_embedding is not None:
                    # Combine old and new embeddings (simple average - you could use more sophisticated methods)
                    hybrid_embedding = (existing_embedding + new_embedding) / 2.0

                    # Store the hybrid embedding
                    try:
                        self._store_hybrid_embedding(
                            track["id"], hybrid_embedding, new_embedding_text
                        )
                        successful_count += 1
                        logger.debug(
                            f"Created hybrid embedding for track {track['id']}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error storing hybrid embedding for track {track['id']}: {e}"
                        )
            except Exception as e:
                logger.error(
                    f"Error creating hybrid embedding for track {track['id']}: {e}"
                )

        return successful_count

    def _store_hybrid_embedding(
        self, track_id: int, embedding: np.ndarray, embedding_text: str
    ) -> None:
        """Store a hybrid embedding in the database."""
        cursor = self.embedder.conn.cursor()
        embedding_blob = embedding.tobytes()
        cursor.execute(
            INSERT_OR_REPLACE_TRACK_EMBEDDING,
            (track_id, embedding_blob, embedding_text),
        )
        self.embedder.conn.commit()

    def _get_tracks_by_ids(self, track_ids: List[int]) -> List[Dict[str, Any]]:
        if not track_ids:
            return []
        cursor = self.embedder.conn.cursor()
        query = build_get_tracks_by_ids_query(track_ids)
        cursor.execute(query, track_ids)
        return [dict(row) for row in cursor.fetchall()]

    def _clear_embeddings_for_tracks(self, track_ids: List[int]) -> None:
        if not track_ids:
            return
        cursor = self.embedder.conn.cursor()
        query = build_delete_embeddings_by_ids_query(track_ids)
        cursor.execute(query, track_ids)
        self.embedder.conn.commit()
        logger.info(f"🗑️  Cleared {cursor.rowcount} existing embeddings.")

    def close(self) -> None:
        pass


def parse_ids_from_file(file_path: str) -> List[int]:
    ids = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.isdigit():
                ids.append(int(line))
    return ids


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Update embeddings for given track IDs."
    )
    parser.add_argument("ids", nargs="*", type=int, help="Track IDs to update")
    parser.add_argument(
        "--file", type=str, help="File containing track IDs (one per line)"
    )
    parser.add_argument(
        "--db", type=str, default="music_library.db", help="Path to SQLite DB"
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--mode",
        type=str,
        default="preserve",
        choices=["preserve", "hybrid"],
        help="Update mode: preserve (only if needed), hybrid (combine old+new)",
    )
    args = parser.parse_args()

    track_ids = args.ids or []
    if args.file:
        track_ids += parse_ids_from_file(args.file)
    if not track_ids:
        print("No track IDs provided. Use positional args or --file.")
        sys.exit(1)
    updater = EmbeddingUpdater(args.db)
    try:
        stats = updater.update_embeddings_for_tracks(
            track_ids,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            mode=args.mode,
        )
        print("Embedding update completed successfully!")
        print(f"Statistics: {stats}")
    except Exception as e:
        logger.error(f"Embedding update failed: {e}")
        sys.exit(1)
    finally:
        updater.close()


if __name__ == "__main__":
    main()
