#!/usr/bin/env python3
"""
Embedding updater for tracks modified during CSV ingestion.
Updates embeddings for tracks with new play count and rating data.
Supports both preservation mode and hybrid mode for handling existing embeddings.
"""

import argparse
import logging
import sys
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

from tonal_hortator.core.database.queries import (
    GET_EMBEDDING_INFO,
    GET_TRACK_EMBEDDING,
    INSERT_OR_REPLACE_TRACK_EMBEDDING,
)
from tonal_hortator.core.database.query_helpers import (
    build_delete_embeddings_by_ids_query,
    build_get_tracks_by_ids_query,
)
from tonal_hortator.core.embeddings.track_embedder import LocalTrackEmbedder
from tonal_hortator.core.models import Track

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class HybridStrategy(Enum):
    """Available strategies for combining embeddings in hybrid mode."""

    SIMPLE_AVERAGE = "simple_average"
    AGE_WEIGHTED = "age_weighted"
    METADATA_WEIGHTED = "metadata_weighted"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    EXPONENTIAL_DECAY = "exponential_decay"


class EmbeddingUpdater:
    """
    Update embeddings for tracks in the database.
    Supports both preservation mode and hybrid mode for handling existing embeddings.
    """

    def __init__(self, db_path: str = "music_library.db"):
        """Initialize the embedding updater."""
        self.embedder = LocalTrackEmbedder(db_path)

    def update_embeddings_for_tracks(
        self,
        track_ids: List[int],
        batch_size: int = 500,
        max_workers: int = 4,
        mode: str = "preserve",  # "preserve" or "hybrid"
        hybrid_strategy: Union[str, HybridStrategy] = HybridStrategy.AGE_WEIGHTED,
        hybrid_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Update embeddings for the given track IDs.

        Args:
            track_ids: List of track IDs to update
            batch_size: Number of tracks to process in each batch
            max_workers: Maximum number of worker threads
            mode: Update mode ("preserve" or "hybrid")
            hybrid_strategy: Strategy for combining embeddings in hybrid mode
            hybrid_config: Configuration for the hybrid strategy

        Returns:
            Dictionary with update statistics
        """
        if not track_ids:
            return {"updated": 0, "preserved": 0, "hybrid": 0, "errors": 0}

        # Normalize hybrid strategy
        if isinstance(hybrid_strategy, str):
            try:
                hybrid_strategy = HybridStrategy(hybrid_strategy)
            except ValueError:
                logger.warning(
                    f"Invalid hybrid strategy '{hybrid_strategy}', using AGE_WEIGHTED"
                )
                hybrid_strategy = HybridStrategy.AGE_WEIGHTED

        # Set default hybrid config
        if hybrid_config is None:
            hybrid_config = self._get_default_hybrid_config(hybrid_strategy)

        logger.info(
            f"🔄 Updating embeddings for {len(track_ids)} tracks in {mode} mode"
            + (f" with {hybrid_strategy.value} strategy" if mode == "hybrid" else "")
        )

        stats: Dict[str, Any] = {
            "total_tracks": len(track_ids),
            "updated": 0,
            "preserved": 0,
            "hybrid": 0,
            "errors": 0,
            "mode": mode,
            "hybrid_strategy": hybrid_strategy.value if mode == "hybrid" else None,
        }

        # Process tracks in batches
        for i in range(0, len(track_ids), batch_size):
            batch_ids = track_ids[i : i + batch_size]
            batch_tracks = self._get_tracks_by_ids(batch_ids)

            if mode == "preserve":
                self._update_embeddings_preserve_mode(
                    batch_tracks, batch_size, max_workers, stats
                )
            elif mode == "hybrid":
                self._update_embeddings_hybrid_mode(
                    batch_tracks,
                    batch_size,
                    max_workers,
                    stats,
                    hybrid_strategy,
                    hybrid_config,
                )

        logger.info(f"✅ Embedding update complete: {stats}")
        return stats

    def _get_default_hybrid_config(self, strategy: HybridStrategy) -> Dict[str, Any]:
        """Get default configuration for a hybrid strategy."""
        configs = {
            HybridStrategy.SIMPLE_AVERAGE: {
                "new_weight": 0.5,
                "old_weight": 0.5,
            },
            HybridStrategy.AGE_WEIGHTED: {
                "age_decay_days": 365,  # Full decay after 1 year
                "min_old_weight": 0.1,  # Minimum weight for old embeddings
                "max_old_weight": 0.8,  # Maximum weight for old embeddings
            },
            HybridStrategy.METADATA_WEIGHTED: {
                "completeness_threshold": 0.7,  # Threshold for "complete" metadata
                "new_weight_bonus": 0.2,  # Bonus weight for new metadata
                "base_new_weight": 0.5,
            },
            HybridStrategy.CONFIDENCE_WEIGHTED: {
                "play_count_weight": 0.3,
                "rating_weight": 0.3,
                "metadata_completeness_weight": 0.4,
                "min_confidence": 0.1,
            },
            HybridStrategy.EXPONENTIAL_DECAY: {
                "half_life_days": 180,  # Half-life of embedding relevance
                "min_weight": 0.05,
                "max_weight": 0.95,
            },
        }
        return configs.get(strategy, configs[HybridStrategy.AGE_WEIGHTED])

    def _update_embeddings_preserve_mode(
        self,
        tracks: List[Track],
        batch_size: int,
        max_workers: int,
        stats: Dict[str, Any],
    ) -> None:
        """Preserve mode: Only update embeddings if metadata has changed significantly."""
        tracks_to_update = []
        tracks_to_preserve = []

        for track in tracks:
            if self._should_update_embedding(track):
                tracks_to_update.append(track)
            else:
                tracks_to_preserve.append(track)

        # Update tracks that need updating
        if tracks_to_update:
            updated_count = self.embedder.embed_tracks_batch(
                tracks_to_update, max_workers=max_workers
            )
            stats["updated"] += updated_count

        stats["preserved"] += len(tracks_to_preserve)

    def _update_embeddings_hybrid_mode(
        self,
        tracks: List[Track],
        batch_size: int,
        max_workers: int,
        stats: Dict[str, Any],
        strategy: HybridStrategy,
        config: Dict[str, Any],
    ) -> None:
        """Hybrid mode: Combine existing embeddings with new metadata."""
        tracks_with_embeddings = []
        tracks_without_embeddings = []

        # Separate tracks with and without existing embeddings
        for track in tracks:
            if track.id is None:
                tracks_without_embeddings.append(track)
                continue
            existing_embedding = self._get_existing_embedding(track.id)
            if existing_embedding is not None:
                tracks_with_embeddings.append((track, existing_embedding))
            else:
                tracks_without_embeddings.append(track)

        # Create new embeddings for tracks without existing ones
        if tracks_without_embeddings:
            new_count = self.embedder.embed_tracks_batch(
                tracks_without_embeddings, max_workers=max_workers
            )
            stats["updated"] += new_count

        # Create hybrid embeddings for tracks with existing ones
        if tracks_with_embeddings:
            logger.info(
                f"Hybrid mode: {len(tracks_with_embeddings)} tracks with existing embeddings, {len(tracks_without_embeddings)} new"
            )
            hybrid_count = self._create_hybrid_embeddings(
                tracks_with_embeddings, batch_size, max_workers, strategy, config
            )
            stats["hybrid"] += hybrid_count

    def _should_update_embedding(self, track: Track) -> bool:
        """Determine if an embedding should be updated based on metadata changes."""
        if track.id is None:
            return True
        embedding_info = self._get_embedding_info(track.id)
        if not embedding_info:
            return True  # No existing embedding, create new one

        return self._has_metadata_changed(track, embedding_info)

    def _get_embedding_info(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Get embedding information for a track."""
        try:
            cursor = self.embedder.conn.cursor()
            cursor.execute(GET_EMBEDDING_INFO, (track_id,))
            result = cursor.fetchone()
            if result:
                return {"embedding_text": result[0], "created_at": result[1]}
            return None
        except Exception as e:
            logger.warning(f"Error getting embedding info for track {track_id}: {e}")
            return None

    def _has_metadata_changed(
        self, track: Track, embedding_info: Dict[str, Any]
    ) -> bool:
        """Check if metadata has changed significantly since the embedding was created."""
        # Key fields to check for changes
        basic_fields = ["name", "artist", "album", "genre", "year"]
        musical_fields = ["bpm", "musical_key", "key_scale", "mood"]
        production_fields = ["label", "producer", "arranger", "lyricist"]
        engagement_fields = ["play_count"]

        key_fields = (
            basic_fields + musical_fields + production_fields + engagement_fields
        )

        # Extract current metadata for comparison
        current_metadata = {
            field: str(getattr(track, field, "")).lower().strip()
            for field in key_fields
        }

        # Try to extract metadata from embedding text (if it contains metadata)
        # For now, we'll use a simple heuristic: if embedding is older than 30 days, update it
        try:
            created_at = datetime.fromisoformat(
                embedding_info["created_at"].replace("Z", "+00:00")
            )
            age_days = (datetime.now() - created_at).days

            # Update if embedding is older than 30 days
            if age_days > 30:
                logger.debug(
                    f"Track {track.id} embedding is {age_days} days old, updating"
                )
                return True

            # Update if any key metadata fields are missing or significantly different
            embedding_text = embedding_info.get("embedding_text", "").lower()

            # Check if key metadata appears in embedding text
            for field, value in current_metadata.items():
                if value and value not in embedding_text:
                    logger.debug(
                        f"Track {track.id} missing {field} in embedding, updating"
                    )
                    return True

            logger.debug(f"Track {track.id} metadata unchanged, preserving embedding")
            return False

        except Exception as e:
            logger.warning(f"Error checking metadata changes for track {track.id}: {e}")
            return True  # Update to be safe

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
        tracks_with_embeddings: List[Tuple[Track, np.ndarray]],
        batch_size: int,
        max_workers: int,
        strategy: HybridStrategy,
        config: Dict[str, Any],
    ) -> int:
        """Create hybrid embeddings using the specified strategy."""
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
                    # Calculate weights based on strategy
                    old_weight, new_weight = self._calculate_weights(
                        track, strategy, config
                    )

                    # Combine embeddings using calculated weights
                    hybrid_embedding = (
                        old_weight * existing_embedding + new_weight * new_embedding
                    )

                    # Store the hybrid embedding
                    try:
                        if track.id is not None:
                            self._store_hybrid_embedding(
                                track.id, hybrid_embedding, new_embedding_text
                            )
                        successful_count += 1
                        logger.debug(
                            f"Created hybrid embedding for track {track.id} "
                            f"(weights: old={old_weight:.2f}, new={new_weight:.2f})"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error storing hybrid embedding for track {track.id}: {e}"
                        )
            except Exception as e:
                logger.error(
                    f"Error creating hybrid embedding for track {track.id}: {e}"
                )

        return successful_count

    def _calculate_weights(
        self,
        track: Track,
        strategy: HybridStrategy,
        config: Dict[str, Any],
    ) -> Tuple[float, float]:
        """Calculate weights for combining old and new embeddings."""
        if strategy == HybridStrategy.SIMPLE_AVERAGE:
            return config["old_weight"], config["new_weight"]

        elif strategy == HybridStrategy.AGE_WEIGHTED:
            return self._calculate_age_weights(track, config)

        elif strategy == HybridStrategy.METADATA_WEIGHTED:
            return self._calculate_metadata_weights(track, config)

        elif strategy == HybridStrategy.CONFIDENCE_WEIGHTED:
            return self._calculate_confidence_weights(track, config)

        elif strategy == HybridStrategy.EXPONENTIAL_DECAY:
            return self._calculate_exponential_decay_weights(track, config)

        # This should never be reached, but mypy needs it for type safety
        assert False, f"Unhandled strategy: {strategy}"
        return 0.5, 0.5  # type: ignore

    def _calculate_age_weights(
        self, track: Track, config: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Calculate weights based on embedding age."""
        if track.id is None:
            return 0.0, 1.0
        embedding_info = self._get_embedding_info(track.id)
        if not embedding_info:
            return 0.0, 1.0  # No existing embedding, use only new

        try:
            created_at = datetime.fromisoformat(
                embedding_info["created_at"].replace("Z", "+00:00")
            )
            age_days = (datetime.now() - created_at).days

            # Calculate old weight based on age
            age_decay_days = config.get("age_decay_days", 365)
            min_old_weight = config.get("min_old_weight", 0.1)
            max_old_weight = config.get("max_old_weight", 0.8)

            # Linear decay from max_old_weight to min_old_weight over age_decay_days
            if age_days >= age_decay_days:
                old_weight = min_old_weight
            else:
                decay_factor = age_days / age_decay_days
                old_weight = (
                    max_old_weight - (max_old_weight - min_old_weight) * decay_factor
                )

            new_weight = 1.0 - old_weight
            return old_weight, new_weight

        except Exception as e:
            logger.warning(f"Error calculating age weights for track {track.id}: {e}")
            return 0.5, 0.5

    def _calculate_metadata_weights(
        self, track: Track, config: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Calculate weights based on metadata completeness."""
        # Key metadata fields to check
        key_fields = [
            "name",
            "artist",
            "album",
            "genre",
            "year",
            "bpm",
            "musical_key",
            "mood",
            "label",
            "producer",
            "play_count",
        ]

        # Calculate completeness of current metadata
        filled_fields = sum(1 for field in key_fields if getattr(track, field, None))
        completeness = filled_fields / len(key_fields)

        # Get existing embedding info to compare
        if track.id is None:
            return 0.0, 1.0
        embedding_info = self._get_embedding_info(track.id)
        if not embedding_info:
            return 0.0, 1.0

        # Check if new metadata is more complete
        embedding_text = embedding_info.get("embedding_text", "").lower()
        old_completeness = sum(
            1
            for field in key_fields
            if str(getattr(track, field, "")).lower() in embedding_text
        ) / len(key_fields)

        # Calculate weights
        base_new_weight = config.get("base_new_weight", 0.5)
        new_weight_bonus = config.get("new_weight_bonus", 0.2)
        completeness_threshold = config.get("completeness_threshold", 0.7)

        if completeness > old_completeness and completeness > completeness_threshold:
            new_weight = base_new_weight + new_weight_bonus
        else:
            new_weight = base_new_weight

        new_weight = min(0.9, max(0.1, new_weight))  # Clamp between 0.1 and 0.9
        old_weight = 1.0 - new_weight

        return old_weight, new_weight

    def _calculate_confidence_weights(
        self, track: Track, config: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Calculate weights based on confidence indicators."""
        # Calculate confidence scores for old and new embeddings
        old_confidence = self._calculate_old_embedding_confidence(track, config)
        new_confidence = self._calculate_new_embedding_confidence(track, config)

        # Normalize to sum to 1.0
        total_confidence = old_confidence + new_confidence
        if total_confidence == 0:
            return 0.5, 0.5

        min_confidence = config.get("min_confidence", 0.1)
        old_weight = max(min_confidence, old_confidence / total_confidence)
        new_weight = max(min_confidence, new_confidence / total_confidence)

        # Renormalize
        total_weight = old_weight + new_weight
        old_weight /= total_weight
        new_weight /= total_weight

        return old_weight, new_weight

    def _calculate_old_embedding_confidence(
        self, track: Track, config: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for existing embedding."""
        confidence = 0.0

        # Play count confidence (more plays = higher confidence)
        play_count = track.play_count or 0
        if play_count > 0:
            play_count_weight = config.get("play_count_weight", 0.3)
            confidence += play_count_weight * min(1.0, play_count / 100.0)

        # Rating confidence (higher ratings = higher confidence)
        avg_rating = track.avg_rating or 0
        if avg_rating > 0:
            rating_weight = config.get("rating_weight", 0.3)
            confidence += rating_weight * (avg_rating / 5.0)

        # Age confidence (newer = higher confidence, but with diminishing returns)
        if track.id is None:
            return 0.0
        embedding_info = self._get_embedding_info(track.id)
        if embedding_info:
            try:
                created_at = datetime.fromisoformat(
                    embedding_info["created_at"].replace("Z", "+00:00")
                )
                age_days = (datetime.now() - created_at).days
                # Exponential decay: newer embeddings get higher confidence
                age_confidence = np.exp(-age_days / 365.0)  # Decay over 1 year
                confidence += 0.4 * age_confidence
            except Exception:
                confidence += 0.2  # Default confidence for old embeddings

        return confidence

    def _calculate_new_embedding_confidence(
        self, track: Track, config: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for new embedding."""
        confidence = 0.0

        # Metadata completeness confidence
        key_fields = [
            "name",
            "artist",
            "album",
            "genre",
            "year",
            "bpm",
            "musical_key",
            "mood",
        ]
        filled_fields = sum(1 for field in key_fields if getattr(track, field, None))
        completeness = filled_fields / len(key_fields)

        metadata_weight = float(config.get("metadata_completeness_weight", 0.4))
        confidence += metadata_weight * completeness

        # Fresh data confidence (new metadata is always "fresh")
        confidence += 0.3

        # Quality indicators
        if (track.play_count or 0) > 0:
            confidence += 0.1
        if (track.avg_rating or 0) > 0:
            confidence += 0.1

        return confidence

    def _calculate_exponential_decay_weights(
        self, track: Track, config: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Calculate weights using exponential decay based on age."""
        if track.id is None:
            return 0.0, 1.0
        embedding_info = self._get_embedding_info(track.id)
        if not embedding_info:
            return 0.0, 1.0

        try:
            created_at = datetime.fromisoformat(
                embedding_info["created_at"].replace("Z", "+00:00")
            )
            age_days = (datetime.now() - created_at).days

            half_life_days = config.get("half_life_days", 180)
            min_weight = config.get("min_weight", 0.05)
            max_weight = config.get("max_weight", 0.95)

            # Exponential decay: weight = max_weight * (0.5 ^ (age / half_life))
            decay_factor = 0.5 ** (age_days / half_life_days)
            old_weight = max_weight * decay_factor
            old_weight = max(min_weight, min(max_weight, old_weight))

            new_weight = 1.0 - old_weight
            return old_weight, new_weight

        except Exception as e:
            logger.warning(
                f"Error calculating exponential decay weights for track {track.id}: {e}"
            )
            return 0.5, 0.5

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

    def _get_tracks_by_ids(self, track_ids: List[int]) -> List[Track]:
        if not track_ids:
            return []
        cursor = self.embedder.conn.cursor()
        query = build_get_tracks_by_ids_query(track_ids)
        cursor.execute(query, track_ids)

        # Use batch constructor for better performance
        rows = cursor.fetchall()
        data_list = [dict(row) for row in rows]
        return Track.from_dict_batch(data_list)

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
    parser.add_argument(
        "--hybrid-strategy",
        type=str,
        default="age_weighted",
        choices=[s.value for s in HybridStrategy],
        help="Strategy for combining embeddings in hybrid mode",
    )
    parser.add_argument(
        "--age-decay-days",
        type=int,
        default=365,
        help="Days for full age decay (age_weighted strategy)",
    )
    parser.add_argument(
        "--half-life-days",
        type=int,
        default=180,
        help="Half-life in days for exponential decay strategy",
    )
    parser.add_argument(
        "--min-old-weight",
        type=float,
        default=0.1,
        help="Minimum weight for old embeddings",
    )
    parser.add_argument(
        "--max-old-weight",
        type=float,
        default=0.8,
        help="Maximum weight for old embeddings",
    )
    args = parser.parse_args()

    track_ids = args.ids or []
    if args.file:
        track_ids += parse_ids_from_file(args.file)
    if not track_ids:
        print("No track IDs provided. Use positional args or --file.")
        sys.exit(1)

    # Build hybrid configuration based on strategy
    hybrid_config = None
    if args.mode == "hybrid":
        hybrid_config = {
            "age_decay_days": args.age_decay_days,
            "half_life_days": args.half_life_days,
            "min_old_weight": args.min_old_weight,
            "max_old_weight": args.max_old_weight,
        }

    updater = EmbeddingUpdater(args.db)
    try:
        stats = updater.update_embeddings_for_tracks(
            track_ids,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            mode=args.mode,
            hybrid_strategy=args.hybrid_strategy,
            hybrid_config=hybrid_config,
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
