"""
Playlist filtering functionality for Tonal Hortator.

Handles genre filtering, boosting, and track filtering logic.
"""

import logging
from typing import Any, Dict, List

from tonal_hortator.core.config import get_config

logger = logging.getLogger(__name__)


class PlaylistFilter:
    """Handles playlist filtering and genre boosting operations."""

    def __init__(self) -> None:
        """Initialize the PlaylistFilter."""
        self.config = get_config()

    def apply_genre_filtering(
        self, genre_keywords: List[str], tracks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply genre filtering and boosting to tracks.

        Args:
            genre_keywords: List of genre keywords to filter/boost
            tracks: List of track dictionaries

        Returns:
            Filtered and boosted track list
        """
        if not genre_keywords:
            return tracks

        # Get genre boost score from configuration
        genre_boost_score: float = self.config.get("similarity.genre_boost_score", 0.1)

        logger.info(f"🎸 Applying genre filtering for: {genre_keywords}")

        # Filter tracks matching genre keywords and boost their scores
        matching_tracks = []
        for track in tracks:
            track_genre = track.get("genre", "").lower()

            # Check if track genre matches any of the specified keywords
            for keyword in genre_keywords:
                if keyword.lower() in track_genre:
                    # Create a copy and boost the similarity score
                    boosted_track = track.copy()
                    similarity_score = track.get("similarity_score", 0)
                    max_score: float = self.config.get(
                        "similarity.perfect_match_score", 1.0
                    )
                    boosted_score = min(max_score, similarity_score + genre_boost_score)
                    boosted_track["similarity_score"] = boosted_score
                    boosted_track["genre_boosted"] = True
                    matching_tracks.append(boosted_track)
                    break

        logger.info(
            f"🎵 Genre filtering: {len(tracks)} → {len(matching_tracks)} tracks"
        )
        return matching_tracks
