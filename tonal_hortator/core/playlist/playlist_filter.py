"""
Playlist filtering functionality for Tonal Hortator.

Handles genre filtering, boosting, and track filtering logic.
"""

import logging
from typing import List

from tonal_hortator.core.config import get_config
from tonal_hortator.core.models import Track

logger = logging.getLogger(__name__)


class PlaylistFilter:
    """Handles playlist filtering and genre boosting operations."""

    def __init__(self) -> None:
        """Initialize the PlaylistFilter."""
        self.config = get_config()

    def apply_genre_filtering(
        self, genre_keywords: List[str], tracks: List[Track]
    ) -> List[Track]:
        """
        Apply genre filtering and boosting to tracks while maintaining diversity.

        Boosts similarity scores for tracks matching genre keywords, but also
        includes non-matching tracks to maintain playlist diversity.

        Args:
            genre_keywords: List of genre keywords to filter/boost
            tracks: List of track dictionaries

        Returns:
            Combined list of boosted genre matches and other tracks for diversity
        """
        if not genre_keywords:
            logger.info(
                "🎵 No specific genre detected in query, using semantic similarity only"
            )
            return tracks

        # Get genre boost score from configuration
        genre_boost_score: float = self.config.get("similarity.genre_boost_score", 0.1)
        max_score: float = self.config.get("similarity.perfect_match_score", 1.0)

        logger.info(f"🎸 Detected genre keywords: {genre_keywords}")

        genre_matches = []
        other_tracks = []

        # Separate tracks into genre matches and others
        for track in tracks:
            track_genre = (track.genre or "").lower()
            similarity_score = track.similarity_score or 0

            # Check if track genre matches any of the specified keywords
            genre_match = any(
                keyword.lower() in track_genre for keyword in genre_keywords
            )

            if genre_match:
                # Create a new Track with boosted similarity score
                boosted_score = min(max_score, similarity_score + genre_boost_score)
                track_dict = track.to_dict()
                track_dict["similarity_score"] = boosted_score
                track_dict["genre_boosted"] = True
                boosted_track = Track.from_dict(track_dict)
                genre_matches.append(boosted_track)
            else:
                # Keep track for diversity (no boosting)
                other_tracks.append(track)

        # Include genre matches plus some other tracks for diversity
        # Use at least 5 other tracks or half the number of genre matches
        max_other_tracks = max(5, len(genre_matches) // 2)
        combined_results = genre_matches + other_tracks[:max_other_tracks]

        logger.info(
            f"🎵 Genre filtering: {len(genre_matches)} genre matches, "
            f"{len(other_tracks[:max_other_tracks])} other tracks for diversity"
        )

        return combined_results
