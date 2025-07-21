"""
Playlist filtering functionality for Tonal Hortator.

Handles genre filtering, boosting, and other filtering logic for playlists.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PlaylistFilter:
    """Handles filtering logic for playlists (genre, similarity, etc.)."""

    def __init__(self) -> None:
        pass

    def apply_genre_filtering(
        self,
        genres: List[str],
        results: List[Dict[str, Any]],
        logger: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        if not results:
            return results

        genre_keywords = [g.lower() for g in genres if isinstance(g, str)]

        if not genre_keywords:
            logger and logger.info(
                "🎵 No specific genre detected in query, using semantic similarity only"
            )
            return results

        logger and logger.info(f"🎵 Detected genre keywords: {genre_keywords}")

        genre_matches = []
        other_tracks = []

        for track in results:
            track_genre = track.get("genre", "").lower() if track.get("genre") else ""
            similarity_score = track.get("similarity_score", 0)
            genre_match = any(keyword in track_genre for keyword in genre_keywords)
            if genre_match:
                boosted_score = min(1.0, similarity_score + 0.1)
                track["similarity_score"] = boosted_score
                track["genre_boosted"] = True
                genre_matches.append(track)
            else:
                track["genre_boosted"] = False
                other_tracks.append(track)
        max_other_tracks = max(5, len(genre_matches) // 2)
        combined_results = genre_matches + other_tracks[:max_other_tracks]
        logger and logger.info(
            f"🎵 Genre filtering: {len(genre_matches)} genre matches, {len(other_tracks[:max_other_tracks])} other tracks"
        )
        return combined_results
