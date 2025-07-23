"""
Playlist deduplication functionality for Tonal Hortator.

Handles deduplication logic for playlists using multiple strategies.
"""

import logging
import secrets
from typing import Any, Callable, List, Optional, cast

from tonal_hortator.core.models import Track

logger = logging.getLogger(__name__)


class PlaylistDeduplicator:
    """Handles deduplication logic for playlists."""

    def __init__(self) -> None:
        pass

    def filter_and_deduplicate_results(
        self,
        results: List[Track],
        min_similarity: float,
        max_tracks: int,
        is_artist_specific: bool,
        max_artist_ratio: float = 0.3,
        sample_with_randomization: Optional[Callable] = None,
        smart_name_deduplication: Optional[Callable] = None,
        enforce_artist_diversity: Optional[Callable] = None,
        distribute_artists: Optional[Callable] = None,
        logger: Any = None,
    ) -> List[Track]:
        """Filter results by similarity and remove duplicates using multiple strategies"""
        from .playlist_utils import normalize_file_location_static

        # Filter by similarity threshold
        filtered = [
            track
            for track in results
            if (track.similarity_score or 0) >= min_similarity
        ]

        logger and logger and logger.info(
            f"🔍 Starting deduplication on {len(filtered)} tracks (similarity ≥ {min_similarity})"
        )

        # Strategy 1: Deduplicate by file location (normalized path)
        seen_locations = set()
        location_deduplicated = []

        for track in filtered:
            location = normalize_file_location_static(track.location or "")
            if location and location not in seen_locations:
                seen_locations.add(location)
                location_deduplicated.append(track)
            elif not location:  # Include tracks without location
                location_deduplicated.append(track)

        logger and logger and logger.info(
            f"📍 Location deduplication: {len(filtered)} → {len(location_deduplicated)} tracks"
        )

        # Strategy 2: Deduplicate by title/artist combination
        seen_combinations = set()
        combination_deduplicated = []

        for track in location_deduplicated:
            title = (track.name or "").strip().lower()
            artist = (track.artist or "").strip().lower()
            combination = f"{title}|{artist}"

            if combination and combination not in seen_combinations:
                seen_combinations.add(combination)
                combination_deduplicated.append(track)
            elif not combination:  # Include tracks without title/artist
                combination_deduplicated.append(track)

        logger and logger.info(
            f"🎵 Title/Artist deduplication: {len(location_deduplicated)} → {len(combination_deduplicated)} tracks"
        )

        # Strategy 3: Deduplicate by track ID (shouldn't happen, but just in case)
        seen_ids = set()
        final_deduplicated = []

        for track in combination_deduplicated:
            track_id = track.id
            if track_id not in seen_ids:
                seen_ids.add(track_id)
                final_deduplicated.append(track)

        logger and logger.info(
            f"🆔 Track ID deduplication: {len(combination_deduplicated)} → {len(final_deduplicated)} tracks"
        )

        # Strategy 4: Smart deduplication for similar titles with slight variations
        if is_artist_specific:
            # For artist-specific queries, skip smart name deduplication
            smart_deduplicated = final_deduplicated
        else:
            # For general queries, apply smart name deduplication
            if smart_name_deduplication is not None:
                smart_deduplicated = smart_name_deduplication(final_deduplicated)
            else:
                smart_deduplicated = final_deduplicated

        logger and logger.info(
            f"🧠 Smart name deduplication: {len(final_deduplicated)} → {len(smart_deduplicated)} tracks"
        )

        # Strategy 5: Enforce artist diversity
        if is_artist_specific:
            logger and logger.info(
                "🎤 Artist-specific query detected, skipping artist diversity enforcement"
            )
            diverse_tracks = smart_deduplicated
        else:
            max_tracks_per_artist = max(2, int(max_tracks * max_artist_ratio))
            logger and logger and logger.info(
                f"🎵 General query, allowing up to {max_tracks_per_artist} tracks per artist (ratio {max_artist_ratio})"
            )
            if enforce_artist_diversity is not None:
                diverse_tracks = enforce_artist_diversity(
                    smart_deduplicated, max_tracks, max_tracks_per_artist
                )
            else:
                diverse_tracks = smart_deduplicated

        logger and logger and logger.info(
            f"✅ Final deduplication summary: {len(results)} → {len(diverse_tracks)} tracks"
        )

        # After artist diversity, use weighted random sampling for variance
        # Ensure we return exactly max_tracks
        top_k = max_tracks * 10
        if sample_with_randomization is not None:
            candidates = sample_with_randomization(diverse_tracks, top_k)
        else:
            candidates = diverse_tracks[:top_k]

        if len(candidates) < max_tracks:
            logger and logger and logger.warning(
                f"⚠️ Only {len(candidates)} tracks available, requested {max_tracks}"
            )
            return cast(List[Track], candidates[:max_tracks])

        # Assign weights proportional to similarity score (shifted to be >=0)
        min_similarity_score = min(
            (t.get("similarity_score", 0) for t in candidates), default=0
        )
        weights = [
            max(0.0, t.get("similarity_score", 0) - min_similarity_score + 1e-6)
            for t in candidates
        ]

        # Sample exactly max_tracks unique tracks
        selected = set()
        final_tracks: List[Track] = []
        attempts = 0
        max_attempts = top_k * 3
        rng = secrets.SystemRandom()

        while len(final_tracks) < max_tracks and attempts < max_attempts:
            if weights and any(w > 0 for w in weights):
                pick = rng.choices(candidates, weights=weights, k=1)[0]
            else:
                pick = rng.choice(candidates)

            pick_id = (
                pick.get("id") or f"{pick.get('name', '')}-{pick.get('artist', '')}"
            )
            if pick_id not in selected:
                final_tracks.append(pick)
                selected.add(pick_id)
            attempts += 1

        if len(final_tracks) < max_tracks:
            logger and logger and logger.info(
                f"🎲 Weighted sampling gave {len(final_tracks)} tracks, filling with randomized remaining tracks"
            )
            remaining_candidates = [
                t
                for t in candidates
                if (t.get("id") or f"{t.get('name', '')}-{t.get('artist', '')}")
                not in selected
            ]
            rng.shuffle(remaining_candidates)
            for t in remaining_candidates:
                t_id = t.get("id") or f"{t.get('name', '')}-{t.get('artist', '')}"
                if t_id not in selected:
                    final_tracks.append(t)
                    selected.add(t_id)
                if len(final_tracks) >= max_tracks:
                    break
        final_tracks = final_tracks[:max_tracks]
        logger and logger and logger.info(
            f"🎲 Final playlist: {len(final_tracks)} tracks (requested: {max_tracks})"
        )
        return final_tracks
