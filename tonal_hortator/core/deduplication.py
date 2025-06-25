#!/usr/bin/env python3
"""
Deduplication utilities for playlist generation
Handles multiple deduplication strategies for track lists
"""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class TrackDeduplicator:
    """Handle track deduplication using multiple strategies"""

    def __init__(self) -> None:
        self.suffixes_to_remove = [
            " (remix)",
            " (remastered)",
            " (live)",
            " (acoustic)",
            " (radio edit)",
            " (extended)",
            " (clean)",
            " (explicit)",
            " (original mix)",
            " (club mix)",
            " (dub mix)",
            " - remix",
            " - remastered",
            " - live",
            " - acoustic",
        ]

    def deduplicate_tracks(
        self, tracks: List[Dict[str, Any]], min_similarity: float, max_tracks: int
    ) -> List[Dict[str, Any]]:
        """Apply comprehensive deduplication to track list"""
        # Filter by similarity threshold
        filtered = [
            track
            for track in tracks
            if track.get("similarity_score", 0) >= min_similarity
        ]

        logger.info(
            f"🔍 Starting deduplication on {len(filtered)} tracks (similarity ≥ {min_similarity})"
        )

        # Apply multiple deduplication strategies
        location_deduplicated = self._deduplicate_by_location(filtered)
        combination_deduplicated = self._deduplicate_by_name_artist(
            location_deduplicated
        )
        id_deduplicated = self._deduplicate_by_track_id(combination_deduplicated)
        smart_deduplicated = self._smart_name_deduplication(id_deduplicated)

        logger.info(
            f"✅ Final deduplication summary: {len(tracks)} → {len(smart_deduplicated)} tracks"
        )

        return smart_deduplicated

    def _deduplicate_by_location(
        self, tracks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicate by file location (normalized path)"""
        seen_locations = set()
        location_deduplicated = []

        for track in tracks:
            location = self._normalize_file_location(track.get("location", ""))
            if location and location not in seen_locations:
                seen_locations.add(location)
                location_deduplicated.append(track)
            elif not location:  # Include tracks without location
                location_deduplicated.append(track)

        logger.info(
            f"📍 Location deduplication: {len(tracks)} → {len(location_deduplicated)} tracks"
        )
        return location_deduplicated

    def _deduplicate_by_name_artist(
        self, tracks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicate by name/artist combination"""
        seen_combinations = set()
        deduplicated = []

        for track in tracks:
            name = track.get("name", "").strip().lower()
            artist = track.get("artist", "").strip().lower()
            combination = f"{name}|{artist}"

            if combination not in seen_combinations:
                seen_combinations.add(combination)
                deduplicated.append(track)
            elif not combination:  # Include tracks without name/artist
                deduplicated.append(track)

        logger.info(
            f"🎵 Name/Artist deduplication: {len(tracks)} → {len(deduplicated)} tracks"
        )
        return deduplicated

    def _deduplicate_by_track_id(
        self, tracks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicate by track ID (shouldn't happen, but just in case)"""
        seen_ids = set()
        final_deduplicated = []

        for track in tracks:
            track_id = track.get("id")
            if track_id not in seen_ids:
                seen_ids.add(track_id)
                final_deduplicated.append(track)

        logger.info(
            f"🆔 Track ID deduplication: {len(tracks)} → {len(final_deduplicated)} tracks"
        )
        return final_deduplicated

    def _smart_name_deduplication(
        self, tracks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Smart deduplication for names with slight variations (e.g., 'Song (Remix)' vs 'Song')"""
        if not tracks:
            return tracks

        # Group tracks by artist for more accurate deduplication
        artist_tracks: Dict[str, List[Dict[str, Any]]] = {}
        for track in tracks:
            artist = track.get("artist", "").strip().lower()
            if artist not in artist_tracks:
                artist_tracks[artist] = []
            artist_tracks[artist].append(track)

        deduplicated = []
        for artist, artist_track_list in artist_tracks.items():
            deduplicated.extend(
                self._deduplicate_artist_tracks_by_name(artist_track_list)
            )

        logger.info(
            f"🧠 Smart name deduplication: {len(tracks)} → {len(deduplicated)} tracks"
        )
        return deduplicated

    def _deduplicate_artist_tracks_by_name(
        self, tracks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicate tracks by the same artist using smart name matching"""
        if not tracks:
            return tracks

        # Sort by similarity score (descending) to keep the best matches
        tracks.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)

        processed_names = set()
        deduplicated = []

        for track in tracks:
            name = track.get("name", "").strip().lower()
            base_name = self._extract_base_name(name)

            if base_name and base_name not in processed_names:
                processed_names.add(base_name)
                best_track = self._find_best_track_for_base_name(tracks, base_name)
                deduplicated.append(best_track)
            elif not base_name:
                deduplicated.append(track)

        return deduplicated

    def _find_best_track_for_base_name(
        self, tracks: List[Dict[str, Any]], base_name: str
    ) -> Dict[str, Any]:
        """Find the track with the highest similarity score for a given base name"""
        best_track = None
        best_score = -1

        for track in tracks:
            name = track.get("name", "").strip().lower()
            track_base = self._extract_base_name(name)
            if track_base == base_name:
                score = track.get("similarity_score", 0)
                if score > best_score:
                    best_score = score
                    best_track = track

        return best_track or tracks[0]

    def _extract_base_name(self, name: str) -> str:
        """Extract base name by removing common suffixes and variations"""
        if not name:
            return ""

        # Remove common suffixes like (Remix), (Live), etc.
        base = re.sub(r"\s*\([^)]*\)\s*$", "", name)

        # Remove common suffixes
        for suffix in self.suffixes_to_remove:
            if base.lower().endswith(suffix.lower()):
                base = base[: -len(suffix)]
                break

        return base.strip()

    def _normalize_file_location(self, location: str) -> str:
        """Normalize file location for deduplication and Apple Music compatibility"""
        if not location:
            return ""

        # Convert to lowercase and normalize separators
        normalized = location.lower().replace("\\", "/")

        # Remove common OS-specific prefixes with username
        # Match /users/<username>/, /home/<username>/, c:/users/<username>/, etc.
        match = re.match(
            r"^(?:/users/|/home/|c:/users/|d:/users/|e:/users/)([^/]+)/(.+)$",
            normalized,
        )
        if match:
            return match.group(2)
        return normalized
