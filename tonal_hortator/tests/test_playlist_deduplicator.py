#!/usr/bin/env python3
"""
Tests for tonal_hortator.core.playlist.playlist_deduplicator
"""

import unittest
from typing import Any, Dict, List

from tonal_hortator.core.playlist.playlist_deduplicator import PlaylistDeduplicator


class TestPlaylistDeduplicator(unittest.TestCase):
    """Test PlaylistDeduplicator"""

    def test_filter_and_deduplicate_results(self) -> None:
        """Test filtering and deduplication of results"""
        deduplicator = PlaylistDeduplicator()

        tracks = [
            {"id": 1, "name": "Song1", "artist": "Artist1", "similarity_score": 0.8},
            {"id": 2, "name": "Song1", "artist": "Artist1", "similarity_score": 0.7},
            {"id": 3, "name": "Song2", "artist": "Artist2", "similarity_score": 0.6},
            {"id": 4, "name": "Song3", "artist": "Artist3", "similarity_score": 0.5},
        ]

        # Mock the required callback functions
        def mock_sample_with_randomization(
            tracks: List[Dict[str, Any]], top_k: int
        ) -> List[Dict[str, Any]]:
            return tracks[:top_k]

        def mock_smart_name_deduplication(
            tracks: List[Dict[str, Any]],
        ) -> List[Dict[str, Any]]:
            # Simple deduplication: keep only one track per name
            seen_names = set()
            deduplicated = []
            for track in tracks:
                name = track.get("name", "")
                if name not in seen_names:
                    seen_names.add(name)
                    deduplicated.append(track)
            return deduplicated

        def mock_enforce_artist_diversity(
            tracks: List[Dict[str, Any]], max_tracks: int, max_per_artist: int
        ) -> List[Dict[str, Any]]:
            return tracks[:max_tracks]

        def mock_distribute_artists(
            tracks: List[Dict[str, Any]], max_tracks: int
        ) -> List[Dict[str, Any]]:
            return tracks[:max_tracks]

        result = deduplicator.filter_and_deduplicate_results(
            tracks,
            min_similarity=0.5,
            max_tracks=3,
            is_artist_specific=False,
            sample_with_randomization=mock_sample_with_randomization,
            smart_name_deduplication=mock_smart_name_deduplication,
            enforce_artist_diversity=mock_enforce_artist_diversity,
            distribute_artists=mock_distribute_artists,
        )

        # Should deduplicate and limit to max_tracks
        self.assertLessEqual(len(result), 3)

    def test_filter_and_deduplicate_results_artist_specific(self) -> None:
        """Test filtering and deduplication for artist-specific queries"""
        deduplicator = PlaylistDeduplicator()

        tracks = [
            {"id": 1, "name": "Song1", "artist": "Artist1", "similarity_score": 0.8},
            {"id": 2, "name": "Song2", "artist": "Artist1", "similarity_score": 0.7},
        ]

        def mock_sample_with_randomization(
            tracks: List[Dict[str, Any]], top_k: int
        ) -> List[Dict[str, Any]]:
            return tracks[:top_k]

        result = deduplicator.filter_and_deduplicate_results(
            tracks,
            min_similarity=0.5,
            max_tracks=2,
            is_artist_specific=True,
            sample_with_randomization=mock_sample_with_randomization,
        )

        # Should return all tracks for artist-specific queries
        self.assertEqual(len(result), 2)

    def test_filter_and_deduplicate_results_min_similarity(self) -> None:
        """Test filtering by minimum similarity score"""
        deduplicator = PlaylistDeduplicator()

        tracks = [
            {"id": 1, "name": "Song1", "artist": "Artist1", "similarity_score": 0.8},
            {
                "id": 2,
                "name": "Song2",
                "artist": "Artist2",
                "similarity_score": 0.3,
            },  # Below threshold
            {"id": 3, "name": "Song3", "artist": "Artist3", "similarity_score": 0.6},
        ]

        def mock_sample_with_randomization(
            tracks: List[Dict[str, Any]], top_k: int
        ) -> List[Dict[str, Any]]:
            return tracks[:top_k]

        result = deduplicator.filter_and_deduplicate_results(
            tracks,
            min_similarity=0.5,
            max_tracks=3,
            is_artist_specific=False,
            sample_with_randomization=mock_sample_with_randomization,
        )

        # Should filter out tracks below min_similarity
        for track in result:
            self.assertGreaterEqual(track.get("similarity_score", 0), 0.5)

    def test_filter_and_deduplicate_results_empty_input(self) -> None:
        """Test filtering and deduplication with empty input"""
        deduplicator = PlaylistDeduplicator()

        result = deduplicator.filter_and_deduplicate_results(
            [],
            min_similarity=0.5,
            max_tracks=3,
            is_artist_specific=False,
        )

        # Should return empty list
        self.assertEqual(result, [])

    def test_filter_and_deduplicate_results_no_callbacks(self) -> None:
        """Test filtering and deduplication with no callback functions"""
        deduplicator = PlaylistDeduplicator()

        tracks = [
            {"id": 1, "name": "Song1", "artist": "Artist1", "similarity_score": 0.8},
            {"id": 2, "name": "Song2", "artist": "Artist2", "similarity_score": 0.7},
        ]

        result = deduplicator.filter_and_deduplicate_results(
            tracks,
            min_similarity=0.5,
            max_tracks=2,
            is_artist_specific=False,
        )

        # Should still work with None callbacks (fallback behavior)
        self.assertLessEqual(len(result), 2)
        # All tracks should meet similarity threshold
        for track in result:
            self.assertGreaterEqual(track.get("similarity_score", 0), 0.5)


if __name__ == "__main__":
    unittest.main()
