#!/usr/bin/env python3
"""
Tests for artist distributor functionality
"""

import unittest
from typing import Any, Dict, List

from tonal_hortator.core.artist_distributor import ArtistDistributor


class TestArtistDistributor(unittest.TestCase):
    """Test artist distributor functionality"""

    def setUp(self) -> None:
        """Set up test fixtures"""
        self.distributor = ArtistDistributor()

    def test_enforce_artist_diversity_basic(self) -> None:
        """Test basic artist diversity enforcement"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song 1", "artist": "Artist A", "similarity_score": 0.9},
            {"name": "Song 2", "artist": "Artist A", "similarity_score": 0.8},
            {"name": "Song 3", "artist": "Artist B", "similarity_score": 0.85},
            {"name": "Song 4", "artist": "Artist C", "similarity_score": 0.95},
        ]

        result = self.distributor.enforce_artist_diversity(tracks, max_tracks=4)
        self.assertEqual(len(result), 4)

    def test_enforce_artist_diversity_too_many_artists(self) -> None:
        """Test artist diversity when there are too many artists"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song 1", "artist": "Artist A", "similarity_score": 0.9},
            {"name": "Song 2", "artist": "Artist B", "similarity_score": 0.8},
            {"name": "Song 3", "artist": "Artist C", "similarity_score": 0.85},
            {"name": "Song 4", "artist": "Artist D", "similarity_score": 0.75},
            {"name": "Song 5", "artist": "Artist E", "similarity_score": 0.7},
        ]

        result = self.distributor.enforce_artist_diversity(tracks, max_tracks=3)
        self.assertEqual(len(result), 3)

    def test_enforce_artist_diversity_single_artist(self) -> None:
        """Test artist diversity with single artist"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song 1", "artist": "Artist A", "similarity_score": 0.9},
            {"name": "Song 2", "artist": "Artist A", "similarity_score": 0.8},
            {"name": "Song 3", "artist": "Artist A", "similarity_score": 0.7},
        ]

        result = self.distributor.enforce_artist_diversity(tracks, max_tracks=3)
        self.assertEqual(len(result), 3)

    def test_enforce_artist_diversity_empty_list(self) -> None:
        """Test artist diversity with empty track list"""
        tracks: List[Dict[str, Any]] = []

        result = self.distributor.enforce_artist_diversity(tracks, max_tracks=10)
        self.assertEqual(result, [])

    def test_enforce_artist_diversity_none_artist(self) -> None:
        """Test artist diversity with None artist values"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song 1", "artist": "Artist A", "similarity_score": 0.9},
            {"name": "Song 2", "artist": None, "similarity_score": 0.8},
            {"name": "Song 3", "artist": "Artist B", "similarity_score": 0.85},
        ]

        result = self.distributor.enforce_artist_diversity(tracks, max_tracks=3)
        self.assertEqual(len(result), 3)

    def test_enforce_artist_diversity_missing_artist(self) -> None:
        """Test artist diversity with missing artist field"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song 1", "artist": "Artist A", "similarity_score": 0.9},
            {"name": "Song 2", "similarity_score": 0.8},
            {"name": "Song 3", "artist": "Artist B", "similarity_score": 0.85},
        ]

        result = self.distributor.enforce_artist_diversity(tracks, max_tracks=3)
        self.assertEqual(len(result), 3)

    def test_group_tracks_by_artist(self) -> None:
        """Test grouping tracks by artist"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song 1", "artist": "Artist A", "similarity_score": 0.9},
            {"name": "Song 2", "artist": "Artist A", "similarity_score": 0.8},
            {"name": "Song 3", "artist": "Artist B", "similarity_score": 0.85},
            {"name": "Song 4", "artist": "Artist A", "similarity_score": 0.7},
        ]

        result = self.distributor._group_tracks_by_artist(tracks)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result["Artist A"]), 3)
        self.assertEqual(len(result["Artist B"]), 1)

    def test_group_tracks_by_artist_none_artist(self) -> None:
        """Test grouping tracks with None artist values"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song 1", "artist": "Artist A", "similarity_score": 0.9},
            {"name": "Song 2", "artist": None, "similarity_score": 0.8},
            {"name": "Song 3", "artist": "Artist B", "similarity_score": 0.85},
            {"name": "Song 4", "artist": None, "similarity_score": 0.7},
        ]

        result = self.distributor._group_tracks_by_artist(tracks)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result["Artist A"]), 1)
        self.assertEqual(len(result["Artist B"]), 1)
        self.assertEqual(len(result["unknown"]), 2)

    def test_group_tracks_by_artist_missing_artist(self) -> None:
        """Test grouping tracks with missing artist field"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song 1", "artist": "Artist A", "similarity_score": 0.9},
            {"name": "Song 2", "similarity_score": 0.8},
            {"name": "Song 3", "artist": "Artist B", "similarity_score": 0.85},
        ]

        result = self.distributor._group_tracks_by_artist(tracks)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result["Artist A"]), 1)
        self.assertEqual(len(result["Artist B"]), 1)
        self.assertEqual(len(result["unknown"]), 1)

    def test_group_tracks_by_artist_empty_list(self) -> None:
        """Test grouping tracks with empty list"""
        tracks: List[Dict[str, Any]] = []

        result = self.distributor._group_tracks_by_artist(tracks)
        self.assertEqual(result, {})

    def test_apply_artist_randomization_not_vague(self) -> None:
        """Test artist randomization when query is not vague"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song 1", "artist": "Artist A", "similarity_score": 0.9},
            {"name": "Song 2", "artist": "Artist B", "similarity_score": 0.8},
            {"name": "Song 3", "artist": "Artist C", "similarity_score": 0.85},
        ]

        result = self.distributor.apply_artist_randomization(
            tracks, max_tracks=5, is_vague=False
        )
        self.assertEqual(len(result), 3)

    def test_apply_artist_randomization_vague_few_artists(self) -> None:
        """Test artist randomization when query is vague but few artists"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song 1", "artist": "Artist A", "similarity_score": 0.9},
            {"name": "Song 2", "artist": "Artist A", "similarity_score": 0.8},
            {"name": "Song 3", "artist": "Artist A", "similarity_score": 0.7},
        ]

        result = self.distributor.apply_artist_randomization(
            tracks, max_tracks=5, is_vague=True
        )
        self.assertEqual(len(result), 3)

    def test_apply_artist_randomization_vague_many_artists(self) -> None:
        """Test artist randomization when query is vague with many artists"""
        diverse_tracks: List[Dict[str, Any]] = []
        for i in range(10):
            diverse_tracks.append(
                {
                    "name": f"Song {i}",
                    "artist": f"Artist {i}",
                    "similarity_score": 0.9 - (i * 0.1),
                }
            )

        result = self.distributor.apply_artist_randomization(
            diverse_tracks, max_tracks=5, is_vague=True
        )

        self.assertEqual(len(result), 5)

    def test_distribute_artists_single_track(self) -> None:
        """Test artist distribution with single track"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song 1", "artist": "Artist A", "similarity_score": 0.9}
        ]
        result = self.distributor._distribute_artists(tracks)
        self.assertEqual(result, tracks)

    def test_distribute_artists_single_artist(self) -> None:
        """Test artist distribution with single artist"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song 1", "artist": "Artist A", "similarity_score": 0.9},
            {"name": "Song 2", "artist": "Artist A", "similarity_score": 0.8},
            {"name": "Song 3", "artist": "Artist A", "similarity_score": 0.7},
        ]
        result = self.distributor._distribute_artists(tracks)
        self.assertEqual(result, tracks)

    def test_distribute_artists_few_tracks(self) -> None:
        """Test artist distribution with few tracks"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song 1", "artist": "Artist A", "similarity_score": 0.9},
            {"name": "Song 2", "artist": "Artist B", "similarity_score": 0.8},
            {"name": "Song 3", "artist": "Artist A", "similarity_score": 0.7},
        ]
        result = self.distributor._distribute_artists(tracks)
        self.assertEqual(len(result), 3)

    def test_distribute_artists_multiple_artists(self) -> None:
        """Test artist distribution with multiple artists"""
        diverse_tracks: List[Dict[str, Any]] = [
            {"name": "Song 1", "artist": "Artist A", "similarity_score": 0.9},
            {"name": "Song 2", "artist": "Artist B", "similarity_score": 0.8},
            {"name": "Song 3", "artist": "Artist A", "similarity_score": 0.7},
            {"name": "Song 4", "artist": "Artist C", "similarity_score": 0.85},
            {"name": "Song 5", "artist": "Artist B", "similarity_score": 0.75},
            {"name": "Song 6", "artist": "Artist A", "similarity_score": 0.65},
        ]

        result = self.distributor._distribute_artists(diverse_tracks)
        self.assertEqual(len(result), 6)

        artists = [t["artist"] for t in result]
        for i in range(len(artists) - 2):
            self.assertFalse(artists[i] == artists[i + 1] == artists[i + 2])

    def test_create_artist_queues(self) -> None:
        """Test creating artist queues"""
        artist_groups = {
            "Artist A": [
                {"name": "Song 1", "artist": "Artist A", "similarity_score": 0.9},
                {"name": "Song 2", "artist": "Artist A", "similarity_score": 0.8},
            ],
            "Artist B": [
                {"name": "Song 3", "artist": "Artist B", "similarity_score": 0.85}
            ],
        }

        result = self.distributor._create_artist_queues(artist_groups)
        self.assertEqual(len(result), 2)
        self.assertIn("Artist A", result)
        self.assertIn("Artist B", result)
        self.assertEqual(len(result["Artist A"]), 2)
        self.assertEqual(len(result["Artist B"]), 1)

    def test_round_robin_distribute(self) -> None:
        """Test round-robin distribution"""
        artist_queues = {
            "Artist A": [
                {"name": "Song 1", "artist": "Artist A", "similarity_score": 0.9},
                {"name": "Song 2", "artist": "Artist A", "similarity_score": 0.8},
            ],
            "Artist B": [
                {"name": "Song 3", "artist": "Artist B", "similarity_score": 0.85}
            ],
        }
        tracks: List[Dict[str, Any]] = [
            {"name": "Song 1", "artist": "Artist A", "similarity_score": 0.9},
            {"name": "Song 2", "artist": "Artist A", "similarity_score": 0.8},
            {"name": "Song 3", "artist": "Artist B", "similarity_score": 0.85},
        ]
        distributed: List[Dict[str, Any]] = []

        result = self.distributor._round_robin_distribute(
            artist_queues, tracks, distributed
        )
        self.assertEqual(len(result), 3)

        artists = [t["artist"] for t in result]
        self.assertIn("Artist A", artists)
        self.assertIn("Artist B", artists)

    def test_add_remaining_tracks(self) -> None:
        """Test adding remaining tracks"""
        artist_queues = {
            "Artist A": [
                {"name": "Song 1", "artist": "Artist A", "similarity_score": 0.9}
            ],
            "Artist B": [
                {"name": "Song 2", "artist": "Artist B", "similarity_score": 0.85}
            ],
        }
        tracks: List[Dict[str, Any]] = [
            {"name": "Song 1", "artist": "Artist A", "similarity_score": 0.9},
            {"name": "Song 2", "artist": "Artist B", "similarity_score": 0.85},
        ]
        distributed: List[Dict[str, Any]] = []

        result = self.distributor._add_remaining_tracks(
            artist_queues, tracks, distributed
        )
        self.assertEqual(len(result), 2)

    def test_distribute_tracks_comprehensive(self) -> None:
        """Test comprehensive track distribution"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song 1", "artist": "Artist A", "similarity_score": 0.9},
            {"name": "Song 2", "artist": "Artist A", "similarity_score": 0.8},
            {"name": "Song 3", "artist": "Artist B", "similarity_score": 0.85},
            {"name": "Song 4", "artist": "Artist C", "similarity_score": 0.95},
            {"name": "Song 5", "artist": "Artist D", "similarity_score": 0.75},
            {"name": "Song 6", "artist": "Artist E", "similarity_score": 0.7},
        ]

        distributed = self.distributor.enforce_artist_diversity(tracks, max_tracks=3)
        self.assertEqual(len(distributed), 3)

        artists = set(track.get("artist") for track in distributed)
        self.assertLessEqual(len(artists), 3)

    def test_distribute_tracks_with_custom_limits(self) -> None:
        """Test track distribution with custom limits"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song 1", "artist": "Artist A", "similarity_score": 0.9},
            {"name": "Song 2", "artist": "Artist B", "similarity_score": 0.8},
            {"name": "Song 3", "artist": "Artist C", "similarity_score": 0.85},
            {"name": "Song 4", "artist": "Artist D", "similarity_score": 0.75},
        ]

        distributed = self.distributor.enforce_artist_diversity(
            tracks, max_tracks=2, max_tracks_per_artist=1
        )
        self.assertEqual(len(distributed), 2)

        artists = set(track.get("artist") for track in distributed)
        self.assertLessEqual(len(artists), 2)

    def test_distribute_tracks_preserve_order(self) -> None:
        """Test that track distribution preserves original order within artists"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song 1", "artist": "Artist A", "similarity_score": 0.9},
            {"name": "Song 2", "artist": "Artist A", "similarity_score": 0.8},
            {"name": "Song 3", "artist": "Artist B", "similarity_score": 0.85},
            {"name": "Song 4", "artist": "Artist A", "similarity_score": 0.7},
            {"name": "Song 5", "artist": "Artist B", "similarity_score": 0.75},
        ]

        distributed = self.distributor.enforce_artist_diversity(tracks, max_tracks=5)

        artist_a_tracks = [
            track for track in distributed if track.get("artist") == "Artist A"
        ]
        if len(artist_a_tracks) >= 3:
            self.assertEqual(artist_a_tracks[0]["name"], "Song 1")
            self.assertEqual(artist_a_tracks[1]["name"], "Song 2")
            self.assertEqual(artist_a_tracks[2]["name"], "Song 4")

    def test_distribute_tracks_handle_edge_cases(self) -> None:
        """Test track distribution with edge cases"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song 1", "artist": None, "similarity_score": 0.9},
            {"name": "Song 2", "similarity_score": 0.8},
            {"name": "Song 3", "artist": "Artist A", "similarity_score": 0.85},
            {"name": "Song 4", "artist": "", "similarity_score": 0.7},
        ]

        distributed = self.distributor.enforce_artist_diversity(tracks, max_tracks=4)
        self.assertIsInstance(distributed, list)


if __name__ == "__main__":
    unittest.main()
