#!/usr/bin/env python3
"""
Tests for deduplication functionality
"""

import unittest
from typing import Any, Dict, List

from tonal_hortator.core.deduplication import TrackDeduplicator


class TestTrackDeduplicator(unittest.TestCase):
    """Test cases for TrackDeduplicator class"""

    def setUp(self) -> None:
        """Set up test fixtures"""
        self.deduplicator = TrackDeduplicator()

    def test_init(self) -> None:
        """Test deduplicator initialization"""
        deduplicator = TrackDeduplicator()
        self.assertIsInstance(deduplicator.suffixes_to_remove, list)
        self.assertGreater(len(deduplicator.suffixes_to_remove), 0)
        self.assertTrue(
            all(isinstance(suffix, str) for suffix in deduplicator.suffixes_to_remove)
        )

    def test_deduplicate_tracks_empty_list(self) -> None:
        """Test deduplication with empty track list"""
        result = self.deduplicator.deduplicate_tracks(
            [], min_similarity=0.5, max_tracks=10
        )
        self.assertEqual(result, [])

    def test_deduplicate_tracks_single_track(self) -> None:
        """Test deduplication with single track"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song", "artist": "Artist", "similarity_score": 0.8}
        ]
        result = self.deduplicator.deduplicate_tracks(
            tracks, min_similarity=0.5, max_tracks=10
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "Song")

    def test_deduplicate_tracks_filters_by_similarity(self) -> None:
        """Test that deduplication filters by similarity threshold"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song1", "artist": "Artist1", "similarity_score": 0.9},
            {
                "name": "Song2",
                "artist": "Artist2",
                "similarity_score": 0.3,
            },  # Below threshold
            {"name": "Song3", "artist": "Artist3", "similarity_score": 0.7},
        ]
        result = self.deduplicator.deduplicate_tracks(
            tracks, min_similarity=0.5, max_tracks=10
        )
        self.assertEqual(len(result), 1)  # Only Song1 remains after deduplication
        self.assertTrue(all(track["similarity_score"] >= 0.5 for track in result))

    def test_deduplicate_tracks_without_similarity_score(self) -> None:
        """Test deduplication with tracks missing similarity_score"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song1", "artist": "Artist1"},  # No similarity_score
            {"name": "Song2", "artist": "Artist2", "similarity_score": 0.8},
        ]
        result = self.deduplicator.deduplicate_tracks(
            tracks, min_similarity=0.5, max_tracks=10
        )
        # Should filter out tracks without similarity_score
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "Song2")

    def test_deduplicate_by_location(self) -> None:
        """Test location-based deduplication"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song1", "artist": "Artist1", "location": "/path/to/song1.mp3"},
            {
                "name": "Song2",
                "artist": "Artist2",
                "location": "/path/to/song1.mp3",
            },  # Same location
            {"name": "Song3", "artist": "Artist3", "location": "/path/to/song3.mp3"},
        ]
        result = self.deduplicator._deduplicate_by_location(tracks)
        self.assertEqual(len(result), 2)  # Should remove duplicate location

    def test_deduplicate_by_location_normalized(self) -> None:
        """Test location deduplication with normalized paths"""
        tracks: List[Dict[str, Any]] = [
            {
                "name": "Song1",
                "artist": "Artist1",
                "location": "/Users/username/Music/song.mp3",
            },
            {
                "name": "Song2",
                "artist": "Artist2",
                "location": "C:\\Users\\username\\Music\\song.mp3",
            },  # Different format
        ]
        result = self.deduplicator._deduplicate_by_location(tracks)
        self.assertEqual(len(result), 1)  # Should normalize and deduplicate

    def test_deduplicate_by_location_empty_location(self) -> None:
        """Test location deduplication with empty locations"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song1", "artist": "Artist1", "location": ""},
            {"name": "Song2", "artist": "Artist2", "location": "/path/to/song.mp3"},
            {"name": "Song3", "artist": "Artist3"},  # No location field
        ]
        result = self.deduplicator._deduplicate_by_location(tracks)
        self.assertEqual(len(result), 3)  # Should include tracks without location

    def test_deduplicate_by_name_artist(self) -> None:
        """Test title/artist combination deduplication"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song1", "artist": "Artist1"},
            {"name": "Song1", "artist": "Artist1"},  # Duplicate
            {"name": "Song1", "artist": "Artist2"},  # Different artist
            {"name": "Song2", "artist": "Artist1"},  # Different title
        ]
        result = self.deduplicator._deduplicate_by_name_artist(tracks)
        self.assertEqual(len(result), 3)  # Should remove exact duplicate

    def test_deduplicate_by_name_artist_case_insensitive(self) -> None:
        """Test title/artist deduplication is case insensitive"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song1", "artist": "Artist1"},
            {"name": "SONG1", "artist": "ARTIST1"},  # Different case
        ]
        result = self.deduplicator._deduplicate_by_name_artist(tracks)
        self.assertEqual(len(result), 1)  # Should treat as duplicate

    def test_deduplicate_by_name_artist_empty_fields(self) -> None:
        """Test title/artist deduplication with empty fields"""
        tracks: List[Dict[str, Any]] = [
            {"name": "", "artist": "Artist1"},
            {"name": "Song1", "artist": ""},
            {"name": "", "artist": ""},
            {"name": "Song2", "artist": "Artist2"},
        ]
        result = self.deduplicator._deduplicate_by_name_artist(tracks)
        self.assertEqual(len(result), 4)  # Should include tracks with empty fields

    def test_deduplicate_by_track_id(self) -> None:
        """Test track ID deduplication"""
        tracks: List[Dict[str, Any]] = [
            {"id": 1, "name": "Song1", "artist": "Artist1"},
            {"id": 1, "name": "Song2", "artist": "Artist2"},  # Same ID
            {"id": 2, "name": "Song3", "artist": "Artist3"},
        ]
        result = self.deduplicator._deduplicate_by_track_id(tracks)
        self.assertEqual(len(result), 2)  # Should remove duplicate ID

    def test_deduplicate_by_track_id_missing_id(self) -> None:
        """Test track ID deduplication with missing IDs"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song1", "artist": "Artist1"},  # No ID
            {"id": 1, "name": "Song2", "artist": "Artist2"},
            {"name": "Song3", "artist": "Artist3"},  # No ID
        ]
        result = self.deduplicator._deduplicate_by_track_id(tracks)
        self.assertEqual(len(result), 2)  # Should deduplicate tracks without ID as one

    def test_smart_name_deduplication_single_track(self) -> None:
        """Test smart title deduplication with single track"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song", "artist": "Artist", "similarity_score": 0.8}
        ]
        result = self.deduplicator._smart_name_deduplication(tracks)
        self.assertEqual(result, tracks)

    def test_smart_name_deduplication_empty_list(self) -> None:
        """Test smart title deduplication with empty list"""
        result = self.deduplicator._smart_name_deduplication([])
        self.assertEqual(result, [])

    def test_smart_name_deduplication_different_artists(self) -> None:
        """Test smart title deduplication with different artists"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song (Remix)", "artist": "Artist1", "similarity_score": 0.8},
            {"name": "Song (Live)", "artist": "Artist2", "similarity_score": 0.7},
            {"name": "Song", "artist": "Artist3", "similarity_score": 0.9},
        ]
        result = self.deduplicator._smart_name_deduplication(tracks)
        # Should keep all since they have different artists
        self.assertEqual(len(result), 3)

    def test_smart_name_deduplication_same_artist_variations(self) -> None:
        """Test smart title deduplication with same artist and title variations"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song (Remix)", "artist": "Artist1", "similarity_score": 0.8},
            {"name": "Song (Live)", "artist": "Artist1", "similarity_score": 0.7},
            {
                "name": "Song",
                "artist": "Artist1",
                "similarity_score": 0.9,
            },  # Highest score
        ]
        result = self.deduplicator._smart_name_deduplication(tracks)
        # Should keep only the highest scoring variation
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["similarity_score"], 0.9)

    def test_find_best_track_for_base_name(self) -> None:
        """Test finding best track for base title"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song (Remix)", "artist": "Artist1", "similarity_score": 0.8},
            {"name": "Song (Live)", "artist": "Artist1", "similarity_score": 0.7},
            {"name": "Song", "artist": "Artist1", "similarity_score": 0.9},
        ]
        result = self.deduplicator._find_best_track_for_base_name(tracks, "song")
        self.assertIsNotNone(result)
        if result is not None:  # Type guard for mypy
            self.assertEqual(result["similarity_score"], 0.9)

    def test_find_best_track_for_base_name_no_match(self) -> None:
        """Test finding best track when no match exists"""
        tracks: List[Dict[str, Any]] = [
            {"name": "Song (Remix)", "artist": "Artist1", "similarity_score": 0.8},
            {"name": "Song (Live)", "artist": "Artist1", "similarity_score": 0.7},
        ]
        result = self.deduplicator._find_best_track_for_base_name(tracks, "different")
        # Should return first track as fallback when no match is found
        self.assertEqual(result, tracks[0])

    def test_extract_base_name_empty(self) -> None:
        """Test extracting base title from empty string"""
        result = self.deduplicator._extract_base_name("")
        self.assertEqual(result, "")

    def test_extract_base_name_no_suffixes(self) -> None:
        """Test extracting base title without suffixes"""
        result = self.deduplicator._extract_base_name("Simple Song")
        self.assertEqual(result, "Simple Song")

    def test_extract_base_name_with_remix_suffix(self) -> None:
        """Test extracting base title with remix suffix"""
        result = self.deduplicator._extract_base_name("Song (Remix)")
        self.assertEqual(result, "Song")

    def test_extract_base_name_with_live_suffix(self) -> None:
        """Test extracting base title with live suffix"""
        result = self.deduplicator._extract_base_name("Song (Live)")
        self.assertEqual(result, "Song")

    def test_extract_base_name_with_acoustic_suffix(self) -> None:
        """Test extracting base title with acoustic suffix"""
        result = self.deduplicator._extract_base_name("Song (Acoustic)")
        self.assertEqual(result, "Song")

    def test_extract_base_name_with_dash_suffix(self) -> None:
        """Test extracting base title with dash suffix"""
        result = self.deduplicator._extract_base_name("Song - Remix")
        self.assertEqual(result, "Song")

    def test_extract_base_name_with_multiple_suffixes(self) -> None:
        """Test extracting base title with multiple suffix variations"""
        test_cases = [
            ("Song (Remastered)", "Song"),
            ("Song (Radio Edit)", "Song"),
            ("Song (Extended)", "Song"),
            ("Song (Clean)", "Song"),
            ("Song (Explicit)", "Song"),
            ("Song (Original Mix)", "Song"),
            ("Song (Club Mix)", "Song"),
            ("Song (Dub Mix)", "Song"),
        ]
        for title, expected in test_cases:
            with self.subTest(title=title):
                result = self.deduplicator._extract_base_name(title)
                self.assertEqual(result, expected)

    def test_extract_base_name_with_parentheses_content(self) -> None:
        """Test extracting base title with parentheses content that's not a suffix"""
        result = self.deduplicator._extract_base_name("Song (feat. Artist)")
        self.assertEqual(result, "Song")

    def test_extract_base_name_case_insensitive_suffixes(self) -> None:
        """Test extracting base title with case insensitive suffix matching"""
        result = self.deduplicator._extract_base_name("Song (REMIX)")
        self.assertEqual(result, "Song")

    def test_normalize_file_location_empty(self) -> None:
        """Test normalizing empty file location"""
        result = self.deduplicator._normalize_file_location("")
        self.assertEqual(result, "")

    def test_normalize_file_location_simple_path(self) -> None:
        """Test normalizing simple file path"""
        result = self.deduplicator._normalize_file_location("/path/to/song.mp3")
        self.assertEqual(result, "/path/to/song.mp3")

    def test_normalize_file_location_windows_path(self) -> None:
        """Test normalizing Windows file path"""
        result = self.deduplicator._normalize_file_location(
            "C:\\Users\\username\\Music\\song.mp3"
        )
        self.assertEqual(result, "music/song.mp3")

    def test_normalize_file_location_removes_username_prefix_mac(self) -> None:
        """Test normalizing Mac file path with username prefix"""
        result = self.deduplicator._normalize_file_location(
            "/Users/username/Music/song.mp3"
        )
        self.assertEqual(result, "music/song.mp3")

    def test_normalize_file_location_removes_username_prefix_linux(self) -> None:
        """Test normalizing Linux file path with username prefix"""
        result = self.deduplicator._normalize_file_location(
            "/home/username/Music/song.mp3"
        )
        self.assertEqual(result, "music/song.mp3")

    def test_normalize_file_location_removes_username_prefix_windows(self) -> None:
        """Test normalizing Windows file path with username prefix"""
        result = self.deduplicator._normalize_file_location(
            "C:/Users/username/Music/song.mp3"
        )
        self.assertEqual(result, "music/song.mp3")

    def test_normalize_file_location_different_drive_letters(self) -> None:
        """Test normalizing file paths with different drive letters"""
        test_cases = [
            ("D:/Users/username/Music/song.mp3", "music/song.mp3"),
            ("E:/Users/username/Music/song.mp3", "music/song.mp3"),
        ]
        for path, expected in test_cases:
            with self.subTest(path=path):
                result = self.deduplicator._normalize_file_location(path)
                self.assertEqual(result, expected)

    def test_normalize_file_location_no_username_match(self) -> None:
        """Test normalizing file path that doesn't match username pattern"""
        result = self.deduplicator._normalize_file_location("/var/lib/music/song.mp3")
        self.assertEqual(result, "/var/lib/music/song.mp3")

    def test_comprehensive_deduplication_workflow(self) -> None:
        """Test the complete deduplication workflow with realistic data"""
        tracks: List[Dict[str, Any]] = [
            # Duplicate locations
            {
                "id": 1,
                "name": "Song1",
                "artist": "Artist1",
                "location": "/Users/user/Music/song1.mp3",
                "similarity_score": 0.9,
            },
            {
                "id": 2,
                "name": "Song1",
                "artist": "Artist1",
                "location": "C:/Users/user/Music/song1.mp3",
                "similarity_score": 0.8,
            },
            # Duplicate title/artist combinations
            {
                "id": 3,
                "name": "Song2",
                "artist": "Artist2",
                "location": "/path/to/song2.mp3",
                "similarity_score": 0.7,
            },
            {
                "id": 4,
                "name": "Song2",
                "artist": "Artist2",
                "location": "/path/to/song2_alt.mp3",
                "similarity_score": 0.6,
            },
            # Title variations for same artist
            {
                "id": 5,
                "name": "Song3 (Remix)",
                "artist": "Artist3",
                "location": "/path/to/song3_remix.mp3",
                "similarity_score": 0.8,
            },
            {
                "id": 6,
                "name": "Song3 (Live)",
                "artist": "Artist3",
                "location": "/path/to/song3_live.mp3",
                "similarity_score": 0.9,
            },
            {
                "id": 7,
                "name": "Song3",
                "artist": "Artist3",
                "location": "/path/to/song3.mp3",
                "similarity_score": 0.7,
            },
            # Below similarity threshold
            {
                "id": 8,
                "name": "Song4",
                "artist": "Artist4",
                "location": "/path/to/song4.mp3",
                "similarity_score": 0.3,
            },
            # Unique track
            {
                "id": 9,
                "name": "Song5",
                "artist": "Artist5",
                "location": "/path/to/song5.mp3",
                "similarity_score": 0.85,
            },
        ]

        result = self.deduplicator.deduplicate_tracks(
            tracks, min_similarity=0.5, max_tracks=10
        )

        # Should have:
        # - 1 track from location deduplication (normalized paths)
        # - 1 track from title/artist deduplication
        # - 1 track from smart title deduplication (highest score variation)
        # - 1 unique track
        # - Filter out low similarity track
        self.assertEqual(len(result), 4)

        # Verify the highest scoring variations were kept
        track_names = [track["name"] for track in result]
        self.assertIn("Song1", track_names)  # From location deduplication
        self.assertIn("Song2", track_names)  # From title/artist deduplication
        self.assertIn("Song3 (Live)", track_names)  # Highest scoring variation
        self.assertIn("Song5", track_names)  # Unique track


if __name__ == "__main__":
    unittest.main()
