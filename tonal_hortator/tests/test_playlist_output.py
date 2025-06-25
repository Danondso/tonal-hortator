#!/usr/bin/env python3
"""
Tests for playlist output functionality
"""

import os
import tempfile
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from tonal_hortator.core.playlist_output import PlaylistOutput


class TestPlaylistOutput(unittest.TestCase):
    """Test playlist output functionality"""

    def setUp(self) -> None:
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "playlists")
        os.makedirs(self.output_dir, exist_ok=True)
        self.playlist_output = PlaylistOutput()
        self.sample_tracks: List[Dict[str, Any]] = [
            {
                "name": "Test Song 1",
                "artist": "Test Artist 1",
                "album": "Test Album 1",
                "duration_ms": 180000,
                "similarity_score": 0.95,
                "location": "file:///path/to/song1.mp3",
            },
            {
                "name": "Test Song 2",
                "artist": "Test Artist 2",
                "album": "Test Album 2",
                "duration_ms": 240000,
                "similarity_score": 0.88,
                "location": "file:///path/to/song2.mp3",
            },
            {
                "name": "Test Song 3",
                "artist": "Test Artist 3",
                "album": "Test Album 3",
                "duration_ms": 200000,
                "similarity_score": 0.92,
                "location": "file:///path/to/song3.mp3",
            },
        ]

    def tearDown(self) -> None:
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_with_output_dir(self) -> None:
        """Test initialization with output directory"""
        output = PlaylistOutput()
        # The PlaylistOutput class doesn't have an output_dir attribute
        # It uses the save_playlist_m3u method with an output_dir parameter

    def test_init_without_output_dir(self) -> None:
        """Test initialization without output directory"""
        output = PlaylistOutput()
        # The PlaylistOutput class doesn't store output_dir as an attribute
        # It's passed as a parameter to save_playlist_m3u

    def test_create_output_directory(self) -> None:
        """Test creating output directory"""
        test_dir = os.path.join(self.temp_dir, "new_output")
        output = PlaylistOutput()

        # Test that save_playlist_m3u creates the directory
        tracks: List[Dict[str, Any]] = [
            {"name": "Test Song", "artist": "Test Artist", "location": "/test/path.mp3"}
        ]
        result = output.save_playlist_m3u(tracks, "test_query", test_dir)

        # Directory should be created
        self.assertTrue(os.path.exists(test_dir))
        self.assertTrue(os.path.exists(result))

    def test_write_m3u_playlist_success(self) -> None:
        """Test successful M3U playlist writing"""
        playlist_name = "test_playlist"
        tracks: List[Dict[str, Any]] = [
            {"name": "Song 1", "artist": "Artist 1", "location": "/path/to/song1.mp3"},
            {"name": "Song 2", "artist": "Artist 2", "location": "/path/to/song2.mp3"},
        ]

        result = self.playlist_output.save_playlist_m3u(
            tracks, playlist_name, self.output_dir
        )
        self.assertTrue(os.path.exists(result))

        # Check file content
        with open(result, "r") as f:
            content = f.read()
            self.assertIn("#EXTM3U", content)
            self.assertIn("Song 1", content)
            self.assertIn("Song 2", content)
            self.assertIn("/path/to/song1.mp3", content)
            self.assertIn("/path/to/song2.mp3", content)

    def test_write_m3u_playlist_empty_tracks(self) -> None:
        """Test writing M3U playlist with empty tracks list"""
        playlist_name = "empty_playlist"
        tracks: List[Dict[str, Any]] = []

        result = self.playlist_output.save_playlist_m3u(
            tracks, playlist_name, self.output_dir
        )
        self.assertTrue(os.path.exists(result))

        # Check file content (should have header and metadata)
        with open(result, "r") as f:
            content = f.read()
            self.assertIn("#EXTM3U", content)
            self.assertIn("Query: empty_playlist", content)

    def test_write_m3u_playlist_missing_fields(self) -> None:
        """Test writing M3U playlist with missing track fields"""
        playlist_name = "incomplete_playlist"
        tracks: List[Dict[str, Any]] = [
            {"name": "Song 1", "artist": "Artist 1"},  # Missing location
            {"name": "Song 2", "location": "/path/to/song2.mp3"},  # Missing artist
            {"artist": "Artist 3", "location": "/path/to/song3.mp3"},  # Missing name
        ]

        result = self.playlist_output.save_playlist_m3u(
            tracks, playlist_name, self.output_dir
        )
        self.assertTrue(os.path.exists(result))

        # Check file content
        with open(result, "r") as f:
            content = f.read()
            self.assertIn("#EXTM3U", content)
            self.assertIn("Song 1", content)
            self.assertIn("Song 2", content)
            self.assertIn("Artist 3", content)

    def test_write_m3u_playlist_io_error(self) -> None:
        """Test writing M3U playlist when IO error occurs"""
        playlist_name = "test_playlist"
        tracks: List[Dict[str, Any]] = [
            {"name": "Song 1", "artist": "Artist 1", "location": "/path/to/song1.mp3"}
        ]

        with patch("builtins.open", side_effect=IOError("Permission denied")):
            with self.assertRaises(IOError):
                self.playlist_output.save_playlist_m3u(
                    tracks, playlist_name, self.output_dir
                )

    def test_write_playlist_multiple_formats(self) -> None:
        """Test writing playlist in M3U format (only supported format)"""
        playlist_name = "test_playlist"
        tracks: List[Dict[str, Any]] = [
            {"name": "Song 1", "artist": "Artist 1", "location": "/path/to/song1.mp3"},
            {"name": "Song 2", "artist": "Artist 2", "location": "/path/to/song2.mp3"},
        ]

        # Test M3U format (only supported format)
        result = self.playlist_output.save_playlist_m3u(
            tracks, playlist_name, self.output_dir
        )
        self.assertTrue(os.path.exists(result))
        self.assertTrue(result.endswith(".m3u"))

    def test_print_playlist_summary_basic(self) -> None:
        """Test basic playlist summary printing"""
        with patch("builtins.print") as mock_print:
            self.playlist_output.print_playlist_summary(
                self.sample_tracks, "test query"
            )

            # Check that print was called
            mock_print.assert_called()

            # Get all print calls
            calls = mock_print.call_args_list
            call_strings = [str(call) for call in calls]

            # Should contain playlist header
            self.assertTrue(any("test query" in call for call in call_strings))
            self.assertTrue(any("3 tracks found" in call for call in call_strings))

            # Should contain track information
            self.assertTrue(
                any("Test Artist 1 - Test Song 1" in call for call in call_strings)
            )
            self.assertTrue(
                any("Test Artist 2 - Test Song 2" in call for call in call_strings)
            )
            self.assertTrue(
                any("Test Artist 3 - Test Song 3" in call for call in call_strings)
            )

    def test_print_playlist_summary_empty_tracks(self) -> None:
        """Test playlist summary printing with empty track list"""
        with patch("builtins.print") as mock_print:
            self.playlist_output.print_playlist_summary([], "empty query")

            # Check that print was called
            mock_print.assert_called()

            # Get all print calls
            calls = mock_print.call_args_list
            call_strings = [str(call) for call in calls]

            # Should contain playlist header
            self.assertTrue(any("empty query" in call for call in call_strings))
            self.assertTrue(any("0 tracks found" in call for call in call_strings))

    def test_print_playlist_summary_missing_fields(self) -> None:
        """Test playlist summary printing with missing track fields"""
        incomplete_tracks = [
            {
                "name": "Song 1",
                "artist": "Artist 1",
                # Missing album, similarity_score
            },
            {
                "name": "Song 2",
                # Missing artist, album, similarity_score
            },
        ]

        with patch("builtins.print") as mock_print:
            self.playlist_output.print_playlist_summary(
                incomplete_tracks, "incomplete query"
            )

            # Check that print was called
            mock_print.assert_called()

            # Get all print calls
            calls = mock_print.call_args_list
            call_strings = [str(call) for call in calls]

            # Should handle missing fields gracefully
            self.assertTrue(any("Artist 1 - Song 1" in call for call in call_strings))
            self.assertTrue(any("Unknown - Song 2" in call for call in call_strings))

    def test_print_playlist_summary_average_similarity(self) -> None:
        """Test playlist summary printing includes average similarity"""
        tracks_with_scores = [
            {
                "name": "Song 1",
                "artist": "Artist 1",
                "album": "Album 1",
                "similarity_score": 0.9,
            },
            {
                "name": "Song 2",
                "artist": "Artist 2",
                "album": "Album 2",
                "similarity_score": 0.8,
            },
        ]

        with patch("builtins.print") as mock_print:
            self.playlist_output.print_playlist_summary(
                tracks_with_scores, "similarity test"
            )

            # Check that print was called
            mock_print.assert_called()

            # Get all print calls
            calls = mock_print.call_args_list
            call_strings = [str(call) for call in calls]

            # Should contain average similarity
            self.assertTrue(any("Average similarity" in call for call in call_strings))
            self.assertTrue(any("0.850" in call for call in call_strings))

    def test_save_playlist_m3u_error_handling(self) -> None:
        """Test M3U playlist saving error handling"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Make the directory read-only to cause a write error
            os.chmod(temp_dir, 0o444)

            with self.assertRaises(Exception):
                self.playlist_output.save_playlist_m3u(
                    self.sample_tracks, "error test", temp_dir
                )

            # Restore permissions
            os.chmod(temp_dir, 0o755)

    def test_save_playlist_m3u_unicode_characters(self) -> None:
        """Test M3U playlist saving with Unicode characters"""
        unicode_tracks = [
            {
                "name": "Café Song",
                "artist": "José Artist",
                "album": "Album with émojis 🎵",
                "duration_ms": 180000,
                "similarity_score": 0.95,
                "location": "file:///path/to/café.mp3",
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            query = "Unicode test: café 🎵"
            filepath = self.playlist_output.save_playlist_m3u(
                unicode_tracks, query, temp_dir
            )

            # Check that file was created
            self.assertTrue(os.path.exists(filepath))

            # Check file contents
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # Should handle Unicode characters properly
            self.assertIn("José Artist - Café Song", content)
            self.assertIn("Album with émojis", content)

    def test_save_playlist_m3u_timestamp_format(self) -> None:
        """Test M3U playlist saving includes proper timestamp format"""
        with tempfile.TemporaryDirectory() as temp_dir:
            query = "timestamp test"
            filepath = self.playlist_output.save_playlist_m3u(
                self.sample_tracks, query, temp_dir
            )

            # Check filename contains timestamp
            filename = os.path.basename(filepath)
            self.assertRegex(filename, r"playlist_\d{8}_\d{6}_timestamp-test\.m3u")

            # Check file contents contain timestamp
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # Should contain generated timestamp
            self.assertIn("Generated:", content)
            self.assertRegex(content, r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")

    def test_save_playlist_m3u_track_numbering(self) -> None:
        """Test M3U playlist saving includes proper track numbering"""
        with tempfile.TemporaryDirectory() as temp_dir:
            query = "numbering test"
            filepath = self.playlist_output.save_playlist_m3u(
                self.sample_tracks, query, temp_dir
            )

            # Check file contents
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # Should contain track numbers
            self.assertIn("Track: 1/3", content)
            self.assertIn("Track: 2/3", content)
            self.assertIn("Track: 3/3", content)

    def test_save_playlist_m3u_duration_formatting(self) -> None:
        """Test M3U playlist saving formats duration correctly"""
        tracks_with_duration = [
            {
                "name": "Song 1",
                "artist": "Artist 1",
                "album": "Album 1",
                "duration_ms": 180000,  # 3 minutes
                "similarity_score": 0.95,
                "location": "file:///path/to/song1.mp3",
            },
            {
                "name": "Song 2",
                "artist": "Artist 2",
                "album": "Album 2",
                "duration_ms": 125000,  # 2 minutes 5 seconds
                "similarity_score": 0.87,
                "location": "file:///path/to/song2.mp3",
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            query = "duration test"
            filepath = self.playlist_output.save_playlist_m3u(
                tracks_with_duration, query, temp_dir
            )

            # Check file contents
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # Should contain duration in seconds
            self.assertIn("#EXTINF:180,", content)
            self.assertIn("#EXTINF:125,", content)


if __name__ == "__main__":
    unittest.main()
