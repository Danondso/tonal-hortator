#!/usr/bin/env python3
"""
Tests for playlist output functionality
"""

import os
import tempfile
import unittest

from tonal_hortator.core.playlist_output import PlaylistOutput


class TestPlaylistOutput(unittest.TestCase):
    """Test playlist output functionality"""

    def setUp(self) -> None:
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.playlist_output = PlaylistOutput()

        # Sample track data
        self.sample_tracks = [
            {
                "name": "Test Song 1",
                "artist": "Test Artist 1",
                "album": "Test Album 1",
                "similarity_score": 0.95,
            },
            {
                "name": "Test Song 2",
                "artist": "Test Artist 2",
                "album": "Test Album 2",
                "similarity_score": 0.85,
            },
            {
                "name": "Test Song 3",
                "artist": "Test Artist 3",
                "album": "Test Album 3",
                "similarity_score": 0.75,
            },
        ]

    def tearDown(self) -> None:
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_playlist_m3u_success(self) -> None:
        """Test successful M3U playlist saving"""
        query = "test query"
        file_path = self.playlist_output.save_playlist_m3u(
            self.sample_tracks, query, self.temp_dir
        )

        # Verify file was created
        self.assertTrue(os.path.exists(file_path))
        self.assertTrue(file_path.endswith(".m3u"))

        # Verify file contents
        with open(file_path, "r") as f:
            content = f.read()

        # Should contain M3U header
        self.assertIn("#EXTM3U", content)
        # Should contain query in comment
        self.assertIn(f"Query: {query}", content)
        # Should contain track entries
        self.assertIn("Test Song 1", content)
        self.assertIn("Test Artist 1", content)

    def test_save_playlist_m3u_empty_tracks(self) -> None:
        """Test M3U playlist saving with empty track list"""
        query = "empty query"
        file_path = self.playlist_output.save_playlist_m3u([], query, self.temp_dir)

        # Verify file was created
        self.assertTrue(os.path.exists(file_path))

        # Verify file contents
        with open(file_path, "r") as f:
            content = f.read()

        # Should contain M3U header and query
        self.assertIn("#EXTM3U", content)
        self.assertIn(f"Query: {query}", content)
        # Should not contain any track entries
        self.assertNotIn("EXTINF", content)

    def test_save_playlist_m3u_missing_fields(self) -> None:
        """Test M3U playlist saving with missing track fields"""
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

        query = "incomplete query"
        file_path = self.playlist_output.save_playlist_m3u(
            incomplete_tracks, query, self.temp_dir
        )

        # Verify file was created
        self.assertTrue(os.path.exists(file_path))

        # Verify file contents
        with open(file_path, "r") as f:
            content = f.read()

        # Should handle missing fields gracefully
        self.assertIn("Song 1", content)
        self.assertIn("Song 2", content)


if __name__ == "__main__":
    unittest.main()
