#!/usr/bin/env python3
"""
Tests for tonal_hortator.core.playlist.playlist_exporter
"""

import os
import tempfile
import unittest

from tonal_hortator.core.playlist.playlist_exporter import PlaylistExporter


class TestPlaylistExporter(unittest.TestCase):
    """Test PlaylistExporter"""

    def test_save_playlist_m3u(self) -> None:
        """Test saving playlist as M3U file"""
        exporter = PlaylistExporter()

        tracks = [
            {
                "name": "Song1",
                "artist": "Artist1",
                "album": "Album1",
                "duration_ms": 180000,
                "similarity_score": 0.8,
                "location": "/path/to/song1.mp3",
            },
            {
                "name": "Song2",
                "artist": "Artist2",
                "album": "Album2",
                "duration_ms": 200000,
                "similarity_score": 0.7,
                "location": "/path/to/song2.mp3",
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = exporter.save_playlist_m3u(tracks, "test query", temp_dir)

            # Check that file was created
            self.assertTrue(os.path.exists(filepath))

            # Check file content
            with open(filepath, "r") as f:
                content = f.read()
                self.assertIn("#EXTM3U", content)
                self.assertIn("# Query: test query", content)
                self.assertIn("Artist1 - Song1", content)
                self.assertIn("Artist2 - Song2", content)

    def test_save_playlist_m3u_creates_clean_filename(self) -> None:
        """Test that save_playlist_m3u creates clean filenames"""
        exporter = PlaylistExporter()

        # Mock track data
        tracks = [
            {
                "artist": "Test Artist",
                "name": "Test Song",
                "album": "Test Album",
                "duration_ms": 180000,
                "similarity_score": 0.8,
                "location": "/path/to/song.mp3",
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test saving playlist
            filepath = exporter.save_playlist_m3u(tracks, "jazz for studying", temp_dir)

            # Check that filename is clean (no timestamps or underscores)
            filename = os.path.basename(filepath)
            self.assertNotIn("playlist_", filename)
            self.assertNotIn("_", filename)
            self.assertTrue(filename.startswith("Jazz Studying"))
            self.assertTrue(filename.endswith(".m3u"))

            # Check that file exists and has content
            self.assertTrue(os.path.exists(filepath))
            with open(filepath, "r") as f:
                content = f.read()
                self.assertIn("# Playlist: Jazz Studying", content)
                self.assertIn("# Query: jazz for studying", content)

    def test_save_playlist_m3u_handles_special_characters(self) -> None:
        """Test that save_playlist_m3u handles special characters in filenames"""
        exporter = PlaylistExporter()

        tracks = [
            {
                "artist": "Test Artist",
                "name": "Test Song",
                "album": "Test Album",
                "duration_ms": 180000,
                "similarity_score": 0.8,
                "location": "/path/to/song.mp3",
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with special characters that should be removed
            filepath = exporter.save_playlist_m3u(tracks, "rock & roll! @#$%", temp_dir)

            filename = os.path.basename(filepath)
            # Should not contain special characters
            self.assertNotIn("&", filename)
            self.assertNotIn("!", filename)
            self.assertNotIn("@", filename)
            self.assertNotIn("#", filename)
            self.assertNotIn("$", filename)
            self.assertNotIn("%", filename)

            # Should contain cleaned name
            self.assertIn("Rock", filename)
            self.assertIn("Roll", filename)

    def test_save_playlist_m3u_handles_missing_location(self) -> None:
        """Test that save_playlist_m3u handles tracks with missing location"""
        exporter = PlaylistExporter()

        tracks = [
            {
                "artist": "Test Artist",
                "name": "Test Song",
                "album": "Test Album",
                "duration_ms": 180000,
                "similarity_score": 0.8,
                "location": "",  # Missing location
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = exporter.save_playlist_m3u(tracks, "test query", temp_dir)

            # Check that file was created
            self.assertTrue(os.path.exists(filepath))

            # Check file content handles missing location
            with open(filepath, "r") as f:
                content = f.read()
                self.assertIn(
                    "# Missing file location for: Test Artist - Test Song", content
                )

    def test_save_playlist_m3u_handles_file_protocol(self) -> None:
        """Test that save_playlist_m3u handles file:// protocol URLs"""
        exporter = PlaylistExporter()

        tracks = [
            {
                "artist": "Test Artist",
                "name": "Test Song",
                "album": "Test Album",
                "duration_ms": 180000,
                "similarity_score": 0.8,
                "location": "file:///Users/test/Music/song.mp3",
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = exporter.save_playlist_m3u(tracks, "test query", temp_dir)

            # Check that file was created
            self.assertTrue(os.path.exists(filepath))

            # Check file content converts file:// URLs
            with open(filepath, "r") as f:
                content = f.read()
                self.assertIn("/Users/test/Music/song.mp3", content)
                self.assertNotIn("file://", content)


if __name__ == "__main__":
    unittest.main()
