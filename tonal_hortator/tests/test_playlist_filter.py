#!/usr/bin/env python3
"""
Tests for tonal_hortator.core.playlist.playlist_filter
"""

import unittest

from tonal_hortator.core.playlist.playlist_filter import PlaylistFilter


class TestPlaylistFilter(unittest.TestCase):
    """Test PlaylistFilter"""

    def test_apply_genre_filtering(self) -> None:
        """Test genre filtering functionality in PlaylistFilter"""
        filter_obj = PlaylistFilter()

        tracks = [
            {"name": "Song1", "genre": "Rock", "similarity_score": 0.8},
            {"name": "Song2", "genre": "Jazz", "similarity_score": 0.7},
            {"name": "Song3", "genre": "Rock", "similarity_score": 0.6},
        ]

        result = filter_obj.apply_genre_filtering(["rock"], tracks)

        # Should filter to only rock tracks (with boosting)
        self.assertTrue(any(track["genre"].lower() == "rock" for track in result))
        # Should have boosted similarity scores for rock tracks
        rock_tracks = [track for track in result if track["genre"].lower() == "rock"]
        self.assertTrue(all(track.get("genre_boosted", False) for track in rock_tracks))

    def test_apply_genre_filtering_no_genre_keywords(self) -> None:
        """Test genre filtering with no genre keywords"""
        filter_obj = PlaylistFilter()

        tracks = [
            {"name": "Song1", "genre": "Rock", "similarity_score": 0.8},
            {"name": "Song2", "genre": "Jazz", "similarity_score": 0.7},
        ]

        result = filter_obj.apply_genre_filtering([], tracks)

        # Should return all tracks unchanged when no genres specified
        self.assertEqual(len(result), 2)
        self.assertEqual(result, tracks)

    def test_apply_genre_filtering_empty_tracks(self) -> None:
        """Test genre filtering with empty track list"""
        filter_obj = PlaylistFilter()

        result = filter_obj.apply_genre_filtering(["rock"], [])

        # Should return empty list
        self.assertEqual(result, [])

    def test_apply_genre_filtering_mixed_genres(self) -> None:
        """Test genre filtering with multiple genres"""
        filter_obj = PlaylistFilter()

        tracks = [
            {"name": "Song1", "genre": "Rock", "similarity_score": 0.8},
            {"name": "Song2", "genre": "Jazz", "similarity_score": 0.7},
            {"name": "Song3", "genre": "Pop", "similarity_score": 0.6},
            {"name": "Song4", "genre": "Electronic", "similarity_score": 0.5},
        ]

        result = filter_obj.apply_genre_filtering(["rock", "jazz"], tracks)

        # Should include rock and jazz tracks
        genres_in_result = {track["genre"].lower() for track in result}
        self.assertIn("rock", genres_in_result)
        self.assertIn("jazz", genres_in_result)

        # Should boost matching tracks
        rock_and_jazz_tracks = [
            track for track in result if track["genre"].lower() in ["rock", "jazz"]
        ]
        self.assertTrue(
            all(track.get("genre_boosted", False) for track in rock_and_jazz_tracks)
        )


if __name__ == "__main__":
    unittest.main()
