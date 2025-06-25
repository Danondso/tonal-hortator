#!/usr/bin/env python3
"""
Tests for playlist generator functionality
"""

import os
import tempfile
import unittest
from unittest.mock import Mock

from tonal_hortator.core.playlist_generator import LocalPlaylistGenerator


class TestLocalPlaylistGenerator(unittest.TestCase):
    """Test Local Playlist Generator functionality"""

    def setUp(self) -> None:
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_generator.db")

        # Create generator with actual implementation
        self.generator = LocalPlaylistGenerator(db_path=self.db_path)

        # Create mock components and replace the real ones
        self.mock_query_processor = Mock()
        self.mock_track_embedder = Mock()
        self.mock_artist_distributor = Mock()
        self.mock_deduplicator = Mock()
        self.mock_embedding_service = Mock()

        # Replace the real components with mocks
        self.generator.query_processor = self.mock_query_processor
        self.generator.track_embedder = self.mock_track_embedder
        self.generator.artist_distributor = self.mock_artist_distributor
        self.generator.deduplicator = self.mock_deduplicator
        self.generator.embedding_service = self.mock_embedding_service

    def tearDown(self) -> None:
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_playlist_success(self) -> None:
        """Test successful playlist generation"""
        # Mock query processing
        mock_query = "test query"
        self.mock_query_processor.extract_track_count.return_value = None
        self.mock_query_processor.extract_artist_from_query.return_value = None
        self.mock_query_processor.is_vague_query.return_value = False
        self.mock_query_processor.extract_genre_keywords.return_value = []

        # Mock embeddings and track data
        mock_embeddings = [[0.1] * 384] * 5
        mock_track_data = [
            {"id": i, "name": f"Track {i}", "artist": f"Artist {i}"} for i in range(5)
        ]
        self.mock_track_embedder.get_all_embeddings.return_value = (
            mock_embeddings,
            mock_track_data,
        )

        # Mock similarity search results
        mock_tracks = [
            {"name": "Song 1", "artist": "Artist 1", "similarity_score": 0.9},
            {"name": "Song 2", "artist": "Artist 2", "similarity_score": 0.8},
            {"name": "Song 3", "artist": "Artist 3", "similarity_score": 0.7},
        ]
        self.mock_embedding_service.similarity_search.return_value = mock_tracks

        # Mock deduplication and artist randomization
        self.mock_deduplicator.deduplicate_tracks.return_value = mock_tracks
        self.mock_artist_distributor.apply_artist_randomization.return_value = (
            mock_tracks
        )

        # Generate playlist
        result = self.generator.generate_playlist(mock_query, max_tracks=3)

        # Verify result
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["name"], "Song 1")

        # Verify method calls
        self.mock_query_processor.extract_track_count.assert_called_once_with(
            mock_query
        )
        self.mock_track_embedder.get_all_embeddings.assert_called_once()
        self.mock_embedding_service.similarity_search.assert_called_once()
        self.mock_deduplicator.deduplicate_tracks.assert_called_once()
        self.mock_artist_distributor.apply_artist_randomization.assert_called_once()

    def test_generate_playlist_no_tracks_found(self) -> None:
        """Test playlist generation when no tracks are found"""
        # Mock query processing
        mock_query = "test query"
        self.mock_query_processor.extract_track_count.return_value = None
        self.mock_query_processor.extract_artist_from_query.return_value = None
        self.mock_query_processor.is_vague_query.return_value = False
        self.mock_query_processor.extract_genre_keywords.return_value = []

        # Mock no embeddings found
        self.mock_track_embedder.get_all_embeddings.return_value = (None, None)
        self.mock_track_embedder.embed_all_tracks.return_value = 0

        # Generate playlist
        result = self.generator.generate_playlist(mock_query, max_tracks=3)

        # Verify result
        self.assertEqual(result, [])

    def test_generate_playlist_vague_query(self) -> None:
        """Test playlist generation with vague query"""
        # Mock query processing
        mock_query = "test query"
        self.mock_query_processor.extract_track_count.return_value = None
        self.mock_query_processor.extract_artist_from_query.return_value = None
        self.mock_query_processor.is_vague_query.return_value = True
        self.mock_query_processor.extract_genre_keywords.return_value = []

        # Mock embeddings and track data
        mock_embeddings = [[0.1] * 384] * 5
        mock_track_data = [
            {"id": i, "name": f"Track {i}", "artist": f"Artist {i}"} for i in range(5)
        ]
        self.mock_track_embedder.get_all_embeddings.return_value = (
            mock_embeddings,
            mock_track_data,
        )

        # Mock similarity search results
        mock_tracks = [
            {"name": "Song 1", "artist": "Artist 1", "similarity_score": 0.9},
            {"name": "Song 2", "artist": "Artist 2", "similarity_score": 0.8},
        ]
        self.mock_embedding_service.similarity_search.return_value = mock_tracks

        # Mock deduplication and artist randomization
        self.mock_deduplicator.deduplicate_tracks.return_value = mock_tracks
        self.mock_artist_distributor.apply_artist_randomization.return_value = (
            mock_tracks
        )

        # Generate playlist
        result = self.generator.generate_playlist(mock_query, max_tracks=2)

        # Verify result
        self.assertEqual(len(result), 2)

        # Verify that artist randomization was called with is_vague=True
        self.mock_artist_distributor.apply_artist_randomization.assert_called_once_with(
            mock_tracks, 2, True
        )

    def test_generate_playlist_exception_handling(self) -> None:
        """Test playlist generation with exception handling"""
        # Mock query processing to raise exception
        mock_query = "test query"
        self.mock_track_embedder.get_all_embeddings.side_effect = Exception(
            "Database error"
        )

        # Generate playlist
        with self.assertRaises(Exception):
            self.generator.generate_playlist(mock_query, max_tracks=3)

    def test_generate_playlist_with_custom_max_tracks(self) -> None:
        """Test playlist generation with custom max_tracks"""
        # Mock query processing
        mock_query = "test query"
        self.mock_query_processor.extract_track_count.return_value = None
        self.mock_query_processor.extract_artist_from_query.return_value = None
        self.mock_query_processor.is_vague_query.return_value = False
        self.mock_query_processor.extract_genre_keywords.return_value = []

        # Mock embeddings and track data
        mock_embeddings = [[0.1] * 384] * 5
        mock_track_data = [
            {"id": i, "name": f"Track {i}", "artist": f"Artist {i}"} for i in range(5)
        ]
        self.mock_track_embedder.get_all_embeddings.return_value = (
            mock_embeddings,
            mock_track_data,
        )

        # Mock similarity search results
        mock_tracks = [
            {"name": "Song 1", "artist": "Artist 1", "similarity_score": 0.9},
            {"name": "Song 2", "artist": "Artist 2", "similarity_score": 0.8},
            {"name": "Song 3", "artist": "Artist 3", "similarity_score": 0.7},
            {"name": "Song 4", "artist": "Artist 4", "similarity_score": 0.6},
            {"name": "Song 5", "artist": "Artist 5", "similarity_score": 0.5},
        ]
        self.mock_embedding_service.similarity_search.return_value = mock_tracks

        # Mock deduplication and artist randomization
        self.mock_deduplicator.deduplicate_tracks.return_value = mock_tracks
        self.mock_artist_distributor.apply_artist_randomization.return_value = (
            mock_tracks
        )

        # Generate playlist with custom max_tracks
        result = self.generator.generate_playlist(mock_query, max_tracks=5)

        # Verify result
        self.assertEqual(len(result), 5)

        # Verify that similarity search was called with correct top_k
        self.mock_embedding_service.similarity_search.assert_called_once_with(
            mock_query, mock_embeddings, mock_track_data, top_k=25
        )

    def test_extract_playlist_parameters_with_count(self) -> None:
        """Test parameter extraction when track count is found in query"""
        mock_query = "15 tracks of rock music"
        self.mock_query_processor.extract_track_count.return_value = 15

        result = self.generator._extract_playlist_parameters(mock_query, 20)

        self.assertEqual(result, 15)
        self.mock_query_processor.extract_track_count.assert_called_once_with(
            mock_query
        )

    def test_filter_by_artist_with_artist(self) -> None:
        """Test filtering by artist when artist is found in query"""
        mock_query = "songs by Queen"
        self.mock_query_processor.extract_artist_from_query.return_value = "Queen"

        mock_results = [
            {"name": "Bohemian Rhapsody", "artist": "Queen", "similarity_score": 0.8},
            {
                "name": "Stairway to Heaven",
                "artist": "Led Zeppelin",
                "similarity_score": 0.7,
            },
        ]

        result = self.generator._filter_by_artist(mock_query, mock_results)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["artist"], "Queen")

    def test_apply_genre_filtering_with_genre(self) -> None:
        """Test genre filtering when genre keywords are found"""
        mock_query = "rock music"
        self.mock_query_processor.extract_genre_keywords.return_value = ["rock"]

        mock_results = [
            {
                "name": "Rock Song",
                "artist": "Rock Artist",
                "genre": "rock",
                "similarity_score": 0.8,
            },
            {
                "name": "Jazz Song",
                "artist": "Jazz Artist",
                "genre": "jazz",
                "similarity_score": 0.7,
            },
        ]

        result = self.generator._apply_genre_filtering(mock_query, mock_results)

        # Should boost rock songs
        self.assertEqual(len(result), 2)
        self.assertTrue(result[0]["genre_boosted"])
        self.assertFalse(result[1]["genre_boosted"])


if __name__ == "__main__":
    unittest.main()
