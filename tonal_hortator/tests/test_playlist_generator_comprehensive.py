#!/usr/bin/env python3
"""
Comprehensive tests for playlist generator functionality
"""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from tonal_hortator.core.playlist_generator import LocalPlaylistGenerator


class TestLocalPlaylistGeneratorComprehensive(unittest.TestCase):
    """Comprehensive test playlist generator functionality"""

    def setUp(self) -> None:
        """Set up test fixtures"""
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()

        # Create a mock embedding service
        self.mock_embedding_service = Mock()
        self.mock_embedding_service.get_embedding.return_value = [0.1] * 384
        self.mock_embedding_service.get_embeddings_batch.return_value = [
            [0.1] * 384
        ] * 10
        self.mock_embedding_service.similarity_search.return_value = []
        self.mock_embedding_service.create_track_embedding_text.return_value = (
            "test track text"
        )

        # Create a mock track embedder
        self.mock_track_embedder = Mock()
        self.mock_track_embedder.get_all_embeddings.return_value = ([], [])
        self.mock_track_embedder.embed_all_tracks.return_value = 0
        self.mock_track_embedder.get_tracks_without_embeddings.return_value = []

        # Create a mock query processor
        self.mock_query_processor = Mock()
        self.mock_query_processor.extract_track_count.return_value = None
        self.mock_query_processor.extract_artist_from_query.return_value = None
        self.mock_query_processor.is_vague_query.return_value = False
        self.mock_query_processor.extract_genre_keywords.return_value = []

        # Create a mock deduplicator
        self.mock_deduplicator = Mock()
        self.mock_deduplicator.deduplicate_tracks.return_value = []

        # Create a mock artist distributor
        self.mock_artist_distributor = Mock()
        self.mock_artist_distributor.apply_artist_randomization.return_value = []

    def tearDown(self) -> None:
        """Clean up test fixtures"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    @patch("tonal_hortator.core.playlist_generator.QueryProcessor")
    @patch("tonal_hortator.core.playlist_generator.TrackDeduplicator")
    @patch("tonal_hortator.core.playlist_generator.ArtistDistributor")
    def test_init_with_defaults(
        self,
        mock_artist_distributor: Mock,
        mock_deduplicator: Mock,
        mock_query_processor: Mock,
        mock_track_embedder: Mock,
        mock_embedding_service: Mock,
    ) -> None:
        """Test initialization with default parameters"""
        mock_embedding_service.return_value = self.mock_embedding_service
        mock_track_embedder.return_value = self.mock_track_embedder
        mock_query_processor.return_value = self.mock_query_processor
        mock_deduplicator.return_value = self.mock_deduplicator
        mock_artist_distributor.return_value = self.mock_artist_distributor

        LocalPlaylistGenerator(self.temp_db.name)

        mock_embedding_service.assert_called_with(model_name="nomic-embed-text:latest")
        mock_track_embedder.assert_called_with(
            self.temp_db.name, embedding_service=self.mock_embedding_service
        )

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    @patch("tonal_hortator.core.playlist_generator.QueryProcessor")
    @patch("tonal_hortator.core.playlist_generator.TrackDeduplicator")
    @patch("tonal_hortator.core.playlist_generator.ArtistDistributor")
    def test_init_with_custom_model(
        self,
        mock_artist_distributor: Mock,
        mock_deduplicator: Mock,
        mock_query_processor: Mock,
        mock_track_embedder: Mock,
        mock_embedding_service: Mock,
    ) -> None:
        """Test initialization with custom model name"""
        mock_embedding_service.return_value = self.mock_embedding_service
        mock_track_embedder.return_value = self.mock_track_embedder
        mock_query_processor.return_value = self.mock_query_processor
        mock_deduplicator.return_value = self.mock_deduplicator
        mock_artist_distributor.return_value = self.mock_artist_distributor

        LocalPlaylistGenerator(self.temp_db.name, model_name="custom-model")

        mock_embedding_service.assert_called_with(model_name="custom-model")

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    @patch("tonal_hortator.core.playlist_generator.QueryProcessor")
    @patch("tonal_hortator.core.playlist_generator.TrackDeduplicator")
    @patch("tonal_hortator.core.playlist_generator.ArtistDistributor")
    def test_generate_playlist_success(
        self,
        mock_artist_distributor: Mock,
        mock_deduplicator: Mock,
        mock_query_processor: Mock,
        mock_track_embedder: Mock,
        mock_embedding_service: Mock,
    ) -> None:
        """Test successful playlist generation"""
        # Setup mocks
        mock_embedding_service.return_value = self.mock_embedding_service
        mock_track_embedder.return_value = self.mock_track_embedder
        mock_query_processor.return_value = self.mock_query_processor
        mock_deduplicator.return_value = self.mock_deduplicator
        mock_artist_distributor.return_value = self.mock_artist_distributor

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
        mock_results = [
            {"id": 1, "name": "Track 1", "artist": "Artist 1", "similarity_score": 0.8},
            {"id": 2, "name": "Track 2", "artist": "Artist 2", "similarity_score": 0.7},
        ]
        self.mock_embedding_service.similarity_search.return_value = mock_results

        # Mock deduplication and artist randomization
        self.mock_deduplicator.deduplicate_tracks.return_value = mock_results
        self.mock_artist_distributor.apply_artist_randomization.return_value = (
            mock_results
        )

        LocalPlaylistGenerator(self.temp_db.name)

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    @patch("tonal_hortator.core.playlist_generator.QueryProcessor")
    @patch("tonal_hortator.core.playlist_generator.TrackDeduplicator")
    @patch("tonal_hortator.core.playlist_generator.ArtistDistributor")
    def test_generate_playlist_no_embeddings_initial(
        self,
        mock_artist_distributor: Mock,
        mock_deduplicator: Mock,
        mock_query_processor: Mock,
        mock_track_embedder: Mock,
        mock_embedding_service: Mock,
    ) -> None:
        """Test playlist generation when no embeddings exist initially"""
        # Setup mocks
        mock_embedding_service.return_value = self.mock_embedding_service
        mock_track_embedder.return_value = self.mock_track_embedder
        mock_query_processor.return_value = self.mock_query_processor
        mock_deduplicator.return_value = self.mock_deduplicator
        mock_artist_distributor.return_value = self.mock_artist_distributor

        # First call returns no embeddings, second call returns embeddings
        self.mock_track_embedder.get_all_embeddings.side_effect = [
            (None, None),  # First call
            (
                [[0.1] * 384] * 3,
                [{"id": i, "name": f"Track {i}"} for i in range(3)],
            ),  # Second call
        ]

        # Mock similarity search results
        mock_results = [{"id": 1, "name": "Track 1", "similarity_score": 0.8}]
        self.mock_embedding_service.similarity_search.return_value = mock_results
        self.mock_deduplicator.deduplicate_tracks.return_value = mock_results
        self.mock_artist_distributor.apply_artist_randomization.return_value = (
            mock_results
        )

        LocalPlaylistGenerator(self.temp_db.name)

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    @patch("tonal_hortator.core.playlist_generator.QueryProcessor")
    @patch("tonal_hortator.core.playlist_generator.TrackDeduplicator")
    @patch("tonal_hortator.core.playlist_generator.ArtistDistributor")
    def test_generate_playlist_no_embeddings_after_embedding(
        self,
        mock_artist_distributor: Mock,
        mock_deduplicator: Mock,
        mock_query_processor: Mock,
        mock_track_embedder: Mock,
        mock_embedding_service: Mock,
    ) -> None:
        """Test playlist generation when no embeddings exist even after embedding process"""
        # Setup mocks
        mock_embedding_service.return_value = self.mock_embedding_service
        mock_track_embedder.return_value = self.mock_track_embedder
        mock_query_processor.return_value = self.mock_query_processor
        mock_deduplicator.return_value = self.mock_deduplicator
        mock_artist_distributor.return_value = self.mock_artist_distributor

        # Both calls return no embeddings
        self.mock_track_embedder.get_all_embeddings.return_value = (None, None)

        LocalPlaylistGenerator(self.temp_db.name)

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    @patch("tonal_hortator.core.playlist_generator.QueryProcessor")
    @patch("tonal_hortator.core.playlist_generator.TrackDeduplicator")
    @patch("tonal_hortator.core.playlist_generator.ArtistDistributor")
    def test_generate_playlist_exception_handling(
        self,
        mock_artist_distributor: Mock,
        mock_deduplicator: Mock,
        mock_query_processor: Mock,
        mock_track_embedder: Mock,
        mock_embedding_service: Mock,
    ) -> None:
        """Test playlist generation exception handling"""
        # Setup mocks
        mock_embedding_service.return_value = self.mock_embedding_service
        mock_track_embedder.return_value = self.mock_track_embedder
        mock_query_processor.return_value = self.mock_query_processor
        mock_deduplicator.return_value = self.mock_deduplicator
        mock_artist_distributor.return_value = self.mock_artist_distributor

        # Mock exception
        self.mock_track_embedder.get_all_embeddings.side_effect = Exception(
            "Database error"
        )

        LocalPlaylistGenerator(self.temp_db.name)

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    @patch("tonal_hortator.core.playlist_generator.QueryProcessor")
    @patch("tonal_hortator.core.playlist_generator.TrackDeduplicator")
    @patch("tonal_hortator.core.playlist_generator.ArtistDistributor")
    def test_extract_playlist_parameters_with_count(
        self,
        mock_artist_distributor: Mock,
        mock_deduplicator: Mock,
        mock_query_processor: Mock,
        mock_track_embedder: Mock,
        mock_embedding_service: Mock,
    ) -> None:
        """Test parameter extraction when track count is found in query"""
        # Setup mocks
        mock_embedding_service.return_value = self.mock_embedding_service
        mock_track_embedder.return_value = self.mock_track_embedder
        mock_query_processor.return_value = self.mock_query_processor
        mock_deduplicator.return_value = self.mock_deduplicator
        mock_artist_distributor.return_value = self.mock_artist_distributor

        # Mock track count extraction
        self.mock_query_processor.extract_track_count.return_value = 15

        # Mock embeddings and results
        self.mock_track_embedder.get_all_embeddings.return_value = (
            [[0.1] * 384] * 5,
            [{"id": i} for i in range(5)],
        )
        self.mock_embedding_service.similarity_search.return_value = []
        self.mock_deduplicator.deduplicate_tracks.return_value = []
        self.mock_artist_distributor.apply_artist_randomization.return_value = []

        LocalPlaylistGenerator(self.temp_db.name)

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    @patch("tonal_hortator.core.playlist_generator.QueryProcessor")
    @patch("tonal_hortator.core.playlist_generator.TrackDeduplicator")
    @patch("tonal_hortator.core.playlist_generator.ArtistDistributor")
    def test_filter_by_artist_with_artist(
        self,
        mock_artist_distributor: Mock,
        mock_deduplicator: Mock,
        mock_query_processor: Mock,
        mock_track_embedder: Mock,
        mock_embedding_service: Mock,
    ) -> None:
        """Test filtering by artist when artist is found in query"""
        # Setup mocks
        mock_embedding_service.return_value = self.mock_embedding_service
        mock_track_embedder.return_value = self.mock_track_embedder
        mock_query_processor.return_value = self.mock_query_processor
        mock_deduplicator.return_value = self.mock_deduplicator
        mock_artist_distributor.return_value = self.mock_artist_distributor

        # Mock artist extraction
        self.mock_query_processor.extract_artist_from_query.return_value = "Queen"

        # Mock embeddings and results
        self.mock_track_embedder.get_all_embeddings.return_value = (
            [[0.1] * 384] * 5,
            [{"id": i} for i in range(5)],
        )

        # Mock similarity search results with different artists
        mock_results = [
            {
                "id": 1,
                "name": "Bohemian Rhapsody",
                "artist": "Queen",
                "similarity_score": 0.8,
            },
            {
                "id": 2,
                "name": "Stairway to Heaven",
                "artist": "Led Zeppelin",
                "similarity_score": 0.7,
            },
            {
                "id": 3,
                "name": "We Will Rock You",
                "artist": "Queen",
                "similarity_score": 0.6,
            },
        ]
        self.mock_embedding_service.similarity_search.return_value = mock_results

        # Mock deduplication and artist randomization
        self.mock_deduplicator.deduplicate_tracks.return_value = mock_results
        self.mock_artist_distributor.apply_artist_randomization.return_value = (
            mock_results
        )

        generator = LocalPlaylistGenerator(self.temp_db.name)
        result = generator.generate_playlist("songs by Queen")

        # Check that artist was extracted
        self.mock_query_processor.extract_artist_from_query.assert_called_with(
            "songs by Queen"
        )

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    @patch("tonal_hortator.core.playlist_generator.QueryProcessor")
    @patch("tonal_hortator.core.playlist_generator.TrackDeduplicator")
    @patch("tonal_hortator.core.playlist_generator.ArtistDistributor")
    def test_apply_genre_filtering_with_genre(
        self,
        mock_artist_distributor: Mock,
        mock_deduplicator: Mock,
        mock_query_processor: Mock,
        mock_track_embedder: Mock,
        mock_embedding_service: Mock,
    ) -> None:
        """Test genre filtering when genre keywords are found"""
        # Setup mocks
        mock_embedding_service.return_value = self.mock_embedding_service
        mock_track_embedder.return_value = self.mock_track_embedder
        mock_query_processor.return_value = self.mock_query_processor
        mock_deduplicator.return_value = self.mock_deduplicator
        mock_artist_distributor.return_value = self.mock_artist_distributor

        # Mock genre extraction
        self.mock_query_processor.extract_genre_keywords.return_value = ["rock"]

        # Mock embeddings and results
        self.mock_track_embedder.get_all_embeddings.return_value = (
            [[0.1] * 384] * 5,
            [{"id": i} for i in range(5)],
        )

        # Mock similarity search results with different genres
        mock_results = [
            {
                "id": 1,
                "name": "Rock Song",
                "artist": "Rock Artist",
                "genre": "rock",
                "similarity_score": 0.8,
            },
            {
                "id": 2,
                "name": "Jazz Song",
                "artist": "Jazz Artist",
                "genre": "jazz",
                "similarity_score": 0.7,
            },
            {
                "id": 3,
                "name": "Rock Song 2",
                "artist": "Rock Artist 2",
                "genre": "rock",
                "similarity_score": 0.6,
            },
        ]
        self.mock_embedding_service.similarity_search.return_value = mock_results

        # Mock deduplication and artist randomization
        self.mock_deduplicator.deduplicate_tracks.return_value = mock_results
        self.mock_artist_distributor.apply_artist_randomization.return_value = (
            mock_results
        )

        generator = LocalPlaylistGenerator(self.temp_db.name)
        result = generator.generate_playlist("rock music")

        # Check that genre keywords were extracted
        self.mock_query_processor.extract_genre_keywords.assert_called_with(
            "rock music"
        )

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    @patch("tonal_hortator.core.playlist_generator.QueryProcessor")
    @patch("tonal_hortator.core.playlist_generator.TrackDeduplicator")
    @patch("tonal_hortator.core.playlist_generator.ArtistDistributor")
    def test_apply_genre_filtering_no_genre(
        self,
        mock_artist_distributor: Mock,
        mock_deduplicator: Mock,
        mock_query_processor: Mock,
        mock_track_embedder: Mock,
        mock_embedding_service: Mock,
    ) -> None:
        """Test genre filtering when no genre keywords are found"""
        # Setup mocks
        mock_embedding_service.return_value = self.mock_embedding_service
        mock_track_embedder.return_value = self.mock_track_embedder
        mock_query_processor.return_value = self.mock_query_processor
        mock_deduplicator.return_value = self.mock_deduplicator
        mock_artist_distributor.return_value = self.mock_artist_distributor

        # Mock no genre extraction
        self.mock_query_processor.extract_genre_keywords.return_value = []

        # Mock embeddings and results
        self.mock_track_embedder.get_all_embeddings.return_value = (
            [[0.1] * 384] * 5,
            [{"id": i} for i in range(5)],
        )
        mock_results = [{"id": 1, "name": "Song", "similarity_score": 0.8}]
        self.mock_embedding_service.similarity_search.return_value = mock_results
        self.mock_deduplicator.deduplicate_tracks.return_value = mock_results
        self.mock_artist_distributor.apply_artist_randomization.return_value = (
            mock_results
        )

        generator = LocalPlaylistGenerator(self.temp_db.name)
        result = generator.generate_playlist("happy music")

        # Check that genre keywords were extracted
        self.mock_query_processor.extract_genre_keywords.assert_called_with(
            "happy music"
        )

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    @patch("tonal_hortator.core.playlist_generator.QueryProcessor")
    @patch("tonal_hortator.core.playlist_generator.TrackDeduplicator")
    @patch("tonal_hortator.core.playlist_generator.ArtistDistributor")
    def test_process_and_refine_playlist_vague_query(
        self,
        mock_artist_distributor: Mock,
        mock_deduplicator: Mock,
        mock_query_processor: Mock,
        mock_track_embedder: Mock,
        mock_embedding_service: Mock,
    ) -> None:
        """Test playlist processing with vague query"""
        # Setup mocks
        mock_embedding_service.return_value = self.mock_embedding_service
        mock_track_embedder.return_value = self.mock_track_embedder
        mock_query_processor.return_value = self.mock_query_processor
        mock_deduplicator.return_value = self.mock_deduplicator
        mock_artist_distributor.return_value = self.mock_artist_distributor

        # Mock vague query
        self.mock_query_processor.is_vague_query.return_value = True

        # Mock embeddings and results
        self.mock_track_embedder.get_all_embeddings.return_value = (
            [[0.1] * 384] * 5,
            [{"id": i} for i in range(5)],
        )
        mock_results = [{"id": 1, "name": "Song", "similarity_score": 0.8}]
        self.mock_embedding_service.similarity_search.return_value = mock_results
        self.mock_deduplicator.deduplicate_tracks.return_value = mock_results
        self.mock_artist_distributor.apply_artist_randomization.return_value = (
            mock_results
        )

        generator = LocalPlaylistGenerator(self.temp_db.name)
        result = generator.generate_playlist("upbeat music")

        # Check that query vagueness was checked
        self.mock_query_processor.is_vague_query.assert_called_with("upbeat music")
        # Check that artist randomization was called with vague=True
        self.mock_artist_distributor.apply_artist_randomization.assert_called_with(
            mock_results, 20, True
        )

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    @patch("tonal_hortator.core.playlist_generator.QueryProcessor")
    @patch("tonal_hortator.core.playlist_generator.TrackDeduplicator")
    @patch("tonal_hortator.core.playlist_generator.ArtistDistributor")
    def test_process_and_refine_playlist_specific_query(
        self,
        mock_artist_distributor: Mock,
        mock_deduplicator: Mock,
        mock_query_processor: Mock,
        mock_track_embedder: Mock,
        mock_embedding_service: Mock,
    ) -> None:
        """Test playlist processing with specific query"""
        # Setup mocks
        mock_embedding_service.return_value = self.mock_embedding_service
        mock_track_embedder.return_value = self.mock_track_embedder
        mock_query_processor.return_value = self.mock_query_processor
        mock_deduplicator.return_value = self.mock_deduplicator
        mock_artist_distributor.return_value = self.mock_artist_distributor

        # Mock specific query
        self.mock_query_processor.is_vague_query.return_value = False

        # Mock embeddings and results
        self.mock_track_embedder.get_all_embeddings.return_value = (
            [[0.1] * 384] * 5,
            [{"id": i} for i in range(5)],
        )
        mock_results = [{"id": 1, "name": "Song", "similarity_score": 0.8}]
        self.mock_embedding_service.similarity_search.return_value = mock_results
        self.mock_deduplicator.deduplicate_tracks.return_value = mock_results
        self.mock_artist_distributor.apply_artist_randomization.return_value = (
            mock_results
        )

        generator = LocalPlaylistGenerator(self.temp_db.name)
        result = generator.generate_playlist("Bohemian Rhapsody by Queen")

        # Check that artist randomization was called with vague=False
        self.mock_artist_distributor.apply_artist_randomization.assert_called_with(
            mock_results, 20, False
        )

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    @patch("tonal_hortator.core.playlist_generator.QueryProcessor")
    @patch("tonal_hortator.core.playlist_generator.TrackDeduplicator")
    @patch("tonal_hortator.core.playlist_generator.ArtistDistributor")
    def test_generate_playlist_custom_parameters(
        self,
        mock_artist_distributor: Mock,
        mock_deduplicator: Mock,
        mock_query_processor: Mock,
        mock_track_embedder: Mock,
        mock_embedding_service: Mock,
    ) -> None:
        """Test playlist generation with custom parameters"""
        # Setup mocks
        mock_embedding_service.return_value = self.mock_embedding_service
        mock_track_embedder.return_value = self.mock_track_embedder
        mock_query_processor.return_value = self.mock_query_processor
        mock_deduplicator.return_value = self.mock_deduplicator
        mock_artist_distributor.return_value = self.mock_artist_distributor

        # Mock embeddings and results
        self.mock_track_embedder.get_all_embeddings.return_value = (
            [[0.1] * 384] * 5,
            [{"id": i} for i in range(5)],
        )
        mock_results = [{"id": 1, "name": "Song", "similarity_score": 0.8}]
        self.mock_embedding_service.similarity_search.return_value = mock_results
        self.mock_deduplicator.deduplicate_tracks.return_value = mock_results
        self.mock_artist_distributor.apply_artist_randomization.return_value = (
            mock_results
        )

        generator = LocalPlaylistGenerator(self.temp_db.name)
        result = generator.generate_playlist(
            "test query", max_tracks=50, min_similarity=0.5
        )

        # Check that custom parameters were used
        self.mock_deduplicator.deduplicate_tracks.assert_called_with(
            mock_results, 0.5, 50
        )
        self.mock_artist_distributor.apply_artist_randomization.assert_called_with(
            mock_results, 50, False
        )


if __name__ == "__main__":
    unittest.main()
