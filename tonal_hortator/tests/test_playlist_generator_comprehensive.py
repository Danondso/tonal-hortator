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


if __name__ == "__main__":
    unittest.main()
