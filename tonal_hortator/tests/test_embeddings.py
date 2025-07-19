#!/usr/bin/env python3
"""
Tests for tonal_hortator.core.embeddings
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import pytest

from tonal_hortator.core.embeddings.embeddings import OllamaEmbeddingService


class TestOllamaEmbeddingService(unittest.TestCase):
    """Test OllamaEmbeddingService"""

    @patch("tonal_hortator.core.embeddings.embeddings.ollama.Client")
    def test_init_success(self, mock_client_class: Mock) -> None:
        """Test successful initialization"""
        # Mock the client and its responses
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock models response
        mock_models_response = Mock()
        mock_models_response.models = [Mock(model="nomic-embed-text:latest")]
        mock_client.list.return_value = mock_models_response

        # Mock embedding response
        mock_embedding_response = {"embedding": [0.1] * 384}
        mock_client.embeddings.return_value = mock_embedding_response

        service = OllamaEmbeddingService()

        self.assertEqual(service.model_name, "nomic-embed-text:latest")
        self.assertIsNotNone(service.client)
        self.assertEqual(service._embedding_dimension, 384)

    @patch("tonal_hortator.core.embeddings.embeddings.ollama.Client")
    def test_init_with_custom_host(self, mock_client_class: Mock) -> None:
        """Test initialization with custom host"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_models_response = Mock()
        mock_models_response.models = [Mock(model="nomic-embed-text:latest")]
        mock_client.list.return_value = mock_models_response

        mock_embedding_response = {"embedding": [0.1] * 384}
        mock_client.embeddings.return_value = mock_embedding_response

        service = OllamaEmbeddingService(host="http://localhost:11434")

        self.assertEqual(service.host, "http://localhost:11434")
        mock_client_class.assert_called_with(host="http://localhost:11434")

    @patch("tonal_hortator.core.embeddings.embeddings.ollama.Client")
    def test_init_model_not_found(self, mock_client_class: Mock) -> None:
        """Test initialization with model not found"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_models_response = Mock()
        mock_models_response.models = [Mock(model="other-model")]
        mock_client.list.return_value = mock_models_response

        with pytest.raises(
            Exception, match="Model 'nomic-embed-text:latest' not available"
        ):
            OllamaEmbeddingService()

    @patch("tonal_hortator.core.embeddings.embeddings.ollama.Client")
    def test_init_dict_response_format(self, mock_client_class: Mock) -> None:
        """Test initialization with dict response format"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock dict response format
        mock_models_response = {"models": [{"name": "nomic-embed-text:latest"}]}
        mock_client.list.return_value = mock_models_response

        mock_embedding_response = {"embedding": [0.1] * 384}
        mock_client.embeddings.return_value = mock_embedding_response

        service = OllamaEmbeddingService()

        self.assertEqual(service.model_name, "nomic-embed-text:latest")

    @patch("tonal_hortator.core.embeddings.embeddings.ollama.Client")
    def test_init_list_response_format(self, mock_client_class: Mock) -> None:
        """Test initialization with list response format"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock list response format
        mock_models_response = ["nomic-embed-text:latest"]
        mock_client.list.return_value = mock_models_response

        mock_embedding_response = {"embedding": [0.1] * 384}
        mock_client.embeddings.return_value = mock_embedding_response

        service = OllamaEmbeddingService()

        self.assertEqual(service.model_name, "nomic-embed-text:latest")

    @patch("tonal_hortator.core.embeddings.embeddings.ollama.Client")
    def test_init_unexpected_response_format(self, mock_client_class: Mock) -> None:
        """Test initialization with unexpected response format"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock unexpected response format
        mock_client.list.return_value = "unexpected"

        with pytest.raises(Exception, match="Unexpected models response format"):
            OllamaEmbeddingService()

    @patch("tonal_hortator.core.embeddings.embeddings.ollama.Client")
    def test_get_embedding_success(self, mock_client_class: Mock) -> None:
        """Test successful embedding generation"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_models_response = Mock()
        mock_models_response.models = [Mock(model="nomic-embed-text:latest")]
        mock_client.list.return_value = mock_models_response

        mock_embedding_response = {"embedding": [0.1, 0.2, 0.3]}
        mock_client.embeddings.return_value = mock_embedding_response

        service = OllamaEmbeddingService()

        result = service.get_embedding("test text")

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 3)
        self.assertEqual(result.dtype, np.float32)
        mock_client.embeddings.assert_called_with(
            model="nomic-embed-text:latest", prompt="test text"
        )

    @patch("tonal_hortator.core.embeddings.embeddings.ollama.Client")
    def test_get_embedding_empty_or_whitespace_text(
        self, mock_client_class: Mock
    ) -> None:
        """Test embedding generation with empty or whitespace text"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_models_response = Mock()
        mock_models_response.models = [Mock(model="nomic-embed-text:latest")]
        mock_client.list.return_value = mock_models_response

        mock_embedding_response = {"embedding": [0.1] * 384}
        mock_client.embeddings.return_value = mock_embedding_response

        service = OllamaEmbeddingService()

        # Test empty text
        result = service.get_embedding("")
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 384)
        self.assertTrue(np.all(result == 0))

        # Test whitespace text
        result = service.get_embedding("   ")
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 384)
        self.assertTrue(np.all(result == 0))

    @patch("tonal_hortator.core.embeddings.embeddings.ollama.Client")
    def test_get_embedding_error(self, mock_client_class: Mock) -> None:
        """Test embedding generation with error"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_models_response = Mock()
        mock_models_response.models = [Mock(model="nomic-embed-text:latest")]
        mock_client.list.return_value = mock_models_response

        mock_client.embeddings.side_effect = Exception("Embedding error")

        service = OllamaEmbeddingService()

        with pytest.raises(Exception, match="Embedding error"):
            service.get_embedding("test text")

    @patch("tonal_hortator.core.embeddings.embeddings.ollama.Client")
    def test_get_embeddings_batch_success(self, mock_client_class: Mock) -> None:
        """Test successful batch embedding generation"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_models_response = Mock()
        mock_models_response.models = [Mock(model="nomic-embed-text:latest")]
        mock_client.list.return_value = mock_models_response

        mock_embedding_response = {"embedding": [0.1, 0.2, 0.3]}
        mock_client.embeddings.return_value = mock_embedding_response

        service = OllamaEmbeddingService()

        texts = ["text1", "text2", "text3"]
        results = service.get_embeddings_batch(texts)

        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(len(result), 3)
            self.assertEqual(result.dtype, np.float32)
        # 1 call for dimension detection, 3 for batch
        self.assertEqual(mock_client.embeddings.call_count, 4)

    @patch("tonal_hortator.core.embeddings.embeddings.ollama.Client")
    def test_get_embeddings_batch_error_fallback(self, mock_client_class: Mock) -> None:
        """Test batch embedding generation with error fallback"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_models_response = Mock()
        mock_models_response.models = [Mock(model="nomic-embed-text:latest")]
        mock_client.list.return_value = mock_models_response

        # First call succeeds, second fails, third succeeds
        mock_client.embeddings.side_effect = [
            {"embedding": [0.1, 0.2, 0.3]},
            Exception("Embedding error"),
            {"embedding": [0.4, 0.5, 0.6]},
        ]

        service = OllamaEmbeddingService()

        texts = ["text1", "text2", "text3"]
        results = service.get_embeddings_batch(texts)

        self.assertEqual(len(results), 3)
        self.assertIsInstance(results[0], np.ndarray)
        self.assertIsInstance(results[1], np.ndarray)  # Fallback to zero vector
        self.assertIsInstance(results[2], np.ndarray)

        # Check that the failed embedding is a zero vector
        self.assertTrue(np.all(results[1] == 0))

    @patch("tonal_hortator.core.embeddings.embeddings.ollama.Client")
    def test_create_track_embedding_text(self, mock_client_class: Mock) -> None:
        """Test track embedding text creation"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_models_response = Mock()
        mock_models_response.models = [Mock(model="nomic-embed-text:latest")]
        mock_client.list.return_value = mock_models_response

        service = OllamaEmbeddingService()

        track_data = {
            "name": "Test Song",
            "artist": "Test Artist",
            "album": "Test Album",
            "genre": "Rock",
            "year": "2020",
        }

        result = service.create_track_embedding_text(track_data)

        expected = "Test Song, Test Artist, Test Album, Rock, 2020"
        self.assertEqual(result, expected)

    @patch("tonal_hortator.core.embeddings.embeddings.ollama.Client")
    def test_create_track_embedding_text_missing_fields(
        self, mock_client_class: Mock
    ) -> None:
        """Test track embedding text creation with missing fields"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_models_response = Mock()
        mock_models_response.models = [Mock(model="nomic-embed-text:latest")]
        mock_client.list.return_value = mock_models_response

        service = OllamaEmbeddingService()

        track_data = {"name": "Test Song", "artist": "Test Artist"}

        result = service.create_track_embedding_text(track_data)

        expected = "Test Song, Test Artist"
        self.assertEqual(result, expected)

    @patch("tonal_hortator.core.embeddings.embeddings.ollama.Client")
    def test_similarity_search(self, mock_client_class: Mock) -> None:
        """Test similarity search functionality"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_models_response = Mock()
        mock_models_response.models = [Mock(model="nomic-embed-text:latest")]
        mock_client.list.return_value = mock_models_response

        service = OllamaEmbeddingService()

        # Mock embeddings and track data
        embeddings = [
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            np.array([0.9, 0.8, 0.7], dtype=np.float32),
            np.array([0.2, 0.3, 0.4], dtype=np.float32),
        ]
        track_data = [
            {"name": "Track 1", "artist": "Artist 1"},
            {"name": "Track 2", "artist": "Artist 2"},
            {"name": "Track 3", "artist": "Artist 3"},
        ]

        # Mock the embedding generation for the query
        mock_client.embeddings.side_effect = [
            {"embedding": [0.1, 0.2, 0.3]},  # Query embedding
        ]

        results = service.similarity_search(
            "test query", embeddings, track_data, top_k=2
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["name"], "Track 1")  # Highest similarity
        self.assertEqual(results[1]["name"], "Track 3")  # Second highest similarity

    @patch("tonal_hortator.core.embeddings.embeddings.ollama.Client")
    def test_similarity_search_empty_data(self, mock_client_class: Mock) -> None:
        """Test similarity search with empty data"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_models_response = Mock()
        mock_models_response.models = [Mock(model="nomic-embed-text:latest")]
        mock_client.list.return_value = mock_models_response

        service = OllamaEmbeddingService()

        results = service.similarity_search("test query", [], [], top_k=5)

        self.assertEqual(results, [])

    @patch("tonal_hortator.core.embeddings.embeddings.ollama.Client")
    def test_get_embedding_dimension_fallback(self, mock_client_class: Mock) -> None:
        """Test embedding dimension fallback when detection fails"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_models_response = Mock()
        mock_models_response.models = [Mock(model="nomic-embed-text:latest")]
        mock_client.list.return_value = mock_models_response

        # Mock embedding response with unexpected dimension
        mock_embedding_response = {"embedding": [0.1, 0.2]}  # Only 2 dimensions
        mock_client.embeddings.return_value = mock_embedding_response

        service = OllamaEmbeddingService()

        # Should use the detected dimension (2)
        self.assertEqual(service._embedding_dimension, 2)

    # Test data
    test_tracks = [
        {
            "id": 1,
            "name": "Test Song 1",
            "artist": "Test Artist 1",
            "album": "Test Album 1",
            "genre": "Rock",
            "year": 2020,
            "play_count": 10,
            "album_artist": "Test Album Artist 1",
            "composer": "Composer 1",
        },
        {
            "id": 2,
            "name": "Test Song 2",
            "artist": "Test Artist 2",
            "album": "Test Album 2",
            "genre": "Jazz",
            "year": 2021,
            "play_count": 5,
            "album_artist": "Test Album Artist 2",
            "composer": "Composer 2",
        },
    ]

    @patch("tonal_hortator.core.embeddings.embeddings.ollama.Client")
    def test_create_track_embedding_text_with_engagement_data(
        self, mock_client_class: Mock
    ) -> None:
        """Test track embedding text creation with engagement data"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_models_response = Mock()
        mock_models_response.models = [Mock(model="nomic-embed-text:latest")]
        mock_client.list.return_value = mock_models_response

        service = OllamaEmbeddingService()

        # Test with play count
        track_data = {
            "name": "Popular Song",
            "artist": "Popular Artist",
            "play_count": 150,
        }
        result = service.create_track_embedding_text(track_data)
        self.assertIn("frequently played", result)

        # Test with track rating
        track_data = {
            "name": "Highly Rated Song",
            "artist": "Great Artist",
            "track_rating": 4.8,
        }
        result = service.create_track_embedding_text(track_data)
        self.assertIn("highly rated", result)

        # Test with average rating
        track_data = {
            "name": "User Favorite",
            "artist": "Beloved Artist",
            "avg_rating": 4.7,
        }
        result = service.create_track_embedding_text(track_data)
        self.assertIn("user favorite", result)

    @patch("tonal_hortator.core.embeddings.embeddings.ollama.Client")
    def test_create_track_embedding_text_play_count_categories(
        self, mock_client_class: Mock
    ) -> None:
        """Test play count categorization in embedding text"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_models_response = Mock()
        mock_models_response.models = [Mock(model="nomic-embed-text:latest")]
        mock_client.list.return_value = mock_models_response

        service = OllamaEmbeddingService()

        # Test different play count categories
        test_cases = [
            (200, "frequently played"),
            (75, "moderately played"),
            (25, "occasionally played"),
            (5, "rarely played"),
            (0, None),  # Should not add any play count text
            (None, None),  # Should not add any play count text
        ]

        for play_count, expected_text in test_cases:
            track_data = {
                "name": "Test Song",
                "artist": "Test Artist",
                "play_count": play_count,
            }
            result = service.create_track_embedding_text(track_data)

            if expected_text:
                self.assertIn(expected_text, result)
            else:
                # Should not contain any play count related text
                self.assertNotIn("played", result)

    @patch("tonal_hortator.core.embeddings.embeddings.ollama.Client")
    def test_create_track_embedding_text_rating_categories(
        self, mock_client_class: Mock
    ) -> None:
        """Test rating categorization in embedding text"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_models_response = Mock()
        mock_models_response.models = [Mock(model="nomic-embed-text:latest")]
        mock_client.list.return_value = mock_models_response

        service = OllamaEmbeddingService()

        # Test track_rating categories
        track_rating_cases = [
            (4.8, "highly rated"),
            (4.2, "well rated"),
            (3.5, "moderately rated"),
            (1.5, "poorly rated"),
            (2.5, None),  # Should not add rating text for middle range
        ]

        for rating, expected_text in track_rating_cases:
            track_data = {
                "name": "Test Song",
                "artist": "Test Artist",
                "track_rating": rating,
            }
            result = service.create_track_embedding_text(track_data)

            if expected_text:
                self.assertIn(expected_text, result)
            else:
                # Should not contain any rating related text
                self.assertNotIn("rated", result)

        # Test avg_rating categories
        avg_rating_cases = [
            (4.8, "user favorite"),
            (4.3, "user liked"),
            (1.5, "user disliked"),
            (3.0, None),  # Should not add rating text for middle range
        ]

        for rating, expected_text in avg_rating_cases:
            track_data = {
                "name": "Test Song",
                "artist": "Test Artist",
                "avg_rating": rating,
            }
            result = service.create_track_embedding_text(track_data)

            if expected_text:
                self.assertIn(expected_text, result)
            else:
                # Should not contain any user rating related text
                self.assertNotIn("user", result)

    @patch("tonal_hortator.core.embeddings.embeddings.ollama.Client")
    def test_create_track_embedding_text_comprehensive_engagement(
        self, mock_client_class: Mock
    ) -> None:
        """Test comprehensive engagement data in embedding text"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_models_response = Mock()
        mock_models_response.models = [Mock(model="nomic-embed-text:latest")]
        mock_client.list.return_value = mock_models_response

        service = OllamaEmbeddingService()

        # Test with all engagement fields
        track_data = {
            "name": "Amazing Song",
            "artist": "Amazing Artist",
            "album": "Amazing Album",
            "genre": "Rock",
            "year": "2023",
            "play_count": 200,
            "track_rating": 4.9,
            "avg_rating": 4.8,
        }

        result = service.create_track_embedding_text(track_data)

        # Should contain all engagement indicators
        self.assertIn("frequently played", result)
        self.assertIn("highly rated", result)
        self.assertIn("user favorite", result)

        # Should contain basic metadata
        self.assertIn("Amazing Song", result)
        self.assertIn("Amazing Artist", result)
        self.assertIn("Amazing Album", result)
        self.assertIn("Rock", result)
        self.assertIn("2023", result)

    @patch("tonal_hortator.core.embeddings.embeddings.ollama.Client")
    def test_create_track_embedding_text_invalid_rating_data(
        self, mock_client_class: Mock
    ) -> None:
        """Test handling of invalid rating data"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_models_response = Mock()
        mock_models_response.models = [Mock(model="nomic-embed-text:latest")]
        mock_client.list.return_value = mock_models_response

        service = OllamaEmbeddingService()

        # Test with invalid rating data
        track_data = {
            "name": "Test Song",
            "artist": "Test Artist",
            "track_rating": "invalid",
            "avg_rating": "not_a_number",
        }

        result = service.create_track_embedding_text(track_data)

        # Should not contain any rating text due to invalid data
        self.assertNotIn("rated", result)
        self.assertNotIn("user", result)

        # Should still contain basic metadata
        self.assertIn("Test Song", result)
        self.assertIn("Test Artist", result)
