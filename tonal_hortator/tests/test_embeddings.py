#!/usr/bin/env python3
"""
Tests for tonal_hortator.core.embeddings
"""

from typing import List
from unittest.mock import Mock, patch

import numpy as np
import pytest

from tonal_hortator.core.embeddings import OllamaEmbeddingService


class TestOllamaEmbeddingService:
    """Test OllamaEmbeddingService"""

    @patch("tonal_hortator.core.embeddings.ollama.Client")
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

        assert service.model_name == "nomic-embed-text:latest"
        assert service.client is not None
        assert service._embedding_dimension == 384

    @patch("tonal_hortator.core.embeddings.ollama.Client")
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

        assert service.host == "http://localhost:11434"
        mock_client_class.assert_called_with(host="http://localhost:11434")

    @patch("tonal_hortator.core.embeddings.ollama.Client")
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

    @patch("tonal_hortator.core.embeddings.ollama.Client")
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

        assert service.model_name == "nomic-embed-text:latest"

    @patch("tonal_hortator.core.embeddings.ollama.Client")
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

        assert service.model_name == "nomic-embed-text:latest"

    @patch("tonal_hortator.core.embeddings.ollama.Client")
    def test_init_unexpected_response_format(self, mock_client_class: Mock) -> None:
        """Test initialization with unexpected response format"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock unexpected response format
        mock_client.list.return_value = "unexpected"

        with pytest.raises(Exception, match="Unexpected models response format"):
            OllamaEmbeddingService()

    @patch("tonal_hortator.core.embeddings.ollama.Client")
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

        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert result.dtype == np.float32
        mock_client.embeddings.assert_called_with(
            model="nomic-embed-text:latest", prompt="test text"
        )

    @patch("tonal_hortator.core.embeddings.ollama.Client")
    def test_get_embedding_empty_text(self, mock_client_class: Mock) -> None:
        """Test embedding generation with empty text"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_models_response = Mock()
        mock_models_response.models = [Mock(model="nomic-embed-text:latest")]
        mock_client.list.return_value = mock_models_response

        mock_embedding_response = {"embedding": [0.1] * 384}
        mock_client.embeddings.return_value = mock_embedding_response

        service = OllamaEmbeddingService()

        result = service.get_embedding("")

        assert isinstance(result, np.ndarray)
        assert len(result) == 384
        assert np.all(result == 0)

    @patch("tonal_hortator.core.embeddings.ollama.Client")
    def test_get_embedding_whitespace_text(self, mock_client_class: Mock) -> None:
        """Test embedding generation with whitespace text"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_models_response = Mock()
        mock_models_response.models = [Mock(model="nomic-embed-text:latest")]
        mock_client.list.return_value = mock_models_response

        mock_embedding_response = {"embedding": [0.1] * 384}
        mock_client.embeddings.return_value = mock_embedding_response

        service = OllamaEmbeddingService()

        result = service.get_embedding("   ")

        assert isinstance(result, np.ndarray)
        assert len(result) == 384
        assert np.all(result == 0)

    @patch("tonal_hortator.core.embeddings.ollama.Client")
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

    @patch("tonal_hortator.core.embeddings.ollama.Client")
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

        assert len(results) == 3
        for result in results:
            assert isinstance(result, np.ndarray)
            assert len(result) == 3
            assert result.dtype == np.float32
        # 1 call for dimension detection, 3 for batch
        assert mock_client.embeddings.call_count == 4

    @patch("tonal_hortator.core.embeddings.ollama.Client")
    def test_get_embeddings_batch_empty(self, mock_client_class: Mock) -> None:
        """Test batch embedding generation with empty list"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_models_response = Mock()
        mock_models_response.models = [Mock(model="nomic-embed-text:latest")]
        mock_client.list.return_value = mock_models_response

        service = OllamaEmbeddingService()

        results = service.get_embeddings_batch([])

        assert results == []
        # 1 call for dimension detection
        assert mock_client.embeddings.call_count == 1

    @patch("tonal_hortator.core.embeddings.ollama.Client")
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

        assert len(results) == 3
        assert isinstance(results[0], np.ndarray)
        assert isinstance(results[1], np.ndarray)  # Fallback to zero vector
        assert isinstance(results[2], np.ndarray)

        # Check that the failed embedding is a zero vector
        assert np.all(results[1] == 0)

    @patch("tonal_hortator.core.embeddings.ollama.Client")
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
        assert result == expected

    @patch("tonal_hortator.core.embeddings.ollama.Client")
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
        assert result == expected

    @patch("tonal_hortator.core.embeddings.ollama.Client")
    def test_similarity_search(self, mock_client_class: Mock) -> None:
        """Test similarity search functionality"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_models_response = Mock()
        mock_models_response.models = [Mock(model="nomic-embed-text:latest")]
        mock_client.list.return_value = mock_models_response

        service = OllamaEmbeddingService()

        # Mock embeddings and track data
        embeddings: List[np.ndarray] = [
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

        assert len(results) == 2
        assert results[0]["name"] == "Track 1"  # Highest similarity
        assert results[1]["name"] == "Track 3"  # Second highest similarity

    @patch("tonal_hortator.core.embeddings.ollama.Client")
    def test_similarity_search_empty_data(self, mock_client_class: Mock) -> None:
        """Test similarity search with empty data"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_models_response = Mock()
        mock_models_response.models = [Mock(model="nomic-embed-text:latest")]
        mock_client.list.return_value = mock_models_response

        service = OllamaEmbeddingService()

        results = service.similarity_search("test query", [], [], top_k=5)

        assert results == []

    @patch("tonal_hortator.core.embeddings.ollama.Client")
    def test_close(self, mock_client_class: Mock) -> None:
        """Test service cleanup"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_models_response = Mock()
        mock_models_response.models = [Mock(model="nomic-embed-text:latest")]
        mock_client.list.return_value = mock_models_response

        service = OllamaEmbeddingService()
        service.close()
        # No assertion needed; just ensure no exception is raised

    @patch("tonal_hortator.core.embeddings.ollama.Client")
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
        assert service._embedding_dimension == 2
