#!/usr/bin/env python3
"""
Local embedding service using Ollama
Provides embeddings for music tracks without requiring internet or HuggingFace
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import ollama
from loguru import logger


class OllamaEmbeddingService:
    """Service for generating embeddings using Ollama."""

    def __init__(
        self, model_name: str = "nomic-embed-text:latest", host: Optional[str] = None
    ):
        self.model_name = model_name
        self.host = host
        self.client: Optional[ollama.Client] = None
        self._embedding_dimension: Optional[int] = None  # Cache for embedding dimension
        self._test_connection()

    def _extract_model_list(self, models_response: object) -> list[Any]:
        """Extract model list from various response formats"""
        if hasattr(models_response, "models"):
            # ListResponse object
            return list(models_response.models)
        elif isinstance(models_response, dict) and "models" in models_response:
            return list(models_response["models"])
        elif isinstance(models_response, list):
            return models_response
        else:
            raise Exception(
                f"Unexpected models response format: {type(models_response)}"
            )

    def _extract_model_names(self, model_list: list[Any]) -> list[str]:
        """Extract model names from model list"""
        model_names: list[str] = []
        for model in model_list:
            if hasattr(model, "model"):
                # Model object
                model_names.append(model.model)
            elif isinstance(model, dict) and "name" in model:
                model_names.append(model["name"])
            elif isinstance(model, str):
                model_names.append(model)
            else:
                logger.warning(f"Skipping invalid model entry: {model}")
        return model_names

    def _validate_model_availability(self, model_names: List[str]) -> None:
        """Validate that the required model is available"""
        if self.model_name not in model_names:
            logger.warning(
                f"⚠️  Model '{self.model_name}' not found in available models: {model_names}"
            )
            logger.info("💡 Available models:")
            for name in model_names:
                logger.info(f"   - {name}")
            raise Exception(f"Model '{self.model_name}' not available")

    def _test_connection(self) -> None:
        """Test connection to Ollama and validate model availability."""
        try:
            self.client = ollama.Client(host=self.host)
            if self.client is None:
                raise Exception("Failed to initialize Ollama client")

            models = self.client.list()
            logger.debug(f"Ollama models response: {models}")

            # Extract model list and names
            model_list = self._extract_model_list(models)
            model_names = self._extract_model_names(model_list)

            # Validate model availability
            self._validate_model_availability(model_names)

            logger.info(f"✅ Connected to Ollama with model: {self.model_name}")

            # Get embedding dimension from a test embedding
            self._get_embedding_dimension()

        except Exception as e:
            logger.error(f"❌ Failed to connect to Ollama: {e}")
            raise

    def _get_embedding_dimension(self) -> int:
        """Get the embedding dimension from the model by making a test embedding."""
        if self._embedding_dimension is not None:
            return self._embedding_dimension

        if self.client is None:
            raise Exception("Ollama client not initialized")

        try:
            # Use a simple test text to get the embedding dimension
            test_text = "test"
            result = self.client.embeddings(model=self.model_name, prompt=test_text)
            self._embedding_dimension = len(result["embedding"])
            logger.info(f"📏 Detected embedding dimension: {self._embedding_dimension}")
            return self._embedding_dimension
        except Exception as e:
            logger.error(f"❌ Failed to get embedding dimension: {e}")
            # Fallback to a reasonable default
            self._embedding_dimension = 384
            logger.warning(
                f"⚠️  Using fallback embedding dimension: {self._embedding_dimension}"
            )
            return self._embedding_dimension

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text string

        Args:
            text: Text string to embed

        Returns:
            Numpy array representing the embedding
        """
        if self.client is None:
            raise Exception("Ollama client not initialized")

        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return np.zeros(self._get_embedding_dimension(), dtype=np.float32)

        try:
            start_time = time.time()
            result = self.client.embeddings(model=self.model_name, prompt=text)
            embedding = np.array(result["embedding"], dtype=np.float32)
            embedding_time = time.time() - start_time

            logger.debug(
                f"Generated embedding in {embedding_time:.3f}s (dim: {len(embedding)})"
            )
            return embedding

        except Exception as e:
            logger.error(f"❌ Error getting embedding for text '{text[:50]}...': {e}")
            raise

    def _process_batch(self, batch_texts: List[str]) -> List[np.ndarray]:
        if self.client is None:
            raise Exception("Ollama client not initialized")

        batch_embeddings = []
        for text in batch_texts:
            result = self.client.embeddings(model=self.model_name, prompt=text)
            batch_embeddings.append(np.array(result["embedding"], dtype=np.float32))
        return batch_embeddings

    def _fallback_individual_embeddings(
        self, batch_texts: List[str]
    ) -> List[np.ndarray]:
        embeddings = []
        embedding_dim = self._get_embedding_dimension()
        for text in batch_texts:
            try:
                embedding = self.get_embedding(text)
                embeddings.append(embedding)
            except Exception as e_ind:
                logger.error(f"❌ Failed to embed text: '{text[:50]}...': {e_ind}")
                embeddings.append(np.zeros(embedding_dim, dtype=np.float32))
        return embeddings

    def get_embeddings_batch(
        self, texts: List[str], batch_size: int = 100
    ) -> List[np.ndarray]:
        """
        Get embeddings for multiple texts using Ollama's batch API

        Args:
            texts: List of text strings to embed
            batch_size: The size of batches to send to Ollama.

        Returns:
            A list of numpy arrays representing the embeddings
        """
        if not self.client:
            raise Exception("Ollama client not initialized")
        if not texts:
            logger.info("✅ Generated 0 embeddings total")
            return []

        all_embeddings = []
        logger.info(
            f"🔄 Generating embeddings for {len(texts)} texts in batches of {batch_size}"
        )

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size

            logger.info(
                f"📦 Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)"
            )
            batch_start = time.time()

            try:
                batch_embeddings = self._process_batch(batch_texts)
                if len(batch_embeddings) == len(batch_texts):
                    all_embeddings.extend(batch_embeddings)
                    batch_time = time.time() - batch_start
                    logger.info(f"⏱️  Batch {batch_num} completed in {batch_time:.2f}s")
                else:
                    logger.error(
                        f"❌ Mismatched embedding count. Got {len(batch_embeddings)} for {len(batch_texts)} prompts."
                    )
                    raise Exception("Incomplete batch response from Ollama library")
            except Exception as e:
                logger.error(
                    f"❌ Batch embedding request failed for batch {batch_num}: {e}"
                )
                logger.info(
                    "🔄 Falling back to individual embeddings for this batch (redundant, but safe)..."
                )
                all_embeddings.extend(self._fallback_individual_embeddings(batch_texts))

        logger.info(f"✅ Generated {len(all_embeddings)} embeddings total")
        return all_embeddings

    def create_track_embedding_text(self, track: Dict[str, Any]) -> str:
        """Create text representation of track for embedding"""
        parts = [
            track.get("artist"),
            track.get("album"),
            track.get("album_artist"),
            track.get("composer"),
            track.get("genre"),
            str(track.get("year")) if track.get("year") else None,
        ]
        return ", ".join(filter(None, parts))

    def similarity_search(
        self,
        query: str,
        embeddings: List[np.ndarray],
        track_data: List[Dict[str, Any]],
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search using cosine similarity

        Args:
            query: Search query
            embeddings: List of track embeddings
            track_data: List of track data dictionaries
            top_k: Number of top results to return

        Returns:
            List of track data dictionaries with similarity scores
        """
        if not embeddings or not track_data:
            logger.warning("No embeddings or track data provided for search")
            return []

        # Get query embedding
        query_embedding = self.get_embedding(query)

        # Calculate cosine similarities
        similarities = []
        for i, track_embedding in enumerate(embeddings):
            # Normalize embeddings for cosine similarity
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            track_norm = track_embedding / (np.linalg.norm(track_embedding) + 1e-8)

            # Calculate cosine similarity
            similarity = np.dot(query_norm, track_norm)
            similarities.append((similarity, i))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Get top results
        results = []
        for similarity, idx in similarities[:top_k]:
            track = track_data[idx].copy()
            track["similarity_score"] = float(similarity)
            results.append(track)

        logger.info(f"🔍 Found {len(results)} similar tracks for query: '{query}'")
        return results

    def close(self) -> None:
        """Close the Ollama embedding service"""
        pass


def test_ollama_embeddings() -> bool:
    """Test the Ollama embedding service"""
    try:
        # Initialize service
        service = OllamaEmbeddingService()

        # Test single embedding
        test_text = "Jazz music by Miles Davis"
        embedding = service.get_embedding(test_text)
        print(f"✅ Single embedding test passed. Shape: {embedding.shape}")

        # Test batch embeddings
        test_texts = [
            "Rock music by Led Zeppelin",
            "Classical music by Beethoven",
            "Electronic music by Daft Punk",
            "Hip hop by Kendrick Lamar",
        ]

        embeddings = service.get_embeddings_batch(test_texts, batch_size=2)
        print(f"✅ Batch embedding test passed. Generated {len(embeddings)} embeddings")

        # Test track embedding text creation
        test_track = {
            "name": "Bohemian Rhapsody",
            "artist": "Queen",
            "album": "A Night at the Opera",
            "genre": "Rock",
            "year": "1975",
            "play_count": 42,
        }

        track_text = service.create_track_embedding_text(test_track)
        print(f"✅ Track text creation test passed: {track_text}")

        print("🎉 All Ollama embedding tests passed!")
        return True

    except Exception as e:
        print(f"❌ Ollama embedding test failed: {e}")
        return False


if __name__ == "__main__":
    test_ollama_embeddings()
