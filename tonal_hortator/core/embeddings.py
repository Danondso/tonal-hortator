#!/usr/bin/env python3
"""
Local embedding service using Ollama
Provides embeddings for music tracks without requiring internet or HuggingFace
"""

import json
import logging
import ollama
import numpy as np
from typing import List, Dict, Any, Optional
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaEmbeddingService:
    """Local embedding service using Ollama"""

    def __init__(
        self, model_name: str = "nomic-embed-text:latest", host: Optional[str] = None
    ):
        """
        Initialize the Ollama embedding service

        Args:
            model_name: Name of the embedding model in Ollama
            host: Ollama API host URL (e.g., http://localhost:11434)
        """
        self.model_name = model_name
        self.client = ollama.Client(host=host)

        # Test connection
        self._test_connection()

    def _test_connection(self):
        """Test connection to Ollama service"""
        try:
            models = self.client.list().get("models", [])
            model_names = [model.get("name", "") for model in models]
            if self.model_name in model_names:
                logger.info(
                    f"✅ Connected to Ollama and found model: {self.model_name}"
                )
            else:
                logger.warning(
                    f"⚠️  Model {self.model_name} not found in Ollama. Available models: {model_names}"
                )
                logger.info(f"Attempting to pull model '{self.model_name}'...")
                self.client.pull(self.model_name)
                logger.info(f"✅ Successfully pulled model '{self.model_name}'")
        except Exception as e:
            logger.error(f"❌ Error connecting to Ollama or pulling model: {e}")
            raise

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text string

        Args:
            text: Text to embed

        Returns:
            numpy array of the embedding
        """
        try:
            start_time = time.time()
            result = self.client.embeddings(model=self.model_name, prompt=text)
            embedding_time = time.time() - start_time

            embedding = np.array(result["embedding"], dtype=np.float32)
            logger.debug(
                f"Generated embedding in {embedding_time:.3f}s (dim: {len(embedding)})"
            )
            return embedding

        except Exception as e:
            logger.error(f"❌ Error getting embedding for text '{text[:50]}...': {e}")
            raise

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
                # Process texts individually within the batch since the library's batching has issues.
                batch_embeddings = []
                for text in batch_texts:
                    # This call gets a single embedding
                    result = self.client.embeddings(model=self.model_name, prompt=text)
                    batch_embeddings.append(
                        np.array(result["embedding"], dtype=np.float32)
                    )

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
                for text in batch_texts:
                    try:
                        embedding = self.get_embedding(text)
                        all_embeddings.append(embedding)
                    except Exception as e_ind:
                        logger.error(
                            f"❌ Failed to embed text: '{text[:50]}...': {e_ind}"
                        )
                        # Assuming nomic-embed-text which has dimension 768
                        all_embeddings.append(np.zeros(768, dtype=np.float32))

        logger.info(f"✅ Generated {len(all_embeddings)} embeddings total")
        return all_embeddings

    def create_track_embedding_text(self, track: Dict[str, Any]) -> str:
        """Create text representation of track for embedding"""
        parts = [
            track.get("name"),
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


def test_ollama_embeddings():
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
