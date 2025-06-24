#!/usr/bin/env python3
"""
Generate music playlists using local Ollama embeddings
This script creates playlists based on semantic search queries
"""

import logging
import time
from typing import Any, Dict, List, Optional

from tonal_hortator.core.artist_distributor import ArtistDistributor
from tonal_hortator.core.deduplication import TrackDeduplicator
from tonal_hortator.core.embeddings import OllamaEmbeddingService
from tonal_hortator.core.query_processor import QueryProcessor
from tonal_hortator.core.track_embedder import LocalTrackEmbedder

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LocalPlaylistGenerator:
    """Generate playlists using local Ollama embeddings"""

    def __init__(
        self,
        db_path: str = "music_library.db",
        model_name: str = "nomic-embed-text:latest",
    ):
        """
        Initialize the local playlist generator

        Args:
            db_path: Path to SQLite database
            model_name: Name of the embedding model to use
        """
        self.db_path = db_path
        self.embedding_service = OllamaEmbeddingService(model_name=model_name)
        self.track_embedder = LocalTrackEmbedder(
            db_path, embedding_service=self.embedding_service
        )

        # Initialize utility classes
        self.query_processor = QueryProcessor()
        self.deduplicator = TrackDeduplicator()
        self.artist_distributor = ArtistDistributor()

    def generate_playlist(
        self, query: str, max_tracks: int = 20, min_similarity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Generate a playlist based on a semantic query

        Args:
            query: Search query (e.g., "upbeat rock songs")
            max_tracks: Maximum number of tracks in playlist
            min_similarity: Minimum similarity score threshold

        Returns:
            List of track dictionaries with similarity scores
        """
        try:
            logger.info(f"🎵 Generating playlist for query: '{query}'")
            start_time = time.time()

            # 1. Extract parameters from query
            max_tracks = self._extract_playlist_parameters(query, max_tracks)

            # 2. Get embeddings and track data
            embeddings, track_data = self._get_embeddings_for_search()
            if not embeddings:
                return []

            # 3. Perform similarity search
            if track_data:
                results = self.embedding_service.similarity_search(
                    query, embeddings, track_data, top_k=max_tracks * 5
                )
            else:
                results = []

            # 4. Filter by artist if specified
            results = self._filter_by_artist(query, results)

            # 5. Process and refine the playlist
            final_playlist = self._process_and_refine_playlist(
                query, results, min_similarity, max_tracks
            )

            generation_time = time.time() - start_time
            logger.info(
                f"✅ Generated playlist with {len(final_playlist)} tracks in {generation_time:.2f}s"
            )

            return final_playlist

        except Exception as e:
            logger.error(f"❌ Error generating playlist: {e}")
            raise

    def _extract_playlist_parameters(self, query: str, default_max_tracks: int) -> int:
        """Extract playlist parameters from the query"""
        extracted_count = self.query_processor.extract_track_count(query)
        if extracted_count:
            logger.info(
                f"🔢 Found track count in query, setting max_tracks to {extracted_count}"
            )
            return extracted_count
        return default_max_tracks

    def _get_embeddings_for_search(
        self,
    ) -> tuple[Optional[List[Any]], Optional[List[Dict[str, Any]]]]:
        """Get embeddings from the database, generating them if necessary"""
        embeddings, track_data = self.track_embedder.get_all_embeddings()
        if not embeddings or not track_data:
            logger.warning("No embeddings found. Running embedding process first...")
            self.track_embedder.embed_all_tracks()
            embeddings, track_data = self.track_embedder.get_all_embeddings()
            if not embeddings:
                logger.error("❌ Still no embeddings available after embedding process")
                return None, None
        logger.info(f"📊 Using {len(embeddings)} track embeddings for search")
        return embeddings, track_data

    def _filter_by_artist(
        self, query: str, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter results by artist if specified in the query"""
        artist = self.query_processor.extract_artist_from_query(query)
        if artist:
            logger.info(f"🎤 Found artist in query: '{artist}'. Filtering results.")
            return [
                track
                for track in results
                if artist.lower() in track.get("artist", "").lower()
            ]
        return results

    def _process_and_refine_playlist(
        self,
        query: str,
        results: List[Dict[str, Any]],
        min_similarity: float,
        max_tracks: int,
    ) -> List[Dict[str, Any]]:
        """Apply genre filtering, deduplication, and artist randomization"""
        # Apply genre filtering and boosting
        results = self._apply_genre_filtering(query, results)

        # Filter by similarity threshold and deduplicate
        results = self.deduplicator.deduplicate_tracks(
            results, min_similarity, max_tracks
        )

        # Apply artist randomization
        is_vague = self.query_processor.is_vague_query(query)
        results = self.artist_distributor.apply_artist_randomization(
            results, max_tracks, is_vague
        )

        return results

    def _apply_genre_filtering(
        self, query: str, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply genre-aware filtering and boosting to search results

        Args:
            query: Original search query
            results: List of track results from similarity search

        Returns:
            Filtered and boosted results
        """
        if not results:
            return results

        # Extract potential genres from query
        query_lower = query.lower()
        genre_keywords = self.query_processor.extract_genre_keywords(query_lower)

        if not genre_keywords:
            logger.info(
                "🎵 No specific genre detected in query, using semantic similarity only"
            )
            return results

        logger.info(f"🎵 Detected genre keywords: {genre_keywords}")

        # Separate tracks by genre match
        genre_matches = []
        other_tracks = []

        for track in results:
            track_genre = track.get("genre", "").lower() if track.get("genre") else ""
            similarity_score = track.get("similarity_score", 0)

            # Check if track genre matches any of the detected genre keywords
            genre_match = any(keyword in track_genre for keyword in genre_keywords)

            if genre_match:
                # Boost similarity score for genre matches
                boosted_score = min(
                    1.0, similarity_score + 0.1
                )  # Boost by 0.1, cap at 1.0
                track["similarity_score"] = boosted_score
                track["genre_boosted"] = True
                genre_matches.append(track)
            else:
                track["genre_boosted"] = False
                other_tracks.append(track)

        # Combine results: genre matches first, then others
        # But only include a limited number of non-genre matches to maintain quality
        max_other_tracks = max(
            5, len(genre_matches) // 2
        )  # At most half as many non-genre tracks
        combined_results = genre_matches + other_tracks[:max_other_tracks]

        logger.info(
            f"🎵 Genre filtering: {len(genre_matches)} genre matches, {len(other_tracks[:max_other_tracks])} other tracks"
        )

        return combined_results
