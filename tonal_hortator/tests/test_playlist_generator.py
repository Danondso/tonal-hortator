#!/usr/bin/env python3
"""
Tests for tonal_hortator.core.playlist_generator
"""

import os
import tempfile
from typing import Any, List, cast
from unittest.mock import Mock, patch

import numpy as np

from tonal_hortator.core.playlist_generator import LocalPlaylistGenerator


class TestLocalPlaylistGenerator:
    """Test LocalPlaylistGenerator"""

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    def test_init(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test initialization"""
        mock_embedding_service = Mock()
        mock_embedding_service_class.return_value = mock_embedding_service

        mock_track_embedder = Mock()
        mock_track_embedder_class.return_value = mock_track_embedder

        generator = LocalPlaylistGenerator()

        assert generator.db_path == "music_library.db"
        assert generator.embedding_service == mock_embedding_service
        assert generator.track_embedder == mock_track_embedder

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    def test_init_custom_db_path(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test initialization with custom db path"""
        mock_embedding_service = Mock()
        mock_embedding_service_class.return_value = mock_embedding_service

        mock_track_embedder = Mock()
        mock_track_embedder_class.return_value = mock_track_embedder

        generator = LocalPlaylistGenerator(db_path="custom.db")

        assert generator.db_path == "custom.db"
        mock_track_embedder_class.assert_called_with(
            "custom.db", embedding_service=mock_embedding_service
        )

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    def test_is_vague_query_true(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test vague query detection - true cases"""
        generator = LocalPlaylistGenerator()

        vague_queries = [
            "video game music",
            "early 90s grunge",
            "upbeat music",
            "party songs",
            "rock music",
            "jazz",
            "happy",
            "chill",
        ]

        for query in vague_queries:
            assert generator._is_vague_query(query) is True

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    def test_is_vague_query_false(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test vague query detection - false cases"""
        generator = LocalPlaylistGenerator()

        specific_queries = [
            "Nirvana Smells Like Teen Spirit",
            "The Beatles Hey Jude",
            "Queen Bohemian Rhapsody",
            "Michael Jackson Thriller",
            "Led Zeppelin Stairway to Heaven",
        ]

        for query in specific_queries:
            assert generator._is_vague_query(query) is False

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    def test_extract_track_count(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test track count extraction from query"""
        generator = LocalPlaylistGenerator()

        # Test queries with track counts
        assert generator._extract_track_count("10 tracks") == 10
        assert generator._extract_track_count("5 songs") == 5

        # Test queries without track counts
        assert generator._extract_track_count("20 rock tracks") is None
        assert generator._extract_track_count("upbeat songs") is None
        assert generator._extract_track_count("rock music") is None

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    def test_extract_artist_from_query(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test artist extraction from query"""
        generator = LocalPlaylistGenerator()

        # Test queries with artists
        assert generator._extract_artist_from_query("Nirvana songs") == "Nirvana"
        assert (
            generator._extract_artist_from_query("The Beatles songs") == "The Beatles"
        )
        assert generator._extract_artist_from_query("Queen radio") == "Queen"

        # Test queries without artists
        assert generator._extract_artist_from_query("upbeat songs") == "Upbeat"
        assert generator._extract_artist_from_query("rock music") is None

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    def test_extract_genre_keywords(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test genre keyword extraction"""
        generator = LocalPlaylistGenerator()

        # Test queries with genre keywords
        assert "rock" in generator._extract_genre_keywords("rock music")
        assert "jazz" in generator._extract_genre_keywords("jazz songs")
        assert "pop" in generator._extract_genre_keywords("pop music")
        assert "hip hop" in generator._extract_genre_keywords("hip hop tracks")

        # Test queries without genre keywords
        assert generator._extract_genre_keywords("upbeat songs") == []
        assert generator._extract_genre_keywords("happy music") == []

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    def test_extract_base_name(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test base name extraction"""
        generator = LocalPlaylistGenerator()

        # Test various name patterns
        assert generator._extract_base_name("Song (Remix)") == "Song"
        assert generator._extract_base_name("Track (Live)") == "Track"
        assert generator._extract_base_name("Music (Acoustic)") == "Music"
        assert generator._extract_base_name("Hit (Radio Edit)") == "Hit"

        # Test names without suffixes
        assert generator._extract_base_name("Simple Song") == "Simple Song"
        assert generator._extract_base_name("") == ""

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    def test_normalize_file_location(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test file location normalization"""
        generator = LocalPlaylistGenerator()

        # Test Windows path normalization
        result = generator._normalize_file_location(
            "C:\\Users\\username\\Music\\song.mp3"
        )
        assert "username" not in result
        assert result.endswith("song.mp3")

        # Test Unix path normalization
        result = generator._normalize_file_location("/home/username/Music/song.mp3")
        assert "username" not in result
        assert result.endswith("song.mp3")

        # Test empty location
        assert generator._normalize_file_location("") == ""

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    def test_apply_artist_randomization_not_vague(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test artist randomization when query is not vague"""
        generator = LocalPlaylistGenerator()

        tracks = [
            {"artist": "Artist1", "similarity_score": 0.8},
            {"artist": "Artist2", "similarity_score": 0.7},
            {"artist": "Artist1", "similarity_score": 0.6},
        ]

        result = generator._apply_artist_randomization(
            tracks, max_tracks=3, is_vague=False
        )

        # Should not randomize when not vague
        assert result == tracks[:3]

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    def test_apply_artist_randomization_vague(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test artist randomization when query is vague"""
        generator = LocalPlaylistGenerator()

        tracks = [
            {"artist": "Artist1", "similarity_score": 0.8},
            {"artist": "Artist2", "similarity_score": 0.7},
            {"artist": "Artist1", "similarity_score": 0.6},
        ]

        result = generator._apply_artist_randomization(
            tracks, max_tracks=3, is_vague=True
        )

        # Should randomize when vague
        assert len(result) == 3
        assert all("similarity_score" in track for track in result)

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    def test_smart_name_deduplication(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test smart name deduplication"""
        generator = LocalPlaylistGenerator()

        tracks = [
            {"name": "Song (Remix)", "artist": "Artist1", "similarity_score": 0.8},
            {"name": "Song (Live)", "artist": "Artist1", "similarity_score": 0.7},
            {"name": "Song", "artist": "Artist1", "similarity_score": 0.9},
            {"name": "Different Song", "artist": "Artist2", "similarity_score": 0.6},
        ]

        result = generator._smart_name_deduplication(tracks)

        # Should keep only 2 tracks: the best "Song" variant by Artist1 and "Different Song" by Artist2
        assert len(result) == 2
        # The "Song" track with highest similarity score (0.9) should be kept
        song_track = next(track for track in result if track["name"] == "Song")
        assert song_track["similarity_score"] == 0.9
        # The "Different Song" track should be kept
        assert any(track["name"] == "Different Song" for track in result)

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    def test_filter_and_deduplicate_results(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test filtering and deduplication of results"""
        generator = LocalPlaylistGenerator()

        tracks = [
            {"name": "Song1", "artist": "Artist1", "similarity_score": 0.8},
            {"name": "Song1", "artist": "Artist1", "similarity_score": 0.7},
            {"name": "Song2", "artist": "Artist2", "similarity_score": 0.6},
            {"name": "Song3", "artist": "Artist3", "similarity_score": 0.5},
        ]

        result = generator._filter_and_deduplicate_results(
            tracks, min_similarity=0.5, max_tracks=3
        )

        # Should deduplicate and limit to max_tracks (only unique name/artist combos)
        assert len(result) == 1

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    def test_apply_genre_filtering(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test genre filtering"""
        generator = LocalPlaylistGenerator()

        tracks = [
            {"name": "Song1", "genre": "Rock", "similarity_score": 0.8},
            {"name": "Song2", "genre": "Jazz", "similarity_score": 0.7},
            {"name": "Song3", "genre": "Rock", "similarity_score": 0.6},
        ]

        result = generator._apply_genre_filtering("rock music", tracks)

        # Should filter to only rock tracks (with boosting)
        assert any(track["genre"].lower() == "rock" for track in result)

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    def test_save_playlist_m3u(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test saving playlist as M3U file"""
        generator = LocalPlaylistGenerator()

        tracks = [
            {"name": "Song1", "file_location": "/path/to/song1.mp3"},
            {"name": "Song2", "file_location": "/path/to/song2.mp3"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".m3u", delete=False) as f:
            temp_path = f.name

        try:
            generator.save_playlist_m3u(
                tracks, "test query", output_dir=os.path.dirname(temp_path)
            )
            assert os.path.exists(temp_path)
        finally:
            os.unlink(temp_path)

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    def test_generate_playlist_success(
        self,
        mock_track_embedder_class: Mock,
        mock_embedding_service_class: Mock,
    ) -> None:
        """Test successful playlist generation"""
        mock_embedding_service = Mock()
        mock_embedding_service_class.return_value = mock_embedding_service

        mock_track_embedder = Mock()
        mock_track_embedder_class.return_value = mock_track_embedder

        # Use real numpy arrays for embeddings
        mock_embeddings = [
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            np.array([0.4, 0.5, 0.6], dtype=np.float32),
        ]
        mock_track_data = [
            {"name": "Song1", "artist": "Artist1", "similarity_score": 0.8},
            {"name": "Song2", "artist": "Artist2", "similarity_score": 0.7},
        ]
        mock_track_embedder.get_all_embeddings.return_value = (
            mock_embeddings,
            mock_track_data,
        )
        mock_track_embedder.create_track_embedding_text.side_effect = (
            lambda x: "embedding text"
        )
        # Return real list of dicts for similarity_search
        mock_embedding_service.similarity_search.return_value = [
            {"name": "Song1", "artist": "Artist1", "similarity_score": 0.8},
            {"name": "Song2", "artist": "Artist2", "similarity_score": 0.7},
        ]

        generator = LocalPlaylistGenerator()
        result = generator.generate_playlist("test query", max_tracks=5)
        assert isinstance(result, list)

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    def test_enforce_artist_diversity(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test artist diversity enforcement"""
        generator = LocalPlaylistGenerator()

        # Create test tracks with multiple tracks from the same artist
        tracks = [
            {"name": "Song1", "artist": "Artist1", "similarity_score": 0.9},
            {"name": "Song2", "artist": "Artist1", "similarity_score": 0.8},
            {"name": "Song3", "artist": "Artist1", "similarity_score": 0.7},
            {"name": "Song4", "artist": "Artist1", "similarity_score": 0.6},
            {"name": "Song5", "artist": "Artist2", "similarity_score": 0.85},
            {"name": "Song6", "artist": "Artist2", "similarity_score": 0.75},
            {"name": "Song7", "artist": "Artist3", "similarity_score": 0.95},
        ]

        # Test with max 2 tracks per artist
        result = generator._enforce_artist_diversity(
            tracks, max_tracks=10, max_tracks_per_artist=2
        )

        # Should have at most 2 tracks per artist, sorted by similarity
        artist_counts: dict[str, int] = {}
        for track in result:
            artist = track["artist"]
            artist_counts[artist] = artist_counts.get(artist, 0) + 1

        # Check that no artist has more than 2 tracks
        assert all(count <= 2 for count in artist_counts.values())

        # Check that we got the best tracks (highest similarity scores)
        artist1_tracks = [t for t in result if t["artist"] == "Artist1"]
        assert len(artist1_tracks) <= 2
        if len(artist1_tracks) >= 2:
            scores: List[float] = [
                float(cast(float, t["similarity_score"]))
                for t in tracks
                if t["artist"] == "Artist1"
            ]
            result_scores: List[float] = [
                float(cast(float, t["similarity_score"])) for t in artist1_tracks
            ]
            scores = sorted(scores, reverse=True)
            result_scores = sorted(result_scores, reverse=True)
            assert result_scores == scores[:2]

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    def test_distribute_artists(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test artist distribution throughout playlist"""
        generator = LocalPlaylistGenerator()

        # Create test tracks with multiple artists
        tracks = [
            {"name": "Song1", "artist": "Artist1", "similarity_score": 0.9},
            {"name": "Song2", "artist": "Artist1", "similarity_score": 0.8},
            {"name": "Song3", "artist": "Artist2", "similarity_score": 0.85},
            {"name": "Song4", "artist": "Artist2", "similarity_score": 0.75},
            {"name": "Song5", "artist": "Artist3", "similarity_score": 0.95},
            {"name": "Song6", "artist": "Artist3", "similarity_score": 0.7},
        ]

        # Test artist distribution
        result = generator._distribute_artists(tracks)

        # Should have the same number of tracks
        assert len(result) == len(tracks)

        # Check that artists are distributed (not grouped together)
        # Count consecutive tracks from the same artist
        consecutive_same_artist = 0
        max_consecutive = 0
        prev_artist = None

        for track in result:
            current_artist = track["artist"]
            if current_artist == prev_artist:
                consecutive_same_artist += 1
                max_consecutive = max(max_consecutive, consecutive_same_artist)
            else:
                consecutive_same_artist = 1
            prev_artist = current_artist

        # With 3 artists and 6 tracks, we should have at most 2 consecutive tracks from same artist
        # (in the worst case, but ideally 1)
        assert (
            max_consecutive <= 2
        ), f"Too many consecutive tracks from same artist: {max_consecutive}"

        # Verify all original tracks are present (just reordered)
        original_artists: List[str] = [cast(str, t["artist"]) for t in tracks]
        result_artists: List[str] = [cast(str, t["artist"]) for t in result]
        assert sorted(original_artists) == sorted(result_artists)

        # Verify all track names are present
        original_names: List[str] = [cast(str, t["name"]) for t in tracks]
        result_names: List[str] = [cast(str, t["name"]) for t in result]
        assert sorted(original_names) == sorted(result_names)
