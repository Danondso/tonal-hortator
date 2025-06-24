#!/usr/bin/env python3
"""
Tests for playlist generation functionality
"""

import tempfile
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from tonal_hortator.core.artist_distributor import ArtistDistributor
from tonal_hortator.core.deduplication import TrackDeduplicator
from tonal_hortator.core.playlist_generator import LocalPlaylistGenerator
from tonal_hortator.core.playlist_output import PlaylistOutput
from tonal_hortator.core.query_processor import QueryProcessor


class TestLocalPlaylistGenerator:
    """Test the LocalPlaylistGenerator class"""

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    def test_init(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test initialization with default parameters"""
        generator = LocalPlaylistGenerator()

        assert generator.db_path == "music_library.db"
        assert generator.embedding_service is not None
        assert generator.track_embedder is not None
        assert generator.query_processor is not None
        assert generator.deduplicator is not None
        assert generator.artist_distributor is not None

        mock_embedding_service_class.assert_called_once_with(
            model_name="nomic-embed-text:latest"
        )
        mock_track_embedder_class.assert_called_once()

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    def test_init_custom_db_path(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test initialization with custom database path"""
        generator = LocalPlaylistGenerator(
            db_path="custom.db", model_name="custom-model:latest"
        )

        assert generator.db_path == "custom.db"
        mock_embedding_service_class.assert_called_once_with(
            model_name="custom-model:latest"
        )

    def test_is_vague_query_true(self) -> None:
        processor = QueryProcessor()
        assert processor.is_vague_query("something fun") is True

    def test_is_vague_query_false(self) -> None:
        processor = QueryProcessor()
        assert processor.is_vague_query("songs by Nirvana") is False

    def test_extract_track_count(self) -> None:
        processor = QueryProcessor()
        assert processor.extract_track_count("10 rock songs") is None
        assert processor.extract_track_count("a playlist") is None

    def test_extract_artist_from_query(self) -> None:
        processor = QueryProcessor()
        assert processor.extract_artist_from_query("songs by Nirvana") == "Nirvana"
        assert processor.extract_artist_from_query("just music") is None

    def test_extract_genre_keywords(self) -> None:
        processor = QueryProcessor()
        assert processor.extract_genre_keywords("jazz and blues") == ["jazz", "blues"]
        assert processor.extract_genre_keywords("no genre here") == []

    def test_extract_base_title(self) -> None:
        """Test base title extraction"""
        deduplicator = TrackDeduplicator()

        # Test with common suffixes
        assert deduplicator._extract_base_title("Song (Remix)") == "Song"
        assert deduplicator._extract_base_title("Track (Live)") == "Track"
        assert deduplicator._extract_base_title("Music (Acoustic)") == "Music"
        assert deduplicator._extract_base_title("Hit (Radio Edit)") == "Hit"

        # Test without suffixes
        assert deduplicator._extract_base_title("Simple Song") == "Simple Song"
        assert deduplicator._extract_base_title("") == ""

    def test_normalize_file_location(self) -> None:
        """Test file location normalization"""
        deduplicator = TrackDeduplicator()

        # Test Windows path normalization
        result = deduplicator._normalize_file_location(
            "C:\\Users\\username\\Music\\song.mp3"
        )
        assert "username" not in result
        assert result.endswith("song.mp3")

        # Test Unix path normalization
        result = deduplicator._normalize_file_location("/home/username/Music/song.mp3")
        assert "username" not in result
        assert result.endswith("song.mp3")

        # Test empty location
        assert deduplicator._normalize_file_location("") == ""

    def test_apply_artist_randomization_not_vague(self) -> None:
        """Test artist randomization when query is not vague"""
        distributor = ArtistDistributor()

        tracks = [
            {"artist": "Artist1", "similarity_score": 0.8},
            {"artist": "Artist2", "similarity_score": 0.7},
            {"artist": "Artist1", "similarity_score": 0.6},
        ]

        result = distributor.apply_artist_randomization(
            tracks, max_tracks=3, is_vague=False
        )

        # Should not randomize when not vague
        assert result == tracks[:3]

    def test_apply_artist_randomization_vague(self) -> None:
        """Test artist randomization when query is vague"""
        distributor = ArtistDistributor()

        tracks = [
            {"artist": "Artist1", "similarity_score": 0.8},
            {"artist": "Artist2", "similarity_score": 0.7},
            {"artist": "Artist1", "similarity_score": 0.6},
        ]

        result = distributor.apply_artist_randomization(
            tracks, max_tracks=3, is_vague=True
        )

        # Should randomize when vague
        assert len(result) == 3
        assert all("similarity_score" in track for track in result)

    def test_smart_title_deduplication(self) -> None:
        """Test smart title deduplication"""
        deduplicator = TrackDeduplicator()

        tracks = [
            {"name": "Song (Remix)", "artist": "Artist1", "similarity_score": 0.8},
            {"name": "Song (Live)", "artist": "Artist2", "similarity_score": 0.7},
            {"name": "Song", "artist": "Artist3", "similarity_score": 0.9},
            {"name": "Different Song", "artist": "Artist4", "similarity_score": 0.6},
        ]

        result = deduplicator._smart_title_deduplication(tracks)

        # Should keep all tracks since they have different artists or base titles
        assert len(result) == 4

    def test_filter_and_deduplicate_results(self) -> None:
        """Test filtering and deduplication of results"""
        deduplicator = TrackDeduplicator()

        tracks = [
            {"name": "Song1", "artist": "Artist1", "similarity_score": 0.8},
            {"name": "Song1", "artist": "Artist1", "similarity_score": 0.7},
            {"name": "Song2", "artist": "Artist2", "similarity_score": 0.6},
            {"name": "Song3", "artist": "Artist3", "similarity_score": 0.5},
        ]

        result = deduplicator.deduplicate_tracks(
            tracks, min_similarity=0.6, max_tracks=3
        )

        # Should remove duplicates and filter by similarity
        assert len(result) <= 3
        assert all(track["similarity_score"] >= 0.6 for track in result)

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    def test_apply_genre_filtering(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test genre filtering in playlist generation"""
        generator = LocalPlaylistGenerator()

        tracks = [
            {"genre": "Rock", "similarity_score": 0.8},
            {"genre": "Pop", "similarity_score": 0.7},
            {"genre": "Jazz", "similarity_score": 0.6},
        ]

        result = generator._apply_genre_filtering("rock music", tracks)

        # Should boost rock tracks
        rock_tracks = [t for t in result if t.get("genre_boosted")]
        assert len(rock_tracks) > 0

    def test_save_playlist_m3u(self) -> None:
        """Test M3U playlist saving"""
        output = PlaylistOutput()

        tracks = [
            {
                "name": "Test Song",
                "artist": "Test Artist",
                "album": "Test Album",
                "duration_ms": 180000,
                "similarity_score": 0.8,
                "location": "/path/to/song.mp3",
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = output.save_playlist_m3u(tracks, "test query", temp_dir)
            assert filepath.endswith(".m3u")
            assert "test-query" in filepath

    def test_print_playlist_summary(self, capsys: Any) -> None:
        """Test playlist summary printing"""
        output = PlaylistOutput()

        tracks = [
            {
                "name": "Test Song",
                "artist": "Test Artist",
                "album": "Test Album",
                "similarity_score": 0.8,
            }
        ]

        output.print_playlist_summary(tracks, "test query")
        captured = capsys.readouterr()
        assert "Test Artist - Test Song" in captured.out
        assert "test query" in captured.out

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    def test_generate_playlist_success(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test successful playlist generation"""
        # Mock the embedding service
        mock_embedding_service = MagicMock()
        mock_embedding_service_class.return_value = mock_embedding_service

        # Mock the track embedder
        mock_track_embedder = MagicMock()
        mock_track_embedder_class.return_value = mock_track_embedder

        # Mock embeddings and track data
        mock_embeddings = [[0.1, 0.2, 0.3]]
        mock_track_data = [
            {
                "id": 1,
                "name": "Test Song",
                "artist": "Test Artist",
                "album": "Test Album",
                "genre": "Rock",
                "location": "/path/to/song.mp3",
            }
        ]

        mock_track_embedder.get_all_embeddings.return_value = (
            mock_embeddings,
            mock_track_data,
        )

        # Mock similarity search results
        mock_results = [
            {
                "id": 1,
                "name": "Test Song",
                "artist": "Test Artist",
                "album": "Test Album",
                "genre": "Rock",
                "location": "/path/to/song.mp3",
                "similarity_score": 0.8,
            }
        ]

        mock_embedding_service.similarity_search.return_value = mock_results

        generator = LocalPlaylistGenerator()

        result = generator.generate_playlist("rock music", max_tracks=5)

        assert len(result) > 0
        assert all("similarity_score" in track for track in result)

    @patch("tonal_hortator.core.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist_generator.LocalTrackEmbedder")
    def test_generate_playlist_with_artist_filter(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test playlist generation with artist filtering"""
        # Mock the embedding service
        mock_embedding_service = MagicMock()
        mock_embedding_service_class.return_value = mock_embedding_service

        # Mock the track embedder
        mock_track_embedder = MagicMock()
        mock_track_embedder_class.return_value = mock_track_embedder

        # Mock embeddings and track data
        mock_embeddings = [[0.1, 0.2, 0.3]]
        mock_track_data = [
            {
                "id": 1,
                "name": "Test Song",
                "artist": "Nirvana",
                "album": "Test Album",
                "genre": "Rock",
                "location": "/path/to/song.mp3",
            }
        ]

        mock_track_embedder.get_all_embeddings.return_value = (
            mock_embeddings,
            mock_track_data,
        )

        # Mock similarity search results
        mock_results = [
            {
                "id": 1,
                "name": "Test Song",
                "artist": "Nirvana",
                "album": "Test Album",
                "genre": "Rock",
                "location": "/path/to/song.mp3",
                "similarity_score": 0.8,
            }
        ]

        mock_embedding_service.similarity_search.return_value = mock_results

        generator = LocalPlaylistGenerator()

        result = generator.generate_playlist("Nirvana songs", max_tracks=5)

        assert len(result) > 0
        assert all("Nirvana" in track["artist"] for track in result)

    def test_enforce_artist_diversity(self) -> None:
        """Test artist diversity enforcement"""
        distributor = ArtistDistributor()

        tracks = [
            {"artist": "Artist1", "similarity_score": 0.8},
            {"artist": "Artist1", "similarity_score": 0.7},
            {"artist": "Artist1", "similarity_score": 0.6},
            {"artist": "Artist2", "similarity_score": 0.5},
            {"artist": "Artist2", "similarity_score": 0.4},
        ]

        result = distributor.enforce_artist_diversity(
            tracks, max_tracks=4, max_tracks_per_artist=2
        )

        # Should limit tracks per artist
        assert len(result) <= 4
        artist_counts: dict[str, int] = {}
        for track in result:
            artist = track["artist"]
            artist_counts[artist] = artist_counts.get(artist, 0) + 1

        # Each artist should have at most 2 tracks
        assert all(count <= 2 for count in artist_counts.values())

    def test_distribute_artists(self) -> None:
        """Test artist distribution"""
        distributor = ArtistDistributor()

        tracks = [
            {"artist": "Artist1", "similarity_score": 0.8},
            {"artist": "Artist1", "similarity_score": 0.7},
            {"artist": "Artist2", "similarity_score": 0.6},
            {"artist": "Artist2", "similarity_score": 0.5},
            {"artist": "Artist3", "similarity_score": 0.4},
        ]

        result = distributor._distribute_artists(tracks)

        # Should distribute artists throughout the playlist
        assert len(result) == 5
        # Check that artists are not all grouped together
        artists = [track["artist"] for track in result]
        # Simple check that we don't have all Artist1 tracks at the beginning
        assert not (artists[0] == "Artist1" and artists[1] == "Artist1")
