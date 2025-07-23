#!/usr/bin/env python3
"""
Tests for tonal_hortator.core.playlist.playlist_generator
"""

import os
import unittest
from typing import List, cast
from unittest.mock import Mock, patch

import numpy as np
import pytest

from tonal_hortator.core.models import Track
from tonal_hortator.core.playlist.playlist_generator import LocalPlaylistGenerator


class TestLocalPlaylistGenerator(unittest.TestCase):
    """Test LocalPlaylistGenerator"""

    @patch("tonal_hortator.core.playlist.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist.playlist_generator.LocalTrackEmbedder")
    def test_init(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test initialization"""
        mock_embedding_service = Mock()
        mock_embedding_service_class.return_value = mock_embedding_service

        mock_track_embedder = Mock()
        mock_track_embedder_class.return_value = mock_track_embedder

        generator = LocalPlaylistGenerator()

        self.assertEqual(generator.db_path, "music_library.db")
        self.assertEqual(generator.embedding_service, mock_embedding_service)
        self.assertEqual(generator.track_embedder, mock_track_embedder)

    @patch("tonal_hortator.core.playlist.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist.playlist_generator.LocalTrackEmbedder")
    def test_init_custom_db_path(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test initialization with custom db path"""
        mock_embedding_service = Mock()
        mock_embedding_service_class.return_value = mock_embedding_service

        mock_track_embedder = Mock()
        mock_track_embedder_class.return_value = mock_track_embedder

        generator = LocalPlaylistGenerator(db_path="custom.db")

        self.assertEqual(generator.db_path, "custom.db")
        mock_track_embedder_class.assert_called_with(
            "custom.db", embedding_service=mock_embedding_service
        )

    @patch("tonal_hortator.core.playlist.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist.playlist_generator.LocalTrackEmbedder")
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
            self.assertTrue(generator._is_vague_query(query))

    @patch("tonal_hortator.core.playlist.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist.playlist_generator.LocalTrackEmbedder")
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
            self.assertFalse(generator._is_vague_query(query))

    @patch("tonal_hortator.core.playlist.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist.playlist_generator.LocalTrackEmbedder")
    def test_extract_track_count(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test track count extraction from query"""
        generator = LocalPlaylistGenerator()

        # Test queries with track counts
        self.assertEqual(generator._extract_track_count("10 tracks"), 10)
        self.assertEqual(generator._extract_track_count("5 songs"), 5)
        self.assertEqual(generator._extract_track_count("20 rock tracks"), 20)

        # Test queries without track counts
        self.assertIsNone(generator._extract_track_count("upbeat songs"))
        self.assertIsNone(generator._extract_track_count("rock music"))

    @patch("tonal_hortator.core.playlist.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist.playlist_generator.LocalTrackEmbedder")
    def test_extract_artist_from_query(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test artist extraction from query"""
        generator = LocalPlaylistGenerator()

        # Test queries with artists
        self.assertEqual(
            generator._extract_artist_from_query("Nirvana songs"), "Nirvana"
        )
        self.assertEqual(
            generator._extract_artist_from_query("The Beatles songs"), "The Beatles"
        )
        self.assertEqual(generator._extract_artist_from_query("Queen radio"), "Queen")

        # Test queries without artists
        self.assertIsNone(generator._extract_artist_from_query("upbeat songs"))
        self.assertIsNone(generator._extract_artist_from_query("rock music"))

    @patch("tonal_hortator.core.playlist.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist.playlist_generator.LocalTrackEmbedder")
    def test_extract_genre_keywords(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test genre keyword extraction"""
        generator = LocalPlaylistGenerator()

        # Test queries with genre keywords
        self.assertIn("rock", generator._extract_genre_keywords("rock music"))
        self.assertIn("jazz", generator._extract_genre_keywords("jazz songs"))
        self.assertIn("pop", generator._extract_genre_keywords("pop music"))
        self.assertIn("hip hop", generator._extract_genre_keywords("hip hop tracks"))

        # Test queries without genre keywords
        self.assertEqual(generator._extract_genre_keywords("upbeat songs"), [])
        self.assertEqual(generator._extract_genre_keywords("happy music"), [])

    @patch("tonal_hortator.core.playlist.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist.playlist_generator.LocalTrackEmbedder")
    def test_extract_base_name(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test base name extraction"""
        generator = LocalPlaylistGenerator()

        # Test various name patterns
        self.assertEqual(generator._extract_base_name("Song (Remix)"), "Song")
        self.assertEqual(generator._extract_base_name("Track (Live)"), "Track")
        self.assertEqual(generator._extract_base_name("Music (Acoustic)"), "Music")
        self.assertEqual(generator._extract_base_name("Hit (Radio Edit)"), "Hit")

        # Test names without suffixes
        self.assertEqual(generator._extract_base_name("Simple Song"), "Simple Song")
        self.assertEqual(generator._extract_base_name(""), "")

    @patch("tonal_hortator.core.playlist.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist.playlist_generator.LocalTrackEmbedder")
    def test_normalize_file_location(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test file location normalization"""
        generator = LocalPlaylistGenerator()

        # Test Windows path normalization
        result = generator._normalize_file_location(
            "C:\\Users\\username\\Music\\song.mp3"
        )
        self.assertNotIn("username", result)
        self.assertTrue(result.endswith("song.mp3"))

        # Test Unix path normalization
        result = generator._normalize_file_location("/home/username/Music/song.mp3")
        self.assertNotIn("username", result)
        self.assertTrue(result.endswith("song.mp3"))

        # Test empty location
        self.assertEqual(generator._normalize_file_location(""), "")

    @patch("tonal_hortator.core.playlist.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist.playlist_generator.LocalTrackEmbedder")
    def test_apply_artist_randomization_not_vague(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test artist randomization when query is not vague"""
        generator = LocalPlaylistGenerator()

        tracks = [
            Track.from_dict({"artist": "Artist1", "similarity_score": 0.8}),
            Track.from_dict({"artist": "Artist2", "similarity_score": 0.7}),
            Track.from_dict({"artist": "Artist1", "similarity_score": 0.6}),
        ]

        result = generator._apply_artist_randomization(
            tracks, max_tracks=3, is_vague=False
        )

        # Should not randomize when not vague
        self.assertEqual(result, tracks[:3])

    @patch("tonal_hortator.core.playlist.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist.playlist_generator.LocalTrackEmbedder")
    def test_apply_artist_randomization_vague(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test artist randomization when query is vague"""
        generator = LocalPlaylistGenerator()

        tracks = [
            Track.from_dict({"artist": "Artist1", "similarity_score": 0.8}),
            Track.from_dict({"artist": "Artist2", "similarity_score": 0.7}),
            Track.from_dict({"artist": "Artist1", "similarity_score": 0.6}),
        ]

        result = generator._apply_artist_randomization(
            tracks, max_tracks=3, is_vague=True
        )

        # Should randomize when vague
        self.assertEqual(len(result), 3)
        self.assertTrue(
            all(
                hasattr(track, "similarity_score")
                and track.similarity_score is not None
                for track in result
            )
        )

    @patch("tonal_hortator.core.playlist.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist.playlist_generator.LocalTrackEmbedder")
    def test_smart_name_deduplication(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test smart name deduplication"""
        generator = LocalPlaylistGenerator()

        tracks = [
            Track.from_dict(
                {"name": "Song (Remix)", "artist": "Artist1", "similarity_score": 0.8}
            ),
            Track.from_dict(
                {"name": "Song (Live)", "artist": "Artist1", "similarity_score": 0.7}
            ),
            Track.from_dict(
                {"name": "Song", "artist": "Artist1", "similarity_score": 0.9}
            ),
            Track.from_dict(
                {"name": "Different Song", "artist": "Artist2", "similarity_score": 0.6}
            ),
        ]

        result = generator._smart_name_deduplication(tracks)

        # Should keep only 2 tracks: the best "Song" variant by Artist1 and "Different Song" by Artist2
        self.assertEqual(len(result), 2)
        # The "Song" track with highest similarity score (0.9) should be kept
        song_track = next(track for track in result if track.name == "Song")
        self.assertEqual(song_track.similarity_score, 0.9)
        # The "Different Song" track should be kept
        self.assertTrue(any(track.name == "Different Song" for track in result))

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skipping playlist generation test in CI (LLM query parser requires Ollama)",
    )
    @patch("tonal_hortator.core.playlist.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist.playlist_generator.LocalTrackEmbedder")
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
            {"id": 1, "name": "Song1", "artist": "Artist1", "similarity_score": 0.8},
            {"id": 2, "name": "Song2", "artist": "Artist2", "similarity_score": 0.7},
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
            Track.from_dict(
                {"id": 1, "name": "Song1", "artist": "Artist1", "similarity_score": 0.8}
            ),
            Track.from_dict(
                {"id": 2, "name": "Song2", "artist": "Artist2", "similarity_score": 0.7}
            ),
        ]

        generator = LocalPlaylistGenerator()
        result = generator.generate_playlist("test query", max_tracks=5)
        self.assertIsInstance(result, list)

    @patch("tonal_hortator.core.playlist.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist.playlist_generator.LocalTrackEmbedder")
    def test_enforce_artist_diversity(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test artist diversity enforcement"""
        generator = LocalPlaylistGenerator()

        # Create test tracks with multiple tracks from the same artist
        tracks = [
            Track.from_dict(
                {"name": "Song1", "artist": "Artist1", "similarity_score": 0.9}
            ),
            Track.from_dict(
                {"name": "Song2", "artist": "Artist1", "similarity_score": 0.8}
            ),
            Track.from_dict(
                {"name": "Song3", "artist": "Artist1", "similarity_score": 0.7}
            ),
            Track.from_dict(
                {"name": "Song4", "artist": "Artist1", "similarity_score": 0.6}
            ),
            Track.from_dict(
                {"name": "Song5", "artist": "Artist2", "similarity_score": 0.85}
            ),
            Track.from_dict(
                {"name": "Song6", "artist": "Artist2", "similarity_score": 0.75}
            ),
            Track.from_dict(
                {"name": "Song7", "artist": "Artist3", "similarity_score": 0.95}
            ),
        ]

        # Test with max 2 tracks per artist
        result = generator._enforce_artist_diversity(
            tracks, max_tracks=10, max_tracks_per_artist=2
        )

        # Should have at most 2 tracks per artist, sorted by similarity
        artist_counts: dict[str, int] = {}
        for track in result:
            artist = track.artist or ""
            artist_counts[artist] = artist_counts.get(artist, 0) + 1

        # Check that no artist has more than 2 tracks
        self.assertTrue(all(count <= 2 for count in artist_counts.values()))

        # Check that we got the best tracks (highest similarity scores)
        artist1_tracks = [t for t in result if t.artist == "Artist1"]
        self.assertLessEqual(len(artist1_tracks), 2)
        if len(artist1_tracks) >= 2:
            scores: List[float] = [
                float(cast(float, t.similarity_score))
                for t in tracks
                if t.artist == "Artist1"
            ]
            result_scores: List[float] = [
                float(cast(float, t.similarity_score)) for t in artist1_tracks
            ]
            scores = sorted(scores, reverse=True)
            result_scores = sorted(result_scores, reverse=True)
            self.assertEqual(result_scores, scores[:2])

    @patch("tonal_hortator.core.playlist.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist.playlist_generator.LocalTrackEmbedder")
    def test_distribute_artists(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test artist distribution"""
        generator = LocalPlaylistGenerator()

        # Create test tracks with different artists
        tracks = [
            Track.from_dict(
                {"artist": "Artist1", "name": "Song1", "similarity_score": 0.8}
            ),
            Track.from_dict(
                {"artist": "Artist2", "name": "Song2", "similarity_score": 0.7}
            ),
            Track.from_dict(
                {"artist": "Artist1", "name": "Song3", "similarity_score": 0.6}
            ),
            Track.from_dict(
                {"artist": "Artist3", "name": "Song4", "similarity_score": 0.5}
            ),
        ]

        result = generator._distribute_artists(tracks, max_tracks=3)

        # Should return 3 tracks with distributed artists
        self.assertEqual(len(result), 3)

        # Check that artists are distributed (not all from same artist)
        artists = [track.artist or "" for track in result]
        self.assertGreater(len(set(artists)), 1)

    @patch("tonal_hortator.core.playlist.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist.playlist_generator.LocalTrackEmbedder")
    def test_create_playlist_name_basic_queries(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test basic playlist name creation"""
        generator = LocalPlaylistGenerator()

        # Test basic queries
        self.assertEqual(
            generator._create_playlist_name("jazz for studying"), "Jazz for Studying"
        )
        self.assertEqual(
            generator._create_playlist_name("5 grunge songs"), "5 Grunge Songs"
        )
        self.assertEqual(generator._create_playlist_name("rock"), "Rock Mix")
        self.assertEqual(
            generator._create_playlist_name("electronic"), "Electronic Mix"
        )

    @patch("tonal_hortator.core.playlist.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist.playlist_generator.LocalTrackEmbedder")
    def test_create_playlist_name_with_prefixes(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test playlist name creation with common prefixes"""
        generator = LocalPlaylistGenerator()

        # Test queries with prefixes that should be removed
        self.assertEqual(
            generator._create_playlist_name("generate jazz for studying"),
            "Jazz for Studying",
        )
        self.assertEqual(
            generator._create_playlist_name("create 5 grunge songs"), "5 Grunge Songs"
        )
        self.assertEqual(
            generator._create_playlist_name("find me some upbeat music"), "Upbeat Music"
        )
        self.assertEqual(
            generator._create_playlist_name("make a playlist of 10 songs"),
            "Playlist Of 10 Songs",
        )
        self.assertEqual(
            generator._create_playlist_name("get rock music"), "Rock Music"
        )
        self.assertEqual(generator._create_playlist_name("show me jazz"), "Jazz Mix")
        self.assertEqual(
            generator._create_playlist_name("give me electronic"), "Electronic Mix"
        )

    @patch("tonal_hortator.core.playlist.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist.playlist_generator.LocalTrackEmbedder")
    def test_create_playlist_name_with_quotes(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test playlist name creation with quoted queries"""
        generator = LocalPlaylistGenerator()

        # Test queries with quotes
        self.assertEqual(generator._create_playlist_name("bedroom pop"), "Bedroom Pop")
        self.assertEqual(
            generator._create_playlist_name("90s alternative"), "90S Alternative"
        )
        self.assertEqual(
            generator._create_playlist_name("jazz for studying"), "Jazz for Studying"
        )

    @patch("tonal_hortator.core.playlist.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist.playlist_generator.LocalTrackEmbedder")
    def test_create_playlist_name_with_filler_words(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test playlist name creation with filler words"""
        generator = LocalPlaylistGenerator()

        # Test queries with filler words that should be removed
        self.assertEqual(
            generator._create_playlist_name("find me some upbeat music"), "Upbeat Music"
        )
        self.assertEqual(
            generator._create_playlist_name("get me some rock songs"), "Rock Songs"
        )
        self.assertEqual(
            generator._create_playlist_name("show me some jazz"), "Jazz Mix"
        )

    @patch("tonal_hortator.core.playlist.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist.playlist_generator.LocalTrackEmbedder")
    def test_create_playlist_name_with_artist_queries(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test playlist name creation with artist-specific queries"""
        generator = LocalPlaylistGenerator()

        # Test queries with artists
        self.assertEqual(
            generator._create_playlist_name("20 songs by my chemical romance"),
            "20 Songs By My Chemical Romance",
        )
        self.assertEqual(
            generator._create_playlist_name("generate 15 tracks by nirvana"),
            "15 Tracks By Nirvana",
        )
        self.assertEqual(
            generator._create_playlist_name("find songs by the beatles"),
            "Songs By The Beatles",
        )

    @patch("tonal_hortator.core.playlist.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist.playlist_generator.LocalTrackEmbedder")
    def test_create_playlist_name_with_context_queries(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test playlist name creation with context-specific queries"""
        generator = LocalPlaylistGenerator()

        # Test queries with context
        self.assertEqual(
            generator._create_playlist_name("classic rock for driving"),
            "Classic Rock for Driving",
        )
        self.assertEqual(
            generator._create_playlist_name("upbeat music for working out"),
            "Upbeat Music for Working Out",
        )
        self.assertEqual(
            generator._create_playlist_name("chill music for studying"),
            "Chill Music for Studying",
        )
        self.assertEqual(
            generator._create_playlist_name("party music for dancing"),
            "Party Music for Dancing",
        )

    @patch("tonal_hortator.core.playlist.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist.playlist_generator.LocalTrackEmbedder")
    def test_create_playlist_name_edge_cases(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test playlist name creation with edge cases"""
        generator = LocalPlaylistGenerator()

        # Test edge cases
        self.assertEqual(generator._create_playlist_name(""), "Mix")  # Empty string
        self.assertEqual(generator._create_playlist_name("a"), "A Mix")  # Single letter
        self.assertEqual(
            generator._create_playlist_name("the"), "The Mix"
        )  # Single word
        self.assertEqual(
            generator._create_playlist_name("   jazz   "), "Jazz Mix"
        )  # Extra whitespace
        self.assertEqual(
            generator._create_playlist_name("generate"), "Generate Mix"
        )  # Just prefix

    @patch("tonal_hortator.core.playlist.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist.playlist_generator.LocalTrackEmbedder")
    def test_create_playlist_name_preserves_important_words(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test that important words are preserved in playlist names"""
        generator = LocalPlaylistGenerator()

        # Test that important words are preserved
        self.assertEqual(
            generator._create_playlist_name("early 90s grunge"), "Early 90S Grunge"
        )
        self.assertEqual(
            generator._create_playlist_name("late 80s rock"), "Late 80S Rock"
        )
        self.assertEqual(
            generator._create_playlist_name("video game music"), "Video Game Music"
        )
        self.assertEqual(
            generator._create_playlist_name("background ambient music"),
            "Background Ambient Music",
        )

    @patch("tonal_hortator.core.playlist.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist.playlist_generator.LocalTrackEmbedder")
    def test_create_playlist_name_music_terms(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test that music-related terms are handled correctly"""
        generator = LocalPlaylistGenerator()

        # Test music-related terms
        self.assertEqual(generator._create_playlist_name("rock music"), "Rock Music")
        self.assertEqual(generator._create_playlist_name("jazz songs"), "Jazz Songs")
        self.assertEqual(generator._create_playlist_name("pop tracks"), "Pop Tracks")
        self.assertEqual(
            generator._create_playlist_name("electronic playlist"),
            "Electronic Playlist",
        )

    @patch("tonal_hortator.core.playlist.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist.playlist_generator.LocalTrackEmbedder")
    def test_playlist_preview_matches_export_count(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test that playlist preview count matches exported file count.

        This test validates that the bug where CLI showed X tracks but only Y tracks
        were saved to the file has been fixed. Both should use the same deduplication method.
        """
        # Setup mocks
        mock_embedding_service = Mock()
        mock_embedding_service_class.return_value = mock_embedding_service

        mock_track_embedder = Mock()
        mock_track_embedder_class.return_value = mock_track_embedder

        # Create generator instance
        generator = LocalPlaylistGenerator()

        # Create test tracks with similarity scores
        mock_tracks = [
            Track(
                id=1,
                name="Desert Highway",
                artist="Desert Rock Band",
                album="Desert Album",
                genre="Rock",
                location="/path/to/track1.mp3",
                similarity_score=0.95,
            ),
            Track(
                id=2,
                name="Sandstorm",
                artist="Desert Rock Band",
                album="Desert Album",
                genre="Rock",
                location="/path/to/track2.mp3",
                similarity_score=0.92,
            ),
            Track(
                id=3,
                name="Cactus Blues",
                artist="Different Artist",
                album="Different Album",
                genre="Blues",
                location="/path/to/track3.mp3",
                similarity_score=0.88,
            ),
            Track(
                id=4,
                name="Mesa Drive",
                artist="Desert Rock Band",
                album="Desert Album",
                genre="Rock",
                location="/path/to/track4.mp3",
                similarity_score=0.85,
            ),
            Track(
                id=5,
                name="Dune Rider",
                artist="Another Artist",
                album="Another Album",
                genre="Electronic",
                location="/path/to/track5.mp3",
                similarity_score=0.82,
            ),
        ]

        # Test parameters
        max_tracks = 3
        min_similarity = 0.3

        # Test the internal deduplication method directly
        # This simulates what happens during preview
        preview_tracks = generator._filter_and_deduplicate_results(
            mock_tracks, min_similarity, max_tracks, False, 0.5
        )

        # Test the external deduplicator method (the old buggy path)
        # This simulates what was happening during export
        # Pass the same callback functions that the internal method uses
        export_tracks = generator.deduplicator.filter_and_deduplicate_results(
            mock_tracks,
            min_similarity,
            max_tracks,
            False,
            0.5,
            sample_with_randomization=generator._sample_with_randomization,
            smart_name_deduplication=generator._smart_name_deduplication,
            enforce_artist_diversity=generator._enforce_artist_diversity,
            distribute_artists=generator._distribute_artists,
            logger=generator.logger if hasattr(generator, "logger") else None,
        )

        # The key assertion: both methods should return the same number of tracks
        self.assertEqual(
            len(preview_tracks),
            len(export_tracks),
            f"Preview count ({len(preview_tracks)}) should match export count ({len(export_tracks)})",
        )

        # Additional validation: both should respect max_tracks
        self.assertLessEqual(
            len(preview_tracks),
            max_tracks,
            f"Preview should not exceed max_tracks ({max_tracks})",
        )
        self.assertLessEqual(
            len(export_tracks),
            max_tracks,
            f"Export should not exceed max_tracks ({max_tracks})",
        )

        # Verify that all returned tracks are from the input set and unique
        input_ids = {track.id for track in mock_tracks}
        preview_ids = {track.id for track in preview_tracks}
        export_ids = {track.id for track in export_tracks}
        self.assertTrue(
            preview_ids.issubset(input_ids), "Preview tracks must be from input set"
        )
        self.assertTrue(
            export_ids.issubset(input_ids), "Export tracks must be from input set"
        )
        self.assertEqual(
            len(preview_ids), len(preview_tracks), "Preview tracks must be unique"
        )
        self.assertEqual(
            len(export_ids), len(export_tracks), "Export tracks must be unique"
        )

        # Verify that both methods return some tracks (not empty)
        self.assertGreater(len(preview_tracks), 0, "Preview should return some tracks")
        self.assertGreater(len(export_tracks), 0, "Export should return some tracks")

        # Additional validation: both should return exactly max_tracks (since we have enough tracks)
        self.assertEqual(
            len(preview_tracks),
            max_tracks,
            f"Preview should return exactly {max_tracks} tracks",
        )
        self.assertEqual(
            len(export_tracks),
            max_tracks,
            f"Export should return exactly {max_tracks} tracks",
        )

    @patch("tonal_hortator.core.playlist.playlist_generator.OllamaEmbeddingService")
    @patch("tonal_hortator.core.playlist.playlist_generator.LocalTrackEmbedder")
    def test_playlist_generation_consistency_with_different_parameters(
        self, mock_track_embedder_class: Mock, mock_embedding_service_class: Mock
    ) -> None:
        """Test playlist generation consistency across different parameter combinations."""

        # Setup mocks
        mock_embedding_service = Mock()
        mock_embedding_service_class.return_value = mock_embedding_service

        mock_track_embedder = Mock()
        mock_track_embedder_class.return_value = mock_track_embedder

        # Mock the track embedder methods that are called during playlist generation
        mock_track_embedder.get_all_embeddings.return_value = (
            [],
            [],
        )  # Empty embeddings and track data
        mock_track_embedder.get_tracks_without_embeddings.return_value = []

        # Create generator instance
        generator = LocalPlaylistGenerator()

        # Create a larger set of mock tracks for more realistic testing
        mock_tracks = []
        for i in range(20):
            mock_tracks.append(
                Track(
                    id=i + 1,
                    name=f"Track {i + 1}",
                    artist=f"Artist {(i % 5) + 1}",  # 5 different artists
                    album=f"Album {(i % 3) + 1}",  # 3 different albums
                    genre=f"Genre {(i % 4) + 1}",  # 4 different genres
                    location=f"/path/to/track{i + 1}.mp3",
                    similarity_score=0.9 - (i * 0.02),  # Decreasing similarity scores
                )
            )

        mock_embedding_service.similarity_search.return_value = mock_tracks

        # Test different parameter combinations
        test_cases = [
            {"max_tracks": 5, "min_similarity": 0.3, "search_breadth": 10},
            {"max_tracks": 10, "min_similarity": 0.5, "search_breadth": 15},
            {"max_tracks": 3, "min_similarity": 0.7, "search_breadth": 5},
            {"max_tracks": 15, "min_similarity": 0.2, "search_breadth": 20},
        ]

        for params in test_cases:
            with self.subTest(**params):
                # Generate playlist with current parameters
                result = generator.generate_playlist(
                    query="test query",
                    max_tracks=int(params["max_tracks"]),
                    min_similarity=params["min_similarity"],
                    search_breadth_factor=int(params["search_breadth"]),
                )

                # Verify result respects max_tracks
                self.assertLessEqual(
                    len(result),
                    params["max_tracks"],
                    f"Result length {len(result)} should not exceed max_tracks {params['max_tracks']}",
                )

                # Verify all tracks meet minimum similarity
                for track in result:
                    self.assertGreaterEqual(
                        track.similarity_score or 0,
                        params["min_similarity"],
                        f"Track {track.id} similarity {track.similarity_score} should be >= {params['min_similarity']}",
                    )

                # Test internal vs external deduplication consistency
                preview_tracks = generator._filter_and_deduplicate_results(
                    mock_tracks,
                    params["min_similarity"],
                    int(params["max_tracks"]),
                    False,
                    0.5,
                )

                export_tracks = generator.deduplicator.filter_and_deduplicate_results(
                    mock_tracks,
                    params["min_similarity"],
                    int(params["max_tracks"]),
                    False,
                    0.5,
                    sample_with_randomization=generator._sample_with_randomization,
                    smart_name_deduplication=generator._smart_name_deduplication,
                    enforce_artist_diversity=generator._enforce_artist_diversity,
                    distribute_artists=generator._distribute_artists,
                    logger=generator.logger if hasattr(generator, "logger") else None,
                )

                # Both should return the same count
                self.assertEqual(
                    len(preview_tracks),
                    len(export_tracks),
                    f"Preview/export count mismatch for params {params}",
                )


if __name__ == "__main__":
    unittest.main()
