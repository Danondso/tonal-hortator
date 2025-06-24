#!/usr/bin/env python3
"""
Tests for track embedder module
"""

import os
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tonal_hortator.core.track_embedder import LocalTrackEmbedder


class TestLocalTrackEmbedder(unittest.TestCase):
    """Test Local Track Embedder functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_embedder.db")

        # Create database connection
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Create tracks table
        self.cursor.execute(
            """
            CREATE TABLE tracks (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                artist TEXT,
                album TEXT,
                genre TEXT,
                year INTEGER,
                play_count INTEGER DEFAULT 0,
                album_artist TEXT,
                composer TEXT,
                bpm INTEGER,
                location TEXT UNIQUE
            )
        """
        )

        # Create track_embeddings table
        self.cursor.execute(
            """
            CREATE TABLE track_embeddings (
                track_id INTEGER PRIMARY KEY,
                embedding BLOB,
                embedding_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Insert some test tracks
        test_tracks = [
            (
                1,
                "Test Song 1",
                "Test Artist 1",
                "Test Album 1",
                "Rock",
                2023,
                42,
                "Test Artist 1",
                "Test Composer 1",
                120,
                "/path/to/song1.mp3",
            ),
            (
                2,
                "Test Song 2",
                "Test Artist 2",
                "Test Album 2",
                "Jazz",
                2022,
                15,
                "Test Artist 2",
                "Test Composer 2",
                90,
                "/path/to/song2.mp3",
            ),
            (
                3,
                "Test Song 3",
                "Test Artist 3",
                "Test Album 3",
                "Pop",
                2021,
                30,
                "Test Artist 3",
                "Test Composer 3",
                110,
                "/path/to/song3.mp3",
            ),
        ]

        self.cursor.executemany(
            """
            INSERT INTO tracks (id, title, artist, album, genre, year, play_count, album_artist, composer, bpm, location)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            test_tracks,
        )

        self.conn.commit()

    def tearDown(self):
        """Clean up test fixtures"""
        self.conn.close()
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("tonal_hortator.core.embeddings.ollama")
    def test_init_with_existing_connection(self, mock_ollama):
        """Test initialization with existing database connection"""
        # Mock the embedding service
        mock_client = Mock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {
            "models": [{"name": "nomic-embed-text:latest"}]
        }
        mock_client.embeddings.return_value = {
            "embedding": np.random.rand(384).tolist()
        }

        # Create embedder with existing connection
        embedder = LocalTrackEmbedder(db_path=self.db_path, conn=self.conn)

        self.assertEqual(embedder.db_path, self.db_path)
        self.assertEqual(embedder.conn, self.conn)

    @patch("tonal_hortator.core.embeddings.ollama")
    def test_init_with_embedding_service(self, mock_ollama):
        """Test initialization with provided embedding service"""
        # Mock the embedding service
        mock_service = Mock()
        mock_client = Mock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {
            "models": [{"name": "nomic-embed-text:latest"}]
        }
        mock_client.embeddings.return_value = {
            "embedding": np.random.rand(384).tolist()
        }

        # Create embedder with provided service
        embedder = LocalTrackEmbedder(
            db_path=self.db_path, embedding_service=mock_service
        )

        self.assertEqual(embedder.embedding_service, mock_service)

    @patch("tonal_hortator.core.embeddings.ollama")
    def test_ensure_embeddings_table_creates_table(self, mock_ollama):
        """Test that embeddings table is created if it doesn't exist"""
        # Mock the embedding service
        mock_client = Mock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {
            "models": [{"name": "nomic-embed-text:latest"}]
        }
        mock_client.embeddings.return_value = {
            "embedding": np.random.rand(384).tolist()
        }

        # Create embedder (should create table)
        LocalTrackEmbedder(db_path=self.db_path, conn=self.conn)

        # Check that table was created
        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='track_embeddings'"
        )
        result = self.cursor.fetchone()
        self.assertIsNotNone(result)

    @patch("tonal_hortator.core.embeddings.ollama")
    def test_ensure_embeddings_table_exists(self, mock_ollama):
        """Test that embeddings table is not recreated if it exists"""
        # Mock the embedding service
        mock_client = Mock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {
            "models": [{"name": "nomic-embed-text:latest"}]
        }
        mock_client.embeddings.return_value = {
            "embedding": np.random.rand(384).tolist()
        }

        # Create embedder first time
        embedder1 = LocalTrackEmbedder(db_path=self.db_path, conn=self.conn)

        # Create embedder second time (table should already exist)
        embedder2 = LocalTrackEmbedder(db_path=self.db_path, conn=self.conn)

        # Both should work without error
        self.assertIsNotNone(embedder1)
        self.assertIsNotNone(embedder2)

    def test_get_tracks_without_embeddings(self):
        """Test getting tracks without embeddings"""
        embedder = LocalTrackEmbedder(db_path=self.db_path, conn=self.conn)

        tracks = embedder.get_tracks_without_embeddings()

        # Should return all tracks since none have embeddings
        self.assertEqual(len(tracks), 3)

        # Check track data
        track_names = [t["name"] for t in tracks]  # Changed from "title" to "name"
        self.assertIn("Test Song 1", track_names)
        self.assertIn("Test Song 2", track_names)
        self.assertIn("Test Song 3", track_names)

    def test_get_tracks_without_embeddings_some_have_embeddings(self):
        """Test getting tracks without embeddings when some tracks have embeddings"""
        embedder = LocalTrackEmbedder(db_path=self.db_path, conn=self.conn)

        # Insert an embedding for one track
        embedding1 = np.random.rand(384).astype(np.float32).tobytes()
        self.cursor.execute(
            """
            INSERT INTO track_embeddings (track_id, embedding, embedding_text)
            VALUES (?, ?, ?)
        """,
            (1, embedding1, "Test Song 1, Test Artist 1"),
        )
        self.conn.commit()

        tracks = embedder.get_tracks_without_embeddings()

        # Should return 2 tracks (excluding the one with embedding)
        self.assertEqual(len(tracks), 2)

        # Check that the track with embedding is not included
        track_ids = [t["id"] for t in tracks]
        self.assertNotIn(1, track_ids)
        self.assertIn(2, track_ids)
        self.assertIn(3, track_ids)

    def test_create_track_embedding_text(self):
        """Test creating track embedding text"""
        # Mock embedding service
        mock_service = Mock()
        mock_service.create_track_embedding_text.return_value = (
            "Test Song 1, Test Artist 1, Test Album 1"
        )

        embedder = LocalTrackEmbedder(
            db_path=self.db_path, conn=self.conn, embedding_service=mock_service
        )

        track = {
            "title": "Test Song 1",
            "artist": "Test Artist 1",
            "album": "Test Album 1",
        }
        result = embedder.create_track_embedding_text(track)

        self.assertEqual(result, "Test Song 1, Test Artist 1, Test Album 1")
        mock_service.create_track_embedding_text.assert_called_once_with(track)

    @patch("tonal_hortator.core.embeddings.ollama")
    def test_embed_tracks_batch_empty_tracks(self, mock_ollama):
        """Test embedding empty track list"""
        # Mock the embedding service
        mock_client = Mock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {
            "models": [{"name": "nomic-embed-text:latest"}]
        }
        mock_client.embeddings.return_value = {
            "embedding": np.random.rand(384).tolist()
        }

        embedder = LocalTrackEmbedder(db_path=self.db_path, conn=self.conn)

        result = embedder.embed_tracks_batch([])
        self.assertEqual(result, 0)

    @patch("tonal_hortator.core.embeddings.ollama")
    def test_embed_tracks_batch_success(self, mock_ollama):
        """Test successful batch embedding"""
        # Mock the embedding service
        mock_client = Mock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {
            "models": [{"name": "nomic-embed-text:latest"}]
        }
        mock_client.embeddings.return_value = {
            "embedding": np.random.rand(384).tolist()
        }

        embedder = LocalTrackEmbedder(db_path=self.db_path, conn=self.conn)

        # Patch create_track_embedding_text to return a string
        embedder.embedding_service.create_track_embedding_text = (
            lambda track: f"{track['title']}, {track['artist']}"
        )

        # Create test tracks
        tracks = [
            {
                "id": 1,
                "title": "Test Song 1",
                "artist": "Test Artist 1",
                "album": "Test Album 1",
            },
            {
                "id": 2,
                "title": "Test Song 2",
                "artist": "Test Artist 2",
                "album": "Test Album 2",
            },
        ]

        # Mock the embedding service methods
        mock_embedding_service = Mock()
        mock_embedding_service.get_embeddings_batch.return_value = [
            np.random.rand(384).astype(np.float32),
            np.random.rand(384).astype(np.float32),
        ]
        embedder.embedding_service.get_embeddings_batch = (
            mock_embedding_service.get_embeddings_batch
        )

        result = embedder.embed_tracks_batch(tracks, batch_size=1)

        # Should embed both tracks
        self.assertEqual(result, 2)

        # Verify embeddings were stored
        self.cursor.execute("SELECT COUNT(*) FROM track_embeddings")
        count = self.cursor.fetchone()[0]
        self.assertEqual(count, 2)

    @patch("tonal_hortator.core.embeddings.ollama")
    def test_embed_tracks_batch_exception_handling(self, mock_ollama):
        """Test batch embedding with exception handling"""
        # Mock the embedding service
        mock_client = Mock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {
            "models": [{"name": "nomic-embed-text:latest"}]
        }
        mock_client.embeddings.return_value = {
            "embedding": np.random.rand(384).tolist()
        }

        embedder = LocalTrackEmbedder(db_path=self.db_path, conn=self.conn)

        # Create test tracks
        tracks = [
            {
                "id": 1,
                "title": "Test Song 1",
                "artist": "Test Artist 1",
                "album": "Test Album 1",
            },
            {
                "id": 2,
                "title": "Test Song 2",
                "artist": "Test Artist 2",
                "album": "Test Album 2",
            },
        ]

        # Mock the embedding service to raise an exception
        mock_embedding_service = Mock()
        mock_embedding_service.get_embeddings_batch.side_effect = Exception(
            "Embedding failed"
        )
        embedder.embedding_service = mock_embedding_service

        result = embedder.embed_tracks_batch(tracks, batch_size=1)

        # Should return 0 due to exception
        self.assertEqual(result, 0)

    @patch("tonal_hortator.core.embeddings.ollama")
    def test_store_embeddings_batch(self, mock_ollama):
        """Test storing embeddings batch"""
        # Mock the embedding service
        mock_client = Mock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {
            "models": [{"name": "nomic-embed-text:latest"}]
        }
        mock_client.embeddings.return_value = {
            "embedding": np.random.rand(384).tolist()
        }

        embedder = LocalTrackEmbedder(db_path=self.db_path, conn=self.conn)

        # Create test data
        tracks = [
            {"id": 1, "title": "Test Song 1", "artist": "Test Artist 1"},
            {"id": 2, "title": "Test Song 2", "artist": "Test Artist 2"},
        ]
        embeddings = [
            np.random.rand(384).astype(np.float32),
            np.random.rand(384).astype(np.float32),
        ]
        embedding_texts = ["Test Song 1, Test Artist 1", "Test Song 2, Test Artist 2"]

        result = embedder._store_embeddings_batch(tracks, embeddings, embedding_texts)

        # Should store both embeddings
        self.assertEqual(result, 2)

        # Verify embeddings were stored
        self.cursor.execute("SELECT COUNT(*) FROM track_embeddings")
        count = self.cursor.fetchone()[0]
        self.assertEqual(count, 2)

    @patch("tonal_hortator.core.embeddings.ollama")
    def test_store_embeddings_batch_duplicate_track_id(self, mock_ollama):
        """Test storing embeddings with duplicate track ID"""
        # Mock the embedding service
        mock_client = Mock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {
            "models": [{"name": "nomic-embed-text:latest"}]
        }
        mock_client.embeddings.return_value = {
            "embedding": np.random.rand(384).tolist()
        }

        embedder = LocalTrackEmbedder(db_path=self.db_path, conn=self.conn)

        # Create test data with duplicate track ID
        tracks = [
            {"id": 1, "title": "Test Song 1", "artist": "Test Artist 1"},
            {
                "id": 1,
                "title": "Test Song 1",
                "artist": "Test Artist 1",
            },  # Duplicate ID
        ]
        embeddings = [
            np.random.rand(384).astype(np.float32),
            np.random.rand(384).astype(np.float32),
        ]
        embedding_texts = ["Test Song 1, Test Artist 1", "Test Song 1, Test Artist 1"]

        result = embedder._store_embeddings_batch(tracks, embeddings, embedding_texts)

        # Should store both embeddings due to INSERT OR REPLACE
        # The second one will replace the first one
        self.assertEqual(result, 2)

        # Verify only one embedding was stored (the last one)
        self.cursor.execute("SELECT COUNT(*) FROM track_embeddings")
        count = self.cursor.fetchone()[0]
        self.assertEqual(count, 1)

    @patch("tonal_hortator.core.embeddings.ollama")
    def test_get_all_embeddings(self, mock_ollama):
        """Test getting all embeddings"""
        # Mock the embedding service
        mock_client = Mock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {
            "models": [{"name": "nomic-embed-text:latest"}]
        }
        mock_client.embeddings.return_value = {
            "embedding": np.random.rand(384).tolist()
        }

        embedder = LocalTrackEmbedder(db_path=self.db_path, conn=self.conn)

        # Insert test embeddings
        embedding1 = np.random.rand(384).astype(np.float32).tobytes()
        embedding2 = np.random.rand(384).astype(np.float32).tobytes()

        self.cursor.execute(
            """
            INSERT INTO track_embeddings (track_id, embedding, embedding_text)
            VALUES (?, ?, ?)
        """,
            (1, embedding1, "Test Song 1, Test Artist 1"),
        )

        self.cursor.execute(
            """
            INSERT INTO track_embeddings (track_id, embedding, embedding_text)
            VALUES (?, ?, ?)
        """,
            (2, embedding2, "Test Song 2, Test Artist 2"),
        )

        self.conn.commit()

        # Get all embeddings
        embeddings, track_data = embedder.get_all_embeddings()

        # Should return 2 embeddings
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(track_data), 2)

        # Check that embeddings are numpy arrays
        self.assertIsInstance(embeddings[0], np.ndarray)
        self.assertIsInstance(embeddings[1], np.ndarray)

    @patch("tonal_hortator.core.embeddings.ollama")
    def test_get_all_embeddings_empty(self, mock_ollama):
        """Test getting all embeddings when none exist"""
        # Mock the embedding service
        mock_client = Mock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {
            "models": [{"name": "nomic-embed-text:latest"}]
        }
        mock_client.embeddings.return_value = {
            "embedding": np.random.rand(384).tolist()
        }

        embedder = LocalTrackEmbedder(db_path=self.db_path, conn=self.conn)

        # Get all embeddings (none exist)
        embeddings, track_data = embedder.get_all_embeddings()

        # Should return empty lists
        self.assertEqual(len(embeddings), 0)
        self.assertEqual(len(track_data), 0)

    @patch("tonal_hortator.core.embeddings.ollama")
    def test_embed_all_tracks(self, mock_ollama):
        """Test embedding all tracks"""
        # Mock the embedding service
        mock_client = Mock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {
            "models": [{"name": "nomic-embed-text:latest"}]
        }
        mock_client.embeddings.return_value = {
            "embedding": np.random.rand(384).tolist()
        }

        embedder = LocalTrackEmbedder(db_path=self.db_path, conn=self.conn)

        # Mock the get_tracks_without_embeddings method
        embedder.get_tracks_without_embeddings = Mock(
            return_value=[
                {"id": 1, "title": "Test Song 1", "artist": "Test Artist 1"},
                {"id": 2, "title": "Test Song 2", "artist": "Test Artist 2"},
            ]
        )

        # Mock the embed_tracks_batch method
        embedder.embed_tracks_batch = Mock(return_value=2)

        result = embedder.embed_all_tracks()

        # Should embed 2 tracks
        self.assertEqual(result, 2)

        # Verify methods were called
        embedder.get_tracks_without_embeddings.assert_called_once()
        embedder.embed_tracks_batch.assert_called_once()

    @patch("tonal_hortator.core.embeddings.ollama")
    def test_get_embedding_stats(self, mock_ollama):
        """Test getting embedding statistics"""
        # Mock the embedding service
        mock_client = Mock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {
            "models": [{"name": "nomic-embed-text:latest"}]
        }
        mock_client.embeddings.return_value = {
            "embedding": np.random.rand(384).tolist()
        }

        embedder = LocalTrackEmbedder(db_path=self.db_path, conn=self.conn)

        # Insert test embeddings
        embedding1 = np.random.rand(384).astype(np.float32).tobytes()
        self.cursor.execute(
            """
            INSERT INTO track_embeddings (track_id, embedding, embedding_text)
            VALUES (?, ?, ?)
        """,
            (1, embedding1, "Test Song 1, Test Artist 1"),
        )

        self.conn.commit()

        # Get stats
        stats = embedder.get_embedding_stats()

        # Should have correct stats
        self.assertEqual(stats["total_tracks"], 3)
        self.assertEqual(stats["tracks_with_embeddings"], 1)
        self.assertEqual(stats["tracks_without_embeddings"], 2)
        self.assertAlmostEqual(stats["coverage_percentage"], 33.33, places=1)

    @patch("tonal_hortator.core.embeddings.ollama")
    def test_get_embedding_stats_no_embeddings(self, mock_ollama):
        """Test getting embedding statistics when no embeddings exist"""
        # Mock the embedding service
        mock_client = Mock()
        mock_ollama.Client.return_value = mock_client
        mock_client.list.return_value = {
            "models": [{"name": "nomic-embed-text:latest"}]
        }
        mock_client.embeddings.return_value = {
            "embedding": np.random.rand(384).tolist()
        }

        embedder = LocalTrackEmbedder(db_path=self.db_path, conn=self.conn)

        # Get stats (no embeddings exist)
        stats = embedder.get_embedding_stats()

        # Should have correct stats
        self.assertEqual(stats["total_tracks"], 3)
        self.assertEqual(stats["tracks_with_embeddings"], 0)
        self.assertEqual(stats["tracks_without_embeddings"], 3)
        self.assertEqual(stats["coverage_percentage"], 0.0)


if __name__ == "__main__":
    unittest.main()
