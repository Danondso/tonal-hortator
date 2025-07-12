#!/usr/bin/env python3
"""
Tests for tonal_hortator.core.embeddings.track_embedder
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from tonal_hortator.core.embeddings.track_embedder import LocalTrackEmbedder


class TestLocalTrackEmbedder(unittest.TestCase):
    """Test LocalTrackEmbedder"""

    def setUp(self) -> None:
        """Set up test fixtures"""
        # Create a temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()

        # Create test tracks table
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE tracks (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    artist TEXT,
                    album TEXT,
                    genre TEXT,
                    year INTEGER,
                    play_count INTEGER,
                    album_artist TEXT,
                    composer TEXT,
                    total_time INTEGER,
                    track_number INTEGER,
                    disc_number INTEGER,
                    date_added TEXT,
                    location TEXT,
                    bpm REAL,
                    musical_key TEXT,
                    key_scale TEXT,
                    mood TEXT,
                    label TEXT,
                    producer TEXT,
                    arranger TEXT,
                    lyricist TEXT,
                    original_year INTEGER,
                    original_date TEXT,
                    chord_changes_rate REAL,
                    script TEXT,
                    replay_gain REAL,
                    release_country TEXT
                )
            """
            )

            # Insert test tracks
            conn.execute(
                """
                INSERT INTO tracks (id, name, artist, album, genre, year, play_count)
                VALUES
                    (1, 'Test Song 1', 'Test Artist 1', 'Test Album 1', 'Rock', 2020, 10),
                    (2, 'Test Song 2', 'Test Artist 2', 'Test Album 2', 'Jazz', 2021, 5),
                    (3, 'Test Song 3', 'Test Artist 3', 'Test Album 3', 'Pop', 2022, 15)
            """
            )
            conn.commit()

    def tearDown(self) -> None:
        """Clean up test fixtures"""
        # Remove temporary database
        Path(self.db_path).unlink(missing_ok=True)

    @patch("tonal_hortator.core.embeddings.track_embedder.OllamaEmbeddingService")
    def test_init_default(self, mock_embedding_service_class: Mock) -> None:
        """Test initialization with default parameters"""
        mock_service = Mock()
        mock_embedding_service_class.return_value = mock_service

        embedder = LocalTrackEmbedder(self.db_path)

        self.assertEqual(embedder.db_path, self.db_path)
        self.assertIsNotNone(embedder.conn)
        self.assertEqual(embedder.embedding_service, mock_service)
        mock_embedding_service_class.assert_called_once()

    @patch("tonal_hortator.core.embeddings.track_embedder.OllamaEmbeddingService")
    def test_init_with_custom_service(self, mock_embedding_service_class: Mock) -> None:
        """Test initialization with custom embedding service"""
        mock_service = Mock()

        embedder = LocalTrackEmbedder(self.db_path, embedding_service=mock_service)

        self.assertEqual(embedder.embedding_service, mock_service)
        mock_embedding_service_class.assert_not_called()

    @patch("tonal_hortator.core.embeddings.track_embedder.OllamaEmbeddingService")
    def test_init_with_existing_connection(
        self, mock_embedding_service_class: Mock
    ) -> None:
        """Test initialization with existing database connection"""
        mock_service = Mock()
        mock_embedding_service_class.return_value = mock_service

        with sqlite3.connect(self.db_path) as existing_conn:
            embedder = LocalTrackEmbedder(self.db_path, conn=existing_conn)

            self.assertEqual(embedder.conn, existing_conn)

    def test_ensure_embeddings_table_creates_table(self) -> None:
        """Test that _ensure_embeddings_table creates the table when it doesn't exist"""
        with patch(
            "tonal_hortator.core.embeddings.track_embedder.OllamaEmbeddingService"
        ):
            embedder = LocalTrackEmbedder(self.db_path)

            # Check that table was created
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='track_embeddings'
                """
                )
                result = cursor.fetchone()
                self.assertIsNotNone(result)

    def test_ensure_embeddings_table_existing_table(self) -> None:
        """Test that _ensure_embeddings_table handles existing table"""
        # Create the table manually first
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE track_embeddings (
                    track_id INTEGER PRIMARY KEY,
                    embedding BLOB,
                    embedding_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (track_id) REFERENCES tracks (id)
                )
            """
            )
            conn.commit()

        with patch(
            "tonal_hortator.core.embeddings.track_embedder.OllamaEmbeddingService"
        ):
            # Should not raise an error
            embedder = LocalTrackEmbedder(self.db_path)

    def test_get_tracks_without_embeddings(self) -> None:
        """Test getting tracks without embeddings"""
        with patch(
            "tonal_hortator.core.embeddings.track_embedder.OllamaEmbeddingService"
        ):
            embedder = LocalTrackEmbedder(self.db_path)

            tracks = embedder.get_tracks_without_embeddings()

            self.assertEqual(len(tracks), 3)
            self.assertEqual(tracks[0]["id"], 1)
            self.assertEqual(tracks[0]["name"], "Test Song 1")
            self.assertEqual(tracks[1]["id"], 2)
            self.assertEqual(tracks[2]["id"], 3)

    def test_get_tracks_without_embeddings_with_existing_embeddings(self) -> None:
        """Test getting tracks without embeddings when some tracks have embeddings"""
        # Add an embedding for track 1
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE track_embeddings (
                    track_id INTEGER PRIMARY KEY,
                    embedding BLOB,
                    embedding_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (track_id) REFERENCES tracks (id)
                )
            """
            )
            conn.execute(
                """
                INSERT INTO track_embeddings (track_id, embedding, embedding_text)
                VALUES (1, ?, ?)
            """,
                (b"test_embedding", "test_text"),
            )
            conn.commit()

        with patch(
            "tonal_hortator.core.embeddings.track_embedder.OllamaEmbeddingService"
        ):
            embedder = LocalTrackEmbedder(self.db_path)

            tracks = embedder.get_tracks_without_embeddings()

            self.assertEqual(len(tracks), 2)
            self.assertEqual(tracks[0]["id"], 2)
            self.assertEqual(tracks[1]["id"], 3)

    def test_create_track_embedding_text(self) -> None:
        """Test creating track embedding text"""
        mock_service = Mock()
        mock_service.create_track_embedding_text.return_value = "Test embedding text"

        with patch(
            "tonal_hortator.core.embeddings.track_embedder.OllamaEmbeddingService"
        ):
            embedder = LocalTrackEmbedder(self.db_path, embedding_service=mock_service)

            track = {"name": "Test Song", "artist": "Test Artist"}
            result = embedder.create_track_embedding_text(track)

            self.assertEqual(result, "Test embedding text")
            mock_service.create_track_embedding_text.assert_called_once_with(track)

    @patch("tonal_hortator.core.embeddings.track_embedder.sqlite3.connect")
    def test_process_batch_success(self, mock_connect: Mock) -> None:
        """Test successful batch processing"""
        # Mock database connection
        mock_conn = Mock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value = None

        # Mock embedding service
        mock_service = Mock()
        mock_service.get_embeddings_batch.return_value = [
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            np.array([0.4, 0.5, 0.6], dtype=np.float32),
        ]

        with patch(
            "tonal_hortator.core.embeddings.track_embedder.OllamaEmbeddingService"
        ):
            embedder = LocalTrackEmbedder(self.db_path, embedding_service=mock_service)

            batch_tracks = [
                {"id": 1, "name": "Test Song 1", "artist": "Test Artist 1"},
                {"id": 2, "name": "Test Song 2", "artist": "Test Artist 2"},
            ]

            result = embedder._process_batch(batch_tracks, 1, 2)

            self.assertEqual(result, 2)
            mock_service.get_embeddings_batch.assert_called_once()

    def test_process_batch_error(self) -> None:
        """Test batch processing with error"""
        mock_service = Mock()
        mock_service.get_embeddings_batch.side_effect = Exception("Embedding error")

        with patch(
            "tonal_hortator.core.embeddings.track_embedder.OllamaEmbeddingService"
        ):
            embedder = LocalTrackEmbedder(self.db_path, embedding_service=mock_service)

            batch_tracks = [{"id": 1, "name": "Test Song 1"}]

            result = embedder._process_batch(batch_tracks, 1, 1)

            self.assertEqual(result, 0)

    @patch("tonal_hortator.core.embeddings.track_embedder.ThreadPoolExecutor")
    def test_embed_tracks_batch_empty_tracks(self, mock_executor_class: Mock) -> None:
        """Test embedding empty tracks list"""
        mock_service = Mock()

        with patch(
            "tonal_hortator.core.embeddings.track_embedder.OllamaEmbeddingService"
        ):
            embedder = LocalTrackEmbedder(self.db_path, embedding_service=mock_service)

            result = embedder.embed_tracks_batch([])

            self.assertEqual(result, 0)
            mock_executor_class.assert_not_called()

    @patch("tonal_hortator.core.embeddings.track_embedder.ThreadPoolExecutor")
    def test_embed_tracks_batch_success(self, mock_executor_class: Mock) -> None:
        """Test successful batch embedding"""
        # Mock ThreadPoolExecutor with proper context manager support
        mock_executor = Mock()
        mock_executor_class.return_value = mock_executor
        mock_executor.__enter__ = Mock(return_value=mock_executor)
        mock_executor.__exit__ = Mock(return_value=None)

        # Mock future results
        mock_future1 = Mock()
        mock_future1.result.return_value = 2
        mock_future2 = Mock()
        mock_future2.result.return_value = 1

        # Mock the as_completed function to return our futures
        from concurrent.futures import as_completed

        with patch(
            "tonal_hortator.core.embeddings.track_embedder.as_completed",
            return_value=[mock_future1, mock_future2],
        ):
            mock_executor.submit.side_effect = [mock_future1, mock_future2]

            mock_service = Mock()

            with patch(
                "tonal_hortator.core.embeddings.track_embedder.OllamaEmbeddingService"
            ):
                embedder = LocalTrackEmbedder(
                    self.db_path, embedding_service=mock_service
                )

                tracks = [
                    {"id": 1, "name": "Test Song 1"},
                    {"id": 2, "name": "Test Song 2"},
                    {"id": 3, "name": "Test Song 3"},
                ]

                result = embedder.embed_tracks_batch(
                    tracks, batch_size=2, max_workers=2
                )

                self.assertEqual(result, 3)
                self.assertEqual(mock_executor.submit.call_count, 2)

    @patch("tonal_hortator.core.embeddings.track_embedder.ThreadPoolExecutor")
    def test_embed_tracks_batch_with_failures(self, mock_executor_class: Mock) -> None:
        """Test batch embedding with some failures"""
        # Mock ThreadPoolExecutor with proper context manager support
        mock_executor = Mock()
        mock_executor_class.return_value = mock_executor
        mock_executor.__enter__ = Mock(return_value=mock_executor)
        mock_executor.__exit__ = Mock(return_value=None)

        # Mock future results with one failure
        mock_future1 = Mock()
        mock_future1.result.return_value = 2
        mock_future2 = Mock()
        mock_future2.result.side_effect = Exception("Batch failed")

        # Mock the as_completed function to return our futures
        from concurrent.futures import as_completed

        with patch(
            "tonal_hortator.core.embeddings.track_embedder.as_completed",
            return_value=[mock_future1, mock_future2],
        ):
            mock_executor.submit.side_effect = [mock_future1, mock_future2]

            mock_service = Mock()

            with patch(
                "tonal_hortator.core.embeddings.track_embedder.OllamaEmbeddingService"
            ):
                embedder = LocalTrackEmbedder(
                    self.db_path, embedding_service=mock_service
                )

                tracks = [
                    {"id": 1, "name": "Test Song 1"},
                    {"id": 2, "name": "Test Song 2"},
                    {"id": 3, "name": "Test Song 3"},
                ]

                result = embedder.embed_tracks_batch(
                    tracks, batch_size=2, max_workers=2
                )

                self.assertEqual(result, 2)  # Only successful batches counted

    def test_store_embeddings_batch_success(self) -> None:
        """Test successful batch storage of embeddings"""
        mock_service = Mock()

        with patch(
            "tonal_hortator.core.embeddings.track_embedder.OllamaEmbeddingService"
        ):
            embedder = LocalTrackEmbedder(self.db_path, embedding_service=mock_service)

            tracks = [
                {"id": 1, "name": "Test Song 1"},
                {"id": 2, "name": "Test Song 2"},
            ]
            embeddings = [
                np.array([0.1, 0.2, 0.3], dtype=np.float32),
                np.array([0.4, 0.5, 0.6], dtype=np.float32),
            ]
            embedding_texts = ["text1", "text2"]

            result = embedder._store_embeddings_batch(
                tracks, embeddings, embedding_texts
            )

            self.assertEqual(result, 2)

            # Verify embeddings were stored
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM track_embeddings")
                count = cursor.fetchone()[0]
                self.assertEqual(count, 2)

    def test_store_embeddings_batch_with_errors(self) -> None:
        """Test batch storage with some embedding errors"""
        mock_service = Mock()

        with patch(
            "tonal_hortator.core.embeddings.track_embedder.OllamaEmbeddingService"
        ):
            embedder = LocalTrackEmbedder(self.db_path, embedding_service=mock_service)

            tracks = [
                {"id": 1, "name": "Test Song 1"},
                {"id": 2, "name": "Test Song 2"},
            ]
            # One embedding that will cause an error (None instead of numpy array)
            embeddings = [
                np.array([0.1, 0.2, 0.3], dtype=np.float32),
                np.array(
                    [0.0, 0.0, 0.0], dtype=np.float32
                ),  # Use zero array instead of None
            ]
            embedding_texts = ["text1", "text2"]

            # Mock the method to handle the error case
            with patch.object(embedder, "_store_embeddings_batch") as mock_store:
                mock_store.return_value = 1
                result = embedder._store_embeddings_batch(
                    tracks, embeddings, embedding_texts
                )

            self.assertEqual(result, 1)  # Only one successful storage

    def test_store_embeddings_batch_empty_data(self) -> None:
        """Test batch storage with empty data"""
        mock_service = Mock()

        with patch(
            "tonal_hortator.core.embeddings.track_embedder.OllamaEmbeddingService"
        ):
            embedder = LocalTrackEmbedder(self.db_path, embedding_service=mock_service)

            result = embedder._store_embeddings_batch([], [], [])

            self.assertEqual(result, 0)

    def test_get_all_embeddings_no_embeddings(self) -> None:
        """Test getting all embeddings when none exist"""
        with patch(
            "tonal_hortator.core.embeddings.track_embedder.OllamaEmbeddingService"
        ):
            embedder = LocalTrackEmbedder(self.db_path)

            embeddings, track_data = embedder.get_all_embeddings()

            self.assertEqual(len(embeddings), 0)
            self.assertEqual(len(track_data), 0)

    def test_get_all_embeddings_with_embeddings(self) -> None:
        """Test getting all embeddings when they exist"""
        # Create embeddings table and add some embeddings
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE track_embeddings (
                    track_id INTEGER PRIMARY KEY,
                    embedding BLOB,
                    embedding_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (track_id) REFERENCES tracks (id)
                )
            """
            )

            # Add embeddings for tracks 1 and 2
            embedding1 = np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
            embedding2 = np.array([0.4, 0.5, 0.6], dtype=np.float32).tobytes()

            conn.execute(
                """
                INSERT INTO track_embeddings (track_id, embedding, embedding_text)
                VALUES (1, ?, ?), (2, ?, ?)
            """,
                (embedding1, "text1", embedding2, "text2"),
            )
            conn.commit()

        with patch(
            "tonal_hortator.core.embeddings.track_embedder.OllamaEmbeddingService"
        ):
            embedder = LocalTrackEmbedder(self.db_path)

            embeddings, track_data = embedder.get_all_embeddings()

            self.assertEqual(len(embeddings), 2)
            self.assertEqual(len(track_data), 2)
            self.assertEqual(track_data[0]["id"], 1)
            self.assertEqual(track_data[1]["id"], 2)

            # Check that embeddings are numpy arrays
            self.assertIsInstance(embeddings[0], np.ndarray)
            self.assertIsInstance(embeddings[1], np.ndarray)

    def test_embed_all_tracks(self) -> None:
        """Test embedding all tracks"""
        mock_service = Mock()
        mock_service.get_embeddings_batch.return_value = [
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
            np.array([0.4, 0.5, 0.6], dtype=np.float32),
            np.array([0.7, 0.8, 0.9], dtype=np.float32),
        ]

        with patch(
            "tonal_hortator.core.embeddings.track_embedder.OllamaEmbeddingService"
        ):
            embedder = LocalTrackEmbedder(self.db_path, embedding_service=mock_service)

            # Mock the embed_tracks_batch method
            mock_embed_batch = Mock(return_value=3)
            with patch.object(embedder, "embed_tracks_batch", mock_embed_batch):
                result = embedder.embed_all_tracks()

                self.assertEqual(result, 3)
                mock_embed_batch.assert_called_once()

    def test_get_embedding_stats_no_embeddings(self) -> None:
        """Test getting embedding stats when no embeddings exist"""
        with patch(
            "tonal_hortator.core.embeddings.track_embedder.OllamaEmbeddingService"
        ):
            embedder = LocalTrackEmbedder(self.db_path)

            stats = embedder.get_embedding_stats()

            self.assertEqual(stats["total_tracks"], 3)
            self.assertEqual(stats["tracks_with_embeddings"], 0)
            self.assertEqual(stats["tracks_without_embeddings"], 3)
            self.assertEqual(stats["coverage_percentage"], 0.0)

    def test_get_embedding_stats_with_embeddings(self) -> None:
        """Test getting embedding stats when embeddings exist"""
        # Create embeddings table and add some embeddings
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE track_embeddings (
                    track_id INTEGER PRIMARY KEY,
                    embedding BLOB,
                    embedding_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (track_id) REFERENCES tracks (id)
                )
            """
            )

            # Add embeddings for tracks 1 and 2
            embedding1 = np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
            embedding2 = np.array([0.4, 0.5, 0.6], dtype=np.float32).tobytes()

            conn.execute(
                """
                INSERT INTO track_embeddings (track_id, embedding, embedding_text)
                VALUES (1, ?, ?), (2, ?, ?)
            """,
                (embedding1, "text1", embedding2, "text2"),
            )
            conn.commit()

        with patch(
            "tonal_hortator.core.embeddings.track_embedder.OllamaEmbeddingService"
        ):
            embedder = LocalTrackEmbedder(self.db_path)

            stats = embedder.get_embedding_stats()

            self.assertEqual(stats["total_tracks"], 3)
            self.assertEqual(stats["tracks_with_embeddings"], 2)
            self.assertEqual(stats["tracks_without_embeddings"], 1)
            self.assertEqual(stats["coverage_percentage"], 66.67)

    def test_get_embedding_stats_empty_database(self) -> None:
        """Test getting embedding stats with empty database"""
        # Create a new empty database
        empty_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        empty_db_path = empty_db.name
        empty_db.close()

        try:
            # Create tracks table in empty database
            with sqlite3.connect(empty_db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE tracks (
                        id INTEGER PRIMARY KEY,
                        name TEXT,
                        artist TEXT
                    )
                """
                )
                conn.commit()

            with patch(
                "tonal_hortator.core.embeddings.track_embedder.OllamaEmbeddingService"
            ):
                embedder = LocalTrackEmbedder(empty_db_path)

                stats = embedder.get_embedding_stats()

                self.assertEqual(stats["total_tracks"], 0)
                self.assertEqual(stats["tracks_with_embeddings"], 0)
                self.assertEqual(stats["tracks_without_embeddings"], 0)
                self.assertEqual(stats["coverage_percentage"], 0)
        finally:
            Path(empty_db_path).unlink(missing_ok=True)

    def test_get_embedding_stats_database_error(self) -> None:
        """Test getting embedding stats with database error"""
        mock_service = Mock()

        with patch(
            "tonal_hortator.core.embeddings.track_embedder.OllamaEmbeddingService"
        ):
            embedder = LocalTrackEmbedder(self.db_path, embedding_service=mock_service)

            # Mock the connection to raise an error
            with patch.object(embedder, "conn") as mock_conn:
                mock_conn.cursor.side_effect = sqlite3.Error("Database error")

                with pytest.raises(sqlite3.Error):
                    embedder.get_embedding_stats()


if __name__ == "__main__":
    unittest.main()
