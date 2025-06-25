#!/usr/bin/env python3
"""
Tests for track embedder functionality
"""

import os
import sqlite3
import tempfile
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

import numpy as np

from tonal_hortator.core.track_embedder import LocalTrackEmbedder


class TestLocalTrackEmbedder(unittest.TestCase):
    """Test local track embedder functionality"""

    def setUp(self) -> None:
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_embeddings.db")
        self.embedder = LocalTrackEmbedder(db_path=self.db_path)

    def tearDown(self) -> None:
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_default_values(self) -> None:
        """Test initialization with default values"""
        self.assertIsNotNone(self.embedder.db_path)
        self.assertIsNotNone(self.embedder.embedding_service)

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values"""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".db") as temp_db:
            embedder = LocalTrackEmbedder(db_path=temp_db.name)
            self.assertEqual(embedder.db_path, temp_db.name)

    def test_ensure_embeddings_table(self) -> None:
        """Test table creation"""
        # The table should be created in __init__
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check track_embeddings table
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='track_embeddings'"
            )
            result = cursor.fetchone()
            self.assertIsNotNone(result)

    def test_create_track_embedding_text(self) -> None:
        """Test track embedding text creation"""
        track_data = {
            "id": 1,
            "name": "Test Song",
            "artist": "Test Artist",
            "album": "Test Album",
            "genre": "Rock",
            "year": 2020,
        }

        result = self.embedder.create_track_embedding_text(track_data)
        self.assertIsInstance(result, str)
        self.assertIn("Test Song", result)
        self.assertIn("Test Artist", result)

    def test_get_tracks_without_embeddings(self) -> None:
        """Test getting tracks without embeddings"""
        # First, create some tracks in the database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create tracks table if it doesn't exist
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tracks (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    artist TEXT,
                    album_artist TEXT,
                    composer TEXT,
                    album TEXT,
                    genre TEXT,
                    year INTEGER,
                    total_time INTEGER,
                    track_number INTEGER,
                    disc_number INTEGER,
                    play_count INTEGER,
                    bpm INTEGER,
                    location TEXT
                )
                """
            )

            # Insert some test tracks
            cursor.execute(
                """
                INSERT INTO tracks (id, name, artist, album, genre, year, location)
                VALUES (1, 'Test Song 1', 'Test Artist 1', 'Test Album 1', 'Rock', 2023, '/path/to/song1.mp3')
            """
            )
            cursor.execute(
                """
                INSERT INTO tracks (id, name, artist, album, genre, year, location)
                VALUES (2, 'Test Song 2', 'Test Artist 2', 'Test Album 2', 'Jazz', 2022, '/path/to/song2.mp3')
            """
            )
            conn.commit()

        result = self.embedder.get_tracks_without_embeddings()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "Test Song 1")
        self.assertEqual(result[1]["name"], "Test Song 2")

    def test_embed_tracks_batch_success(self) -> None:
        """Test successful batch embedding"""
        tracks = [
            {"id": 1, "name": "Song 1", "artist": "Artist 1"},
            {"id": 2, "name": "Song 2", "artist": "Artist 2"},
        ]

        with patch.object(
            self.embedder, "create_track_embedding_text"
        ) as mock_create_text:
            mock_create_text.side_effect = ["text1", "text2"]

            with patch.object(
                self.embedder.embedding_service, "get_embeddings_batch"
            ) as mock_embeddings:
                mock_embeddings.return_value = [[0.1, 0.2], [0.3, 0.4]]

                with patch.object(
                    self.embedder, "_store_embeddings_batch"
                ) as mock_store:
                    mock_store.return_value = 2

                    result = self.embedder.embed_tracks_batch(tracks)
                    self.assertEqual(result, 2)

    def test_embed_tracks_batch_empty_list(self) -> None:
        """Test batch embedding with empty list"""
        tracks: List[Dict[str, Any]] = []

        result = self.embedder.embed_tracks_batch(tracks)
        self.assertEqual(result, 0)

    def test_embed_tracks_batch_embedding_failure(self) -> None:
        """Test batch embedding when embedding fails"""
        tracks = [
            {"id": 1, "name": "Song 1", "artist": "Artist 1"},
        ]

        with patch.object(
            self.embedder, "create_track_embedding_text"
        ) as mock_create_text:
            mock_create_text.side_effect = ["text1"]

            with patch.object(
                self.embedder.embedding_service, "get_embeddings_batch"
            ) as mock_embeddings:
                mock_embeddings.side_effect = Exception("Embedding failed")

                result = self.embedder.embed_tracks_batch(tracks)
                self.assertEqual(result, 0)

    def test_store_embeddings_batch_success(self) -> None:
        """Test successful embedding storage"""
        tracks = [
            {"id": 1, "name": "Song 1", "artist": "Artist 1"},
            {"id": 2, "name": "Song 2", "artist": "Artist 2"},
        ]
        embeddings: List[np.ndarray] = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        embedding_texts = ["text1", "text2"]

        result = self.embedder._store_embeddings_batch(
            tracks, embeddings, embedding_texts
        )
        self.assertEqual(result, 2)

        # Verify embeddings were stored
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM track_embeddings")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 2)

    def test_store_embeddings_batch_duplicate_ids(self) -> None:
        """Test storing embeddings with duplicate IDs"""
        tracks = [
            {"id": 1, "name": "Song 1", "artist": "Artist 1"},
            {"id": 1, "name": "Song 1", "artist": "Artist 1"},  # Duplicate ID
        ]
        embeddings: List[np.ndarray] = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        embedding_texts = ["text1", "text2"]

        result = self.embedder._store_embeddings_batch(
            tracks, embeddings, embedding_texts
        )
        self.assertEqual(
            result, 2
        )  # Both attempts are counted, but only one row will exist in the table
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM track_embeddings")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)

    def test_store_embeddings_batch_empty_list(self) -> None:
        """Test storing empty embeddings list"""
        tracks: List[Dict[str, Any]] = []
        embeddings: List[np.ndarray] = []
        embedding_texts: List[str] = []

        result = self.embedder._store_embeddings_batch(
            tracks, embeddings, embedding_texts
        )
        self.assertEqual(result, 0)

    def test_embed_all_tracks(self) -> None:
        """Test embedding all tracks"""
        # First, create some tracks in the database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create tracks table if it doesn't exist
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tracks (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    artist TEXT,
                    album_artist TEXT,
                    composer TEXT,
                    album TEXT,
                    genre TEXT,
                    year INTEGER,
                    total_time INTEGER,
                    track_number INTEGER,
                    disc_number INTEGER,
                    play_count INTEGER,
                    bpm INTEGER,
                    location TEXT
                )
                """
            )

            # Insert some test tracks
            cursor.execute(
                """
                INSERT INTO tracks (id, name, artist, album, genre, year, location)
                VALUES (1, 'Test Song 1', 'Test Artist 1', 'Test Album 1', 'Rock', 2023, '/path/to/song1.mp3')
            """
            )
            cursor.execute(
                """
                INSERT INTO tracks (id, name, artist, album, genre, year, location)
                VALUES (2, 'Test Song 2', 'Test Artist 2', 'Test Album 2', 'Jazz', 2022, '/path/to/song2.mp3')
            """
            )
            conn.commit()

        with patch.object(self.embedder, "embed_tracks_batch") as mock_embed_batch:
            mock_embed_batch.return_value = 2

            result = self.embedder.embed_all_tracks()
            self.assertEqual(result, 2)

    def test_get_embedding_stats(self) -> None:
        """Test getting embedding statistics"""
        # First, create some embeddings
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create tracks table if it doesn't exist
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tracks (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    artist TEXT,
                    album_artist TEXT,
                    composer TEXT,
                    album TEXT,
                    genre TEXT,
                    year INTEGER,
                    total_time INTEGER,
                    track_number INTEGER,
                    disc_number INTEGER,
                    play_count INTEGER,
                    bpm INTEGER,
                    location TEXT
                )
                """
            )

            # Insert some test tracks
            cursor.execute(
                """
                INSERT INTO tracks (id, name, artist, album, genre, year, location)
                VALUES (1, 'Test Song 1', 'Test Artist 1', 'Test Album 1', 'Rock', 2023, '/path/to/song1.mp3')
            """
            )
            cursor.execute(
                """
                INSERT INTO tracks (id, name, artist, album, genre, year, location)
                VALUES (2, 'Test Song 2', 'Test Artist 2', 'Test Album 2', 'Jazz', 2022, '/path/to/song2.mp3')
            """
            )

            # Insert some embeddings
            cursor.execute(
                """
                INSERT INTO track_embeddings (track_id, embedding, embedding_text)
                VALUES (1, ?, 'text1')
            """,
                (b"test_embedding_1",),
            )
            conn.commit()

        result = self.embedder.get_embedding_stats()
        self.assertIsInstance(result, dict)
        self.assertIn("total_tracks", result)
        self.assertIn("tracks_with_embeddings", result)
        self.assertIn("coverage_percentage", result)

    def test_get_all_embeddings(self) -> None:
        """Test getting all embeddings"""
        # First, create some embeddings
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create tracks table if it doesn't exist
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tracks (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    artist TEXT,
                    album_artist TEXT,
                    composer TEXT,
                    album TEXT,
                    genre TEXT,
                    year INTEGER,
                    total_time INTEGER,
                    track_number INTEGER,
                    disc_number INTEGER,
                    play_count INTEGER,
                    bpm INTEGER,
                    location TEXT
                )
                """
            )

            # Insert some test tracks
            cursor.execute(
                """
                INSERT INTO tracks (id, name, artist, album, genre, year, location)
                VALUES (1, 'Test Song 1', 'Test Artist 1', 'Test Album 1', 'Rock', 2023, '/path/to/song1.mp3')
            """
            )

            # Insert some embeddings
            cursor.execute(
                """
                INSERT INTO track_embeddings (track_id, embedding, embedding_text)
                VALUES (1, ?, 'text1')
            """,
                (b"test_embedding_1",),
            )
            conn.commit()

        embeddings, tracks = self.embedder.get_all_embeddings()
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(len(tracks), 1)


if __name__ == "__main__":
    unittest.main()
