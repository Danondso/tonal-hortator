import sqlite3
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tonal_hortator.core.playlist_generator import LocalPlaylistGenerator
from tonal_hortator.core.track_embedder import LocalTrackEmbedder


class TestCoreFunctionality(unittest.TestCase):

    def setUp(self) -> None:
        """Set up an in-memory database with a single track for testing."""
        self.db_path = ":memory:"
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Create tracks table with the correct schema
        self.cursor.execute(
            """
            CREATE TABLE tracks (
                id INTEGER PRIMARY KEY,
                title TEXT,
                artist TEXT,
                album TEXT,
                genre TEXT,
                year INTEGER,
                play_count INTEGER,
                album_artist TEXT,
                composer TEXT,
                bpm INTEGER,
                location TEXT
            )
        """
        )

        # Insert a sample track
        self.cursor.execute(
            """
            INSERT INTO tracks (id, title, artist, album, genre, year, play_count, bpm, location)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                1,
                "Test Song",
                "Test Artist",
                "Test Album",
                "Rock",
                2023,
                42,
                120,
                "/path/to/test.mp3",
            ),
        )
        self.conn.commit()

    def tearDown(self) -> None:
        """Close the database connection."""
        self.conn.close()

    @patch("tonal_hortator.core.embeddings.ollama")
    def test_track_embedding(self, mock_ollama: Mock) -> None:
        """Test that a single track can be successfully embedded."""
        # Mock the Ollama client and embeddings
        mock_client = Mock()
        mock_ollama.Client.return_value = mock_client

        # Mock the list() method to return a proper structure
        mock_client.list.return_value = {
            "models": [{"name": "nomic-embed-text:latest"}]
        }

        # Mock the embeddings response
        mock_embedding = np.random.rand(384).astype(
            np.float32
        )  # Typical embedding size
        mock_client.embeddings.return_value = {"embedding": mock_embedding.tolist()}

        embedder = LocalTrackEmbedder(db_path=self.db_path, conn=self.conn)
        embedded_count = embedder.embed_all_tracks()
        self.assertEqual(embedded_count, 1)

        # Verify the embedding was stored
        self.cursor.execute("SELECT COUNT(*) FROM track_embeddings WHERE track_id = 1")
        count = self.cursor.fetchone()[0]
        self.assertEqual(count, 1)

    @patch("tonal_hortator.core.embeddings.ollama")
    def test_playlist_generation(self, mock_ollama: Mock) -> None:
        """Test that a simple playlist can be generated."""
        # Mock the Ollama client and embeddings
        mock_client = Mock()
        mock_ollama.Client.return_value = mock_client

        # Mock the list() method to return a proper structure
        mock_client.list.return_value = {
            "models": [{"name": "nomic-embed-text:latest"}]
        }

        # Mock the embeddings response
        mock_embedding = np.random.rand(384).astype(np.float32)
        mock_client.embeddings.return_value = {"embedding": mock_embedding.tolist()}

        # First, ensure the track is embedded
        embedder = LocalTrackEmbedder(db_path=self.db_path, conn=self.conn)
        embedder.embed_all_tracks()

        # Now, generate a playlist - create a new embedder for the playlist generator
        embedder_for_playlist = LocalTrackEmbedder(db_path=self.db_path, conn=self.conn)
        playlist_generator = LocalPlaylistGenerator(db_path=self.db_path)
        playlist_generator.track_embedder = embedder_for_playlist
        playlist = playlist_generator.generate_playlist("a test song")

        self.assertIsNotNone(playlist)
        self.assertEqual(len(playlist), 1)
        self.assertEqual(playlist[0]["name"], "Test Song")

    @patch("tonal_hortator.core.embeddings.ollama")
    def test_generate_playlist(self, mock_ollama: Mock) -> None:
        """Test that a simple playlist can be generated."""
        # Mock the Ollama client and embeddings
        mock_client = Mock()
        mock_ollama.Client.return_value = mock_client

        # Mock the list() method to return a proper structure
        mock_client.list.return_value = {
            "models": [{"name": "nomic-embed-text:latest"}]
        }

        # Mock the embeddings response
        mock_embedding = np.random.rand(384).astype(np.float32)
        mock_client.embeddings.return_value = {"embedding": mock_embedding.tolist()}

        # First, ensure the track is embedded
        embedder = LocalTrackEmbedder(db_path=self.db_path, conn=self.conn)
        embedder.embed_all_tracks()

        # Now, generate a playlist - create a new embedder for the playlist generator
        embedder_for_playlist = LocalTrackEmbedder(db_path=self.db_path, conn=self.conn)
        playlist_generator = LocalPlaylistGenerator(db_path=self.db_path)
        playlist_generator.track_embedder = embedder_for_playlist
        playlist = playlist_generator.generate_playlist("a test song")

        self.assertIsNotNone(playlist)
        self.assertEqual(len(playlist), 1)
        self.assertEqual(playlist[0]["name"], "Test Song")


if __name__ == "__main__":
    unittest.main()
