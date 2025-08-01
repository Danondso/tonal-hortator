import os
import sqlite3
import tempfile
import unittest
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import numpy as np

from tonal_hortator.core.embeddings.track_embedder import LocalTrackEmbedder
from tonal_hortator.core.playlist.playlist_generator import LocalPlaylistGenerator


class TestCoreFunctionality(unittest.TestCase):

    def setUp(self) -> None:
        """Set up a temp file-based database with test tracks for parallel embedding."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.db_path = self.temp_db.name
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Create tracks table with correct schema
        self.cursor.execute(
            """
            CREATE TABLE tracks (
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
                date_added TEXT,
                location TEXT UNIQUE,
                bpm REAL,
                musical_key TEXT,
                key_scale TEXT,
                mood TEXT,
                release_country TEXT,
                label TEXT,
                arranger TEXT,
                lyricist TEXT,
                producer TEXT,
                original_year INTEGER,
                original_date TEXT,
                chord_changes_rate REAL,
                script TEXT,
                replay_gain TEXT
            )
        """
        )

        # Insert test tracks
        self.cursor.execute(
            """
            INSERT INTO tracks (id, name, artist, album_artist, composer, album, genre, year, total_time,
                               track_number, disc_number, play_count, date_added, location)
            VALUES
                (1, 'Test Song 1', 'Test Artist 1', 'Test Album Artist 1', 'Test Composer 1', 'Test Album 1', 'Rock', 2020,
                    180000, 1, 1, 5, '2024-01-01', '/path/to/test/song1.mp3'),
                (2, 'Test Song 2', 'Test Artist 2', 'Test Album Artist 2', 'Test Composer 2', 'Test Album 2', 'Jazz', 2021,
                    200000, 1, 1, 3, '2024-01-02', '/path/to/test/song2.mp3'),
                (3, 'Test Song 3', 'Test Artist 3', 'Test Album Artist 3', 'Test Composer 3', 'Test Album 3', 'Pop', 2022,
                    160000, 1, 1, 7, '2024-01-03', '/path/to/test/song3.mp3')
            """
        )
        self.conn.commit()

    def tearDown(self) -> None:
        """Close the database connection."""
        self.conn.close()
        self.temp_db.close()
        os.unlink(self.temp_db.name)

    @patch("tonal_hortator.core.embeddings.embeddings.ollama")
    def test_track_embedding(self, mock_ollama: Mock) -> None:
        """Test that tracks can be successfully embedded."""
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
        self.assertEqual(embedded_count, 3)  # All 3 tracks should be embedded

        # Verify the embeddings were stored
        self.cursor.execute("SELECT COUNT(*) FROM track_embeddings")
        count = self.cursor.fetchone()[0]
        self.assertEqual(count, 3)

    @unittest.skipIf(
        os.environ.get("CI") == "true",
        "Skipping playlist generation test in CI (Ollama not running)",
    )
    @patch("tonal_hortator.core.embeddings.embeddings.ollama")
    @patch("tonal_hortator.core.llm.llm_client.ollama")
    def test_playlist_generation(
        self, mock_llm_ollama: Mock, mock_ollama: Mock
    ) -> None:
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

        # Mock the LLM response for query parsing
        mock_llm_ollama.chat.return_value = {
            "message": {
                "content": """{
  "query_type": "general",
  "artist": null,
  "reference_artist": null,
  "genres": ["rock"],
  "mood": null,
  "count": null,
  "unplayed": false,
  "vague": true
}"""
            }
        }

        # First, ensure the tracks are embedded
        embedder = LocalTrackEmbedder(db_path=self.db_path, conn=self.conn)
        embedder.embed_all_tracks()

        # Now, generate a playlist - create a new embedder for the playlist generator
        embedder_for_playlist = LocalTrackEmbedder(db_path=self.db_path, conn=self.conn)
        playlist_generator = LocalPlaylistGenerator(db_path=self.db_path)
        playlist_generator.track_embedder = embedder_for_playlist
        playlist = playlist_generator.generate_playlist("a test song", max_tracks=3)

        self.assertIsNotNone(playlist)
        self.assertGreater(len(playlist), 0)  # Should have at least one track
        # Check that the playlist contains tracks with the expected names
        track_names = [track.name or "" for track in playlist]
        self.assertTrue(any("Test Song" in name for name in track_names))

    @unittest.skipIf(
        os.environ.get("CI") == "true",
        "Skipping parallel embedding test in CI (Ollama not running)",
    )
    def test_parallel_embedding(self) -> None:
        """Test that parallel embedding works correctly with multiple workers."""
        # Test with 2 workers and small batch size to ensure parallel processing
        from tonal_hortator.core.models import Track

        embedder = LocalTrackEmbedder(db_path=self.db_path, conn=self.conn)
        test_tracks = [Track.from_dict(track) for track in self.get_test_tracks()]
        embedded_count = embedder.embed_tracks_batch(
            test_tracks, batch_size=1, max_workers=2
        )

        # Should embed all 3 tracks
        self.assertEqual(embedded_count, 3)

        # Verify embeddings were stored
        self.cursor.execute("SELECT COUNT(*) FROM track_embeddings")
        count = self.cursor.fetchone()[0]
        self.assertEqual(count, 3)

    def get_test_tracks(self) -> List[Dict[str, Any]]:
        """Get test tracks from the database for embedding tests."""
        self.cursor.execute("SELECT * FROM tracks")
        columns = [description[0] for description in self.cursor.description]
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]


if __name__ == "__main__":
    unittest.main()
