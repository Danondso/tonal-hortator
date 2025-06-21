import unittest
import sqlite3
import numpy as np
import os
from tonal_hortator.core.track_embedder import LocalTrackEmbedder
from tonal_hortator.core.playlist_generator import LocalPlaylistGenerator

class TestCoreFunctionality(unittest.TestCase):

    def setUp(self):
        """Set up an in-memory database with a single track for testing."""
        self.db_path = ":memory:"
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Create tracks table
        self.cursor.execute("""
            CREATE TABLE tracks (
                id INTEGER PRIMARY KEY, title TEXT, artist TEXT, album TEXT, genre TEXT,
                year INTEGER, play_count INTEGER, album_artist TEXT, composer TEXT, bpm INTEGER,
                location TEXT
            )
        """)

        # Create track_embeddings table
        self.cursor.execute("""
            CREATE TABLE track_embeddings (
                track_id INTEGER PRIMARY KEY,
                embedding BLOB,
                embedding_text TEXT
            )
        """)

        # Insert a sample track
        self.cursor.execute("""
            INSERT INTO tracks (id, title, artist, album, genre, year, play_count, bpm, location)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (1, 'Test Song', 'Test Artist', 'Test Album', 'Rock', 2023, 42, 120, '/path/to/test.mp3'))
        self.conn.commit()

    def tearDown(self):
        """Close the database connection."""
        self.conn.close()

    def test_track_embedding(self):
        """Test that a single track can be successfully embedded."""
        embedder = LocalTrackEmbedder(db_path=self.db_path)
        embedded_count = embedder.embed_all_tracks()
        self.assertEqual(embedded_count, 1)

        # Verify the embedding was stored
        self.cursor.execute("SELECT COUNT(*) FROM track_embeddings WHERE track_id = 1")
        count = self.cursor.fetchone()[0]
        self.assertEqual(count, 1)

    def test_playlist_generation(self):
        """Test that a simple playlist can be generated."""
        # First, ensure the track is embedded
        embedder = LocalTrackEmbedder(db_path=self.db_path)
        embedder.embed_all_tracks()

        # Now, generate a playlist
        playlist_generator = LocalPlaylistGenerator(db_path=self.db_path)
        playlist = playlist_generator.generate_playlist_from_query("a test song")
        
        self.assertIsNotNone(playlist)
        self.assertEqual(len(playlist), 1)
        self.assertEqual(playlist[0]['name'], 'Test Song')

if __name__ == '__main__':
    unittest.main() 