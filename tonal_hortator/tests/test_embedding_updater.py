import sqlite3
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from tonal_hortator.utils.embedding_updater import EmbeddingUpdater


def create_test_db(db_path: str) -> None:
    """Create a test database with sample data."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create tracks table
    cur.execute(
        """CREATE TABLE IF NOT EXISTS tracks (
        id INTEGER PRIMARY KEY,
        name TEXT,
        artist TEXT,
        album TEXT,
        genre TEXT,
        year INTEGER,
        play_count INTEGER,
        location TEXT
    )"""
    )

    # Create track_embeddings table
    cur.execute(
        """CREATE TABLE IF NOT EXISTS track_embeddings (
        track_id INTEGER PRIMARY KEY,
        embedding BLOB,
        embedding_text TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )"""
    )

    # Insert sample tracks
    cur.execute(
        """INSERT INTO tracks (id, name, artist, album, genre, year, play_count, location)
                   VALUES (1, 'Test Song 1', 'Test Artist 1', 'Test Album 1', 'Rock', 2020, 5, '/path/to/song1.mp3')"""
    )
    cur.execute(
        """INSERT INTO tracks (id, name, artist, album, genre, year, play_count, location)
                   VALUES (2, 'Test Song 2', 'Test Artist 2', 'Test Album 2', 'Jazz', 2021, 10, '/path/to/song2.mp3')"""
    )
    cur.execute(
        """INSERT INTO tracks (id, name, artist, album, genre, year, play_count, location)
                   VALUES (3, 'Test Song 3', 'Test Artist 3', 'Test Album 3', 'Pop', 2022, 0, '/path/to/song3.mp3')"""
    )

    # Insert sample embeddings
    import numpy as np

    embedding1 = np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
    embedding2 = np.array([0.4, 0.5, 0.6], dtype=np.float32).tobytes()

    cur.execute(
        """INSERT INTO track_embeddings (track_id, embedding, embedding_text)
                   VALUES (1, ?, 'Test embedding text 1')""",
        (embedding1,),
    )
    cur.execute(
        """INSERT INTO track_embeddings (track_id, embedding, embedding_text)
                   VALUES (2, ?, 'Test embedding text 2')""",
        (embedding2,),
    )

    conn.commit()
    conn.close()


def test_update_embeddings_for_valid_ids() -> None:
    """Test updating embeddings for valid track IDs."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_file:
        db_path = db_file.name

    try:
        create_test_db(db_path)

        with patch(
            "tonal_hortator.utils.embedding_updater.LocalTrackEmbedder"
        ) as mock_embedder_class:
            mock_embedder = MagicMock()
            mock_embedder_class.return_value = mock_embedder
            mock_embedder.conn = sqlite3.connect(db_path)
            mock_embedder.conn.row_factory = sqlite3.Row
            mock_embedder.embed_tracks_batch.return_value = 2

            updater = EmbeddingUpdater(db_path)
            stats = updater.update_embeddings_for_tracks([1, 2])

            assert stats["total_tracks"] == 2
            assert stats["updated"] == 2
            assert stats["errors"] == 0
            assert stats["skipped"] == 0

            # Verify embed_tracks_batch was called
            mock_embedder.embed_tracks_batch.assert_called_once()

    finally:
        import os

        if os.path.exists(db_path):
            os.unlink(db_path)


def test_update_embeddings_with_missing_ids() -> None:
    """Test updating embeddings for non-existent track IDs."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_file:
        db_path = db_file.name

    try:
        create_test_db(db_path)

        with patch(
            "tonal_hortator.utils.embedding_updater.LocalTrackEmbedder"
        ) as mock_embedder_class:
            mock_embedder = MagicMock()
            mock_embedder_class.return_value = mock_embedder
            mock_embedder.conn = sqlite3.connect(db_path)
            mock_embedder.conn.row_factory = sqlite3.Row
            mock_embedder.embed_tracks_batch.return_value = 0

            updater = EmbeddingUpdater(db_path)
            stats = updater.update_embeddings_for_tracks([999, 1000])

            assert stats["total_tracks"] == 2
            assert stats["updated"] == 0
            assert stats["errors"] == 0
            assert stats["skipped"] == 0

    finally:
        import os

        if os.path.exists(db_path):
            os.unlink(db_path)


def test_update_embeddings_no_ids() -> None:
    """Test updating embeddings with no track IDs."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_file:
        db_path = db_file.name

    try:
        create_test_db(db_path)

        with patch(
            "tonal_hortator.utils.embedding_updater.LocalTrackEmbedder"
        ) as mock_embedder_class:
            mock_embedder = MagicMock()
            mock_embedder_class.return_value = mock_embedder
            mock_embedder.conn = sqlite3.connect(db_path)
            mock_embedder.conn.row_factory = sqlite3.Row

            updater = EmbeddingUpdater(db_path)
            stats = updater.update_embeddings_for_tracks([])

            assert stats["total_tracks"] == 0
            assert stats["updated"] == 0
            assert stats["errors"] == 0
            assert stats["skipped"] == 0

            # Verify embed_tracks_batch was not called
            mock_embedder.embed_tracks_batch.assert_not_called()

    finally:
        import os

        if os.path.exists(db_path):
            os.unlink(db_path)


def test_update_embeddings_file_input() -> None:
    """Test updating embeddings from file input."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_file:
        db_path = db_file.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as ids_file:
        ids_file.write("1\n2\n3\n")
        ids_file_path = ids_file.name

    try:
        create_test_db(db_path)

        with patch(
            "tonal_hortator.utils.embedding_updater.LocalTrackEmbedder"
        ) as mock_embedder_class:
            mock_embedder = MagicMock()
            mock_embedder_class.return_value = mock_embedder
            mock_embedder.conn = sqlite3.connect(db_path)
            mock_embedder.conn.row_factory = sqlite3.Row
            mock_embedder.embed_tracks_batch.return_value = 3

            # Test the parse_ids_from_file function directly
            from tonal_hortator.utils.embedding_updater import parse_ids_from_file

            track_ids = parse_ids_from_file(ids_file_path)
            assert track_ids == [1, 2, 3]

            updater = EmbeddingUpdater(db_path)
            stats = updater.update_embeddings_for_tracks(track_ids)

            assert stats["total_tracks"] == 3
            assert stats["updated"] == 3
            assert stats["errors"] == 0
            assert stats["skipped"] == 0

    finally:
        import os

        if os.path.exists(db_path):
            os.unlink(db_path)
        if os.path.exists(ids_file_path):
            os.unlink(ids_file_path)


def test_update_embeddings_handles_exception() -> None:
    """Test that exceptions are handled gracefully."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_file:
        db_path = db_file.name

    try:
        create_test_db(db_path)

        with patch(
            "tonal_hortator.utils.embedding_updater.LocalTrackEmbedder"
        ) as mock_embedder_class:
            mock_embedder = MagicMock()
            mock_embedder_class.return_value = mock_embedder
            mock_embedder.conn = sqlite3.connect(db_path)
            mock_embedder.conn.row_factory = sqlite3.Row
            mock_embedder.embed_tracks_batch.side_effect = Exception("Test error")

            updater = EmbeddingUpdater(db_path)
            stats = updater.update_embeddings_for_tracks([1, 2])

            assert stats["total_tracks"] == 2
            assert stats["updated"] == 0
            assert stats["errors"] == 2
            assert stats["skipped"] == 0

    finally:
        import os

        if os.path.exists(db_path):
            os.unlink(db_path)
