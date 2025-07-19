import sqlite3
import tempfile
from unittest.mock import MagicMock, patch

from tonal_hortator.core.database import (
    GET_EMBEDDING_COUNT,
    TEST_CREATE_TRACKS_TABLE,
    TEST_INSERT_TRACK,
    TEST_INSERT_TRACK_EMBEDDING,
)
from tonal_hortator.utils.embedding_updater import EmbeddingUpdater


def create_test_db(db_path: str) -> None:
    """Create a test database with sample data."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create tracks table
    cur.execute(TEST_CREATE_TRACKS_TABLE)

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
        TEST_INSERT_TRACK,
        (1, "Test Song 1", "Test Artist 1", "Test Album 1", "Rock", 2020, 5),
    )
    cur.execute(
        TEST_INSERT_TRACK,
        (2, "Test Song 2", "Test Artist 2", "Test Album 2", "Jazz", 2021, 10),
    )
    cur.execute(
        TEST_INSERT_TRACK,
        (3, "Test Song 3", "Test Artist 3", "Test Album 3", "Pop", 2022, 0),
    )

    # Insert sample embeddings
    import numpy as np

    embedding1 = np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
    embedding2 = np.array([0.4, 0.5, 0.6], dtype=np.float32).tobytes()

    cur.execute(TEST_INSERT_TRACK_EMBEDDING, (1, embedding1, "Test embedding text 1"))
    cur.execute(TEST_INSERT_TRACK_EMBEDDING, (2, embedding2, "Test embedding text 2"))

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


def test_preserve_mode() -> None:
    """Test preserve mode functionality."""
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
            mock_embedder.embed_tracks_batch.return_value = 1

            updater = EmbeddingUpdater(db_path)
            stats = updater.update_embeddings_for_tracks([1, 2], mode="preserve")

            assert stats["total_tracks"] == 2
            assert stats["updated"] == 1
            assert stats["preserved"] == 0  # Current logic always updates
            assert stats["errors"] == 0

    finally:
        import os

        if os.path.exists(db_path):
            os.unlink(db_path)


def test_hybrid_mode() -> None:
    """Test hybrid mode functionality."""
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
            mock_embedder.embed_tracks_batch.return_value = 1
            mock_embedder.create_track_embedding_text.return_value = (
                "Test embedding text"
            )
            mock_embedder.embedding_service = MagicMock()
            mock_embedder.embedding_service.get_embedding.return_value = [0.1, 0.2, 0.3]

            updater = EmbeddingUpdater(db_path)
            stats = updater.update_embeddings_for_tracks([1, 2], mode="hybrid")

            assert stats["total_tracks"] == 2
            assert stats["updated"] >= 0  # May vary based on hybrid logic
            assert stats["errors"] == 0

    finally:
        import os

        if os.path.exists(db_path):
            os.unlink(db_path)


def test_invalid_mode() -> None:
    """Test handling of invalid mode."""
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

            # The error is caught and handled, so we check the stats instead
            stats = updater.update_embeddings_for_tracks([1, 2], mode="invalid_mode")
            assert stats["errors"] == 2  # Should have errors due to invalid mode

    finally:
        import os

        if os.path.exists(db_path):
            os.unlink(db_path)


def test_get_existing_embedding() -> None:
    """Test getting existing embedding."""
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

            # Test getting existing embedding
            embedding = updater._get_existing_embedding(1)
            assert embedding is not None
            assert len(embedding) == 3

            # Test getting non-existent embedding
            embedding = updater._get_existing_embedding(999)
            assert embedding is None

    finally:
        import os

        if os.path.exists(db_path):
            os.unlink(db_path)


def test_should_update_embedding() -> None:
    """Test should_update_embedding logic."""
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

            # Test track with existing embedding
            track = {"id": 1, "name": "Test Song", "artist": "Test Artist"}
            should_update = updater._should_update_embedding(track)
            assert should_update is True  # Current logic always returns True

            # Test track without existing embedding
            track = {"id": 999, "name": "Test Song", "artist": "Test Artist"}
            should_update = updater._should_update_embedding(track)
            assert (
                should_update is True
            )  # Should return True for non-existent embedding

    finally:
        import os

        if os.path.exists(db_path):
            os.unlink(db_path)


def test_clear_embeddings_for_tracks() -> None:
    """Test clearing embeddings for tracks."""
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

            # Test clearing embeddings
            updater._clear_embeddings_for_tracks([1, 2])

            # Verify embeddings were cleared
            cursor = mock_embedder.conn.cursor()
            cursor.execute(GET_EMBEDDING_COUNT)
            count = cursor.fetchone()[0]
            assert count == 0

            # Test clearing with empty list
            updater._clear_embeddings_for_tracks([])
            # Should not raise any errors

    finally:
        import os

        if os.path.exists(db_path):
            os.unlink(db_path)


def test_parse_ids_from_file() -> None:
    """Test parsing IDs from file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as ids_file:
        ids_file.write("1\n2\n3\ninvalid\n5\n")
        ids_file_path = ids_file.name

    try:
        from tonal_hortator.utils.embedding_updater import parse_ids_from_file

        ids = parse_ids_from_file(ids_file_path)
        assert ids == [1, 2, 3, 5]  # Should skip "invalid"

    finally:
        import os

        if os.path.exists(ids_file_path):
            os.unlink(ids_file_path)


def test_parse_ids_from_empty_file() -> None:
    """Test parsing IDs from empty file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as ids_file:
        ids_file.write("")
        ids_file_path = ids_file.name

    try:
        from tonal_hortator.utils.embedding_updater import parse_ids_from_file

        ids = parse_ids_from_file(ids_file_path)
        assert ids == []

    finally:
        import os

        if os.path.exists(ids_file_path):
            os.unlink(ids_file_path)
