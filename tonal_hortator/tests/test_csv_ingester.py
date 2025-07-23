import tempfile

import pytest

from tonal_hortator.core.database import GET_TRACK_COUNT, TEST_GET_TRACK
from tonal_hortator.utils.csv_ingester import MusicCSVIngester


def create_test_csv_content() -> str:
    return """Song Title,Artist,Album,Year released,Song duration (seconds),Play count,Genre,Composer,Filename,Added to library on (timestamp)
Test Song 1,Test Artist 1,Test Album 1,2020,180,5,Rock,Test Composer 1,/path/to/song1.mp3,2020-01-01
Test Song 2,Test Artist 2,Test Album 2,2021,200,10,Jazz,Test Composer 2,/path/to/song2.mp3,2021-01-01"""


def create_test_csv_file() -> str:
    content = create_test_csv_content()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(content)
        return f.name


def test_insert_new_tracks() -> None:
    """Test inserting new tracks from CSV."""
    csv_file = create_test_csv_file()
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_file:
            db_path = db_file.name

        ingester = MusicCSVIngester(db_path)
        stats = ingester.ingest_csv(csv_file, dry_run=False)

        assert stats["total_rows"] == 2
        assert stats["inserted"] == 2
        assert stats["updated"] == 0
        assert stats["skipped"] == 0
        assert stats["errors"] == 0
        assert len(stats["tracks_to_update_embeddings"]) == 2

        # Verify tracks were inserted
        cursor = ingester.conn.cursor()
        cursor.execute(GET_TRACK_COUNT)
        count = cursor.fetchone()[0]
        assert count == 2

        cursor.execute(TEST_GET_TRACK, (1,))
        result = cursor.fetchone()
        assert result is not None
        assert result[0] == "Test Song 1"  # name column
        assert result[1] == "Test Artist 1"  # artist column

    finally:
        import os

        if os.path.exists(csv_file):
            os.unlink(csv_file)
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_update_existing_tracks() -> None:
    """Test updating existing tracks from CSV."""
    csv_file = create_test_csv_file()
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_file:
            db_path = db_file.name

        ingester = MusicCSVIngester(db_path)

        # First ingestion
        stats1 = ingester.ingest_csv(csv_file, dry_run=False)
        assert stats1["inserted"] == 2

        # Second ingestion (should update)
        stats2 = ingester.ingest_csv(csv_file, dry_run=False)
        assert stats2["inserted"] == 0
        assert stats2["updated"] == 2
        assert len(stats2["tracks_to_update_embeddings"]) == 2

    finally:
        import os

        if os.path.exists(csv_file):
            os.unlink(csv_file)
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_skip_missing_location() -> None:
    """Test skipping tracks without location."""
    content = """Song Title,Artist,Album,Year released,Song duration (seconds),Play count,Genre,Composer,Filename,Added to library on (timestamp)
Test Song 1,Test Artist 1,Test Album 1,2020,180,5,Rock,Test Composer 1,,2020-01-01"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as csv_file:
        csv_file.write(content)
        csv_file_path = csv_file.name

    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_file:
            db_path = db_file.name

        ingester = MusicCSVIngester(db_path)
        stats = ingester.ingest_csv(csv_file_path, dry_run=False)

        assert stats["total_rows"] == 1
        assert stats["inserted"] == 0
        assert stats["updated"] == 0
        assert stats["skipped"] == 1
        assert stats["errors"] == 0

    finally:
        import os

        if os.path.exists(csv_file_path):
            os.unlink(csv_file_path)
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_dry_run_mode() -> None:
    """Test dry run mode doesn't modify database."""
    csv_file = create_test_csv_file()
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_file:
            db_path = db_file.name

        ingester = MusicCSVIngester(db_path)
        stats = ingester.ingest_csv(csv_file, dry_run=True)

        assert stats["total_rows"] == 2
        assert stats["inserted"] == 2
        assert stats["updated"] == 0
        assert stats["skipped"] == 0
        assert stats["errors"] == 0

        # Verify no tracks were actually inserted
        cursor = ingester.conn.cursor()
        cursor.execute(GET_TRACK_COUNT)
        count = cursor.fetchone()[0]
        assert count == 0

    finally:
        import os

        if os.path.exists(csv_file):
            os.unlink(csv_file)
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_error_handling() -> None:
    """Test error handling for invalid CSV."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as csv_file:
        csv_file.write("Invalid CSV content")
        csv_file_path = csv_file.name

    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_file:
            db_path = db_file.name

        ingester = MusicCSVIngester(db_path)

        with pytest.raises(ValueError):
            ingester.ingest_csv(csv_file_path, dry_run=False)

    finally:
        import os

        if os.path.exists(csv_file_path):
            os.unlink(csv_file_path)
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_empty_csv() -> None:
    """Test handling of empty CSV file."""
    content = """Song Title,Artist,Album,Year released,Song duration (seconds),Play count,Genre,Composer,Filename,Added to library on (timestamp)"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as csv_file:
        csv_file.write(content)
        csv_file_path = csv_file.name

    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_file:
            db_path = db_file.name

        ingester = MusicCSVIngester(db_path)
        stats = ingester.ingest_csv(csv_file_path, dry_run=False)

        assert stats["total_rows"] == 0
        assert stats["inserted"] == 0
        assert stats["updated"] == 0
        assert stats["skipped"] == 0
        assert stats["errors"] == 0
        assert len(stats["tracks_to_update_embeddings"]) == 0

    finally:
        import os

        if os.path.exists(csv_file_path):
            os.unlink(csv_file_path)
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_missing_required_fields() -> None:
    """Test handling of CSV with missing required fields."""
    content = """Song Title,Artist,Album,Year released,Song duration (seconds),Play count,Genre,Composer,Added to library on (timestamp)
Test Song 1,Test Artist 1,Test Album 1,2020,180,5,Rock,Test Composer 1,2020-01-01"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as csv_file:
        csv_file.write(content)
        csv_file_path = csv_file.name

    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_file:
            db_path = db_file.name

        ingester = MusicCSVIngester(db_path)

        with pytest.raises(ValueError, match="Missing required CSV fields"):
            ingester.ingest_csv(csv_file_path, dry_run=False)

    finally:
        import os

        if os.path.exists(csv_file_path):
            os.unlink(csv_file_path)
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_csv_with_no_headers() -> None:
    """Test handling of CSV with no headers."""
    content = """Test Song 1,Test Artist 1,Test Album 1,2020,180,5,Rock,Test Composer 1,/path/to/song1.mp3,2020-01-01"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as csv_file:
        csv_file.write(content)
        csv_file_path = csv_file.name

    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_file:
            db_path = db_file.name

        ingester = MusicCSVIngester(db_path)

        with pytest.raises(ValueError, match="Missing required CSV fields"):
            ingester.ingest_csv(csv_file_path, dry_run=False)

    finally:
        import os

        if os.path.exists(csv_file_path):
            os.unlink(csv_file_path)
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_normalize_track_data() -> None:
    """Test track data normalization."""
    ingester = MusicCSVIngester(":memory:")

    # Test normal data
    csv_row = {
        "Song Title": "Test Song",
        "Artist": "Test Artist",
        "Album": "Test Album",
        "Year released": "2020",
        "Song duration (seconds)": "180",
        "Play count": "5",
        "Genre": "Rock",
        "Composer": "Test Composer",
        "Filename": "/path/to/song.mp3",
        "Added to library on (timestamp)": "2020-01-01",
    }

    normalized = ingester._normalize_track_data(csv_row)

    assert normalized.name == "Test Song"
    assert normalized.artist == "Test Artist"
    assert normalized.album == "Test Album"
    assert normalized.year == 2020
    assert normalized.total_time == 180
    assert normalized.play_count == 5
    assert normalized.genre == "Rock"
    assert normalized.location == "/path/to/song.mp3"


def test_normalize_track_data_with_missing_values() -> None:
    """Test track data normalization with missing values."""
    ingester = MusicCSVIngester(":memory:")

    # Test data with missing values
    csv_row = {
        "Song Title": "",
        "Artist": "",
        "Album": "Test Album",
        "Year released": "",
        "Song duration (seconds)": "",
        "Play count": "",
        "Genre": "Rock",
        "Composer": "Test Composer",
        "Filename": "/path/to/song.mp3",
        "Added to library on (timestamp)": "2020-01-01",
    }

    normalized = ingester._normalize_track_data(csv_row)

    assert normalized.name == "Unknown Track"
    assert normalized.artist == "Unknown Artist"
    assert normalized.album == "Test Album"
    assert normalized.year is None
    assert normalized.total_time is None
    assert normalized.play_count is None
    assert normalized.genre == "Rock"
    assert normalized.location == "/path/to/song.mp3"


def test_file_not_found() -> None:
    """Test handling of non-existent CSV file."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_file:
            db_path = db_file.name

        ingester = MusicCSVIngester(db_path)

        with pytest.raises(FileNotFoundError):
            ingester.ingest_csv("non_existent_file.csv", dry_run=False)

    finally:
        import os

        if os.path.exists(db_path):
            os.unlink(db_path)
