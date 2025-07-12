#!/usr/bin/env python3
"""
Tests for tonal_hortator.utils.metadata_reader
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tonal_hortator.utils.metadata_reader import MetadataReader


class TestMetadataReader(unittest.TestCase):
    """Test MetadataReader"""

    def setUp(self) -> None:
        """Set up test fixtures"""
        # Create a temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()

        # Create test tracks table with metadata columns
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
                    location TEXT,
                    bpm REAL,
                    musical_key TEXT,
                    key_scale TEXT,
                    mood TEXT,
                    label TEXT,
                    producer TEXT,
                    arranger TEXT,
                    lyricist TEXT,
                    original_year INTEGER
                )
            """
            )

            # Create metadata_mappings table
            conn.execute(
                """
                CREATE TABLE metadata_mappings (
                    source_format TEXT,
                    source_tag TEXT,
                    normalized_tag TEXT,
                    data_type TEXT,
                    PRIMARY KEY (source_format, source_tag)
                )
            """
            )

            # Insert some test mappings for all formats
            conn.executemany(
                """
                INSERT INTO metadata_mappings (source_format, source_tag, normalized_tag, data_type)
                VALUES (?, ?, ?, ?)
                """,
                [
                    ("mp3", "title", "name", "string"),
                    ("mp3", "artist", "artist", "string"),
                    ("mp3", "album", "album", "string"),
                    ("flac", "title", "name", "string"),
                    ("flac", "artist", "artist", "string"),
                    ("m4a", "title", "name", "string"),
                    ("m4a", "artist", "artist", "string"),
                    ("ogg", "title", "name", "string"),
                    ("ogg", "artist", "artist", "string"),
                    ("aiff", "title", "name", "string"),
                    ("aiff", "artist", "artist", "string"),
                    ("wav", "title", "name", "string"),
                    ("wav", "artist", "artist", "string"),
                ],
            )

            # Insert test tracks
            conn.execute(
                """
                INSERT INTO tracks (id, name, artist, album, location)
                VALUES 
                    (1, 'Test Song 1', 'Test Artist 1', 'Test Album 1', '/path/to/test1.mp3'),
                    (2, 'Test Song 2', 'Test Artist 2', 'Test Album 2', '/path/to/test2.flac'),
                    (3, 'Test Song 3', 'Test Artist 3', 'Test Album 3', '/path/to/test3.m4a')
            """
            )
            conn.commit()

    def tearDown(self) -> None:
        """Clean up test fixtures"""
        # Remove temporary database
        Path(self.db_path).unlink(missing_ok=True)

    def test_init_default(self) -> None:
        """Test initialization with default parameters"""
        reader = MetadataReader(self.db_path)

        self.assertEqual(reader.db_path, self.db_path)
        self.assertIsNotNone(reader.supported_formats)
        self.assertIsNotNone(reader.metadata_mappings)

    def test_get_supported_extensions(self) -> None:
        """Test getting supported file extensions"""
        reader = MetadataReader(self.db_path)

        extensions = reader.get_supported_extensions()

        expected_extensions = {".mp3", ".flac", ".ogg", ".m4a", ".wav", ".aiff"}
        self.assertEqual(extensions, expected_extensions)

    def test_load_metadata_mappings(self) -> None:
        """Test loading metadata mappings from database"""
        reader = MetadataReader(self.db_path)

        mappings = reader._load_metadata_mappings()

        self.assertIn("mp3", mappings)
        self.assertIn("flac", mappings)
        self.assertEqual(mappings["mp3"]["title"], "name")
        self.assertEqual(mappings["mp3"]["artist"], "artist")

    def test_load_metadata_mappings_no_table(self) -> None:
        """Test loading metadata mappings when table doesn't exist"""
        # Create a new database without the metadata_mappings table
        empty_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        empty_db_path = empty_db.name
        empty_db.close()

        try:
            with sqlite3.connect(empty_db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE tracks (
                        id INTEGER PRIMARY KEY,
                        name TEXT
                    )
                """
                )
                conn.commit()

            reader = MetadataReader(empty_db_path)
            mappings = reader._load_metadata_mappings()

            self.assertEqual(mappings, {})
        finally:
            Path(empty_db_path).unlink(missing_ok=True)

    def test_decode_file_path_normal(self) -> None:
        """Test decoding normal file path"""
        reader = MetadataReader(self.db_path)

        result = reader._decode_file_path("/path/to/song.mp3")

        self.assertEqual(result, "/path/to/song.mp3")

    def test_decode_file_path_url_encoded(self) -> None:
        """Test decoding URL-encoded file path"""
        reader = MetadataReader(self.db_path)

        result = reader._decode_file_path("file:///path/to/song%20with%20spaces.mp3")

        self.assertEqual(result, "/path/to/song with spaces.mp3")

    def test_decode_file_path_file_protocol(self) -> None:
        """Test decoding file:// protocol path"""
        reader = MetadataReader(self.db_path)

        result = reader._decode_file_path("file:///Users/user/Music/song.mp3")

        self.assertEqual(result, "/Users/user/Music/song.mp3")

    @patch("tonal_hortator.utils.metadata_reader.Path")
    def test_read_metadata_file_not_found(self, mock_path: Mock) -> None:
        """Test reading metadata from non-existent file"""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance

        reader = MetadataReader(self.db_path)

        with pytest.raises(FileNotFoundError):
            reader.read_metadata("/path/to/nonexistent.mp3")

    @patch("tonal_hortator.utils.metadata_reader.Path")
    def test_read_metadata_unsupported_format(self, mock_path: Mock) -> None:
        """Test reading metadata from unsupported format"""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.suffix = ".xyz"
        mock_path.return_value = mock_path_instance

        reader = MetadataReader(self.db_path)

        with pytest.raises(ValueError, match="Unsupported file format"):
            reader.read_metadata("/path/to/song.xyz")

    @patch("tonal_hortator.utils.metadata_reader.EasyID3")
    @patch("tonal_hortator.utils.metadata_reader.ID3")
    @patch("tonal_hortator.utils.metadata_reader.MP3")
    @patch("tonal_hortator.utils.metadata_reader.Path")
    def test_read_mp3_metadata_success(
        self, mock_path: Mock, mock_mp3: Mock, mock_id3: Mock, mock_easyid3: Mock
    ) -> None:
        """Test successful MP3 metadata reading"""
        # Mock file path
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.suffix = ".mp3"
        mock_path.return_value = mock_path_instance

        # Mock EasyID3
        mock_easy_audio = Mock()
        mock_easy_audio.items.return_value = [
            ("title", ["Test Song"]),
            ("artist", ["Test Artist"]),
            ("album", ["Test Album"]),
        ]
        mock_easyid3.return_value = mock_easy_audio

        # Mock ID3
        mock_id3_audio = Mock()
        mock_id3_audio.items.return_value = []
        mock_id3.return_value = mock_id3_audio

        # Mock MP3
        mock_mp3_audio = Mock()
        mock_mp3_audio.info.length = 180.0
        mock_mp3_audio.info.bitrate = 320000
        mock_mp3_audio.info.sample_rate = 44100
        mock_mp3.return_value = mock_mp3_audio

        reader = MetadataReader(self.db_path)
        metadata = reader._read_mp3_metadata(mock_path_instance)

        self.assertIn("easyid3_title", metadata)
        self.assertIn("easyid3_artist", metadata)
        self.assertIn("easyid3_album", metadata)
        self.assertIn("length", metadata)
        self.assertEqual(metadata["length"], 180)

    @patch("tonal_hortator.utils.metadata_reader.FLAC")
    @patch("tonal_hortator.utils.metadata_reader.Path")
    def test_read_flac_metadata_success(self, mock_path: Mock, mock_flac: Mock) -> None:
        """Test successful FLAC metadata reading"""
        # Mock file path
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.suffix = ".flac"
        mock_path.return_value = mock_path_instance

        # Mock FLAC
        mock_flac_audio = Mock()
        mock_flac_audio.tags = {
            "title": ["Test Song"],
            "artist": ["Test Artist"],
            "album": ["Test Album"],
        }
        mock_flac_audio.info.length = 240.0
        mock_flac_audio.info.sample_rate = 44100
        mock_flac_audio.info.channels = 2
        mock_flac_audio.info.bits_per_sample = 16
        mock_flac.return_value = mock_flac_audio

        reader = MetadataReader(self.db_path)
        metadata = reader._read_flac_metadata(mock_path_instance)

        self.assertIn("flac_title", metadata)
        self.assertIn("flac_artist", metadata)
        self.assertIn("flac_album", metadata)
        self.assertIn("length", metadata)
        self.assertEqual(metadata["length"], 240)

    @patch("tonal_hortator.utils.metadata_reader.OggVorbis")
    @patch("tonal_hortator.utils.metadata_reader.Path")
    def test_read_ogg_metadata_success(self, mock_path: Mock, mock_ogg: Mock) -> None:
        """Test successful OGG metadata reading"""
        # Mock file path
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.suffix = ".ogg"
        mock_path.return_value = mock_path_instance

        # Mock OGG
        mock_ogg_audio = Mock()
        mock_ogg_audio.tags = {"title": ["Test Song"], "artist": ["Test Artist"]}
        mock_ogg_audio.info.length = 200.0
        mock_ogg_audio.info.bitrate = 192000
        mock_ogg_audio.info.sample_rate = 44100
        mock_ogg.return_value = mock_ogg_audio

        reader = MetadataReader(self.db_path)
        metadata = reader._read_ogg_metadata(mock_path_instance)

        self.assertIn("ogg_title", metadata)
        self.assertIn("ogg_artist", metadata)
        self.assertIn("length", metadata)
        self.assertEqual(metadata["length"], 200)

    @patch("tonal_hortator.utils.metadata_reader.MP4")
    @patch("tonal_hortator.utils.metadata_reader.Path")
    def test_read_m4a_metadata_success(self, mock_path: Mock, mock_mp4: Mock) -> None:
        """Test successful M4A metadata reading"""
        # Mock file path
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.suffix = ".m4a"
        mock_path.return_value = mock_path_instance

        # Mock MP4
        mock_mp4_audio = Mock()
        mock_mp4_audio.tags = {
            "\xa9nam": ["Test Song"],
            "\xa9ART": ["Test Artist"],
            "\xa9alb": ["Test Album"],
        }
        mock_mp4_audio.info.length = 220.0
        mock_mp4_audio.info.sample_rate = 44100
        mock_mp4_audio.info.channels = 2
        mock_mp4.return_value = mock_mp4_audio

        reader = MetadataReader(self.db_path)
        metadata = reader._read_m4a_metadata(mock_path_instance)

        self.assertIn("m4a_\xa9nam", metadata)
        self.assertIn("m4a_\xa9ART", metadata)
        self.assertIn("m4a_\xa9alb", metadata)
        self.assertIn("length", metadata)
        self.assertEqual(metadata["length"], 220)

    @patch("tonal_hortator.utils.metadata_reader.WAVE")
    @patch("tonal_hortator.utils.metadata_reader.Path")
    def test_read_wav_metadata_success(self, mock_path: Mock, mock_wave: Mock) -> None:
        """Test successful WAV metadata reading"""
        # Mock file path
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.suffix = ".wav"
        mock_path.return_value = mock_path_instance

        # Mock WAVE
        mock_wave_audio = Mock()
        mock_wave_audio.info.length = 300.0
        mock_wave_audio.info.sample_rate = 44100
        mock_wave_audio.info.channels = 2
        mock_wave_audio.info.bits_per_sample = 16
        mock_wave.return_value = mock_wave_audio

        reader = MetadataReader(self.db_path)
        metadata = reader._read_wav_metadata(mock_path_instance)

        self.assertIn("length", metadata)
        self.assertIn("sample_rate", metadata)
        self.assertIn("channels", metadata)
        self.assertIn("bits_per_sample", metadata)
        self.assertEqual(metadata["length"], 300)

    @patch("tonal_hortator.utils.metadata_reader.AIFF")
    @patch("tonal_hortator.utils.metadata_reader.Path")
    def test_read_aiff_metadata_success(self, mock_path: Mock, mock_aiff: Mock) -> None:
        """Test successful AIFF metadata reading"""
        # Mock file path
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.suffix = ".aiff"
        mock_path.return_value = mock_path_instance

        # Mock AIFF
        mock_aiff_audio = Mock()
        mock_aiff_audio.info.length = 280.0
        mock_aiff_audio.info.sample_rate = 44100
        mock_aiff_audio.info.channels = 2
        mock_aiff_audio.info.bits_per_sample = 16
        mock_aiff.return_value = mock_aiff_audio

        reader = MetadataReader(self.db_path)
        metadata = reader._read_aiff_metadata(mock_path_instance)

        self.assertIn("length", metadata)
        self.assertIn("sample_rate", metadata)
        self.assertIn("channels", metadata)
        self.assertIn("bits_per_sample", metadata)
        self.assertEqual(metadata["length"], 280)

    def test_normalize_metadata(self) -> None:
        """Test metadata normalization"""
        reader = MetadataReader(self.db_path)

        raw_metadata = {
            "title": "Test Song",
            "artist": "Test Artist",
            "album": "Test Album",
            "unknown_tag": "Unknown Value",
        }

        normalized = reader._normalize_metadata(raw_metadata, ".mp3")

        # Check that mapped fields are normalized
        self.assertIn("name", normalized)  # mapped from title
        self.assertIn("artist", normalized)  # mapped from artist
        self.assertIn("album", normalized)  # mapped from album
        self.assertIn("unknown_tag", normalized)  # kept as original

        self.assertEqual(normalized["name"], "Test Song")
        self.assertEqual(normalized["artist"], "Test Artist")

    def test_normalize_metadata_no_mappings(self) -> None:
        """Test metadata normalization with no mappings"""
        reader = MetadataReader(self.db_path)

        raw_metadata = {"title": "Test Song", "artist": "Test Artist"}

        normalized = reader._normalize_metadata(raw_metadata, ".xyz")

        # Should keep original tags when no mappings exist
        self.assertIn("title", normalized)
        self.assertIn("artist", normalized)
        self.assertEqual(normalized["title"], "Test Song")

    @patch.object(MetadataReader, "read_metadata")
    @patch("os.path.exists")
    def test_update_track_metadata_success(
        self, mock_exists: Mock, mock_read_metadata: Mock
    ) -> None:
        """Test successful track metadata update"""
        mock_exists.return_value = True
        mock_read_metadata.return_value = {
            "name": "Updated Song",
            "artist": "Updated Artist",
            "duration": "180",
            "sample_rate": "44100",
        }

        reader = MetadataReader(self.db_path)

        result = reader.update_track_metadata(1, "/path/to/test.mp3")

        self.assertTrue(result)

        # Verify database was updated
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, artist FROM tracks WHERE id = 1")
            row = cursor.fetchone()

            self.assertEqual(row[0], "Updated Song")
            self.assertEqual(row[1], "Updated Artist")

    @patch.object(MetadataReader, "read_metadata")
    def test_update_track_metadata_with_error(self, mock_read_metadata: Mock) -> None:
        """Test track metadata update with error in metadata"""
        mock_read_metadata.return_value = {"error": "Failed to read metadata"}

        reader = MetadataReader(self.db_path)

        result = reader.update_track_metadata(1, "/path/to/test.mp3")

        self.assertFalse(result)

    @patch.object(MetadataReader, "read_metadata")
    def test_update_track_metadata_exception(self, mock_read_metadata: Mock) -> None:
        """Test track metadata update with exception"""
        mock_read_metadata.side_effect = Exception("File read error")

        reader = MetadataReader(self.db_path)

        result = reader.update_track_metadata(1, "/path/to/test.mp3")

        self.assertFalse(result)

    def test_update_database_metadata_success(self) -> None:
        """Test successful database metadata update"""
        reader = MetadataReader(self.db_path)

        metadata = {
            "name": "Test Song",
            "artist": "Test Artist",
            "duration": "180",
            "sample_rate": "44100",
        }

        result = reader._update_database_metadata(1, metadata, "/path/to/test.mp3")

        self.assertTrue(result)

        # Verify database was updated
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, artist FROM tracks WHERE id = 1")
            row = cursor.fetchone()

            self.assertEqual(row[0], "Test Song")
            self.assertEqual(row[1], "Test Artist")

    def test_update_database_metadata_no_valid_fields(self) -> None:
        """Test database metadata update with no valid fields"""
        reader = MetadataReader(self.db_path)

        metadata = {"invalid_field": "value", "another_invalid": "value"}

        result = reader._update_database_metadata(1, metadata, "/path/to/test.mp3")

        self.assertFalse(result)

    def test_update_database_metadata_invalid_column(self) -> None:
        """Test database metadata update with invalid column"""
        reader = MetadataReader(self.db_path)

        metadata = {
            "name": "Test Song",
            "invalid_column": "value",  # This should be filtered out
        }

        result = reader._update_database_metadata(1, metadata, "/path/to/test.mp3")

        # Should still succeed with valid fields
        self.assertTrue(result)

        # Verify only valid field was updated
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM tracks WHERE id = 1")
            row = cursor.fetchone()
            self.assertEqual(row[0], "Test Song")

    @patch("os.path.exists")
    def test_update_all_tracks_metadata(self, mock_exists: Mock) -> None:
        """Test updating metadata for all tracks"""
        mock_exists.return_value = True
        reader = MetadataReader(self.db_path)

        # Mock the update_track_metadata method
        with patch.object(reader, "update_track_metadata") as mock_update:
            mock_update.return_value = True

            result = reader.update_all_tracks_metadata(batch_size=10)

            # Should have called update for each track
            self.assertEqual(mock_update.call_count, 3)
            self.assertEqual(result, 3)

    @patch("os.path.exists")
    def test_update_all_tracks_metadata_with_missing_files(
        self, mock_exists: Mock
    ) -> None:
        """Test updating metadata for all tracks with missing files"""
        mock_exists.return_value = True
        reader = MetadataReader(self.db_path)

        # Mock the update_track_metadata method to return False for some tracks
        with patch.object(reader, "update_track_metadata") as mock_update:
            mock_update.side_effect = [True, False, True]  # Second track fails

            result = reader.update_all_tracks_metadata(batch_size=10)

            self.assertEqual(result, 2)  # Only 2 successful updates

    def test_get_metadata_stats(self) -> None:
        """Test getting metadata statistics"""
        # Add some metadata to test tracks
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE tracks 
                SET bpm = 120.5, musical_key = 'C', key_scale = 'major', mood = 'happy'
                WHERE id = 1
            """
            )
            conn.execute(
                """
                UPDATE tracks 
                SET bpm = 140.0, musical_key = 'G', key_scale = 'minor'
                WHERE id = 2
            """
            )
            conn.commit()

        reader = MetadataReader(self.db_path)
        stats = reader.get_metadata_stats()

        self.assertEqual(stats["total_tracks"], 3)
        self.assertEqual(stats["bpm_coverage"], 2)
        self.assertEqual(stats["musical_key_coverage"], 2)
        self.assertEqual(stats["key_scale_coverage"], 2)
        self.assertEqual(stats["mood_coverage"], 1)
        self.assertEqual(stats["bpm_percentage"], 66.67)
        self.assertEqual(stats["musical_key_percentage"], 66.67)

    def test_get_metadata_stats_empty_database(self) -> None:
        """Test getting metadata statistics from empty database"""
        # Create a new empty database
        empty_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        empty_db_path = empty_db.name
        empty_db.close()

        try:
            with sqlite3.connect(empty_db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE tracks (
                        id INTEGER PRIMARY KEY,
                        name TEXT
                    )
                """
                )
                conn.commit()

            reader = MetadataReader(empty_db_path)
            stats = reader.get_metadata_stats()

            self.assertEqual(stats["total_tracks"], 0)
        finally:
            Path(empty_db_path).unlink(missing_ok=True)

    @patch.object(MetadataReader, "read_metadata")
    def test_debug_metadata_for_file(self, mock_read_metadata: Mock) -> None:
        """Test debugging metadata for a file"""
        mock_read_metadata.return_value = {
            "title": "Test Song",
            "artist": "Test Artist",
            "length": 180,
            "bitrate": 320000,
        }

        reader = MetadataReader(self.db_path)

        # Should not raise any exceptions
        reader.debug_metadata_for_file("/path/to/test.mp3")

        mock_read_metadata.assert_called_once_with("/path/to/test.mp3")

    def test_read_metadata_mp3_format(self) -> None:
        """Test reading metadata for MP3 format"""
        reader = MetadataReader(self.db_path)

        with patch("tonal_hortator.utils.metadata_reader.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.suffix = ".mp3"
            mock_path.return_value = mock_path_instance

            # Mock the read_metadata method directly
            with patch.object(reader, "read_metadata") as mock_read_metadata:
                mock_read_metadata.return_value = {
                    "easyid3_title": "Test Song",
                    "easyid3_artist": "Test Artist",
                    "length": 180,
                }
                metadata = reader.read_metadata("/path/to/test.mp3")
                self.assertEqual(metadata["easyid3_title"], "Test Song")
                self.assertEqual(metadata["easyid3_artist"], "Test Artist")
                self.assertEqual(metadata["length"], 180)

    def test_read_metadata_flac_format(self) -> None:
        """Test reading metadata for FLAC format"""
        reader = MetadataReader(self.db_path)

        with patch("tonal_hortator.utils.metadata_reader.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.suffix = ".flac"
            mock_path.return_value = mock_path_instance

            # Mock the read_metadata method directly
            with patch.object(reader, "read_metadata") as mock_read_metadata:
                mock_read_metadata.return_value = {
                    "flac_title": "Test Song",
                    "flac_artist": "Test Artist",
                    "length": 240,
                }
                metadata = reader.read_metadata("/path/to/test.flac")
                self.assertEqual(metadata["flac_title"], "Test Song")
                self.assertEqual(metadata["flac_artist"], "Test Artist")
                self.assertEqual(metadata["length"], 240)

    def test_read_metadata_m4a_format(self) -> None:
        """Test reading metadata for M4A format"""
        reader = MetadataReader(self.db_path)

        with patch("tonal_hortator.utils.metadata_reader.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.suffix = ".m4a"
            mock_path.return_value = mock_path_instance

            # Mock the read_metadata method directly
            with patch.object(reader, "read_metadata") as mock_read_metadata:
                mock_read_metadata.return_value = {
                    "m4a_\xa9nam": "Test Song",
                    "m4a_\xa9ART": "Test Artist",
                    "length": 220,
                }
                metadata = reader.read_metadata("/path/to/test.m4a")
                self.assertEqual(metadata["m4a_\xa9nam"], "Test Song")
                self.assertEqual(metadata["m4a_\xa9ART"], "Test Artist")
                self.assertEqual(metadata["length"], 220)

    def test_read_metadata_wav_format(self) -> None:
        """Test reading metadata for WAV format"""
        reader = MetadataReader(self.db_path)

        with patch("tonal_hortator.utils.metadata_reader.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.suffix = ".wav"
            mock_path.return_value = mock_path_instance

            # Mock the read_metadata method directly
            with patch.object(reader, "read_metadata") as mock_read_metadata:
                mock_read_metadata.return_value = {"length": 180, "sample_rate": 44100}
                metadata = reader.read_metadata("/path/to/test.wav")
                self.assertEqual(metadata["length"], 180)
                self.assertEqual(metadata["sample_rate"], 44100)

    def test_read_metadata_aiff_format(self) -> None:
        """Test reading metadata for AIFF format"""
        reader = MetadataReader(self.db_path)

        with patch("tonal_hortator.utils.metadata_reader.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.suffix = ".aiff"
            mock_path.return_value = mock_path_instance

            # Mock the read_metadata method directly
            with patch.object(reader, "read_metadata") as mock_read_metadata:
                mock_read_metadata.return_value = {"length": 240, "sample_rate": 44100}
                metadata = reader.read_metadata("/path/to/test.aiff")
                self.assertEqual(metadata["length"], 240)
                self.assertEqual(metadata["sample_rate"], 44100)

    def test_read_metadata_ogg_format(self) -> None:
        """Test reading metadata for OGG format"""
        reader = MetadataReader(self.db_path)

        with patch("tonal_hortator.utils.metadata_reader.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.suffix = ".ogg"
            mock_path.return_value = mock_path_instance

            # Mock the read_metadata method directly
            with patch.object(reader, "read_metadata") as mock_read_metadata:
                mock_read_metadata.return_value = {
                    "ogg_title": "Test Song",
                    "ogg_artist": "Test Artist",
                    "length": 200,
                }
                metadata = reader.read_metadata("/path/to/test.ogg")
                self.assertEqual(metadata["ogg_title"], "Test Song")
                self.assertEqual(metadata["ogg_artist"], "Test Artist")
                self.assertEqual(metadata["length"], 200)


if __name__ == "__main__":
    unittest.main()
