#!/usr/bin/env python3
"""
Unit tests for the LibraryParser class.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

from tonal_hortator.utils.library_parser import LibraryParser


class TestLibraryParser(unittest.TestCase):
    """Test cases for the LibraryParser class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create a temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()

        # Create a temporary XML file for testing
        self.temp_xml = tempfile.NamedTemporaryFile(suffix=".xml", delete=False)
        self.xml_path = self.temp_xml.name
        self.temp_xml.close()

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        # Remove temporary files
        Path(self.db_path).unlink(missing_ok=True)
        Path(self.xml_path).unlink(missing_ok=True)

    def test_init_default(self) -> None:
        """Test initialization with default database path."""
        with patch(
            "tonal_hortator.utils.library_parser.sqlite3.connect"
        ) as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor

            with patch(
                "tonal_hortator.utils.library_parser.MetadataReader"
            ) as mock_metadata_reader:
                mock_metadata_reader.return_value = Mock()

                parser = LibraryParser()

                self.assertEqual(parser.db_path, "music_library.db")
                self.assertIsNotNone(parser.metadata_reader)
                self.assertIsNotNone(parser._field_processors)
                self.assertIsNotNone(parser._field_mapping)

    def test_init_custom_db_path(self) -> None:
        """Test initialization with custom database path."""
        with patch(
            "tonal_hortator.utils.library_parser.sqlite3.connect"
        ) as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_connect.return_value.__enter__.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor

            with patch(
                "tonal_hortator.utils.library_parser.MetadataReader"
            ) as mock_metadata_reader:
                mock_metadata_reader.return_value = Mock()

                parser = LibraryParser(db_path=self.db_path)

                self.assertEqual(parser.db_path, self.db_path)
                self.assertIsNotNone(parser.metadata_reader)
                mock_metadata_reader.assert_called_once_with(self.db_path)

    def test_create_table_success(self) -> None:
        """Test successful table creation."""
        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()

            LibraryParser(db_path=self.db_path)

            # Verify table was created
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='tracks'"
                )
                result = cursor.fetchone()
                self.assertIsNotNone(result)
                self.assertEqual(result[0], "tracks")

                # Check table schema
                cursor.execute("PRAGMA table_info(tracks)")
                columns = cursor.fetchall()
                column_names = [col[1] for col in columns]
                expected_columns = [
                    "id",
                    "name",
                    "artist",
                    "album_artist",
                    "composer",
                    "album",
                    "genre",
                    "year",
                    "total_time",
                    "track_number",
                    "disc_number",
                    "play_count",
                    "date_added",
                    "location",
                ]
                for col in expected_columns:
                    self.assertIn(col, column_names)

    def test_create_table_database_error(self) -> None:
        """Test table creation with database error."""
        # Use an invalid path to cause database error
        invalid_path = "/invalid/path/database.db"

        with self.assertRaises(sqlite3.OperationalError):
            with patch(
                "tonal_hortator.utils.library_parser.MetadataReader"
            ) as mock_metadata_reader:
                mock_metadata_reader.return_value = Mock()
                LibraryParser(db_path=invalid_path)

    def test_parse_library_file_not_found(self) -> None:
        """Test parsing library with non-existent file."""
        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()
            parser = LibraryParser(db_path=self.db_path)

        result = parser.parse_library("/nonexistent/file.xml")

        self.assertEqual(result, 0)

    def test_parse_library_success(self) -> None:
        """Test successful library parsing."""
        # Create a simple XML file
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Major Version</key>
    <integer>1</integer>
    <key>Minor Version</key>
    <integer>1</integer>
    <key>Date</key>
    <date>2023-01-01T00:00:00Z</date>
    <key>Application Version</key>
    <string>12.0</string>
    <key>Features</key>
    <integer>5</integer>
    <key>Show Content Ratings</key>
    <true/>
    <key>Music Folder</key>
    <string>file:///Users/test/Music/</string>
    <key>Library Persistent ID</key>
    <string>1234567890ABCDEF</string>
    <key>Tracks</key>
    <dict>
        <key>1</key>
        <dict>
            <key>Track ID</key>
            <integer>1</integer>
            <key>Name</key>
            <string>Test Song</string>
            <key>Artist</key>
            <string>Test Artist</string>
            <key>Album</key>
            <string>Test Album</string>
            <key>Genre</key>
            <string>Rock</string>
            <key>Year</key>
            <integer>2020</integer>
            <key>Total Time</key>
            <integer>180000</integer>
            <key>Track Number</key>
            <integer>1</integer>
            <key>Disc Number</key>
            <integer>1</integer>
            <key>Play Count</key>
            <integer>5</integer>
            <key>Date Added</key>
            <date>2023-01-01T00:00:00Z</date>
            <key>Location</key>
            <string>file:///Users/test/Music/test_song.mp3</string>
        </dict>
    </dict>
</dict>
</plist>"""

        with open(self.xml_path, "w") as f:
            f.write(xml_content)

        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()
            parser = LibraryParser(db_path=self.db_path)

        with patch.object(parser.metadata_reader, "update_track_metadata"):
            result = parser.parse_library(self.xml_path)

        self.assertEqual(result, 1)

        # Verify track was inserted
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tracks WHERE name = 'Test Song'")
            track = cursor.fetchone()
            self.assertIsNotNone(track)

    def test_parse_library_xml_parse_error(self) -> None:
        """Test parsing library with XML parse error."""
        # Create invalid XML
        with open(self.xml_path, "w") as f:
            f.write("invalid xml content")

        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()
            parser = LibraryParser(db_path=self.db_path)

        result = parser.parse_library(self.xml_path)

        self.assertEqual(result, 0)

    def test_process_string_field_with_text(self) -> None:
        """Test processing string field with text."""
        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()
            parser = LibraryParser(db_path=self.db_path)

        mock_elem = Mock()
        mock_elem.text = "Test Value"

        result = parser._process_string_field(mock_elem)

        self.assertEqual(result, "Test Value")

    def test_process_string_field_without_text(self) -> None:
        """Test processing string field without text."""
        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()
            parser = LibraryParser(db_path=self.db_path)

        mock_elem = Mock()
        mock_elem.text = None

        result = parser._process_string_field(mock_elem)

        self.assertIsNone(result)

    def test_process_location_field_file_protocol(self) -> None:
        """Test processing location field with file:// protocol."""
        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()
            parser = LibraryParser(db_path=self.db_path)

        mock_elem = Mock()
        mock_elem.text = "file:///Users/test/Music/test%20song.mp3"

        result = parser._process_location_field(mock_elem)

        self.assertEqual(result, "/Users/test/Music/test song.mp3")

    def test_process_location_field_url_encoded(self) -> None:
        """Test processing location field with URL encoding."""
        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()
            parser = LibraryParser(db_path=self.db_path)

        mock_elem = Mock()
        mock_elem.text = "/Users/test/Music/test%20song.mp3"

        result = parser._process_location_field(mock_elem)

        self.assertEqual(result, "/Users/test/Music/test song.mp3")

    def test_process_location_field_normal(self) -> None:
        """Test processing location field without encoding."""
        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()
            parser = LibraryParser(db_path=self.db_path)

        mock_elem = Mock()
        mock_elem.text = "/Users/test/Music/test_song.mp3"

        result = parser._process_location_field(mock_elem)

        self.assertEqual(result, "/Users/test/Music/test_song.mp3")

    def test_process_location_field_empty(self) -> None:
        """Test processing location field with empty text."""
        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()
            parser = LibraryParser(db_path=self.db_path)

        mock_elem = Mock()
        mock_elem.text = ""

        result = parser._process_location_field(mock_elem)

        self.assertIsNone(result)

    def test_process_int_field_with_value(self) -> None:
        """Test processing integer field with value."""
        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()
            parser = LibraryParser(db_path=self.db_path)

        mock_elem = Mock()
        mock_elem.text = "123"

        result = parser._process_int_field(mock_elem)

        self.assertEqual(result, 123)

    def test_process_int_field_empty(self) -> None:
        """Test processing integer field with empty text."""
        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()
            parser = LibraryParser(db_path=self.db_path)

        mock_elem = Mock()
        mock_elem.text = ""

        result = parser._process_int_field(mock_elem)

        self.assertEqual(result, 0)

    def test_process_optional_int_field_with_value(self) -> None:
        """Test processing optional integer field with value."""
        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()
            parser = LibraryParser(db_path=self.db_path)

        mock_elem = Mock()
        mock_elem.text = "2020"

        result = parser._process_optional_int_field(mock_elem)

        self.assertEqual(result, 2020)

    def test_process_optional_int_field_empty(self) -> None:
        """Test processing optional integer field with empty text."""
        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()
            parser = LibraryParser(db_path=self.db_path)

        mock_elem = Mock()
        mock_elem.text = ""

        result = parser._process_optional_int_field(mock_elem)

        self.assertIsNone(result)

    def test_get_field_processors(self) -> None:
        """Test getting field processors."""
        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()
            parser = LibraryParser(db_path=self.db_path)

        processors = parser._get_field_processors()

        expected_fields = [
            "Name",
            "Artist",
            "Album Artist",
            "Composer",
            "Album",
            "Genre",
            "Year",
            "Total Time",
            "Track Number",
            "Disc Number",
            "Play Count",
            "Date Added",
            "Location",
        ]

        for field in expected_fields:
            self.assertIn(field, processors)
            self.assertTrue(callable(processors[field]))

    def test_get_field_mapping(self) -> None:
        """Test getting field mapping."""
        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()
            parser = LibraryParser(db_path=self.db_path)

        mapping = parser._get_field_mapping()

        expected_mapping = {
            "Name": "name",
            "Artist": "artist",
            "Album Artist": "album_artist",
            "Composer": "composer",
            "Album": "album",
            "Genre": "genre",
            "Year": "year",
            "Total Time": "total_time",
            "Track Number": "track_number",
            "Disc Number": "disc_number",
            "Play Count": "play_count",
            "Date Added": "date_added",
            "Location": "location",
        }

        self.assertEqual(mapping, expected_mapping)

    def test_process_track_field_valid(self) -> None:
        """Test processing valid track field."""
        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()
            parser = LibraryParser(db_path=self.db_path)

        data: Dict[str, Any] = {}
        mock_elem = Mock()
        mock_elem.text = "Test Value"

        parser._process_track_field("Name", mock_elem, data)

        self.assertEqual(data["name"], "Test Value")

    def test_process_track_field_invalid(self) -> None:
        """Test processing invalid track field."""
        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()
            parser = LibraryParser(db_path=self.db_path)

        data: Dict[str, Any] = {}
        mock_elem = Mock()
        mock_elem.text = "Test Value"

        parser._process_track_field("InvalidField", mock_elem, data)

        self.assertEqual(data, {})

    def test_extract_track_data_success(self) -> None:
        """Test successful track data extraction."""
        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()
            parser = LibraryParser(db_path=self.db_path)

        # Create mock track element
        mock_track = Mock()
        mock_track.__iter__ = Mock(
            return_value=iter(
                [
                    Mock(text="Name"),
                    Mock(text="Test Song"),
                    Mock(text="Artist"),
                    Mock(text="Test Artist"),
                    Mock(text="Album"),
                    Mock(text="Test Album"),
                ]
            )
        )

        result = parser._extract_track_data(mock_track)
        self.assertIsNotNone(result)
        from typing import Any, Dict, cast

        result = cast(Dict[str, Any], result)
        self.assertEqual(result["name"], "Test Song")
        self.assertEqual(result["artist"], "Test Artist")
        self.assertEqual(result["album"], "Test Album")

    def test_extract_track_data_no_name(self) -> None:
        """Test track data extraction without name."""
        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()
            parser = LibraryParser(db_path=self.db_path)

        # Create mock track element without name
        mock_track = Mock()
        mock_track.__iter__ = Mock(
            return_value=iter(
                [
                    Mock(text="Artist"),
                    Mock(text="Test Artist"),
                    Mock(text="Album"),
                    Mock(text="Test Album"),
                ]
            )
        )

        result = parser._extract_track_data(mock_track)

        self.assertIsNone(result)

    def test_insert_tracks_success(self) -> None:
        """Test successful track insertion."""
        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()
            parser = LibraryParser(db_path=self.db_path)

        tracks = [
            {
                "name": "Test Song 1",
                "artist": "Test Artist 1",
                "album_artist": None,
                "composer": None,
                "album": "Test Album 1",
                "genre": None,
                "year": None,
                "total_time": 0,
                "track_number": 0,
                "disc_number": 0,
                "play_count": 0,
                "date_added": None,
                "location": "/path/to/song1.mp3",
            },
            {
                "name": "Test Song 2",
                "artist": "Test Artist 2",
                "album_artist": None,
                "composer": None,
                "album": "Test Album 2",
                "genre": None,
                "year": None,
                "total_time": 0,
                "track_number": 0,
                "disc_number": 0,
                "play_count": 0,
                "date_added": None,
                "location": "/path/to/song2.mp3",
            },
        ]

        with patch.object(parser.metadata_reader, "update_track_metadata"):
            result = parser._insert_tracks(iter(tracks))

        self.assertEqual(result, 2)

        # Verify tracks were inserted
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM tracks")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 2)

    def test_insert_tracks_duplicate_location(self) -> None:
        """Test track insertion with duplicate location."""
        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()
            parser = LibraryParser(db_path=self.db_path)

        # Insert first track
        tracks1 = [
            {
                "name": "Test Song 1",
                "artist": "Test Artist 1",
                "album_artist": None,
                "composer": None,
                "album": "Test Album 1",
                "genre": None,
                "year": None,
                "total_time": 0,
                "track_number": 0,
                "disc_number": 0,
                "play_count": 0,
                "date_added": None,
                "location": "/path/to/song.mp3",
            }
        ]

        with patch.object(parser.metadata_reader, "update_track_metadata"):
            result1 = parser._insert_tracks(iter(tracks1))

        self.assertEqual(result1, 1)

        # Try to insert track with same location
        tracks2 = [
            {
                "name": "Test Song 2",
                "artist": "Test Artist 2",
                "album_artist": None,
                "composer": None,
                "album": "Test Album 2",
                "genre": None,
                "year": None,
                "total_time": 0,
                "track_number": 0,
                "disc_number": 0,
                "play_count": 0,
                "date_added": None,
                "location": "/path/to/song.mp3",  # Same location
            }
        ]

        with patch.object(parser.metadata_reader, "update_track_metadata"):
            result2 = parser._insert_tracks(iter(tracks2))

        self.assertEqual(result2, 0)

        # Verify only one track exists
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM tracks")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)

    def test_insert_tracks_no_location(self) -> None:
        """Test track insertion without location."""
        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()
            parser = LibraryParser(db_path=self.db_path)

        tracks = [
            {
                "name": "Test Song",
                "artist": "Test Artist",
                "album_artist": None,
                "composer": None,
                "album": "Test Album",
                "genre": None,
                "year": None,
                "total_time": 0,
                "track_number": 0,
                "disc_number": 0,
                "play_count": 0,
                "date_added": None,
                # No location
            }
        ]

        with patch.object(parser.metadata_reader, "update_track_metadata"):
            result = parser._insert_tracks(iter(tracks))

        self.assertEqual(result, 0)

    def test_insert_tracks_database_error(self) -> None:
        """Test track insertion with database error."""
        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()
            parser = LibraryParser(db_path=self.db_path)

        tracks = [
            {
                "name": "Test Song",
                "artist": "Test Artist",
                "album_artist": None,
                "composer": None,
                "album": "Test Album",
                "genre": None,
                "year": None,
                "total_time": 0,
                "track_number": 0,
                "disc_number": 0,
                "play_count": 0,
                "date_added": None,
                "location": "/path/to/song.mp3",
            }
        ]

        with patch(
            "tonal_hortator.utils.library_parser.sqlite3.connect"
        ) as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Database error")

            result = parser._insert_tracks(iter(tracks))

        self.assertEqual(result, 0)

    def test_insert_tracks_metadata_error(self) -> None:
        """Test track insertion with metadata reading error."""
        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()
            parser = LibraryParser(db_path=self.db_path)

        tracks = [
            {
                "name": "Test Song",
                "artist": "Test Artist",
                "album_artist": None,
                "composer": None,
                "album": "Test Album",
                "genre": None,
                "year": None,
                "total_time": 0,
                "track_number": 0,
                "disc_number": 0,
                "play_count": 0,
                "date_added": None,
                "location": "/path/to/song.mp3",
            }
        ]

        with patch.object(
            parser.metadata_reader, "update_track_metadata"
        ) as mock_update:
            mock_update.side_effect = Exception("Metadata error")

            result = parser._insert_tracks(iter(tracks))

        # Should still insert the track even if metadata fails
        self.assertEqual(result, 1)

    def test_parse_tracks_empty_file(self) -> None:
        """Test parsing tracks from empty XML file."""
        # Create empty XML file
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Tracks</key>
    <dict>
    </dict>
</dict>
</plist>"""

        with open(self.xml_path, "w") as f:
            f.write(xml_content)

        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()
            parser = LibraryParser(db_path=self.db_path)

        tracks = list(parser._parse_tracks(self.xml_path))

        self.assertEqual(len(tracks), 0)

    def test_parse_tracks_no_tracks_section(self) -> None:
        """Test parsing tracks from XML without tracks section."""
        # Create XML without tracks section
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Major Version</key>
    <integer>1</integer>
</dict>
</plist>"""

        with open(self.xml_path, "w") as f:
            f.write(xml_content)

        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()
            parser = LibraryParser(db_path=self.db_path)

        tracks = list(parser._parse_tracks(self.xml_path))

        self.assertEqual(len(tracks), 0)

    def test_parse_tracks_malformed_xml(self) -> None:
        """Test parsing tracks from malformed XML."""
        # Create malformed XML
        with open(self.xml_path, "w") as f:
            f.write("<invalid>xml</invalid>")

        with patch(
            "tonal_hortator.utils.library_parser.MetadataReader"
        ) as mock_metadata_reader:
            mock_metadata_reader.return_value = Mock()
            parser = LibraryParser(db_path=self.db_path)

        tracks = list(parser._parse_tracks(self.xml_path))

        self.assertEqual(len(tracks), 0)

    def test_main_function(self) -> None:
        """Test the main function."""
        # Create a simple XML file
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Tracks</key>
    <dict>
        <key>1</key>
        <dict>
            <key>Track ID</key>
            <integer>1</integer>
            <key>Name</key>
            <string>Test Song</string>
            <key>Artist</key>
            <string>Test Artist</string>
            <key>Location</key>
            <string>file:///Users/test/Music/test_song.mp3</string>
        </dict>
    </dict>
</dict>
</plist>"""

        with open(self.xml_path, "w") as f:
            f.write(xml_content)

        with patch(
            "sys.argv", ["library_parser.py", self.xml_path, "--db-path", self.db_path]
        ):
            with patch(
                "tonal_hortator.utils.library_parser.LibraryParser"
            ) as mock_library_parser:
                mock_instance = Mock()
                mock_library_parser.return_value = mock_instance
                mock_instance.parse_library.return_value = 1

                from tonal_hortator.utils.library_parser import main

                main()

                # Accept both positional and keyword argument calls
                called = False
                for call in mock_instance.parse_library.call_args_list:
                    args, kwargs = call
                    if (args and args[0] == self.xml_path) or (
                        kwargs.get("xml_path") == self.xml_path
                    ):
                        called = True
                        break
                self.assertTrue(
                    called,
                    f"parse_library was not called with the expected xml_path: {self.xml_path}",
                )


if __name__ == "__main__":
    unittest.main()
