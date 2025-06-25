#!/usr/bin/env python3
"""
Tests for utility modules
"""

import os
import sqlite3
import tempfile
import unittest
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

from tonal_hortator.utils.apple_music import (
    find_latest_playlist,
    list_available_playlists,
    open_in_apple_music,
)
from tonal_hortator.utils.library_parser import LibraryParser


class TestAppleMusicUtils(unittest.TestCase):
    """Test Apple Music utility functions"""

    def setUp(self) -> None:
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.playlist_dir = os.path.join(self.temp_dir, "playlists")
        os.makedirs(self.playlist_dir, exist_ok=True)

    def tearDown(self) -> None:
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_open_in_apple_music_success(self) -> None:
        """Test successful opening of playlist in Apple Music"""
        # Create a temporary playlist file
        playlist_path = os.path.join(self.temp_dir, "test.m3u")
        with open(playlist_path, "w") as f:
            f.write("#EXTM3U\n#EXTINF:180,Test Song\n/path/to/song.mp3\n")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock()
            result = open_in_apple_music(playlist_path)

        self.assertTrue(result)
        mock_run.assert_called_once()

    def test_open_in_apple_music_file_not_found(self) -> None:
        """Test opening non-existent playlist file"""
        result = open_in_apple_music("/nonexistent/playlist.m3u")
        self.assertFalse(result)

    def test_open_in_apple_music_subprocess_error(self) -> None:
        """Test opening playlist when subprocess fails"""
        # Create a temporary playlist file
        playlist_path = os.path.join(self.temp_dir, "test.m3u")
        with open(playlist_path, "w") as f:
            f.write("#EXTM3U\n")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("Music app not found")
            result = open_in_apple_music(playlist_path)

        self.assertFalse(result)

    def test_find_latest_playlist_success(self) -> None:
        """Test finding the latest playlist"""
        # Create multiple playlist files with different timestamps
        playlist1 = os.path.join(self.playlist_dir, "old.m3u")
        playlist2 = os.path.join(self.playlist_dir, "new.m3u")

        with open(playlist1, "w") as f:
            f.write("#EXTM3U\n")

        # Wait a bit to ensure different timestamps
        import time

        time.sleep(0.1)

        with open(playlist2, "w") as f:
            f.write("#EXTM3U\n")

        result = find_latest_playlist(self.playlist_dir)
        self.assertEqual(result, playlist2)

    def test_find_latest_playlist_no_directory(self) -> None:
        """Test finding latest playlist when directory doesn't exist"""
        result = find_latest_playlist("/nonexistent/directory")
        self.assertIsNone(result)

    def test_find_latest_playlist_no_files(self) -> None:
        """Test finding latest playlist when no M3U files exist"""
        result = find_latest_playlist(self.temp_dir)
        self.assertIsNone(result)

    def test_list_available_playlists_success(self) -> None:
        """Test listing available playlists"""
        # Create multiple playlist files
        playlists = ["playlist1.m3u", "playlist2.m3u", "playlist3.m3u"]
        for playlist in playlists:
            with open(os.path.join(self.playlist_dir, playlist), "w") as f:
                f.write("#EXTM3U\n")

        result = list_available_playlists(self.playlist_dir)
        self.assertEqual(len(result), 3)
        # Check that all playlists are included
        for playlist in playlists:
            self.assertTrue(any(playlist in path for path in result))

    def test_list_available_playlists_no_directory(self) -> None:
        """Test listing playlists when directory doesn't exist"""
        result = list_available_playlists("/nonexistent/directory")
        self.assertEqual(result, [])

    def test_list_available_playlists_no_files(self) -> None:
        """Test listing playlists when no M3U files exist"""
        result = list_available_playlists(self.temp_dir)
        self.assertEqual(result, [])

    def test_list_available_playlists_sorted_by_time(self) -> None:
        """Test that playlists are sorted by modification time"""
        # Create playlists with different timestamps
        playlist1 = os.path.join(self.playlist_dir, "old.m3u")
        playlist2 = os.path.join(self.playlist_dir, "new.m3u")

        with open(playlist1, "w") as f:
            f.write("#EXTM3U\n")

        import time

        time.sleep(0.1)

        with open(playlist2, "w") as f:
            f.write("#EXTM3U\n")

        result = list_available_playlists(self.playlist_dir)
        # Newest should be first
        self.assertIn("new.m3u", result[0])


class TestLibraryParser(unittest.TestCase):
    """Test Library Parser functionality"""

    def setUp(self) -> None:
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_library.db")
        self.xml_path = os.path.join(self.temp_dir, "test_library.xml")
        self.parser = LibraryParser(self.db_path)

    def tearDown(self) -> None:
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_xml(self, tracks_data: List[Dict[str, Any]]) -> None:
        """Helper method to create test XML file"""
        root = ET.Element("plist")
        root.set("version", "1.0")
        dict_elem = ET.SubElement(root, "dict")

        # Add Tracks key
        tracks_key = ET.SubElement(dict_elem, "key")
        tracks_key.text = "Tracks"

        tracks_dict = ET.SubElement(dict_elem, "dict")

        for i, track_data in enumerate(tracks_data):
            # Track ID key
            track_id_key = ET.SubElement(tracks_dict, "key")
            track_id_key.text = str(i + 1)

            # Track dict
            track_dict = ET.SubElement(tracks_dict, "dict")

            # Add Track ID field first (required by parser)
            track_id_key_elem = ET.SubElement(track_dict, "key")
            track_id_key_elem.text = "Track ID"
            track_id_value_elem = ET.SubElement(track_dict, "integer")
            track_id_value_elem.text = str(i + 1)

            # Add track fields
            for field_name, field_value in track_data.items():
                if field_value is not None:
                    key_elem = ET.SubElement(track_dict, "key")
                    key_elem.text = field_name

                    if isinstance(field_value, int):
                        value_elem = ET.SubElement(track_dict, "integer")
                    else:
                        value_elem = ET.SubElement(track_dict, "string")
                    value_elem.text = str(field_value)

        tree = ET.ElementTree(root)
        tree.write(self.xml_path, encoding="utf-8", xml_declaration=True)

    def test_create_table(self) -> None:
        """Test table creation"""
        # The table should be created in __init__
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='tracks'"
            )
            result = cursor.fetchone()
            self.assertIsNotNone(result)

    def test_parse_library_success(self) -> None:
        """Test successful library parsing"""
        tracks_data = [
            {
                "Name": "Test Song 1",
                "Artist": "Test Artist 1",
                "Album": "Test Album 1",
                "Genre": "Rock",
                "Year": 2023,
                "Play Count": 42,
                "Location": "/path/to/song1.mp3",
            },
            {
                "Name": "Test Song 2",
                "Artist": "Test Artist 2",
                "Album": "Test Album 2",
                "Genre": "Jazz",
                "Year": 2022,
                "Play Count": 15,
                "Location": "/path/to/song2.mp3",
            },
        ]

        self._create_test_xml(tracks_data)
        result = self.parser.parse_library(self.xml_path)

        self.assertEqual(result, 2)

        # Verify tracks were inserted
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM tracks")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 2)

    def test_parse_library_file_not_found(self) -> None:
        """Test parsing when XML file doesn't exist"""
        result = self.parser.parse_library("/nonexistent/library.xml")
        self.assertEqual(result, 0)

    def test_parse_library_empty_tracks(self) -> None:
        """Test parsing library with no tracks"""
        tracks_data: List[Dict[str, Any]] = []
        self._create_test_xml(tracks_data)
        result = self.parser.parse_library(self.xml_path)
        self.assertEqual(result, 0)

    def test_parse_library_tracks_without_name(self) -> None:
        """Test parsing tracks without name field"""
        tracks_data = [
            {
                "Artist": "Test Artist 1",
                "Album": "Test Album 1",
                "Genre": "Rock",
                "Year": 2023,
                "Play Count": 42,
                "Location": "/path/to/song1.mp3",
                # No Name field
            }
        ]

        self._create_test_xml(tracks_data)
        result = self.parser.parse_library(self.xml_path)

        self.assertEqual(result, 0)  # Track without name should be filtered out

    def test_field_processors(self) -> None:
        """Test field processing functions"""
        # Test string field processor
        elem = ET.Element("string")
        elem.text = "test string"
        result: Optional[str] = self.parser._process_string_field(elem)
        self.assertEqual(result, "test string")

        # Test int field processor
        elem = ET.Element("integer")
        elem.text = "42"
        result_int: int = self.parser._process_int_field(elem)
        self.assertEqual(result_int, 42)

        # Test optional int field processor
        elem = ET.Element("integer")
        elem.text = "123"
        result_optional_int: Optional[int] = self.parser._process_optional_int_field(
            elem
        )
        self.assertEqual(result_optional_int, 123)

        # Test optional int field with None
        elem = ET.Element("integer")
        elem.text = None
        result_none: Optional[int] = self.parser._process_optional_int_field(elem)
        self.assertIsNone(result_none)

    def test_field_mapping(self) -> None:
        """Test field mapping"""
        mapping = self.parser._get_field_mapping()
        self.assertIn("Name", mapping)
        self.assertIn("Artist", mapping)
        self.assertEqual(mapping["Name"], "name")
        self.assertEqual(mapping["Artist"], "artist")

    def test_process_track_field(self) -> None:
        """Test processing individual track fields"""
        data: Dict[str, Any] = {}
        key_name = "Name"
        value_elem = ET.Element("string")
        value_elem.text = "Test Song"

        self.parser._process_track_field(key_name, value_elem, data)
        self.assertEqual(data["name"], "Test Song")

    def test_process_track_field_unknown_field(self) -> None:
        """Test processing unknown track field"""
        data: Dict[str, Any] = {}
        key_name = "UnknownField"
        value_elem = ET.Element("string")
        value_elem.text = "test"

        self.parser._process_track_field(key_name, value_elem, data)
        # Should not add anything to data
        self.assertEqual(data, {})

    def test_extract_track_data(self) -> None:
        """Test extracting track data from XML element"""
        track_dict = ET.Element("dict")

        # Add track fields
        fields = [
            ("Track ID", "integer", "1"),
            ("Name", "string", "Test Song"),
            ("Artist", "string", "Test Artist"),
            ("Album", "string", "Test Album"),
            ("Genre", "string", "Rock"),
            ("Year", "integer", "2023"),
            ("Play Count", "integer", "42"),
            ("Location", "string", "/path/to/song.mp3"),
        ]

        for field_name, field_type, field_value in fields:
            key_elem = ET.SubElement(track_dict, "key")
            key_elem.text = field_name

            value_elem = ET.SubElement(track_dict, field_type)
            value_elem.text = field_value

        result = self.parser._extract_track_data(track_dict)

        self.assertIsNotNone(result)
        if result is not None:
            self.assertEqual(result["name"], "Test Song")
            self.assertEqual(result["artist"], "Test Artist")
            self.assertEqual(result["album"], "Test Album")
            self.assertEqual(result["genre"], "Rock")
            self.assertEqual(result["year"], 2023)
            self.assertEqual(result["play_count"], 42)
            self.assertEqual(result["location"], "/path/to/song.mp3")

    def test_extract_track_data_no_name(self) -> None:
        """Test extracting track data without name"""
        track_dict = ET.Element("dict")

        # Add track fields without name
        fields = [
            ("Track ID", "integer", "1"),
            ("Artist", "string", "Test Artist"),
            ("Album", "string", "Test Album"),
        ]

        for field_name, field_type, field_value in fields:
            key_elem = ET.SubElement(track_dict, "key")
            key_elem.text = field_name

            value_elem = ET.SubElement(track_dict, field_type)
            value_elem.text = field_value

        result = self.parser._extract_track_data(track_dict)
        self.assertIsNone(result)  # Should return None without name

    def test_insert_tracks(self) -> None:
        """Test inserting tracks into database"""
        tracks_data = [
            {
                "name": "Test Song 1",
                "artist": "Test Artist 1",
                "album_artist": "Test Artist 1",
                "composer": "Test Composer 1",
                "album": "Test Album 1",
                "genre": "Rock",
                "year": 2023,
                "total_time": 180000,
                "track_number": 1,
                "disc_number": 1,
                "play_count": 42,
                "date_added": "2023-01-01",
                "location": "/path/to/song1.mp3",
            },
            {
                "name": "Test Song 2",
                "artist": "Test Artist 2",
                "album_artist": "Test Artist 2",
                "composer": "Test Composer 2",
                "album": "Test Album 2",
                "genre": "Jazz",
                "year": 2022,
                "total_time": 240000,
                "track_number": 2,
                "disc_number": 1,
                "play_count": 15,
                "date_added": "2022-01-01",
                "location": "/path/to/song2.mp3",
            },
        ]

        # Create generator
        def tracks_generator() -> Any:
            for track in tracks_data:
                yield track

        result = self.parser._insert_tracks(tracks_generator())
        self.assertEqual(result, 2)

        # Verify tracks were inserted
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM tracks")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 2)

    def test_insert_tracks_duplicate_location(self) -> None:
        """Test inserting tracks with duplicate location"""
        tracks_data = [
            {
                "name": "Test Song 1",
                "artist": "Test Artist 1",
                "album_artist": "Test Artist 1",
                "composer": "Test Composer 1",
                "album": "Test Album 1",
                "genre": "Rock",
                "year": 2023,
                "total_time": 180000,
                "track_number": 1,
                "disc_number": 1,
                "play_count": 42,
                "date_added": "2023-01-01",
                "location": "/path/to/song.mp3",  # Same location
            },
            {
                "name": "Test Song 2",
                "artist": "Test Artist 2",
                "album_artist": "Test Artist 2",
                "composer": "Test Composer 2",
                "album": "Test Album 2",
                "genre": "Jazz",
                "year": 2022,
                "total_time": 240000,
                "track_number": 2,
                "disc_number": 1,
                "play_count": 15,
                "date_added": "2022-01-01",
                "location": "/path/to/song.mp3",  # Same location
            },
        ]

        # Create generator
        def tracks_generator() -> Any:
            for track in tracks_data:
                yield track

        result = self.parser._insert_tracks(tracks_generator())
        # Should only insert one due to unique constraint on location
        self.assertEqual(result, 1)

    def test_parse_library_with_field_processing(self) -> None:
        """Test library parsing with field processing"""
        tracks_data = [
            {
                "Name": "Test Song",
                "Artist": "Test Artist",
                "Album": "Test Album",
                "Genre": "Rock",
                "Year": 2023,
                "Play Count": 42,
                "Location": "/path/to/song.mp3",
            }
        ]

        self._create_test_xml(tracks_data)
        result = self.parser.parse_library(self.xml_path)

        self.assertEqual(result, 1)

        # Verify track was inserted with correct field processing
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tracks WHERE name = 'Test Song'")
            track = cursor.fetchone()
            self.assertIsNotNone(track)
            if track is not None:
                self.assertEqual(track[1], "Test Song")  # name
                self.assertEqual(track[2], "Test Artist")  # artist
                self.assertEqual(track[5], "Test Album")  # album (index 5)
                self.assertEqual(track[6], "Rock")  # genre (index 6)
                self.assertEqual(track[7], 2023)  # year (index 7)
                self.assertEqual(track[11], 42)  # play_count (index 11)
                self.assertEqual(track[13], "/path/to/song.mp3")  # location (index 13)

    def test_parse_library_with_none_values(self) -> None:
        """Test library parsing with None values"""
        tracks_data = [
            {
                "Name": "Test Song",
                "Artist": None,
                "Album": "Test Album",
                "Genre": None,
                "Year": None,
                "Play Count": None,
                "Location": "/path/to/song.mp3",
            }
        ]

        self._create_test_xml(tracks_data)
        result = self.parser.parse_library(self.xml_path)

        self.assertEqual(result, 1)

        # Verify track was inserted with None values handled correctly
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tracks WHERE name = 'Test Song'")
            track = cursor.fetchone()
            self.assertIsNotNone(track)
            if track is not None:
                self.assertEqual(track[1], "Test Song")  # name
                self.assertIsNone(track[2])  # artist (None)
                self.assertEqual(track[5], "Test Album")  # album
                self.assertIsNone(track[6])  # genre (None)
                self.assertIsNone(track[7])  # year (None)
                self.assertEqual(
                    track[11], 0
                )  # play_count (should be 0 if None provided)
                self.assertEqual(track[13], "/path/to/song.mp3")  # location


if __name__ == "__main__":
    unittest.main()
