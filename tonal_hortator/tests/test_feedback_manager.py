#!/usr/bin/env python3
"""
Tests for the FeedbackManager class.

Tests user feedback collection, preference management, and learning
functionality for playlist generation improvement.
"""

import json
import os
import sqlite3
import tempfile
import unittest
from typing import Any, Dict, List, cast
from unittest.mock import patch

from tonal_hortator.core.feedback.feedback_manager import FeedbackManager


class TestFeedbackManager(unittest.TestCase):
    """Test cases for FeedbackManager functionality"""

    def setUp(self) -> None:
        """Set up a temporary database for testing"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.db_path = self.temp_db.name
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Create tracks table for track rating tests
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
                location TEXT UNIQUE
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

        # Create feedback manager after database is set up
        self.feedback_manager = FeedbackManager(db_path=self.db_path)

    def tearDown(self) -> None:
        """Clean up temporary database"""
        if hasattr(self, "feedback_manager"):
            delattr(self, "feedback_manager")
        if hasattr(self, "conn"):
            self.conn.close()
        if hasattr(self, "temp_db"):
            self.temp_db.close()
            os.unlink(self.temp_db.name)

    def test_init_creates_tables(self) -> None:
        """Test that initialization creates all required tables"""
        # Check that all tables exist
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in self.cursor.fetchall()}

        expected_tables = {
            "tracks",  # Our test table
            "user_feedback",
            "user_preferences",
            "track_ratings",
            "query_learning",
        }

        self.assertTrue(expected_tables.issubset(tables))

    def test_record_playlist_feedback_success(self) -> None:
        """Test successful recording of playlist feedback with comprehensive validation"""
        query = "songs like Bohemian Rhapsody"
        query_type = "similarity"
        parsed_data = {
            "artist": None,
            "reference_artist": "Queen",
            "genres": ["Rock", "Progressive Rock"],
            "mood": "epic",
        }
        generated_tracks = [
            {"id": 1, "name": "Song 1", "artist": "Artist 1"},
            {"id": 2, "name": "Song 2", "artist": "Artist 2"},
        ]

        result = self.feedback_manager.record_playlist_feedback(
            query=query,
            query_type=query_type,
            parsed_data=parsed_data,
            generated_tracks=generated_tracks,
            user_rating=4,
            user_comments="Great playlist!",
            user_actions=["like", "skip"],
            playlist_length=2,
            requested_length=3,
            similarity_threshold=0.8,
            search_breadth=20,
        )

        # Validate return value
        self.assertTrue(result, "Feedback recording should return True on success")

        # Verify data was stored with comprehensive validation
        self.cursor.execute("SELECT * FROM user_feedback WHERE query = ?", (query,))
        row = self.cursor.fetchone()

        # Validate row exists and has correct structure
        self.assertIsNotNone(row, "Feedback record should be stored in database")
        self.assertIsInstance(row, tuple, "Database row should be a tuple")
        self.assertGreaterEqual(
            len(row), 15, "Feedback record should have at least 15 columns"
        )

        # Validate each field with detailed assertions
        self.assertEqual(row[1], query, "Query should be stored exactly as provided")
        self.assertEqual(row[2], query_type, "Query type should match provided value")
        self.assertIsNone(row[3], "Parsed artist should be None as provided")
        self.assertEqual(row[4], "Queen", "Reference artist should be stored correctly")

        # Validate JSON fields
        stored_genres = json.loads(row[5])
        self.assertIsInstance(stored_genres, list, "Stored genres should be a list")
        self.assertEqual(
            stored_genres, ["Rock", "Progressive Rock"], "Genres should match exactly"
        )

        self.assertEqual(row[6], "epic", "Mood should be stored correctly")

        # Validate track IDs
        stored_tracks = json.loads(row[7])
        self.assertIsInstance(stored_tracks, list, "Stored tracks should be a list")
        self.assertEqual(
            stored_tracks, [1, 2], "Track IDs should be extracted and stored correctly"
        )

        # Validate numeric fields
        self.assertEqual(row[8], 4, "User rating should be stored as integer")
        self.assertIsInstance(
            row[8], int, "User rating should be stored as integer type"
        )

        self.assertEqual(
            row[9], "Great playlist!", "User comments should be stored exactly"
        )

        # Validate user actions
        stored_actions = json.loads(row[10])
        self.assertIsInstance(
            stored_actions, list, "User actions should be stored as list"
        )
        self.assertEqual(
            stored_actions, ["like", "skip"], "User actions should match exactly"
        )

        # Validate playlist metrics
        self.assertEqual(row[11], 2, "Playlist length should be stored correctly")
        self.assertEqual(row[12], 3, "Requested length should be stored correctly")
        self.assertEqual(row[13], 0.8, "Similarity threshold should be stored as float")
        self.assertEqual(row[14], 20, "Search breadth should be stored correctly")

        # Validate timestamp fields
        self.assertIsNotNone(row[15], "Created timestamp should be set")
        self.assertIsInstance(row[15], str, "Created timestamp should be string")

        # Verify no duplicate records were created
        self.cursor.execute(
            "SELECT COUNT(*) FROM user_feedback WHERE query = ?", (query,)
        )
        count = self.cursor.fetchone()[0]
        self.assertEqual(
            count, 1, "Only one feedback record should exist for this query"
        )

    def test_record_playlist_feedback_minimal_data(self) -> None:
        """Test recording feedback with minimal required data"""
        result = self.feedback_manager.record_playlist_feedback(
            query="test query",
            query_type="general",
            parsed_data={},
            generated_tracks=[],
        )

        self.assertTrue(result)

    def test_record_playlist_feedback_edge_cases(self) -> None:
        """Test edge cases and boundary conditions for playlist feedback"""

        # Test with maximum rating
        result = self.feedback_manager.record_playlist_feedback(
            query="max rating test",
            query_type="similarity",
            parsed_data={},
            generated_tracks=[],
            user_rating=5,
        )
        self.assertTrue(result, "Should accept maximum rating (5)")

        # Test with minimum rating
        result = self.feedback_manager.record_playlist_feedback(
            query="min rating test",
            query_type="similarity",
            parsed_data={},
            generated_tracks=[],
            user_rating=1,
        )
        self.assertTrue(result, "Should accept minimum rating (1)")

        # Test with zero rating (edge case)
        result = self.feedback_manager.record_playlist_feedback(
            query="zero rating test",
            query_type="similarity",
            parsed_data={},
            generated_tracks=[],
            user_rating=0,
        )
        self.assertTrue(result, "Should accept zero rating")

        # Test with very long query
        long_query = "a" * 1000  # 1000 character query
        result = self.feedback_manager.record_playlist_feedback(
            query=long_query,
            query_type="similarity",
            parsed_data={},
            generated_tracks=[],
        )
        self.assertTrue(result, "Should handle very long queries")

        # Test with empty string query
        result = self.feedback_manager.record_playlist_feedback(
            query="",
            query_type="similarity",
            parsed_data={},
            generated_tracks=[],
        )
        self.assertTrue(result, "Should handle empty string queries")

        # Test with very large track list
        large_track_list = [{"id": i, "name": f"Track {i}"} for i in range(1000)]
        result = self.feedback_manager.record_playlist_feedback(
            query="large track list test",
            query_type="similarity",
            parsed_data={},
            generated_tracks=large_track_list,
        )
        self.assertTrue(result, "Should handle large track lists")

        # Test with complex nested parsed data
        complex_parsed_data = {
            "artist": "Complex Artist",
            "genres": ["Rock", "Metal", "Progressive", "Alternative"],
            "mood": "aggressive",
            "tempo": "fast",
            "year_range": [1990, 2020],
            "nested": {
                "sub_genres": ["Thrash Metal", "Progressive Rock"],
                "instruments": ["Guitar", "Drums", "Bass"],
            },
        }
        result = self.feedback_manager.record_playlist_feedback(
            query="complex data test",
            query_type="similarity",
            parsed_data=complex_parsed_data,
            generated_tracks=[],
        )
        self.assertTrue(result, "Should handle complex nested parsed data")

        # Verify complex data was stored correctly
        self.cursor.execute(
            "SELECT parsed_genres, parsed_mood FROM user_feedback WHERE query = ?",
            ("complex data test",),
        )
        row = self.cursor.fetchone()
        self.assertIsNotNone(row, "Complex data should be stored")

        stored_genres = json.loads(row[0])
        self.assertEqual(stored_genres, ["Rock", "Metal", "Progressive", "Alternative"])
        self.assertEqual(row[1], "aggressive")

    def test_record_playlist_feedback_invalid_data(self) -> None:
        """Test handling of invalid data in playlist feedback"""

        # Test with None query (should handle gracefully)
        result = self.feedback_manager.record_playlist_feedback(
            query="",
            query_type="similarity",
            parsed_data={},
            generated_tracks=[],
        )
        # Should either return False or handle None gracefully
        self.assertIsInstance(
            result, bool, "Should return boolean result even with None query"
        )

        # Test with negative rating
        result = self.feedback_manager.record_playlist_feedback(
            query="negative rating test",
            query_type="similarity",
            parsed_data={},
            generated_tracks=[],
            user_rating=-1,
        )
        self.assertIsInstance(result, bool, "Should handle negative ratings gracefully")

        # Test with rating above maximum
        result = self.feedback_manager.record_playlist_feedback(
            query="high rating test",
            query_type="similarity",
            parsed_data={},
            generated_tracks=[],
            user_rating=10,
        )
        self.assertIsInstance(
            result, bool, "Should handle ratings above maximum gracefully"
        )

        # Test with invalid JSON data
        invalid_parsed_data = {
            "artist": "Test Artist",
            "genres": ["Rock", "Pop"],  # This should be fine
            "invalid_field": object(),  # This might cause issues
        }
        result = self.feedback_manager.record_playlist_feedback(
            query="invalid data test",
            query_type="similarity",
            parsed_data=invalid_parsed_data,
            generated_tracks=[],
        )
        self.assertIsInstance(result, bool, "Should handle invalid data gracefully")

    def test_record_playlist_feedback_database_error(self) -> None:
        """Test handling of database errors during feedback recording"""
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Database error")

            result = self.feedback_manager.record_playlist_feedback(
                query="test", query_type="general", parsed_data={}, generated_tracks=[]
            )

            self.assertFalse(result)

    def test_record_track_rating_success(self) -> None:
        """Test successful recording of track rating with comprehensive validation"""
        track_id = 1
        rating = 5
        context = "playlist: Summer Vibes"

        result = self.feedback_manager.record_track_rating(
            track_id=track_id, rating=rating, context=context
        )

        # Validate return value
        self.assertTrue(result, "Track rating should return True on success")

        # Verify rating was stored with comprehensive validation
        self.cursor.execute(
            "SELECT * FROM track_ratings WHERE track_id = ?", (track_id,)
        )
        row = self.cursor.fetchone()

        # Validate row exists and has correct structure
        self.assertIsNotNone(row, "Track rating should be stored in database")
        self.assertIsInstance(row, tuple, "Database row should be a tuple")
        self.assertGreaterEqual(
            len(row), 4, "Track rating record should have at least 4 columns"
        )

        # Validate each field with detailed assertions
        self.assertEqual(row[1], track_id, "Track ID should be stored correctly")
        self.assertIsInstance(row[1], int, "Track ID should be stored as integer")

        self.assertEqual(row[2], rating, "Rating should be stored correctly")
        self.assertIsInstance(row[2], int, "Rating should be stored as integer")
        self.assertGreaterEqual(row[2], 0, "Rating should be non-negative")
        self.assertLessEqual(row[2], 5, "Rating should not exceed maximum")

        self.assertEqual(
            row[3], context, "Context should be stored exactly as provided"
        )
        self.assertIsInstance(row[3], str, "Context should be stored as string")

        # Validate timestamp
        self.assertIsNotNone(row[4], "Created timestamp should be set")
        self.assertIsInstance(row[4], str, "Created timestamp should be string")

        # Verify no duplicate ratings for same track
        self.cursor.execute(
            "SELECT COUNT(*) FROM track_ratings WHERE track_id = ?", (track_id,)
        )
        count = self.cursor.fetchone()[0]
        self.assertEqual(count, 1, "Only one rating should exist per track")

    def test_record_track_rating_edge_cases(self) -> None:
        """Test edge cases for track rating functionality"""

        # Test with minimum rating
        result = self.feedback_manager.record_track_rating(
            track_id=2, rating=1, context="min rating test"
        )
        self.assertTrue(result, "Should accept minimum rating (1)")

        # Test with zero rating
        result = self.feedback_manager.record_track_rating(
            track_id=3, rating=0, context="zero rating test"
        )
        self.assertTrue(result, "Should accept zero rating")

        # Test with maximum rating
        result = self.feedback_manager.record_track_rating(
            track_id=4, rating=5, context="max rating test"
        )
        self.assertTrue(result, "Should accept maximum rating (5)")

        # Test with very long context
        long_context = "a" * 500  # 500 character context
        result = self.feedback_manager.record_track_rating(
            track_id=5, rating=3, context=long_context
        )
        self.assertTrue(result, "Should handle very long context strings")

        # Test with empty context
        result = self.feedback_manager.record_track_rating(
            track_id=6, rating=4, context=""
        )
        self.assertTrue(result, "Should handle empty context strings")

        # Test with special characters in context
        special_context = "playlist: 🎵 Rock & Roll 🎸 (2024)"
        result = self.feedback_manager.record_track_rating(
            track_id=7, rating=5, context=special_context
        )
        self.assertTrue(result, "Should handle special characters in context")

        # Verify special characters were stored correctly
        self.cursor.execute(
            "SELECT context FROM track_ratings WHERE track_id = ?", (7,)
        )
        stored_context = self.cursor.fetchone()[0]
        self.assertEqual(
            stored_context, special_context, "Special characters should be preserved"
        )

        # Test multiple ratings for different tracks
        track_ids = [10, 11, 12, 13, 14]
        for i, track_id in enumerate(track_ids):
            result = self.feedback_manager.record_track_rating(
                track_id=track_id, rating=i + 1, context=f"test {i+1}"
            )
            self.assertTrue(result, f"Should record rating for track {track_id}")

        # Verify all ratings were stored
        self.cursor.execute(
            "SELECT COUNT(*) FROM track_ratings WHERE track_id IN (10, 11, 12, 13, 14)"
        )
        count = self.cursor.fetchone()[0]
        self.assertEqual(count, 5, "All 5 ratings should be stored")

    def test_record_track_rating_invalid_data(self) -> None:
        """Test handling of invalid data in track ratings"""

        # Test with negative rating
        result = self.feedback_manager.record_track_rating(
            track_id=20, rating=-1, context="negative test"
        )
        self.assertIsInstance(result, bool, "Should handle negative ratings gracefully")

        # Test with rating above maximum
        result = self.feedback_manager.record_track_rating(
            track_id=21, rating=10, context="high rating test"
        )
        self.assertIsInstance(
            result, bool, "Should handle ratings above maximum gracefully"
        )

        # Test with "" track_id
        result = self.feedback_manager.record_track_rating(
            track_id=0, rating=3, context="none track test"
        )
        self.assertIsInstance(result, bool, "Should handle None track_id gracefully")

        # Test with None context
        result = self.feedback_manager.record_track_rating(
            track_id=22, rating=3, context=None
        )
        self.assertIsInstance(result, bool, "Should handle None context gracefully")

    def test_record_track_rating_database_error(self) -> None:
        """Test handling of database errors during track rating"""
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Database error")

            result = self.feedback_manager.record_track_rating(
                track_id=1, rating=5, context="test"
            )

            self.assertFalse(result, "Should return False on database error")

    def test_set_preference_success(self) -> None:
        """Test successful setting of user preferences with comprehensive validation"""
        # Test different preference types with detailed validation
        test_cases = [
            ("theme", "dark", "string", "UI theme preference"),
            ("max_playlist_length", 20, "integer", "Maximum playlist length"),
            ("similarity_threshold", 0.75, "float", "Default similarity threshold"),
            ("auto_save", True, "boolean", "Auto-save playlists"),
            ("favorite_genres", ["Rock", "Jazz"], "json", "Favorite music genres"),
        ]

        for key, value, pref_type, description in test_cases:
            result = self.feedback_manager.set_preference(
                key=key, value=value, preference_type=pref_type, description=description
            )

            # Validate return value
            self.assertTrue(result, f"Setting preference '{key}' should return True")

            # Verify preference was stored with comprehensive validation
            self.cursor.execute(
                "SELECT preference_value, preference_type, description FROM user_preferences WHERE preference_key = ?",
                (key,),
            )
            row = self.cursor.fetchone()

            # Validate row exists and has correct structure
            self.assertIsNotNone(
                row, f"Preference '{key}' should be stored in database"
            )
            self.assertIsInstance(row, tuple, "Database row should be a tuple")
            self.assertEqual(len(row), 3, "Preference row should have 3 columns")

            # Validate each field based on type
            if pref_type == "json":
                stored_value = json.loads(row[0])
                self.assertIsInstance(
                    stored_value,
                    list,
                    f"JSON preference '{key}' should be stored as list",
                )
                self.assertEqual(
                    stored_value, value, f"JSON preference '{key}' should match exactly"
                )
            elif pref_type == "boolean":
                self.assertEqual(
                    row[0].lower(),
                    str(value).lower(),
                    f"Boolean preference '{key}' should match",
                )
            elif pref_type == "integer":
                self.assertEqual(
                    int(row[0]),
                    value,
                    f"Integer preference '{key}' should match exactly",
                )
            elif pref_type == "float":
                self.assertAlmostEqual(
                    float(row[0]),
                    cast(float, value),
                    places=6,
                    msg=f"Float preference '{key}' should match",
                )
            else:  # string
                self.assertEqual(
                    row[0],
                    str(value),
                    f"String preference '{key}' should match exactly",
                )

            self.assertEqual(
                row[1], pref_type, f"Preference type for '{key}' should match"
            )
            self.assertEqual(
                row[2], description, f"Description for '{key}' should match"
            )

            # Verify no duplicate preferences
            self.cursor.execute(
                "SELECT COUNT(*) FROM user_preferences WHERE preference_key = ?", (key,)
            )
            count = self.cursor.fetchone()[0]
            self.assertEqual(
                count, 1, f"Only one preference should exist for key '{key}'"
            )

    def test_set_preference_edge_cases(self) -> None:
        """Test edge cases for preference setting"""

        # Test with empty string key
        result = self.feedback_manager.set_preference(
            key="", value="empty key test", preference_type="string"
        )
        self.assertTrue(result, "Should handle empty string keys")

        # Test with very long key
        long_key = "a" * 100
        result = self.feedback_manager.set_preference(
            key=long_key, value="long key test", preference_type="string"
        )
        self.assertTrue(result, "Should handle very long keys")

        # Test with special characters in key
        special_key = "pref_with_🎵_chars"
        result = self.feedback_manager.set_preference(
            key=special_key, value="special chars", preference_type="string"
        )
        self.assertTrue(result, "Should handle special characters in keys")

        # Test with very large integer
        large_int = 999999999
        result = self.feedback_manager.set_preference(
            key="large_int", value=large_int, preference_type="integer"
        )
        self.assertTrue(result, "Should handle very large integers")

        # Test with very small float
        small_float = 0.000001
        result = self.feedback_manager.set_preference(
            key="small_float", value=small_float, preference_type="float"
        )
        self.assertTrue(result, "Should handle very small floats")

        # Test with complex JSON data
        complex_json = {
            "nested": {
                "deep": {
                    "value": [1, 2, 3, {"key": "value"}],
                    "boolean": True,
                    "null": None,
                }
            },
            "array": ["string", 123, False, None],
        }
        result = self.feedback_manager.set_preference(
            key="complex_json", value=complex_json, preference_type="json"
        )
        self.assertTrue(result, "Should handle complex nested JSON data")

        # Verify complex JSON was stored correctly
        stored_value = self.feedback_manager.get_preference("complex_json")
        self.assertEqual(
            stored_value, complex_json, "Complex JSON should be preserved exactly"
        )

        # Test with None value
        result = self.feedback_manager.set_preference(
            key="none_value", value=None, preference_type="string"
        )
        self.assertIsInstance(result, bool, "Should handle None values gracefully")

    def test_set_preference_database_error(self) -> None:
        """Test handling of database errors during preference setting"""
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Database error")

            result = self.feedback_manager.set_preference(
                key="test", value="value", preference_type="string"
            )

            self.assertFalse(result, "Should return False on database error")

    def test_get_preference_success(self) -> None:
        """Test successful retrieval of user preferences with comprehensive validation"""
        # Set up test preferences with various types
        test_preferences = [
            ("theme", "dark", "string"),
            ("max_length", 25, "integer"),
            ("threshold", 0.8, "float"),
            ("auto_save", True, "boolean"),
            ("genres", ["Rock", "Pop"], "json"),
        ]

        for key, value, pref_type in test_preferences:
            self.feedback_manager.set_preference(key, value, pref_type)

        # Test retrieval with comprehensive validation
        for key, expected_value, pref_type in test_preferences:
            result = self.feedback_manager.get_preference(key)

            # Validate return type and value
            if pref_type == "boolean":
                self.assertIsInstance(
                    result, bool, f"Boolean preference '{key}' should return bool"
                )
                self.assertEqual(
                    result, expected_value, f"Boolean preference '{key}' should match"
                )
            elif pref_type == "integer":
                self.assertIsInstance(
                    result, int, f"Integer preference '{key}' should return int"
                )
                self.assertEqual(
                    result, expected_value, f"Integer preference '{key}' should match"
                )
            elif pref_type == "float":
                self.assertIsInstance(
                    result, float, f"Float preference '{key}' should return float"
                )
                self.assertAlmostEqual(
                    result,
                    cast(float, expected_value),
                    places=6,
                    msg=f"Float preference '{key}' should match",
                )
            elif pref_type == "json":
                self.assertIsInstance(
                    result, list, f"JSON preference '{key}' should return list"
                )
                self.assertEqual(
                    result, expected_value, f"JSON preference '{key}' should match"
                )
            else:  # string
                self.assertIsInstance(
                    result, str, f"String preference '{key}' should return string"
                )
                self.assertEqual(
                    result, expected_value, f"String preference '{key}' should match"
                )

    def test_get_preference_not_found(self) -> None:
        """Test getting non-existent preference returns default"""
        result = self.feedback_manager.get_preference(
            "nonexistent", default="default_value"
        )
        self.assertEqual(result, "default_value")

    def test_get_preference_database_error(self) -> None:
        """Test handling of database errors during preference retrieval"""
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Database error")

            result = self.feedback_manager.get_preference("test", default="default")
            self.assertEqual(result, "default")

    def test_record_query_learning_success(self) -> None:
        """Test successful recording of query learning data"""
        original_query = "songs like Bohemian Rhapsody"
        llm_parsed_result = {
            "artist": None,
            "reference_artist": "Queen",
            "genres": ["Rock"],
            "mood": "epic",
        }
        user_correction = {
            "artist": None,
            "reference_artist": "Queen",
            "genres": ["Rock", "Progressive Rock"],
            "mood": "epic",
        }

        result = self.feedback_manager.record_query_learning(
            original_query=original_query,
            llm_parsed_result=llm_parsed_result,
            user_correction=user_correction,
            feedback_score=0.8,
        )

        self.assertTrue(result, "Should successfully record query learning data")

        # Verify data was stored correctly
        self.cursor.execute(
            "SELECT original_query, llm_parsed_result, user_correction, feedback_score FROM query_learning WHERE original_query = ?",
            (original_query,),
        )
        row = self.cursor.fetchone()

        self.assertIsNotNone(row, "Query learning data should be stored")
        self.assertEqual(row[0], original_query, "Original query should match")
        self.assertEqual(
            json.loads(row[1]), llm_parsed_result, "LLM parsed result should match"
        )
        self.assertEqual(
            json.loads(row[2]), user_correction, "User correction should match"
        )
        self.assertEqual(row[3], 0.8, "Feedback score should match")

    def test_record_query_learning_minimal_data(self) -> None:
        """Test recording query learning with minimal required data"""
        result = self.feedback_manager.record_query_learning(
            original_query="minimal query",
            llm_parsed_result={"artist": "Test Artist"},
        )
        self.assertTrue(result, "Should record minimal query learning data")

    def test_record_query_learning_database_error(self) -> None:
        """Test handling of database errors during query learning recording"""
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Database error")

            result = self.feedback_manager.record_query_learning(
                original_query="test", llm_parsed_result={"artist": "test"}
            )
            self.assertFalse(result, "Should return False on database error")

    def test_get_feedback_stats_success(self) -> None:
        """Test successful retrieval of feedback statistics with comprehensive validation"""
        # Set up test feedback data
        test_feedback_data = [
            {
                "query": "songs like Bohemian Rhapsody",
                "query_type": "artist_specific",
                "parsed_data": {"artist": "Queen", "genres": ["Rock"]},
                "generated_tracks": [{"id": 1}, {"id": 2}],
                "user_rating": 5,
                "similarity_threshold": 0.8,
                "search_breadth": 20,
            },
            {
                "query": "rock music from the 70s",
                "query_type": "artist_specific",
                "parsed_data": {"artist": None, "genres": ["Rock"]},
                "generated_tracks": [{"id": 3}],
                "user_rating": 4,
                "similarity_threshold": 0.7,
                "search_breadth": 15,
            },
            {
                "query": "similar to Led Zeppelin",
                "query_type": "similarity",
                "parsed_data": {"reference_artist": "Led Zeppelin"},
                "generated_tracks": [{"id": 3}],
                "user_rating": 3,
                "similarity_threshold": 0.7,
                "search_breadth": 15,
            },
        ]

        for feedback in test_feedback_data:
            self.feedback_manager.record_playlist_feedback(
                query=str(feedback["query"]),
                query_type=str(feedback["query_type"]),
                parsed_data=cast(Dict[str, Any], feedback["parsed_data"]),
                generated_tracks=cast(
                    List[Dict[str, Any]], feedback["generated_tracks"]
                ),
                user_rating=(
                    int(cast(int, feedback["user_rating"]))
                    if feedback["user_rating"] is not None
                    else None
                ),
                similarity_threshold=(
                    float(cast(float, feedback["similarity_threshold"]))
                    if feedback["similarity_threshold"] is not None
                    else None
                ),
                search_breadth=(
                    int(cast(int, feedback["search_breadth"]))
                    if feedback["search_breadth"] is not None
                    else None
                ),
            )

        # Get and validate statistics
        stats = self.feedback_manager.get_feedback_stats()

        # Validate return type and structure
        self.assertIsInstance(stats, dict, "Stats should be returned as dictionary")
        self.assertIn("total_feedback", stats, "Stats should contain total_feedback")
        self.assertIn("average_rating", stats, "Stats should contain average_rating")
        self.assertIn(
            "feedback_by_type", stats, "Stats should contain feedback_by_type"
        )
        self.assertIn("recent_feedback", stats, "Stats should contain recent_feedback")

        # Validate specific values
        self.assertEqual(
            stats["total_feedback"], 3.0, "Total feedback count should be 3"
        )
        self.assertAlmostEqual(
            stats["average_rating"], 4.0, places=1, msg="Average rating should be 4.0"
        )

        # Validate feedback_by_type structure
        self.assertIsInstance(
            stats["feedback_by_type"], dict, "feedback_by_type should be dictionary"
        )
        self.assertIn(
            "artist_specific",
            stats["feedback_by_type"],
            "Should have artist_specific type",
        )
        self.assertIn(
            "similarity", stats["feedback_by_type"], "Should have similarity type"
        )
        self.assertEqual(
            stats["feedback_by_type"]["artist_specific"],
            2,
            "Should have 2 artist_specific feedback",
        )
        self.assertEqual(
            stats["feedback_by_type"]["similarity"],
            1,
            "Should have 1 similarity feedback",
        )

        # Validate recent_feedback structure
        self.assertIsInstance(
            stats["recent_feedback"], list, "recent_feedback should be list"
        )
        self.assertEqual(
            len(stats["recent_feedback"]), 3, "Should have 3 recent feedback entries"
        )

        # Validate recent feedback entries (they are tuples, not dictionaries)
        for entry in stats["recent_feedback"]:
            self.assertIsInstance(
                entry, tuple, "Each recent feedback entry should be tuple"
            )
            self.assertGreaterEqual(
                len(entry), 15, "Each feedback tuple should have at least 15 columns"
            )

            # Validate key fields in the tuple (based on user_feedback table structure)
            # entry[0] = id, entry[1] = query, entry[2] = query_type, entry[8] = user_rating, entry[15] = created_at
            self.assertIsInstance(entry[1], str, "Query should be string")
            self.assertIsInstance(entry[2], str, "Query type should be string")
            if entry[8] is not None:  # user_rating can be None
                self.assertIsInstance(entry[8], int, "User rating should be integer")
            self.assertIsInstance(entry[15], str, "Created timestamp should be string")

    def test_get_feedback_stats_empty_database(self) -> None:
        """Test getting feedback stats from empty database"""
        stats = self.feedback_manager.get_feedback_stats()

        self.assertEqual(stats["total_feedback"], 0.0)
        self.assertEqual(stats["average_rating"], 0.0)
        self.assertEqual(stats["feedback_by_type"], {})
        self.assertEqual(stats["recent_feedback"], [])

    def test_get_feedback_stats_database_error(self) -> None:
        """Test handling of database errors during stats retrieval"""
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Database error")

            stats = self.feedback_manager.get_feedback_stats()
            self.assertEqual(stats, {})

    def test_get_user_preferences_success(self) -> None:
        """Test successful retrieval of all user preferences"""
        # Set up test preferences
        self.feedback_manager.set_preference("theme", "dark", "string", "UI theme")
        self.feedback_manager.set_preference(
            "max_length", 20, "integer", "Max playlist length"
        )
        self.feedback_manager.set_preference(
            "auto_save", True, "boolean", "Auto-save setting"
        )

        preferences = self.feedback_manager.get_user_preferences()

        self.assertIsInstance(preferences, list)
        self.assertEqual(len(preferences), 3)

        # Check that all preferences are returned with correct types
        pref_dict = {
            key: (value, pref_type, desc) for key, value, pref_type, desc in preferences
        }

        self.assertIn("theme", pref_dict)
        self.assertEqual(pref_dict["theme"][0], "dark")
        self.assertEqual(pref_dict["theme"][1], "string")
        self.assertEqual(pref_dict["theme"][2], "UI theme")

        self.assertIn("max_length", pref_dict)
        self.assertEqual(pref_dict["max_length"][0], 20)
        self.assertEqual(pref_dict["max_length"][1], "integer")

        self.assertIn("auto_save", pref_dict)
        self.assertEqual(pref_dict["auto_save"][0], True)
        self.assertEqual(pref_dict["auto_save"][1], "boolean")

    def test_get_user_preferences_empty_database(self) -> None:
        """Test getting preferences from empty database"""
        preferences = self.feedback_manager.get_user_preferences()
        self.assertEqual(preferences, [])

    def test_get_user_preferences_database_error(self) -> None:
        """Test handling of database errors during preferences retrieval"""
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Database error")

            preferences = self.feedback_manager.get_user_preferences()
            self.assertEqual(preferences, [])

    def test_get_track_ratings_success(self) -> None:
        """Test successful retrieval of track ratings"""
        self.feedback_manager.record_track_rating(1, 5, "playlist: Rock")
        self.feedback_manager.record_track_rating(2, 3, "playlist: Pop")

        ratings = self.feedback_manager.get_track_ratings()

        self.assertIsInstance(ratings, list)
        self.assertEqual(len(ratings), 2)

        # Check rating data structure
        for rating, context, track_name, artist in ratings:
            self.assertIsInstance(rating, int)
            self.assertIsInstance(context, str)
            self.assertIsInstance(track_name, str)
            self.assertIsInstance(artist, str)

    def test_get_track_ratings_empty_database(self) -> None:
        """Test getting track ratings from empty database"""
        ratings = self.feedback_manager.get_track_ratings()
        self.assertEqual(ratings, [])

    def test_get_track_ratings_database_error(self) -> None:
        """Test handling of database errors during track ratings retrieval"""
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Database error")

            ratings = self.feedback_manager.get_track_ratings()
            self.assertEqual(ratings, [])

    def test_get_learning_data_success(self) -> None:
        """Test successful retrieval of learning data"""
        # Set up test learning data
        self.feedback_manager.record_query_learning(
            original_query="test query 1",
            llm_parsed_result={"artist": "Artist 1"},
            user_correction={"artist": "Artist 1", "genres": ["Rock"]},
            feedback_score=0.8,
        )

        self.feedback_manager.record_query_learning(
            original_query="test query 2",
            llm_parsed_result={"artist": "Artist 2"},
            feedback_score=-0.2,
        )

        learning_data = self.feedback_manager.get_learning_data()

        self.assertIsInstance(learning_data, list)
        self.assertEqual(len(learning_data), 2)

        # Check data structure
        for query, llm_result, user_correction, feedback_score in learning_data:
            self.assertIsInstance(query, str)
            self.assertIsInstance(llm_result, dict)
            self.assertIsInstance(feedback_score, (float, type(None)))
            if user_correction is not None:
                self.assertIsInstance(user_correction, dict)

    def test_get_learning_data_empty_database(self) -> None:
        """Test getting learning data from empty database"""
        learning_data = self.feedback_manager.get_learning_data()
        self.assertEqual(learning_data, [])

    def test_get_learning_data_database_error(self) -> None:
        """Test handling of database errors during learning data retrieval"""
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Database error")

            learning_data = self.feedback_manager.get_learning_data()
            self.assertEqual(learning_data, [])

    def test_mark_learning_applied_success(self) -> None:
        """Test successful marking of learning data as applied"""
        # Set up test learning data
        self.feedback_manager.record_query_learning(
            original_query="test query", llm_parsed_result={"artist": "Artist 1"}
        )

        # Get the learning ID
        self.cursor.execute(
            "SELECT id FROM query_learning WHERE original_query = ?", ("test query",)
        )
        learning_id = self.cursor.fetchone()[0]

        result = self.feedback_manager.mark_learning_applied(learning_id)

        self.assertTrue(result)

        # Verify it was marked as applied
        self.cursor.execute(
            "SELECT learning_applied FROM query_learning WHERE id = ?", (learning_id,)
        )
        applied = self.cursor.fetchone()[0]

        self.assertEqual(applied, True)

    def test_mark_learning_applied_database_error(self) -> None:
        """Test handling of database errors during learning marking"""
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Database error")

            result = self.feedback_manager.mark_learning_applied(1)
            self.assertFalse(result)

    def test_get_recommended_settings_success(self) -> None:
        """Test successful retrieval of recommended settings"""
        # Set up test feedback data with ratings
        self.feedback_manager.record_playlist_feedback(
            query="test query 1",
            query_type="artist_specific",
            parsed_data={"artist": "Artist 1"},
            generated_tracks=[{"id": 1}],
            user_rating=4,
            similarity_threshold=0.8,
            search_breadth=20,
        )

        self.feedback_manager.record_playlist_feedback(
            query="test query 2",
            query_type="artist_specific",
            parsed_data={"artist": "Artist 2"},
            generated_tracks=[{"id": 2}],
            user_rating=5,
            similarity_threshold=0.9,
            search_breadth=25,
        )

        settings = self.feedback_manager.get_recommended_settings("artist_specific")

        self.assertIsInstance(settings, dict)
        self.assertIn("similarity_threshold", settings)
        self.assertIn("search_breadth", settings)
        self.assertIn("average_rating", settings)
        self.assertIn("feedback_count", settings)

        self.assertEqual(settings["feedback_count"], 2)
        self.assertEqual(settings["average_rating"], 4.5)
        self.assertAlmostEqual(settings["similarity_threshold"], 0.85, places=2)
        self.assertEqual(
            settings["search_breadth"], 22
        )  # rounded average (20 + 25) / 2 = 22.5 -> 22

    def test_get_recommended_settings_no_feedback(self) -> None:
        """Test getting recommended settings with no feedback"""
        settings = self.feedback_manager.get_recommended_settings("artist_specific")
        self.assertEqual(settings, {})

    def test_get_recommended_settings_database_error(self) -> None:
        """Test handling of database errors during settings retrieval"""
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Database error")

            settings = self.feedback_manager.get_recommended_settings("artist_specific")
            self.assertEqual(settings, {})

    def test_preference_type_conversion_edge_cases(self) -> None:
        """Test edge cases in preference type conversion"""
        # Test float preference with integer conversion
        self.feedback_manager.set_preference("test_float", 3.7, "float")
        result = self.feedback_manager.get_preference("test_float")
        self.assertEqual(
            result, 3.7
        )  # Float values are preserved, not converted to int

        # Test boolean edge cases
        self.feedback_manager.set_preference("test_bool_true", True, "boolean")
        self.feedback_manager.set_preference("test_bool_false", False, "boolean")

        self.assertEqual(self.feedback_manager.get_preference("test_bool_true"), True)
        self.assertEqual(self.feedback_manager.get_preference("test_bool_false"), False)

        # Test JSON with complex data
        complex_data = {
            "nested": {"list": [1, 2, 3], "string": "test"},
            "boolean": True,
            "number": 42,
        }
        self.feedback_manager.set_preference("complex_json", complex_data, "json")
        result = self.feedback_manager.get_preference("complex_json")
        self.assertEqual(result, complex_data)

    def test_feedback_with_special_characters(self) -> None:
        """Test feedback recording with special characters and unicode"""
        query = "songs like 🎵 Bohemian Rhapsody 🎶"
        user_comments = "Great playlist! 🎉 Love the variety 😊"

        result = self.feedback_manager.record_playlist_feedback(
            query=query,
            query_type="similarity",
            parsed_data={"artist": "Queen 🎸"},
            generated_tracks=[{"id": 1, "name": "Song with émojis 🎵"}],
            user_comments=user_comments,
        )

        self.assertTrue(result)

        # Verify special characters were stored correctly
        self.cursor.execute(
            "SELECT query, user_comments FROM user_feedback WHERE query = ?", (query,)
        )
        row = self.cursor.fetchone()

        self.assertEqual(row[0], query)
        self.assertEqual(row[1], user_comments)

    def test_concurrent_feedback_recording(self) -> None:
        """Test concurrent feedback recording for thread safety"""
        import threading
        import time

        results = []
        errors = []

        def record_feedback(thread_id: int) -> None:
            """Record feedback from a thread"""
            try:
                result = self.feedback_manager.record_playlist_feedback(
                    query=f"concurrent test {thread_id}",
                    query_type="similarity",
                    parsed_data={"artist": f"Artist {thread_id}"},
                    generated_tracks=[{"id": thread_id}],
                    user_rating=thread_id % 5 + 1,
                )
                results.append((thread_id, result))
            except Exception as e:
                errors.append((thread_id, str(e)))

        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=record_feedback, args=(i,))
            threads.append(thread)

        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        end_time = time.time()

        # Validate results
        self.assertEqual(
            len(results), 10, "All 10 threads should complete successfully"
        )
        self.assertEqual(
            len(errors), 0, "No errors should occur during concurrent access"
        )

        # Validate all feedback was recorded
        for thread_id, result in results:
            self.assertTrue(result, f"Thread {thread_id} should return True")

        # Verify data integrity
        self.cursor.execute(
            "SELECT COUNT(*) FROM user_feedback WHERE query LIKE 'concurrent test %'"
        )
        count = self.cursor.fetchone()[0]
        self.assertEqual(
            count, 10, "All 10 concurrent feedback records should be stored"
        )

        # Performance assertion (should complete within reasonable time)
        execution_time = end_time - start_time
        self.assertLess(
            execution_time,
            5.0,
            f"Concurrent operations should complete within 5 seconds, took {execution_time:.2f}s",
        )

    def test_data_integrity_validation(self) -> None:
        """Test data integrity across multiple operations"""

        # Record initial feedback
        self.feedback_manager.record_playlist_feedback(
            query="integrity test",
            query_type="similarity",
            parsed_data={"artist": "Test Artist"},
            generated_tracks=[{"id": 1}],
            user_rating=4,
        )

        # Record track rating
        self.feedback_manager.record_track_rating(
            track_id=1, rating=5, context="integrity test"
        )

        # Set preference
        self.feedback_manager.set_preference(
            key="integrity_test", value="test_value", preference_type="string"
        )

        # Record learning data
        self.feedback_manager.record_query_learning(
            original_query="integrity test query",
            llm_parsed_result={"artist": "Test Artist"},
            user_correction={"artist": "Test Artist", "genres": ["Rock"]},
            feedback_score=0.8,
        )

        # Validate all data is consistent
        stats = self.feedback_manager.get_feedback_stats()
        self.assertEqual(
            stats["total_feedback"], 1.0, "Should have exactly 1 feedback record"
        )

        preferences = self.feedback_manager.get_user_preferences()
        self.assertEqual(len(preferences), 1, "Should have exactly 1 preference")
        self.assertEqual(
            preferences[0][0], "integrity_test", "Preference key should match"
        )

        ratings = self.feedback_manager.get_track_ratings()
        self.assertEqual(len(ratings), 1, "Should have exactly 1 track rating")
        self.assertEqual(ratings[0][0], 5, "Track rating should match")

        learning_data = self.feedback_manager.get_learning_data()
        self.assertEqual(len(learning_data), 1, "Should have exactly 1 learning record")
        self.assertEqual(
            learning_data[0][0], "integrity test query", "Learning query should match"
        )

        # Test data persistence across manager instances
        new_manager = FeedbackManager(db_path=self.db_path)

        new_stats = new_manager.get_feedback_stats()
        self.assertEqual(
            new_stats["total_feedback"],
            1.0,
            "Data should persist across manager instances",
        )

        new_preferences = new_manager.get_user_preferences()
        self.assertEqual(
            len(new_preferences),
            1,
            "Preferences should persist across manager instances",
        )

    def test_performance_benchmarks(self) -> None:
        """Test performance characteristics of feedback operations"""
        import time

        # Benchmark feedback recording
        start_time = time.time()
        for i in range(100):
            self.feedback_manager.record_playlist_feedback(
                query=f"perf test {i}",
                query_type="similarity",
                parsed_data={"artist": f"Artist {i}"},
                generated_tracks=[{"id": i}],
                user_rating=i % 5 + 1,
            )
        feedback_time = time.time() - start_time

        # Benchmark preference operations
        start_time = time.time()
        for i in range(100):
            self.feedback_manager.set_preference(
                key=f"perf_pref_{i}", value=f"value_{i}", preference_type="string"
            )
        preference_time = time.time() - start_time

        # Benchmark retrieval operations
        start_time = time.time()
        for i in range(100):
            self.feedback_manager.get_feedback_stats()
        retrieval_time = time.time() - start_time

        # Performance assertions (reasonable thresholds for SQLite operations)
        self.assertLess(
            feedback_time,
            10.0,
            f"100 feedback recordings should complete within 10s, took {feedback_time:.2f}s",
        )
        self.assertLess(
            preference_time,
            5.0,
            f"100 preference operations should complete within 5s, took {preference_time:.2f}s",
        )
        self.assertLess(
            retrieval_time,
            2.0,
            f"100 retrieval operations should complete within 2s, took {retrieval_time:.2f}s",
        )

        # Validate all operations succeeded
        stats = self.feedback_manager.get_feedback_stats()
        self.assertEqual(
            stats["total_feedback"], 100.0, "All 100 feedback records should be stored"
        )

        preferences = self.feedback_manager.get_user_preferences()
        self.assertEqual(len(preferences), 100, "All 100 preferences should be stored")

    def test_error_recovery_and_consistency(self) -> None:
        """Test error recovery and data consistency after failures"""

        # Record some initial data
        self.feedback_manager.record_playlist_feedback(
            query="recovery test",
            query_type="similarity",
            parsed_data={"artist": "Test Artist"},
            generated_tracks=[{"id": 1}],
            user_rating=4,
        )

        # Simulate a database error and recovery
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Simulated database error")

            # These should fail gracefully
            result1 = self.feedback_manager.record_playlist_feedback(
                query="should fail",
                query_type="similarity",
                parsed_data={},
                generated_tracks=[],
            )
            result2 = self.feedback_manager.set_preference(
                key="should_fail", value="test", preference_type="string"
            )

            self.assertFalse(result1, "Should return False on database error")
            self.assertFalse(result2, "Should return False on database error")

        # After error recovery, system should still work
        result = self.feedback_manager.record_playlist_feedback(
            query="after recovery",
            query_type="similarity",
            parsed_data={"artist": "Recovery Artist"},
            generated_tracks=[{"id": 2}],
            user_rating=5,
        )
        self.assertTrue(result, "System should work after error recovery")

        # Validate data consistency
        stats = self.feedback_manager.get_feedback_stats()
        self.assertEqual(
            stats["total_feedback"],
            2.0,
            "Should have 2 feedback records after recovery",
        )


if __name__ == "__main__":
    unittest.main()
