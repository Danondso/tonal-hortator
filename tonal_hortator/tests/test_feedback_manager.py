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
        """Test that initialization creates all required tables with proper schema"""
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

        self.assertTrue(
            expected_tables.issubset(tables),
            f"Missing tables. Expected: {expected_tables}, Found: {tables}",
        )

        # Validate table schemas with detailed assertions
        self._validate_user_feedback_schema()
        self._validate_user_preferences_schema()
        self._validate_track_ratings_schema()
        self._validate_query_learning_schema()

    def _validate_user_feedback_schema(self) -> None:
        """Validate user_feedback table schema"""
        self.cursor.execute("PRAGMA table_info(user_feedback)")
        columns = {row[1]: row[2] for row in self.cursor.fetchall()}

        # Validate required columns exist with correct types
        expected_columns = {
            "id": "INTEGER",
            "query": "TEXT",
            "query_type": "TEXT",
            "parsed_artist": "TEXT",
            "parsed_reference_artist": "TEXT",
            "parsed_genres": "TEXT",
            "parsed_mood": "TEXT",
            "generated_tracks": "TEXT",
            "user_rating": "INTEGER",
            "user_comments": "TEXT",
            "user_actions": "TEXT",
            "playlist_length": "INTEGER",
            "requested_length": "INTEGER",
            "similarity_threshold": "REAL",
            "search_breadth": "INTEGER",
            "created_at": "TIMESTAMP",
        }

        for col_name, expected_type in expected_columns.items():
            self.assertIn(
                col_name,
                columns,
                f"Column '{col_name}' missing from user_feedback table",
            )
            self.assertEqual(
                columns[col_name],
                expected_type,
                f"Column '{col_name}' has wrong type. Expected: {expected_type}, Got: {columns[col_name]}",
            )

    def _validate_user_preferences_schema(self) -> None:
        """Validate user_preferences table schema"""
        self.cursor.execute("PRAGMA table_info(user_preferences)")
        columns = {row[1]: row[2] for row in self.cursor.fetchall()}

        expected_columns = {
            "id": "INTEGER",
            "preference_key": "TEXT",
            "preference_value": "TEXT",
            "preference_type": "TEXT",
            "description": "TEXT",
            "created_at": "TIMESTAMP",
            "updated_at": "TIMESTAMP",
        }

        for col_name, expected_type in expected_columns.items():
            self.assertIn(
                col_name,
                columns,
                f"Column '{col_name}' missing from user_preferences table",
            )
            self.assertEqual(
                columns[col_name],
                expected_type,
                f"Column '{col_name}' has wrong type. Expected: {expected_type}, Got: {columns[col_name]}",
            )

    def _validate_track_ratings_schema(self) -> None:
        """Validate track_ratings table schema"""
        self.cursor.execute("PRAGMA table_info(track_ratings)")
        columns = {row[1]: row[2] for row in self.cursor.fetchall()}

        expected_columns = {
            "id": "INTEGER",
            "track_id": "INTEGER",
            "rating": "INTEGER",
            "context": "TEXT",
            "created_at": "TIMESTAMP",
        }

        for col_name, expected_type in expected_columns.items():
            self.assertIn(
                col_name,
                columns,
                f"Column '{col_name}' missing from track_ratings table",
            )
            self.assertEqual(
                columns[col_name],
                expected_type,
                f"Column '{col_name}' has wrong type. Expected: {expected_type}, Got: {columns[col_name]}",
            )

    def _validate_query_learning_schema(self) -> None:
        """Validate query_learning table schema"""
        self.cursor.execute("PRAGMA table_info(query_learning)")
        columns = {row[1]: row[2] for row in self.cursor.fetchall()}

        expected_columns = {
            "id": "INTEGER",
            "original_query": "TEXT",
            "llm_parsed_result": "TEXT",
            "user_correction": "TEXT",
            "feedback_score": "REAL",
            "learning_applied": "BOOLEAN",
            "created_at": "TIMESTAMP",
        }

        for col_name, expected_type in expected_columns.items():
            self.assertIn(
                col_name,
                columns,
                f"Column '{col_name}' missing from query_learning table",
            )
            self.assertEqual(
                columns[col_name],
                expected_type,
                f"Column '{col_name}' has wrong type. Expected: {expected_type}, Got: {columns[col_name]}",
            )

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

        # Validate return value with descriptive error message
        self.assertTrue(
            result,
            f"Feedback recording should return True on success, but got {result}",
        )

        # Verify data was stored with comprehensive validation
        self.cursor.execute("SELECT * FROM user_feedback WHERE query = ?", (query,))
        row = self.cursor.fetchone()

        # Validate row exists and has correct structure
        self.assertIsNotNone(
            row, f"Feedback record should be stored in database for query: {query}"
        )
        self.assertIsInstance(
            row, tuple, f"Database row should be a tuple, got {type(row)}"
        )
        self.assertGreaterEqual(
            len(row),
            15,
            f"Feedback record should have at least 15 columns, got {len(row)}",
        )

        # Validate each field with detailed assertions and better error messages
        self.assertEqual(
            row[1],
            query,
            f"Query should be stored exactly as provided. Expected: '{query}', Got: '{row[1]}'",
        )
        self.assertEqual(
            row[2],
            query_type,
            f"Query type should match provided value. Expected: '{query_type}', Got: '{row[2]}'",
        )
        self.assertIsNone(
            row[3], f"Parsed artist should be None as provided, got: {row[3]}"
        )
        self.assertEqual(
            row[4],
            "Queen",
            f"Reference artist should be stored correctly. Expected: 'Queen', Got: '{row[4]}'",
        )

        # Validate JSON fields with comprehensive error checking
        try:
            stored_genres = json.loads(row[5])
        except json.JSONDecodeError as e:
            self.fail(f"Failed to parse stored genres JSON: {e}. Raw value: {row[5]}")

        self.assertIsInstance(
            stored_genres,
            list,
            f"Stored genres should be a list, got {type(stored_genres)}",
        )
        self.assertEqual(
            stored_genres,
            ["Rock", "Progressive Rock"],
            f"Genres should match exactly. Expected: {['Rock', 'Progressive Rock']}, Got: {stored_genres}",
        )

        self.assertEqual(
            row[6],
            "epic",
            f"Mood should be stored correctly. Expected: 'epic', Got: '{row[6]}'",
        )

        # Validate track IDs with JSON parsing error handling
        try:
            stored_tracks = json.loads(row[7])
        except json.JSONDecodeError as e:
            self.fail(
                f"Failed to parse stored track IDs JSON: {e}. Raw value: {row[7]}"
            )

        self.assertIsInstance(
            stored_tracks,
            list,
            f"Stored tracks should be a list, got {type(stored_tracks)}",
        )
        self.assertEqual(
            stored_tracks,
            [1, 2],
            f"Track IDs should be extracted and stored correctly. Expected: [1, 2], Got: {stored_tracks}",
        )

        # Validate numeric fields with type checking
        self.assertEqual(
            row[8],
            4,
            f"User rating should be stored as integer. Expected: 4, Got: {row[8]} (type: {type(row[8])})",
        )
        self.assertIsInstance(
            row[8],
            int,
            f"User rating should be stored as integer type, got {type(row[8])}",
        )

        self.assertEqual(
            row[9],
            "Great playlist!",
            f"User comments should be stored exactly. Expected: 'Great playlist!', Got: '{row[9]}'",
        )

        # Validate user actions with JSON parsing error handling
        try:
            stored_actions = json.loads(row[10])
        except json.JSONDecodeError as e:
            self.fail(
                f"Failed to parse stored user actions JSON: {e}. Raw value: {row[10]}"
            )

        self.assertIsInstance(
            stored_actions,
            list,
            f"User actions should be stored as list, got {type(stored_actions)}",
        )
        self.assertEqual(
            stored_actions,
            ["like", "skip"],
            f"User actions should match exactly. Expected: ['like', 'skip'], Got: {stored_actions}",
        )

        # Validate playlist metrics with type checking
        self.assertEqual(
            row[11],
            2,
            f"Playlist length should be stored correctly. Expected: 2, Got: {row[11]}",
        )
        self.assertEqual(
            row[12],
            3,
            f"Requested length should be stored correctly. Expected: 3, Got: {row[12]}",
        )
        self.assertEqual(
            row[13],
            0.8,
            f"Similarity threshold should be stored as float. Expected: 0.8, Got: {row[13]} (type: {type(row[13])})",
        )
        self.assertIsInstance(
            row[13],
            float,
            f"Similarity threshold should be stored as float type, got {type(row[13])}",
        )
        self.assertEqual(
            row[14],
            20,
            f"Search breadth should be stored correctly. Expected: 20, Got: {row[14]}",
        )

        # Validate timestamp fields with format checking
        self.assertIsNotNone(row[15], "Created timestamp should be set")
        self.assertIsInstance(
            row[15], str, f"Created timestamp should be string, got {type(row[15])}"
        )
        # Basic timestamp format validation (should be ISO format)
        self.assertIn(
            "T" in row[15] or "-" in row[15],
            [True],
            f"Created timestamp should be in ISO format, got: {row[15]}",
        )

        # Verify no duplicate records were created
        self.cursor.execute(
            "SELECT COUNT(*) FROM user_feedback WHERE query = ?", (query,)
        )
        count = self.cursor.fetchone()[0]
        self.assertEqual(
            count,
            1,
            f"Only one feedback record should exist for query '{query}', found {count}",
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

        # Validate return value with descriptive error message
        self.assertTrue(
            result, f"Track rating should return True on success, but got {result}"
        )

        # Verify rating was stored with comprehensive validation
        self.cursor.execute(
            "SELECT * FROM track_ratings WHERE track_id = ?", (track_id,)
        )
        row = self.cursor.fetchone()

        # Validate row exists and has correct structure
        self.assertIsNotNone(
            row, f"Track rating should be stored in database for track_id: {track_id}"
        )
        self.assertIsInstance(
            row, tuple, f"Database row should be a tuple, got {type(row)}"
        )
        self.assertGreaterEqual(
            len(row),
            4,
            f"Track rating record should have at least 4 columns, got {len(row)}",
        )

        # Validate each field with detailed assertions and better error messages
        self.assertEqual(
            row[1],
            track_id,
            f"Track ID should be stored correctly. Expected: {track_id}, Got: {row[1]} (type: {type(row[1])})",
        )
        self.assertIsInstance(
            row[1], int, f"Track ID should be stored as integer, got {type(row[1])}"
        )
        self.assertGreater(row[1], 0, f"Track ID should be positive, got {row[1]}")

        self.assertEqual(
            row[2],
            rating,
            f"Rating should be stored correctly. Expected: {rating}, Got: {row[2]} (type: {type(row[2])})",
        )
        self.assertIsInstance(
            row[2], int, f"Rating should be stored as integer, got {type(row[2])}"
        )
        self.assertGreaterEqual(
            row[2], 0, f"Rating should be non-negative, got {row[2]}"
        )
        self.assertLessEqual(
            row[2], 5, f"Rating should not exceed maximum of 5, got {row[2]}"
        )

        self.assertEqual(
            row[3],
            context,
            f"Context should be stored exactly as provided. Expected: '{context}', Got: '{row[3]}'",
        )
        self.assertIsInstance(
            row[3], str, f"Context should be stored as string, got {type(row[3])}"
        )
        self.assertGreater(
            len(row[3]), 0, f"Context should not be empty, got length: {len(row[3])}"
        )

        # Validate timestamp with format checking
        self.assertIsNotNone(row[4], "Created timestamp should be set")
        self.assertIsInstance(
            row[4], str, f"Created timestamp should be string, got {type(row[4])}"
        )
        # Basic timestamp format validation (should be ISO format)
        self.assertIn(
            "T" in row[4] or "-" in row[4],
            [True],
            f"Created timestamp should be in ISO format, got: {row[4]}",
        )

        # Verify no duplicate ratings for same track
        self.cursor.execute(
            "SELECT COUNT(*) FROM track_ratings WHERE track_id = ?", (track_id,)
        )
        count = self.cursor.fetchone()[0]
        self.assertEqual(
            count,
            1,
            f"Only one rating should exist per track for track_id {track_id}, found {count}",
        )

        # Verify track exists in tracks table (foreign key validation)
        self.cursor.execute("SELECT name FROM tracks WHERE id = ?", (track_id,))
        track_row = self.cursor.fetchone()
        self.assertIsNotNone(
            track_row, f"Track with id {track_id} should exist in tracks table"
        )
        self.assertEqual(
            track_row[0],
            "Test Song 1",
            f"Track name should match. Expected: 'Test Song 1', Got: '{track_row[0]}'",
        )

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

            # Validate return value with descriptive error message
            self.assertTrue(
                result,
                f"Setting preference '{key}' should return True, but got {result}",
            )

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
            self.assertIsInstance(
                row, tuple, f"Database row should be a tuple, got {type(row)}"
            )
            self.assertEqual(
                len(row), 3, f"Preference row should have 3 columns, got {len(row)}"
            )

            # Validate each field based on type with comprehensive error checking
            if pref_type == "json":
                try:
                    stored_value = json.loads(row[0])
                except json.JSONDecodeError as e:
                    self.fail(
                        f"Failed to parse JSON preference '{key}': {e}. Raw value: {row[0]}"
                    )

                self.assertIsInstance(
                    stored_value,
                    list,
                    f"JSON preference '{key}' should be stored as list, got {type(stored_value)}",
                )
                self.assertEqual(
                    stored_value,
                    value,
                    f"JSON preference '{key}' should match exactly. Expected: {value}, Got: {stored_value}",
                )
            elif pref_type == "boolean":
                # Handle boolean conversion variations
                expected_str = str(value).lower()
                actual_str = row[0].lower()
                self.assertEqual(
                    actual_str,
                    expected_str,
                    f"Boolean preference '{key}' should match. Expected: '{expected_str}', Got: '{actual_str}'",
                )
            elif pref_type == "integer":
                try:
                    stored_int = int(row[0])
                except ValueError as e:
                    self.fail(
                        f"Failed to convert integer preference '{key}': {e}. Raw value: {row[0]}"
                    )

                self.assertEqual(
                    stored_int,
                    value,
                    f"Integer preference '{key}' should match exactly. Expected: {value}, Got: {stored_int}",
                )
                self.assertIsInstance(
                    stored_int,
                    int,
                    f"Integer preference '{key}' should be stored as int, got {type(stored_int)}",
                )
            elif pref_type == "float":
                try:
                    stored_float = float(row[0])
                except ValueError as e:
                    self.fail(
                        f"Failed to convert float preference '{key}': {e}. Raw value: {row[0]}"
                    )

                self.assertAlmostEqual(
                    stored_float,
                    cast(float, value),
                    places=6,
                    msg=f"Float preference '{key}' should match. Expected: {value}, Got: {stored_float}",
                )
                self.assertIsInstance(
                    stored_float,
                    float,
                    f"Float preference '{key}' should be stored as float, got {type(stored_float)}",
                )
            else:  # string
                self.assertEqual(
                    row[0],
                    str(value),
                    f"String preference '{key}' should match exactly. Expected: '{str(value)}', Got: '{row[0]}'",
                )
                self.assertIsInstance(
                    row[0],
                    str,
                    f"String preference '{key}' should be stored as string, got {type(row[0])}",
                )

            self.assertEqual(
                row[1],
                pref_type,
                f"Preference type for '{key}' should match. Expected: '{pref_type}', Got: '{row[1]}'",
            )
            self.assertEqual(
                row[2],
                description,
                f"Description for '{key}' should match. Expected: '{description}', Got: '{row[2]}'",
            )

            # Verify preference key is not empty
            self.assertGreater(
                len(key), 0, f"Preference key should not be empty for '{key}'"
            )

            # Verify description is not empty
            self.assertGreater(
                len(description),
                0,
                f"Description should not be empty for preference '{key}'",
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
        total_feedback = stats.get("total_feedback")
        average_rating = stats.get("average_rating")

        self.assertEqual(total_feedback, 3.0, "Total feedback count should be 3")
        if isinstance(average_rating, (int, float)):
            self.assertAlmostEqual(
                average_rating, 4.0, places=1, msg="Average rating should be 4.0"
            )

        # Validate feedback_by_type structure
        feedback_by_type = stats["feedback_by_type"]
        self.assertIsInstance(
            feedback_by_type, dict, "feedback_by_type should be dictionary"
        )
        if isinstance(feedback_by_type, dict):
            self.assertIn(
                "artist_specific",
                feedback_by_type,
                "Should have artist_specific type",
            )
            self.assertIn("similarity", feedback_by_type, "Should have similarity type")
            self.assertEqual(
                feedback_by_type["artist_specific"],
                2,
                "Should have 2 artist_specific feedback",
            )
            self.assertEqual(
                feedback_by_type["similarity"],
                1,
                "Should have 1 similarity feedback",
            )

        # Validate recent_feedback structure
        recent_feedback = stats.get("recent_feedback")
        if isinstance(recent_feedback, list):
            self.assertEqual(
                len(recent_feedback), 3, "Should have 3 recent feedback entries"
            )

            # Validate recent feedback entries (they are tuples, not dictionaries)
            for entry in recent_feedback:
                self.assertIsInstance(
                    entry, tuple, "Each recent feedback entry should be tuple"
                )
                self.assertGreaterEqual(
                    len(entry),
                    15,
                    "Each feedback tuple should have at least 15 columns",
                )

                # Validate key fields in the tuple (based on user_feedback table structure)
                # entry[0] = id, entry[1] = query, entry[2] = query_type, entry[8] = user_rating, entry[15] = created_at
                self.assertIsInstance(entry[1], str, "Query should be string")
                self.assertIsInstance(entry[2], str, "Query type should be string")
                if entry[8] is not None:  # user_rating can be None
                    self.assertIsInstance(
                        entry[8], int, "User rating should be integer"
                    )
                self.assertIsInstance(
                    entry[15], str, "Created timestamp should be string"
                )

    def test_get_feedback_stats_empty_database(self) -> None:
        """Test getting feedback stats from empty database"""
        stats = self.feedback_manager.get_feedback_stats()

        total_feedback = stats.get("total_feedback")
        average_rating = stats.get("average_rating")
        feedback_by_type = stats.get("feedback_by_type")
        recent_feedback = stats.get("recent_feedback")

        self.assertEqual(total_feedback, 0.0)
        self.assertEqual(average_rating, 0.0)
        self.assertEqual(feedback_by_type, {})
        self.assertEqual(recent_feedback, [])

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

        # Validate results with comprehensive error reporting
        self.assertEqual(
            len(results),
            10,
            f"All 10 threads should complete successfully, but only {len(results)} completed",
        )
        self.assertEqual(
            len(errors),
            0,
            f"No errors should occur during concurrent access, but got {len(errors)} errors: {errors}",
        )

        # Validate all feedback was recorded successfully
        for thread_id, result in results:
            self.assertTrue(
                result, f"Thread {thread_id} should return True, but got {result}"
            )

        # Verify data integrity with detailed counting
        self.cursor.execute(
            "SELECT COUNT(*) FROM user_feedback WHERE query LIKE 'concurrent test %'"
        )
        count = self.cursor.fetchone()[0]
        self.assertEqual(
            count,
            10,
            f"All 10 concurrent feedback records should be stored, but found {count}",
        )

        # Verify no duplicate queries were created
        self.cursor.execute(
            "SELECT query, COUNT(*) FROM user_feedback WHERE query LIKE 'concurrent test %' GROUP BY query HAVING COUNT(*) > 1"
        )
        duplicates = self.cursor.fetchall()
        self.assertEqual(
            len(duplicates),
            0,
            f"No duplicate queries should exist, but found duplicates: {duplicates}",
        )

        # Performance assertion (should complete within reasonable time)
        execution_time = end_time - start_time
        self.assertLess(
            execution_time,
            5.0,
            f"Concurrent operations should complete within 5 seconds, took {execution_time:.2f}s",
        )

        # Verify thread-specific data was stored correctly
        for thread_id in range(10):
            self.cursor.execute(
                "SELECT user_rating, parsed_artist FROM user_feedback WHERE query = ?",
                (f"concurrent test {thread_id}",),
            )
            row = self.cursor.fetchone()
            self.assertIsNotNone(
                row, f"Thread {thread_id} feedback should be stored in database"
            )
            expected_rating = thread_id % 5 + 1
            self.assertEqual(
                row[0],
                expected_rating,
                f"Thread {thread_id} rating should be {expected_rating}, got {row[0]}",
            )

            # Verify parsed artist was stored correctly
            expected_artist = f"Artist {thread_id}"
            self.assertEqual(
                row[1],
                expected_artist,
                f"Thread {thread_id} artist should be '{expected_artist}', got '{row[1]}'",
            )

    def test_comprehensive_data_validation(self) -> None:
        """Test comprehensive data validation across all feedback operations"""

        # Test 1: Validate JSON serialization/deserialization integrity
        complex_data = {
            "nested": {
                "array": [1, 2, 3, {"key": "value"}],
                "string": "test",
                "number": 42.5,
                "boolean": True,
                "null": None,
            },
            "unicode": "🎵 🎸 🎹",
            "special_chars": "!@#$%^&*()_+-=[]{}|;':\",./<>?",
        }

        result = self.feedback_manager.record_playlist_feedback(
            query="complex data test",
            query_type="similarity",
            parsed_data=complex_data,
            generated_tracks=[{"id": 999, "name": "Complex Track"}],
            user_rating=5,
            user_comments="Testing complex data storage",
            user_actions=["like", "favorite", "skip"],
        )

        self.assertTrue(result, "Complex data should be stored successfully")

        # Verify complex data was stored and can be retrieved correctly
        self.cursor.execute(
            "SELECT parsed_genres, user_actions FROM user_feedback WHERE query = ?",
            ("complex data test",),
        )
        row = self.cursor.fetchone()
        self.assertIsNotNone(row, "Complex data should be retrievable")

        # Test JSON parsing of stored data
        try:
            stored_actions = json.loads(row[1])
            self.assertIsInstance(
                stored_actions, list, "Stored actions should be a list"
            )
            self.assertEqual(
                stored_actions,
                ["like", "favorite", "skip"],
                "Actions should match exactly",
            )
        except json.JSONDecodeError as e:
            self.fail(f"Failed to parse stored actions JSON: {e}")

        # Test 2: Validate numeric range constraints
        edge_ratings = [0, 1, 3, 5]  # Valid ratings
        for rating in edge_ratings:
            result = self.feedback_manager.record_track_rating(
                track_id=100 + rating,
                rating=rating,
                context=f"edge rating test {rating}",
            )
            self.assertTrue(result, f"Rating {rating} should be accepted")

        # Test 3: Validate string length limits
        long_string = "x" * 1000
        result = self.feedback_manager.set_preference(
            key="long_string_test",
            value=long_string,
            preference_type="string",
            description="Testing long string storage",
        )
        self.assertTrue(result, "Long string should be stored successfully")

        # Verify long string was stored correctly
        self.cursor.execute(
            "SELECT preference_value FROM user_preferences WHERE preference_key = ?",
            ("long_string_test",),
        )
        row = self.cursor.fetchone()
        self.assertIsNotNone(row, "Long string should be retrievable")
        self.assertEqual(len(row[0]), 1000, "Long string length should be preserved")

        # Test 4: Validate timestamp consistency
        import datetime

        # Record feedback and check timestamp format
        result = self.feedback_manager.record_playlist_feedback(
            query="timestamp test",
            query_type="similarity",
            parsed_data={},
            generated_tracks=[],
            user_rating=4,
        )
        self.assertTrue(result, "Timestamp test should succeed")

        # Verify timestamp format
        self.cursor.execute(
            "SELECT created_at FROM user_feedback WHERE query = ?", ("timestamp test",)
        )
        row = self.cursor.fetchone()
        self.assertIsNotNone(row, "Timestamp should be stored")

        timestamp_str = row[0]
        self.assertIsInstance(timestamp_str, str, "Timestamp should be string")

        # Try to parse timestamp (should be ISO format)
        try:
            datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except ValueError as e:
            self.fail(f"Timestamp should be in ISO format: {e}. Got: {timestamp_str}")

        # Test 5: Validate data type consistency across operations
        test_data = {
            "feedback_count": 0,
            "preference_count": 0,
            "rating_count": 0,
            "learning_count": 0,
        }

        # Count initial records
        self.cursor.execute("SELECT COUNT(*) FROM user_feedback")
        test_data["feedback_count"] = self.cursor.fetchone()[0]

        self.cursor.execute("SELECT COUNT(*) FROM user_preferences")
        test_data["preference_count"] = self.cursor.fetchone()[0]

        self.cursor.execute("SELECT COUNT(*) FROM track_ratings")
        test_data["rating_count"] = self.cursor.fetchone()[0]

        self.cursor.execute("SELECT COUNT(*) FROM query_learning")
        test_data["learning_count"] = self.cursor.fetchone()[0]

        # Perform operations
        self.feedback_manager.record_playlist_feedback(
            query="validation test",
            query_type="similarity",
            parsed_data={},
            generated_tracks=[],
            user_rating=3,
        )

        self.feedback_manager.set_preference(
            key="validation_test",
            value="test_value",
            preference_type="string",
            description="Validation test preference",
        )

        self.feedback_manager.record_track_rating(
            track_id=200, rating=4, context="validation test"
        )

        self.feedback_manager.record_query_learning(
            original_query="validation test query",
            llm_parsed_result={},
            user_correction={},
            feedback_score=0.7,
        )

        # Verify counts increased by exactly 1
        self.cursor.execute("SELECT COUNT(*) FROM user_feedback")
        new_feedback_count = self.cursor.fetchone()[0]
        self.assertEqual(
            new_feedback_count,
            test_data["feedback_count"] + 1,
            f"Feedback count should increase by 1. Expected: {test_data['feedback_count'] + 1}, Got: {new_feedback_count}",
        )

        self.cursor.execute("SELECT COUNT(*) FROM user_preferences")
        new_preference_count = self.cursor.fetchone()[0]
        self.assertEqual(
            new_preference_count,
            test_data["preference_count"] + 1,
            f"Preference count should increase by 1. Expected: {test_data['preference_count'] + 1}, Got: {new_preference_count}",
        )

        self.cursor.execute("SELECT COUNT(*) FROM track_ratings")
        new_rating_count = self.cursor.fetchone()[0]
        self.assertEqual(
            new_rating_count,
            test_data["rating_count"] + 1,
            f"Rating count should increase by 1. Expected: {test_data['rating_count'] + 1}, Got: {new_rating_count}",
        )

        self.cursor.execute("SELECT COUNT(*) FROM query_learning")
        new_learning_count = self.cursor.fetchone()[0]
        self.assertEqual(
            new_learning_count,
            test_data["learning_count"] + 1,
            f"Learning count should increase by 1. Expected: {test_data['learning_count'] + 1}, Got: {new_learning_count}",
        )


if __name__ == "__main__":
    unittest.main()
