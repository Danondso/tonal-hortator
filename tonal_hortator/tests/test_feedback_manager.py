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
        """Test successful recording of playlist feedback"""
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

        self.assertTrue(result)

        # Verify data was stored
        self.cursor.execute("SELECT * FROM user_feedback WHERE query = ?", (query,))
        row = self.cursor.fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row[1], query)  # query
        self.assertEqual(row[2], query_type)  # query_type
        self.assertEqual(row[3], None)  # parsed_artist
        self.assertEqual(row[4], "Queen")  # parsed_reference_artist
        self.assertEqual(row[5], '["Rock", "Progressive Rock"]')  # parsed_genres
        self.assertEqual(row[6], "epic")  # parsed_mood
        self.assertEqual(row[7], "[1, 2]")  # generated_tracks
        self.assertEqual(row[8], 4)  # user_rating
        self.assertEqual(row[9], "Great playlist!")  # user_comments
        self.assertEqual(row[10], '["like", "skip"]')  # user_actions
        self.assertEqual(row[11], 2)  # playlist_length
        self.assertEqual(row[12], 3)  # requested_length
        self.assertEqual(row[13], 0.8)  # similarity_threshold
        self.assertEqual(row[14], 20)  # search_breadth

    def test_record_playlist_feedback_minimal_data(self) -> None:
        """Test recording feedback with minimal required data"""
        result = self.feedback_manager.record_playlist_feedback(
            query="test query",
            query_type="general",
            parsed_data={},
            generated_tracks=[],
        )

        self.assertTrue(result)

    def test_record_playlist_feedback_database_error(self) -> None:
        """Test handling of database errors during feedback recording"""
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Database error")

            result = self.feedback_manager.record_playlist_feedback(
                query="test", query_type="general", parsed_data={}, generated_tracks=[]
            )

            self.assertFalse(result)

    def test_record_track_rating_success(self) -> None:
        """Test successful recording of track rating"""
        result = self.feedback_manager.record_track_rating(
            track_id=1, rating=5, context="playlist: Summer Vibes"
        )

        self.assertTrue(result)

        # Verify rating was stored
        self.cursor.execute("SELECT * FROM track_ratings WHERE track_id = ?", (1,))
        row = self.cursor.fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row[1], 1)  # track_id
        self.assertEqual(row[2], 5)  # rating
        self.assertEqual(row[3], "playlist: Summer Vibes")  # context

    def test_record_track_rating_database_error(self) -> None:
        """Test handling of database errors during track rating"""
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Database error")

            result = self.feedback_manager.record_track_rating(
                track_id=1, rating=5, context="test"
            )

            self.assertFalse(result)

    def test_set_preference_success(self) -> None:
        """Test successful setting of user preferences"""
        # Test different preference types
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

            self.assertTrue(result)

            # Verify preference was stored
            self.cursor.execute(
                "SELECT preference_value, preference_type, description FROM user_preferences WHERE preference_key = ?",
                (key,),
            )
            row = self.cursor.fetchone()

            self.assertIsNotNone(row)
            if pref_type == "json":
                self.assertEqual(json.loads(row[0]), value)
            elif pref_type == "boolean":
                self.assertEqual(row[0].lower(), str(value).lower())
            else:
                self.assertEqual(row[0], str(value))
            self.assertEqual(row[1], pref_type)
            self.assertEqual(row[2], description)

    def test_set_preference_database_error(self) -> None:
        """Test handling of database errors during preference setting"""
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Database error")

            result = self.feedback_manager.set_preference(
                key="test", value="value", preference_type="string"
            )

            self.assertFalse(result)

    def test_get_preference_success(self) -> None:
        """Test successful retrieval of user preferences"""
        # Set up test preferences
        self.feedback_manager.set_preference("theme", "dark", "string")
        self.feedback_manager.set_preference("max_length", 25, "integer")
        self.feedback_manager.set_preference("threshold", 0.8, "float")
        self.feedback_manager.set_preference("auto_save", True, "boolean")
        self.feedback_manager.set_preference("genres", ["Rock", "Pop"], "json")

        # Test retrieval
        self.assertEqual(self.feedback_manager.get_preference("theme"), "dark")
        self.assertEqual(self.feedback_manager.get_preference("max_length"), 25)
        self.assertEqual(self.feedback_manager.get_preference("threshold"), 0.8)
        self.assertEqual(self.feedback_manager.get_preference("auto_save"), True)
        self.assertEqual(
            self.feedback_manager.get_preference("genres"), ["Rock", "Pop"]
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

        self.assertTrue(result)

        # Verify data was stored
        self.cursor.execute(
            "SELECT * FROM query_learning WHERE original_query = ?", (original_query,)
        )
        row = self.cursor.fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row[1], original_query)  # original_query
        self.assertEqual(json.loads(row[2]), llm_parsed_result)  # llm_parsed_result
        self.assertEqual(json.loads(row[3]), user_correction)  # user_correction
        self.assertEqual(row[4], 0.8)  # feedback_score
        self.assertEqual(row[5], False)  # learning_applied

    def test_record_query_learning_minimal_data(self) -> None:
        """Test recording query learning with minimal data"""
        result = self.feedback_manager.record_query_learning(
            original_query="test query", llm_parsed_result={}
        )

        self.assertTrue(result)

    def test_record_query_learning_database_error(self) -> None:
        """Test handling of database errors during query learning recording"""
        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = sqlite3.Error("Database error")

            result = self.feedback_manager.record_query_learning(
                original_query="test", llm_parsed_result={}
            )

            self.assertFalse(result)

    def test_get_feedback_stats_success(self) -> None:
        """Test successful retrieval of feedback statistics"""
        # Set up test feedback data
        self.feedback_manager.record_playlist_feedback(
            query="test query 1",
            query_type="artist_specific",
            parsed_data={"artist": "Artist 1"},
            generated_tracks=[{"id": 1}],
            user_rating=4,
        )

        self.feedback_manager.record_playlist_feedback(
            query="test query 2",
            query_type="similarity",
            parsed_data={"artist": "Artist 2"},
            generated_tracks=[{"id": 2}],
            user_rating=5,
        )

        stats = self.feedback_manager.get_feedback_stats()

        self.assertIsInstance(stats, dict)
        self.assertIn("total_feedback", stats)
        self.assertIn("average_rating", stats)
        self.assertIn("feedback_by_type", stats)
        self.assertIn("recent_feedback", stats)

        self.assertEqual(stats["total_feedback"], 2.0)
        self.assertEqual(stats["average_rating"], 4.5)
        self.assertIsInstance(stats["feedback_by_type"], dict)
        self.assertIsInstance(stats["recent_feedback"], list)

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


if __name__ == "__main__":
    unittest.main()
