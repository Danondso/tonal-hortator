#!/usr/bin/env python3
"""
Feedback Manager for Tonal Hortator

Handles user feedback collection, preference management, and learning
to improve playlist generation over time.
"""

import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, cast

from tonal_hortator.core.database import (
    CREATE_QUERY_LEARNING_TABLE,
    CREATE_TRACK_RATINGS_TABLE,
    CREATE_USER_FEEDBACK_TABLE,
    CREATE_USER_PREFERENCES_TABLE,
    GET_QUERY_LEARNING_DATA,
    GET_TRACK_RATINGS,
    GET_USER_FEEDBACK_STATS,
    GET_USER_PREFERENCES,
    INSERT_QUERY_LEARNING,
    INSERT_TRACK_RATING,
    INSERT_USER_FEEDBACK,
    INSERT_USER_PREFERENCE,
    UPDATE_QUERY_LEARNING_APPLIED,
)

logger = logging.getLogger(__name__)


class FeedbackManager:
    """Manages user feedback, preferences, and learning for playlist generation"""

    def __init__(self, db_path: str = "music_library.db"):
        """
        Initialize the feedback manager

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._ensure_tables_exist()

    def _ensure_tables_exist(self) -> None:
        """Ensure all feedback-related tables exist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create tables if they don't exist
                cursor.execute(CREATE_USER_FEEDBACK_TABLE)
                cursor.execute(CREATE_USER_PREFERENCES_TABLE)
                cursor.execute(CREATE_TRACK_RATINGS_TABLE)
                cursor.execute(CREATE_QUERY_LEARNING_TABLE)

                conn.commit()
                logger.info("✅ Feedback tables ensured")

        except sqlite3.Error as e:
            logger.error(f"❌ Error creating feedback tables: {e}")
            raise

    def record_playlist_feedback(
        self,
        query: str,
        query_type: str,
        parsed_data: Dict[str, Any],
        generated_tracks: List[Dict[str, Any]],
        user_rating: Optional[int] = None,
        user_comments: Optional[str] = None,
        user_actions: Optional[List[str]] = None,
        playlist_length: Optional[int] = None,
        requested_length: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        search_breadth: Optional[int] = None,
    ) -> bool:
        """
        Record feedback for a generated playlist

        Args:
            query: Original user query
            query_type: Type of query (artist_specific, similarity, general)
            parsed_data: LLM parsed data
            generated_tracks: List of generated tracks
            user_rating: User rating (1-5 stars)
            user_comments: User comments
            user_actions: List of user actions (skip, like, dislike, etc.)
            playlist_length: Actual playlist length
            requested_length: Requested playlist length
            similarity_threshold: Similarity threshold used
            search_breadth: Search breadth factor used

        Returns:
            True if feedback was recorded successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Extract parsed data
                parsed_artist = parsed_data.get("artist")
                parsed_reference_artist = parsed_data.get("reference_artist")
                parsed_genres = json.dumps(parsed_data.get("genres", []))
                parsed_mood = parsed_data.get("mood")

                # Extract track IDs
                track_ids = [
                    track.get("id") for track in generated_tracks if track.get("id")
                ]
                generated_tracks_json = json.dumps(track_ids)

                # Prepare user actions
                user_actions_json = json.dumps(user_actions or [])

                cursor.execute(
                    INSERT_USER_FEEDBACK,
                    (
                        query,
                        query_type,
                        parsed_artist,
                        parsed_reference_artist,
                        parsed_genres,
                        parsed_mood,
                        generated_tracks_json,
                        user_rating,
                        user_comments,
                        user_actions_json,
                        playlist_length,
                        requested_length,
                        similarity_threshold,
                        search_breadth,
                    ),
                )

                conn.commit()
                logger.info(f"✅ Recorded feedback for query: {query}")
                return True

        except sqlite3.Error as e:
            logger.error(f"❌ Error recording feedback: {e}")
            return False

    def record_track_rating(
        self, track_id: int, rating: int, context: Optional[str] = None
    ) -> bool:
        """
        Record a rating for a specific track

        Args:
            track_id: Track ID
            rating: Rating (1-5 stars)
            context: Context where rating was given (e.g., playlist name)

        Returns:
            True if rating was recorded successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(INSERT_TRACK_RATING, (track_id, rating, context))
                conn.commit()

                logger.info(f"✅ Recorded rating {rating} for track {track_id}")
                return True

        except sqlite3.Error as e:
            logger.error(f"❌ Error recording track rating: {e}")
            return False

    def set_preference(
        self,
        key: str,
        value: Any,
        preference_type: str,
        description: Optional[str] = None,
    ) -> bool:
        """
        Set a user preference

        Args:
            key: Preference key
            value: Preference value
            preference_type: Type (string, integer, float, boolean, json)
            description: Optional description

        Returns:
            True if preference was set successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Convert value to string based on type
                if preference_type == "json" and not isinstance(value, str):
                    value_str = json.dumps(value)
                else:
                    value_str = str(value)

                cursor.execute(
                    INSERT_USER_PREFERENCE,
                    (key, value_str, preference_type, description),
                )
                conn.commit()

                logger.info(f"✅ Set preference {key} = {value}")
                return True

        except sqlite3.Error as e:
            logger.error(f"❌ Error setting preference: {e}")
            return False

    def get_preference(self, key: str, default: Any = None) -> Any:
        """
        Get a user preference

        Args:
            key: Preference key
            default: Default value if preference doesn't exist

        Returns:
            Preference value or default
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT preference_value, preference_type FROM user_preferences WHERE preference_key = ?",
                    (key,),
                )
                result = cursor.fetchone()

                if not result:
                    return default

                value_str, preference_type = result

                # Convert back to original type
                if preference_type == "integer":
                    return int(value_str)
                elif preference_type == "float":
                    return float(value_str)
                elif preference_type == "boolean":
                    return value_str.lower() == "true"
                elif preference_type == "json":
                    return json.loads(value_str)
                else:
                    return value_str

        except sqlite3.Error as e:
            logger.error(f"❌ Error getting preference: {e}")
            return default

    def record_query_learning(
        self,
        original_query: str,
        llm_parsed_result: Dict[str, Any],
        user_correction: Optional[Dict[str, Any]] = None,
        feedback_score: Optional[float] = None,
    ) -> bool:
        """
        Record query learning data for improving LLM prompts

        Args:
            original_query: Original user query
            llm_parsed_result: What the LLM parsed
            user_correction: What the user expected (if provided)
            feedback_score: Feedback score (-1 to 1)

        Returns:
            True if learning data was recorded successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    INSERT_QUERY_LEARNING,
                    (
                        original_query,
                        json.dumps(llm_parsed_result),
                        json.dumps(user_correction) if user_correction else None,
                        feedback_score,
                    ),
                )
                conn.commit()

                logger.info(f"✅ Recorded query learning for: {original_query}")
                return True

        except sqlite3.Error as e:
            logger.error(f"❌ Error recording query learning: {e}")
            return False

    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get feedback statistics

        Returns:
            Dictionary with feedback statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                stats: Dict[str, Any] = {}

                # Get basic stats
                for stat_name, query in GET_USER_FEEDBACK_STATS.items():
                    if stat_name == "feedback_by_type":
                        cursor.execute(query)
                        stats[stat_name] = cast(Any, dict(cursor.fetchall()))
                    elif stat_name == "recent_feedback":
                        cursor.execute(query)
                        stats[stat_name] = cast(Any, cursor.fetchall())
                    else:
                        cursor.execute(query)
                        result = cursor.fetchone()
                        stats[stat_name] = cast(
                            Any,
                            (
                                float(result[0])
                                if result and result[0] is not None
                                else 0.0
                            ),
                        )

                return stats

        except sqlite3.Error as e:
            logger.error(f"❌ Error getting feedback stats: {e}")
            return {}

    def get_user_preferences(self) -> List[Tuple[str, Any, str, str]]:
        """
        Get all user preferences

        Returns:
            List of (key, value, type, description) tuples
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(GET_USER_PREFERENCES)
                results = cursor.fetchall()

                preferences: List[Tuple[str, Any, str, str]] = []
                for key, value_str, preference_type, description in results:
                    # Convert value back to original type
                    if preference_type == "integer":
                        value = int(value_str)
                    elif preference_type == "float":
                        value = int(float(value_str))
                    elif preference_type == "boolean":
                        value = value_str.lower() == "true"
                    elif preference_type == "json":
                        value = json.loads(value_str)
                    else:
                        value = value_str

                    preferences.append((key, value, preference_type, description))

                return preferences

        except sqlite3.Error as e:
            logger.error(f"❌ Error getting user preferences: {e}")
            return []

    def get_track_ratings(self) -> List[Tuple[int, str, str, str]]:
        """
        Get all track ratings

        Returns:
            List of (rating, context, track_name, artist) tuples
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(GET_TRACK_RATINGS)
                return cursor.fetchall()

        except sqlite3.Error as e:
            logger.error(f"❌ Error getting track ratings: {e}")
            return []

    def get_learning_data(
        self,
    ) -> List[Tuple[str, Dict[str, Any], Optional[Dict[str, Any]], Optional[float]]]:
        """
        Get query learning data that hasn't been applied yet

        Returns:
            List of (query, llm_result, user_correction, feedback_score) tuples
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(GET_QUERY_LEARNING_DATA)
                results = cursor.fetchall()

                learning_data = []
                for (
                    query,
                    llm_result_json,
                    user_correction_json,
                    feedback_score,
                ) in results:
                    llm_result = json.loads(llm_result_json)
                    user_correction = (
                        json.loads(user_correction_json)
                        if user_correction_json
                        else None
                    )

                    learning_data.append(
                        (query, llm_result, user_correction, feedback_score)
                    )

                return learning_data

        except sqlite3.Error as e:
            logger.error(f"❌ Error getting learning data: {e}")
            return []

    def mark_learning_applied(self, learning_id: int) -> bool:
        """
        Mark learning data as applied

        Args:
            learning_id: ID of the learning record

        Returns:
            True if marked successfully
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(UPDATE_QUERY_LEARNING_APPLIED, (learning_id,))
                conn.commit()

                logger.info(f"✅ Marked learning {learning_id} as applied")
                return True

        except sqlite3.Error as e:
            logger.error(f"❌ Error marking learning as applied: {e}")
            return False

    def get_recommended_settings(self, query_type: str) -> Dict[str, Any]:
        """
        Get recommended settings based on user feedback

        Args:
            query_type: Type of query (artist_specific, similarity, general)

        Returns:
            Dictionary with recommended settings
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get average ratings for different settings
                cursor.execute(
                    """
                    SELECT
                        AVG(similarity_threshold) as avg_similarity,
                        AVG(search_breadth) as avg_breadth,
                        AVG(user_rating) as avg_rating,
                        COUNT(*) as feedback_count
                    FROM user_feedback
                    WHERE query_type = ? AND user_rating IS NOT NULL
                    """,
                    (query_type,),
                )

                result = cursor.fetchone()
                if not result or result[3] == 0:  # No feedback
                    return {}

                avg_similarity, avg_breadth, avg_rating, feedback_count = result

                return {
                    "similarity_threshold": (
                        round(avg_similarity, 2) if avg_similarity else 0.3
                    ),
                    "search_breadth": round(avg_breadth) if avg_breadth else 15,
                    "average_rating": round(avg_rating, 1) if avg_rating else 0,
                    "feedback_count": feedback_count,
                }

        except sqlite3.Error as e:
            logger.error(f"❌ Error getting recommended settings: {e}")
            return {}

    def get_adjusted_score(self, track_id: int, track_meta: dict) -> float:
        """
        Combine feedback ratings and iTunes play metadata to compute an adjusted score.
        """
        feedback_adj = self._get_feedback_rating_score(track_id)
        itunes_adj = self._get_itunes_score(track_meta)
        return max(-0.2, min(0.2, feedback_adj + itunes_adj))

    def _get_feedback_rating_score(self, track_id: int) -> float:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT rating FROM track_ratings WHERE track_id = ?",
                    (track_id,),
                )
                ratings = [row[0] for row in cursor.fetchall()]

                if not ratings:
                    return 0.0

                avg_rating = sum(ratings) / len(ratings)

                if avg_rating >= 4.5:
                    return 0.15
                elif avg_rating >= 4.0:
                    return 0.1
                elif avg_rating >= 3.0:
                    return 0.05
                elif avg_rating <= 1.5:
                    return -0.15
                elif avg_rating <= 2.0:
                    return -0.1
                else:
                    return 0.0

        except sqlite3.Error as e:
            logger.error(f"❌ Error getting feedback rating score: {e}")
            return 0.0

    def _get_itunes_score(self, track: dict) -> float:
        score = 0.0
        play_count = track.get("play_count", 0)
        rating = track.get("rating", 0)
        skip_count = track.get("skip_count", 0)

        if play_count > 50:
            score += 0.1
        elif play_count > 20:
            score += 0.05

        if rating >= 80:
            score += 0.1
        elif rating >= 60:
            score += 0.05

        if skip_count > 25:
            score -= 0.1
        elif skip_count > 10:
            score -= 0.05

        return score

    def record_user_feedback(
        self, track_id: str, feedback: str, query_context: str = ""
    ) -> None:
        import sqlite3
        from datetime import datetime

        feedback_map = {"like": 0.2, "dislike": -0.2, "block": -1.0}
        adjustment = feedback_map.get(feedback, 0.0)
        timestamp = datetime.now().isoformat()

        conn = sqlite3.connect("feedback.db")
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                track_id TEXT,
                feedback TEXT,
                adjustment REAL,
                timestamp TEXT,
                query_context TEXT,
                source TEXT DEFAULT 'user'
            )
        """
        )
        cur.execute(
            """
            INSERT INTO feedback (track_id, feedback, adjustment, timestamp, query_context, source)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (track_id, feedback, adjustment, timestamp, query_context, "user"),
        )
        conn.commit()
        conn.close()
