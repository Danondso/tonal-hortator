#!/usr/bin/env python3
"""
Feedback service for playlist generation

Provides interfaces and implementations for handling user feedback
during playlist generation with persistent storage and score adjustment.
"""

import os
import sqlite3
from datetime import datetime
from typing import Protocol

from tonal_hortator.core.config import get_config
from tonal_hortator.core.database import GET_FEEDBACK_BY_TRACK_ID


class FeedbackService(Protocol):
    """Protocol defining the interface for feedback services"""

    def record_user_feedback(
        self, track_id: str, feedback: str, query_context: str = ""
    ) -> None:
        """Record user feedback for a track"""
        ...

    def get_adjusted_score(self, track_id: str, track: dict) -> float:
        """Get adjusted score for a track based on feedback"""
        ...


class PlaylistFeedbackService:
    """Feedback service specifically for playlist generation with persistent storage"""

    def __init__(self, db_path: str = "feedback.db"):
        self.db_path = db_path
        self.config = get_config()
        self._ensure_feedback_table()

    def _ensure_feedback_table(self) -> None:
        """Ensure the feedback table exists"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                track_id TEXT,
                feedback TEXT,
                adjustment REAL,
                timestamp TEXT,
                query_context TEXT
            )
        """
        )
        conn.commit()
        conn.close()

    def record_user_feedback(
        self, track_id: str, feedback: str, query_context: str = ""
    ) -> None:
        """Record user feedback for a track with persistent storage"""
        # Get feedback adjustments from configuration
        feedback_adjustments = self.config.feedback_adjustments
        adjustment = feedback_adjustments.get(feedback, 0.0)
        timestamp = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO feedback (track_id, feedback, adjustment, timestamp, query_context)
            VALUES (?, ?, ?, ?, ?)
        """,
            (track_id, feedback, adjustment, timestamp, query_context),
        )
        conn.commit()
        conn.close()

    def get_adjusted_score(self, track_id: str, track: dict) -> float:
        """Calculate adjusted similarity score based on user feedback with time decay"""
        original_score = track.get("similarity_score", 0)

        # If no feedback DB, return original score
        if not os.path.exists(self.db_path):
            return 0.0

        # Get time decay configuration
        time_decay_config = self.config.get_section("feedback").get("time_decay", {})
        weekly_decay_factor = time_decay_config.get("weekly_decay_factor", 0.95)
        days_per_week = time_decay_config.get("days_per_week", 7)

        # Query feedback for this track
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(GET_FEEDBACK_BY_TRACK_ID, (track_id,))
        feedback_rows = cur.fetchall()
        conn.close()

        if not feedback_rows:
            return 0.0

        # Calculate cumulative adjustment with time decay
        total_adjustment = 0.0
        for adjustment, timestamp in feedback_rows:
            # Apply time decay
            weeks_old = (
                datetime.now() - datetime.fromisoformat(timestamp)
            ).days / days_per_week
            decay = weekly_decay_factor**weeks_old
            total_adjustment += adjustment * decay

        return float(original_score + total_adjustment)
