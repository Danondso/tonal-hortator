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
        feedback_map = {"like": 0.2, "dislike": -0.2, "block": -1.0}
        adjustment = feedback_map.get(feedback, 0.0)
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
        """Get adjusted score for a track based on feedback with decay"""
        if not os.path.exists(self.db_path):
            return 0.0

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT adjustment, timestamp FROM feedback WHERE track_id = ?", (track_id,)
        )
        rows = cur.fetchall()
        conn.close()

        total_adjustment = 0.0
        for adjustment, ts in rows:
            try:
                weeks_old = (datetime.now() - datetime.fromisoformat(ts)).days / 7.0
                decay = 0.95**weeks_old
                total_adjustment += adjustment * decay
            except Exception:
                total_adjustment += adjustment
        return total_adjustment
