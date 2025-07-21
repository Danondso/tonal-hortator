#!/usr/bin/env python3
"""
Database Manager for Tonal Hortator

Handles all database interactions for feedback and preferences.
"""

import logging
import sqlite3
from typing import Any, List, Optional, Tuple, cast

from tonal_hortator.core.database.queries import (
    CREATE_FEEDBACK_TABLE,
    CREATE_QUERY_LEARNING_TABLE,
    CREATE_TRACK_RATINGS_TABLE,
    CREATE_USER_FEEDBACK_TABLE,
    CREATE_USER_PREFERENCES_TABLE,
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Handles all database interactions for feedback and preferences."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_tables_exist()

    def _ensure_tables_exist(self) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(CREATE_USER_FEEDBACK_TABLE)
                cursor.execute(CREATE_TRACK_RATINGS_TABLE)
                cursor.execute(CREATE_QUERY_LEARNING_TABLE)
                cursor.execute(CREATE_USER_PREFERENCES_TABLE)
                cursor.execute(CREATE_FEEDBACK_TABLE)
                conn.commit()
                logger.info("✅ Feedback tables ensured")
        except sqlite3.Error as e:
            logger.error(f"❌ Error creating feedback tables: {e}")
            raise

    def execute(
        self, query: str, params: tuple = (), commit: bool = False
    ) -> sqlite3.Cursor:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                if commit:
                    conn.commit()
                return cursor
        except sqlite3.Error as e:
            logger.error(f"❌ Database error: {e}")
            raise

    def execute_fetchone(
        self, query: str, params: tuple = ()
    ) -> Optional[Tuple[Any, ...]]:
        cursor = self.execute(query, params)
        result = cursor.fetchone()
        return cast(Optional[Tuple[Any, ...]], result)

    def execute_fetchall(self, query: str, params: tuple = ()) -> List[Tuple[Any, ...]]:
        cursor = self.execute(query, params)
        return cursor.fetchall()

    def execute_commit(self, query: str, params: tuple = ()) -> None:
        self.execute(query, params, commit=True)
