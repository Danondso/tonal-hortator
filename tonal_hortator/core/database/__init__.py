"""
Database utilities for Tonal Hortator.

This package contains centralized database queries, schema definitions,
and metadata mappings to keep the codebase clean and maintainable.
"""

from .metadata_mappings import *
from .queries import *
from .schema import *

__all__ = [
    # Queries
    "CREATE_TRACKS_TABLE",
    "CREATE_TRACK_EMBEDDINGS_TABLE",
    "CREATE_METADATA_MAPPINGS_TABLE",
    "CREATE_USER_FEEDBACK_TABLE",
    "CREATE_USER_PREFERENCES_TABLE",
    "CREATE_TRACK_RATINGS_TABLE",
    "CREATE_QUERY_LEARNING_TABLE",
    "GET_TRACKS_WITHOUT_EMBEDDINGS",
    "INSERT_TRACK",
    "INSERT_USER_FEEDBACK",
    "INSERT_USER_PREFERENCE",
    "INSERT_TRACK_RATING",
    "INSERT_QUERY_LEARNING",
    "UPDATE_TRACK_METADATA",
    "GET_EMBEDDING_STATS",
    "GET_USER_FEEDBACK_STATS",
    "GET_USER_PREFERENCES",
    "GET_TRACK_RATINGS",
    "GET_QUERY_LEARNING_DATA",
    "UPDATE_QUERY_LEARNING_APPLIED",
    "CHECK_TABLE_EXISTS",
    # Schema
    "TRACKS_TABLE_SCHEMA",
    "METADATA_COLUMNS",
    "USER_FEEDBACK_TABLE_SCHEMA",
    "USER_PREFERENCES_TABLE_SCHEMA",
    "TRACK_RATINGS_TABLE_SCHEMA",
    "QUERY_LEARNING_TABLE_SCHEMA",
    # Metadata mappings
    "METADATA_MAPPINGS",
    "MAPPING_CATEGORIES",
]
