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
    "GET_TRACKS_WITHOUT_EMBEDDINGS",
    "INSERT_TRACK",
    "UPDATE_TRACK_METADATA",
    "GET_EMBEDDING_STATS",
    "CHECK_TABLE_EXISTS",
    # Schema
    "TRACKS_TABLE_SCHEMA",
    "METADATA_COLUMNS",
    # Metadata mappings
    "METADATA_MAPPINGS",
    "MAPPING_CATEGORIES",
]
