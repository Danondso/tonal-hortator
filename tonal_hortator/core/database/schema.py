"""
Database schema definitions for Tonal Hortator.

This module contains database schema definitions, column specifications,
and schema validation functions.
"""

from typing import List, Tuple

# Tracks table schema definition
TRACKS_TABLE_SCHEMA = [
    ("id", "INTEGER PRIMARY KEY"),
    ("name", "TEXT"),
    ("artist", "TEXT"),
    ("album_artist", "TEXT"),
    ("composer", "TEXT"),
    ("album", "TEXT"),
    ("genre", "TEXT"),
    ("year", "INTEGER"),
    ("total_time", "INTEGER"),
    ("track_number", "INTEGER"),
    ("disc_number", "INTEGER"),
    ("play_count", "INTEGER"),
    ("date_added", "TEXT"),
    ("location", "TEXT UNIQUE"),
]

# Metadata columns to add during migration
METADATA_COLUMNS = [
    ("bpm", "REAL"),
    ("musical_key", "TEXT"),
    ("key_scale", "TEXT"),
    ("mood", "TEXT"),
    ("label", "TEXT"),
    ("producer", "TEXT"),
    ("arranger", "TEXT"),
    ("lyricist", "TEXT"),
    ("original_year", "INTEGER"),
    ("original_date", "TEXT"),
    ("chord_changes_rate", "REAL"),
    ("script", "TEXT"),
    ("replay_gain", "REAL"),
    ("release_country", "TEXT"),
    ("catalog_number", "TEXT"),
    ("isrc", "TEXT"),
    ("barcode", "TEXT"),
    ("acoustid_id", "TEXT"),
    ("musicbrainz_track_id", "TEXT"),
    ("musicbrainz_artist_id", "TEXT"),
    ("musicbrainz_album_id", "TEXT"),
    ("media_type", "TEXT"),
    ("analyzed_genre", "TEXT"),
    ("chord_key", "TEXT"),
    ("chord_scale", "TEXT"),
]

# Track embeddings table schema
TRACK_EMBEDDINGS_TABLE_SCHEMA = [
    ("track_id", "INTEGER PRIMARY KEY"),
    ("embedding", "BLOB"),
    ("embedding_text", "TEXT"),
    ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
]

# Metadata mappings table schema
METADATA_MAPPINGS_TABLE_SCHEMA = [
    ("id", "INTEGER PRIMARY KEY"),
    ("source_format", "TEXT NOT NULL"),
    ("source_tag", "TEXT NOT NULL"),
    ("normalized_tag", "TEXT NOT NULL"),
    ("data_type", "TEXT NOT NULL"),
    ("description", "TEXT"),
    ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
]

# Valid column types for validation
VALID_COLUMN_TYPES = {"TEXT", "INTEGER", "REAL", "BLOB", "TIMESTAMP", "BOOLEAN"}

# Valid column names for validation (SQLite keywords to avoid)
INVALID_COLUMN_NAMES = {
    "SELECT",
    "FROM",
    "WHERE",
    "INSERT",
    "UPDATE",
    "DELETE",
    "CREATE",
    "DROP",
    "TABLE",
    "INDEX",
    "PRIMARY",
    "KEY",
    "FOREIGN",
    "REFERENCES",
    "UNIQUE",
    "NOT",
    "NULL",
    "DEFAULT",
    "CHECK",
    "CONSTRAINT",
    "ORDER",
    "BY",
    "GROUP",
    "HAVING",
    "LIMIT",
    "OFFSET",
    "JOIN",
    "LEFT",
    "RIGHT",
    "INNER",
    "OUTER",
    "CROSS",
    "UNION",
    "ALL",
    "DISTINCT",
    "AS",
    "ON",
    "AND",
    "OR",
    "IN",
    "EXISTS",
    "BETWEEN",
    "LIKE",
    "GLOB",
    "REGEXP",
    "IS",
    "CASE",
    "WHEN",
    "THEN",
    "ELSE",
    "END",
    "CAST",
    "COALESCE",
    "IFNULL",
    "LENGTH",
    "LOWER",
    "UPPER",
    "SUBSTR",
    "TRIM",
    "LTRIM",
    "RTRIM",
    "REPLACE",
    "ROUND",
    "ABS",
    "MAX",
    "MIN",
    "SUM",
    "AVG",
    "COUNT",
    "TOTAL",
    "RANDOM",
    "TYPEOF",
    "LAST_INSERT_ROWID",
    "CHANGES",
    "TOTAL_CHANGES",
    "VERSION",
}


def validate_column_name(column_name: str) -> bool:
    """
    Validate that a column name is safe for SQLite.

    Args:
        column_name: The column name to validate

    Returns:
        True if the column name is valid, False otherwise
    """
    if not column_name or not isinstance(column_name, str):
        return False

    # Check for SQLite keywords
    if column_name.upper() in INVALID_COLUMN_NAMES:
        return False

    # Check for valid characters (alphanumeric and underscore only)
    if not column_name.replace("_", "").isalnum():
        return False

    # Must start with a letter or underscore
    if not (column_name[0].isalpha() or column_name[0] == "_"):
        return False

    return True


def validate_column_type(column_type: str) -> bool:
    """
    Validate that a column type is supported by SQLite.

    Args:
        column_type: The column type to validate

    Returns:
        True if the column type is valid, False otherwise
    """
    if not column_type or not isinstance(column_type, str):
        return False

    # Check if it's a valid SQLite type
    base_type = column_type.upper().split()[0]  # Remove size constraints
    return base_type in VALID_COLUMN_TYPES


def get_schema_info() -> dict:
    """
    Get information about the database schema.

    Returns:
        Dictionary containing schema information
    """
    return {
        "tracks_table_columns": len(TRACKS_TABLE_SCHEMA),
        "metadata_columns": len(METADATA_COLUMNS),
        "total_tracks_columns": len(TRACKS_TABLE_SCHEMA) + len(METADATA_COLUMNS),
        "embeddings_table_columns": len(TRACK_EMBEDDINGS_TABLE_SCHEMA),
        "mappings_table_columns": len(METADATA_MAPPINGS_TABLE_SCHEMA),
    }
