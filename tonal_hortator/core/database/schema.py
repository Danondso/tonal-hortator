"""
Database schema definitions for Tonal Hortator.

This module contains database schema definitions, column specifications,
and schema validation functions.
"""

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

# User feedback table schema
USER_FEEDBACK_TABLE_SCHEMA = [
    ("id", "INTEGER PRIMARY KEY"),
    ("query", "TEXT NOT NULL"),
    ("query_type", "TEXT NOT NULL"),  # artist_specific, similarity, general
    ("parsed_artist", "TEXT"),
    ("parsed_reference_artist", "TEXT"),
    ("parsed_genres", "TEXT"),  # JSON array
    ("parsed_mood", "TEXT"),
    ("generated_tracks", "TEXT"),  # JSON array of track IDs
    ("user_rating", "INTEGER"),  # 1-5 stars
    ("user_comments", "TEXT"),
    ("user_actions", "TEXT"),  # JSON array of actions (skip, like, dislike, etc.)
    ("playlist_length", "INTEGER"),
    ("requested_length", "INTEGER"),
    ("similarity_threshold", "REAL"),
    ("search_breadth", "INTEGER"),
    ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
]

# User preferences table schema
USER_PREFERENCES_TABLE_SCHEMA = [
    ("id", "INTEGER PRIMARY KEY"),
    ("preference_key", "TEXT UNIQUE NOT NULL"),
    ("preference_value", "TEXT NOT NULL"),
    ("preference_type", "TEXT NOT NULL"),  # string, integer, float, boolean, json
    ("description", "TEXT"),
    ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
    ("updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
]

# Track ratings table schema
TRACK_RATINGS_TABLE_SCHEMA = [
    ("id", "INTEGER PRIMARY KEY"),
    ("track_id", "INTEGER NOT NULL"),
    ("rating", "INTEGER NOT NULL"),  # 1-5 stars
    ("context", "TEXT"),  # playlist context where rating was given
    ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
    ("FOREIGN KEY (track_id) REFERENCES tracks (id)"),
]

# Query learning table schema (for improving LLM prompts)
QUERY_LEARNING_TABLE_SCHEMA = [
    ("id", "INTEGER PRIMARY KEY"),
    ("original_query", "TEXT NOT NULL"),
    ("llm_parsed_result", "TEXT NOT NULL"),  # JSON
    ("user_correction", "TEXT"),  # JSON - what user expected
    ("feedback_score", "REAL"),  # -1 to 1 (negative = wrong, positive = correct)
    ("learning_applied", "BOOLEAN DEFAULT FALSE"),
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
