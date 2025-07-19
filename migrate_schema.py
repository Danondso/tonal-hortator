#!/usr/bin/env python3
"""
Database migration script to add new metadata fields to the tracks table.
This migration adds musical analysis fields from MusicBrainz Picard processing.
"""

import argparse
import logging
import sqlite3
from pathlib import Path
from typing import List, Tuple

from tonal_hortator.core.database import (
    CREATE_METADATA_MAPPINGS_TABLE,
    INSERT_METADATA_MAPPING,
    METADATA_MAPPINGS,
    validate_column_name,
    validate_column_type,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatabaseMigrator:
    """Handle database schema migrations for Tonal Hortator"""

    def __init__(self, db_path: str = "music_library.db"):
        self.db_path = db_path

    def _validate_column_name(self, column_name: str) -> bool:
        """Validate column name using centralized validation"""
        return validate_column_name(column_name)

    def _validate_column_type(self, column_type: str) -> bool:
        """Validate column type using centralized validation"""
        return validate_column_type(column_type)

    def _safe_add_column(
        self, cursor: sqlite3.Cursor, column_name: str, column_type: str
    ) -> bool:
        """Safely add a column using string formatting with validation"""
        if not self._validate_column_name(column_name):
            logger.error(f"❌ Invalid column name: {column_name}")
            return False

        if not self._validate_column_type(column_type):
            logger.error(f"❌ Invalid column type: {column_type}")
            return False

        try:
            # Use string formatting since SQLite doesn't support parameters for column names/types
            # Column names and types are validated against whitelists for security
            query = f"ALTER TABLE tracks ADD COLUMN {column_name} {column_type}"
            cursor.execute(query)
            return True
        except sqlite3.Error as e:
            logger.error(f"❌ Error adding column {column_name}: {e}")
            return False

    def get_current_schema(self) -> List[Tuple[str, str, str, bool, str, bool]]:
        """Get current table schema from SQLite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(tracks)")
                return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"❌ Error getting current schema: {e}")
            return []

    def add_metadata_columns(self) -> bool:
        """Add new metadata columns to the tracks table"""
        # Use centralized metadata columns with descriptions
        new_columns = [
            ("bpm", "REAL", "Tempo in beats per minute"),
            ("musical_key", "TEXT", "Musical key (e.g., C, G#, F)"),
            ("key_scale", "TEXT", "Scale (major, minor, etc.)"),
            ("mood", "TEXT", "Mood classification (acoustic, electronic, etc.)"),
            ("label", "TEXT", "Record label"),
            ("producer", "TEXT", "Producer information"),
            ("arranger", "TEXT", "Arranger information"),
            ("lyricist", "TEXT", "Lyricist information"),
            ("original_year", "INTEGER", "Original release year"),
            ("original_date", "TEXT", "Original release date"),
            ("chord_changes_rate", "REAL", "Rate of chord changes"),
            ("script", "TEXT", "Script/language code"),
            ("replay_gain", "REAL", "ReplayGain information"),
            ("release_country", "TEXT", "Release country code"),
            ("catalog_number", "TEXT", "Catalog number"),
            ("isrc", "TEXT", "ISRC code"),
            ("barcode", "TEXT", "Album barcode"),
            ("acoustid_id", "TEXT", "AcoustID"),
            ("musicbrainz_track_id", "TEXT", "MusicBrainz track ID"),
            ("musicbrainz_artist_id", "TEXT", "MusicBrainz artist ID"),
            ("musicbrainz_album_id", "TEXT", "MusicBrainz album ID"),
            ("media_type", "TEXT", "Media type"),
            ("analyzed_genre", "TEXT", "Analyzed genre"),
            ("chord_key", "TEXT", "Chord key"),
            ("chord_scale", "TEXT", "Chord scale"),
        ]

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get current schema
                current_schema = self.get_current_schema()
                existing_columns = {col[1] for col in current_schema}

                # Add new columns that don't already exist
                added_count = 0
                for column_name, column_type, description in new_columns:
                    if column_name not in existing_columns:
                        if self._safe_add_column(cursor, column_name, column_type):
                            logger.info(
                                f"✅ Added column: {column_name} ({column_type}) - {description}"
                            )
                            added_count += 1
                        else:
                            logger.warning(
                                f"⚠️  Column {column_name} may already exist or invalid."
                            )
                    else:
                        logger.info(f"ℹ️  Column {column_name} already exists, skipping")

                conn.commit()
                logger.info(f"✅ Migration complete. Added {added_count} new columns.")
                return True

        except sqlite3.Error as e:
            logger.error(f"❌ Error during migration: {e}")
            return False

    def create_metadata_mappings_table(self) -> bool:
        """Create the metadata mappings table for tag normalization"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Check if table already exists
                cursor.execute(
                    """
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='metadata_mappings'
                """
                )

                if cursor.fetchone():
                    logger.info("ℹ️  metadata_mappings table already exists")
                    return True

                # Create the table using centralized query
                cursor.execute(CREATE_METADATA_MAPPINGS_TABLE)

                # Insert mappings using centralized data
                cursor.executemany(INSERT_METADATA_MAPPING, METADATA_MAPPINGS)

                conn.commit()
                logger.info(
                    f"✅ Created metadata_mappings table with {len(METADATA_MAPPINGS)} mappings"
                )
                return True

        except sqlite3.Error as e:
            logger.error(f"❌ Error creating metadata_mappings table: {e}")
            return False

    def run_migration(self) -> bool:
        """Run the complete migration"""
        logger.info("🚀 Starting database migration...")

        # Check if database exists
        if not Path(self.db_path).exists():
            logger.error(f"❌ Database not found: {self.db_path}")
            return False

        # Add new columns to tracks table
        if not self.add_metadata_columns():
            return False

        # Create metadata mappings table
        if not self.create_metadata_mappings_table():
            return False

        logger.info("✅ Migration completed successfully!")
        return True

    def show_schema(self) -> None:
        """Display current database schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Show tracks table schema
                cursor.execute("PRAGMA table_info(tracks)")
                tracks_schema = cursor.fetchall()

                print("\n📊 TRACKS TABLE SCHEMA:")
                print("=" * 50)
                for col in tracks_schema:
                    print(f"  {col[1]} ({col[2]}) - {col[5] or 'No description'}")

                # Show metadata_mappings table schema
                cursor.execute("PRAGMA table_info(metadata_mappings)")
                mappings_schema = cursor.fetchall()

                if mappings_schema:
                    print("\n📋 METADATA_MAPPINGS TABLE SCHEMA:")
                    print("=" * 50)
                    for col in mappings_schema:
                        print(f"  {col[1]} ({col[2]}) - {col[5] or 'No description'}")

                # Show sample mappings
                cursor.execute(
                    "SELECT source_format, source_tag, normalized_tag FROM metadata_mappings LIMIT 10"
                )
                sample_mappings = cursor.fetchall()

                if sample_mappings:
                    print("\n🔗 SAMPLE METADATA MAPPINGS:")
                    print("=" * 50)
                    for fmt, source, normalized in sample_mappings:
                        print(f"  {fmt}: {source} → {normalized}")

        except sqlite3.Error as e:
            logger.error(f"❌ Error showing schema: {e}")


def main() -> int:
    """Main function for running the migration"""
    parser = argparse.ArgumentParser(
        description="Database migration for Tonal Hortator"
    )
    parser.add_argument(
        "--db-path", default="music_library.db", help="Path to SQLite database"
    )
    parser.add_argument(
        "--show-schema",
        action="store_true",
        help="Show current schema without migrating",
    )

    args = parser.parse_args()

    migrator = DatabaseMigrator(args.db_path)

    if args.show_schema:
        migrator.show_schema()
    else:
        if migrator.run_migration():
            print("\n✅ Migration completed successfully!")
            print("\n📊 Updated schema:")
            migrator.show_schema()
        else:
            print("\n❌ Migration failed!")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
