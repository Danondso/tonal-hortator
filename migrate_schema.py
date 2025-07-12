#!/usr/bin/env python3
"""
Database migration script to add new metadata fields to the tracks table.
This migration adds musical analysis fields from MusicBrainz Picard processing.
"""

import logging
import sqlite3
from pathlib import Path
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatabaseMigrator:
    """Handle database schema migrations for Tonal Hortator"""

    def __init__(self, db_path: str = "music_library.db"):
        self.db_path = db_path

        # Whitelist of valid column names and types for security
        self.valid_column_names = {
            "bpm",
            "musical_key",
            "key_scale",
            "mood",
            "release_country",
            "label",
            "composer",
            "arranger",
            "lyricist",
            "producer",
            "original_year",
            "original_date",
            "chord_changes_rate",
            "script",
            "replay_gain",
        }

        self.valid_column_types = {"TEXT", "INTEGER", "REAL", "BLOB"}

    def _validate_column_name(self, column_name: str) -> bool:
        """Validate column name against whitelist"""
        return column_name in self.valid_column_names

    def _validate_column_type(self, column_type: str) -> bool:
        """Validate column type against whitelist"""
        return column_type.upper() in self.valid_column_types

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
        new_columns = [
            ("bpm", "REAL", "Tempo in beats per minute"),
            ("musical_key", "TEXT", "Musical key (e.g., C, G#, F)"),
            ("key_scale", "TEXT", "Scale (major, minor, etc.)"),
            ("mood", "TEXT", "Mood classification (acoustic, electronic, etc.)"),
            ("release_country", "TEXT", "Release country code"),
            ("label", "TEXT", "Record label"),
            ("composer", "TEXT", "Composer information"),
            ("arranger", "TEXT", "Arranger information"),
            ("lyricist", "TEXT", "Lyricist information"),
            ("producer", "TEXT", "Producer information"),
            ("original_year", "INTEGER", "Original release year"),
            ("original_date", "TEXT", "Original release date"),
            ("chord_changes_rate", "REAL", "Rate of chord changes"),
            ("script", "TEXT", "Script/language code"),
            ("replay_gain", "TEXT", "ReplayGain information"),
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

                # Create the table
                cursor.execute(
                    """
                    CREATE TABLE metadata_mappings (
                        id INTEGER PRIMARY KEY,
                        source_format TEXT NOT NULL,
                        source_tag TEXT NOT NULL,
                        normalized_tag TEXT NOT NULL,
                        data_type TEXT NOT NULL,
                        description TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(source_format, source_tag)
                    )
                """
                )

                # Insert common mappings
                mappings = [
                    # MP3 EasyID3 mappings
                    ("mp3", "easyid3_artist", "artist", "string", "Artist name"),
                    ("mp3", "easyid3_album", "album", "string", "Album name"),
                    ("mp3", "easyid3_genre", "genre", "string", "Genre"),
                    ("mp3", "easyid3_title", "title", "string", "Track title"),
                    ("mp3", "easyid3_date", "year", "integer", "Release year"),
                    (
                        "mp3",
                        "easyid3_tracknumber",
                        "track_number",
                        "integer",
                        "Track number",
                    ),
                    (
                        "mp3",
                        "easyid3_discnumber",
                        "disc_number",
                        "integer",
                        "Disc number",
                    ),
                    (
                        "mp3",
                        "easyid3_albumartist",
                        "album_artist",
                        "string",
                        "Album artist",
                    ),
                    ("mp3", "easyid3_composer", "composer", "string", "Composer"),
                    ("mp3", "easyid3_organization", "label", "string", "Record label"),
                    ("mp3", "easyid3_media", "media_type", "string", "Media type"),
                    ("mp3", "easyid3_isrc", "isrc", "string", "ISRC code"),
                    ("mp3", "easyid3_barcode", "barcode", "string", "Album barcode"),
                    (
                        "mp3",
                        "easyid3_catalognumber",
                        "catalog_number",
                        "string",
                        "Catalog number",
                    ),
                    (
                        "mp3",
                        "easyid3_releasecountry",
                        "release_country",
                        "string",
                        "Release country",
                    ),
                    (
                        "mp3",
                        "easyid3_originaldate",
                        "original_date",
                        "string",
                        "Original release date",
                    ),
                    ("mp3", "easyid3_acoustid_id", "acoustid_id", "string", "AcoustID"),
                    (
                        "mp3",
                        "easyid3_musicbrainz_trackid",
                        "musicbrainz_track_id",
                        "string",
                        "MusicBrainz track ID",
                    ),
                    (
                        "mp3",
                        "easyid3_musicbrainz_artistid",
                        "musicbrainz_artist_id",
                        "string",
                        "MusicBrainz artist ID",
                    ),
                    (
                        "mp3",
                        "easyid3_musicbrainz_albumid",
                        "musicbrainz_album_id",
                        "string",
                        "MusicBrainz album ID",
                    ),
                    # MP3 ID3 mappings
                    ("mp3", "id3_TPE1", "artist", "string", "Artist name"),
                    ("mp3", "id3_TALB", "album", "string", "Album name"),
                    ("mp3", "id3_TCON", "genre", "string", "Genre"),
                    ("mp3", "id3_TIT2", "title", "string", "Track title"),
                    ("mp3", "id3_TDRC", "year", "integer", "Release year"),
                    ("mp3", "id3_TRCK", "track_number", "integer", "Track number"),
                    ("mp3", "id3_TPOS", "disc_number", "integer", "Disc number"),
                    ("mp3", "id3_TPE2", "album_artist", "string", "Album artist"),
                    ("mp3", "id3_TCOM", "composer", "string", "Composer"),
                    ("mp3", "id3_TPUB", "label", "string", "Record label"),
                    ("mp3", "id3_TMED", "media_type", "string", "Media type"),
                    ("mp3", "id3_TSRC", "isrc", "string", "ISRC code"),
                    (
                        "mp3",
                        "id3_TDOR",
                        "original_date",
                        "string",
                        "Original release date",
                    ),
                    # MP3 TXXX mappings for musical analysis
                    (
                        "mp3",
                        "id3_TXXX:ab:lo:rhythm:bpm",
                        "bpm",
                        "float",
                        "Tempo in BPM",
                    ),
                    (
                        "mp3",
                        "id3_TXXX:ab:lo:tonal:key_key",
                        "musical_key",
                        "string",
                        "Musical key",
                    ),
                    (
                        "mp3",
                        "id3_TXXX:ab:lo:tonal:key_scale",
                        "key_scale",
                        "string",
                        "Key scale",
                    ),
                    (
                        "mp3",
                        "id3_TXXX:ab:lo:tonal:chords_key",
                        "chord_key",
                        "string",
                        "Chord key",
                    ),
                    (
                        "mp3",
                        "id3_TXXX:ab:lo:tonal:chords_scale",
                        "chord_scale",
                        "string",
                        "Chord scale",
                    ),
                    (
                        "mp3",
                        "id3_TXXX:ab:lo:tonal:chords_changes_rate",
                        "chord_changes_rate",
                        "float",
                        "Chord changes rate",
                    ),
                    (
                        "mp3",
                        "id3_TXXX:ab:mood",
                        "mood",
                        "string",
                        "Mood classification",
                    ),
                    (
                        "mp3",
                        "id3_TXXX:ab:genre",
                        "analyzed_genre",
                        "string",
                        "Analyzed genre",
                    ),
                    (
                        "mp3",
                        "id3_TXXX:Acoustid Id",
                        "acoustid_id",
                        "string",
                        "AcoustID",
                    ),
                    (
                        "mp3",
                        "id3_TXXX:MusicBrainz Track Id",
                        "musicbrainz_track_id",
                        "string",
                        "MusicBrainz track ID",
                    ),
                    (
                        "mp3",
                        "id3_TXXX:MusicBrainz Artist Id",
                        "musicbrainz_artist_id",
                        "string",
                        "MusicBrainz artist ID",
                    ),
                    (
                        "mp3",
                        "id3_TXXX:MusicBrainz Album Id",
                        "musicbrainz_album_id",
                        "string",
                        "MusicBrainz album ID",
                    ),
                    ("mp3", "id3_TXXX:BARCODE", "barcode", "string", "Album barcode"),
                    (
                        "mp3",
                        "id3_TXXX:CATALOGNUMBER",
                        "catalog_number",
                        "string",
                        "Catalog number",
                    ),
                    (
                        "mp3",
                        "id3_TXXX:MusicBrainz Album Release Country",
                        "release_country",
                        "string",
                        "Release country",
                    ),
                    (
                        "mp3",
                        "id3_TXXX:originalyear",
                        "original_year",
                        "integer",
                        "Original year",
                    ),
                    # M4A mappings
                    ("m4a", "m4a_©ART", "artist", "string", "Artist name"),
                    ("m4a", "m4a_©alb", "album", "string", "Album name"),
                    ("m4a", "m4a_©gen", "genre", "string", "Genre"),
                    ("m4a", "m4a_©nam", "title", "string", "Track title"),
                    ("m4a", "m4a_©day", "year", "integer", "Release year"),
                    ("m4a", "m4a_trkn", "track_number", "integer", "Track number"),
                    ("m4a", "m4a_disk", "disc_number", "integer", "Disc number"),
                    ("m4a", "m4a_aART", "album_artist", "string", "Album artist"),
                    ("m4a", "m4a_©wrt", "composer", "string", "Composer"),
                    ("m4a", "m4a_©too", "producer", "string", "Producer"),
                    ("m4a", "m4a_©lyr", "lyricist", "string", "Lyricist"),
                    # M4A iTunes mappings
                    (
                        "m4a",
                        "m4a_----:com.apple.iTunes:ARTISTS",
                        "artist",
                        "string",
                        "Artist name",
                    ),
                    (
                        "m4a",
                        "m4a_----:com.apple.iTunes:ALBUM_ARTISTS",
                        "album_artist",
                        "string",
                        "Album artist",
                    ),
                    (
                        "m4a",
                        "m4a_----:com.apple.iTunes:LABEL",
                        "label",
                        "string",
                        "Record label",
                    ),
                    (
                        "m4a",
                        "m4a_----:com.apple.iTunes:MEDIA",
                        "media_type",
                        "string",
                        "Media type",
                    ),
                    (
                        "m4a",
                        "m4a_----:com.apple.iTunes:ISRC",
                        "isrc",
                        "string",
                        "ISRC code",
                    ),
                    (
                        "m4a",
                        "m4a_----:com.apple.iTunes:BARCODE",
                        "barcode",
                        "string",
                        "Album barcode",
                    ),
                    (
                        "m4a",
                        "m4a_----:com.apple.iTunes:CATALOGNUMBER",
                        "catalog_number",
                        "string",
                        "Catalog number",
                    ),
                    (
                        "m4a",
                        "m4a_----:com.apple.iTunes:MusicBrainz Album Release Country",
                        "release_country",
                        "string",
                        "Release country",
                    ),
                    (
                        "m4a",
                        "m4a_----:com.apple.iTunes:ORIGINALDATE",
                        "original_date",
                        "string",
                        "Original release date",
                    ),
                    (
                        "m4a",
                        "m4a_----:com.apple.iTunes:ORIGINAL YEAR",
                        "original_year",
                        "integer",
                        "Original year",
                    ),
                    (
                        "m4a",
                        "m4a_----:com.apple.iTunes:Acoustid Id",
                        "acoustid_id",
                        "string",
                        "AcoustID",
                    ),
                    (
                        "m4a",
                        "m4a_----:com.apple.iTunes:MusicBrainz Track Id",
                        "musicbrainz_track_id",
                        "string",
                        "MusicBrainz track ID",
                    ),
                    (
                        "m4a",
                        "m4a_----:com.apple.iTunes:MusicBrainz Artist Id",
                        "musicbrainz_artist_id",
                        "string",
                        "MusicBrainz artist ID",
                    ),
                    (
                        "m4a",
                        "m4a_----:com.apple.iTunes:MusicBrainz Album Id",
                        "musicbrainz_album_id",
                        "string",
                        "MusicBrainz album ID",
                    ),
                    (
                        "m4a",
                        "m4a_----:com.apple.iTunes:PRODUCER",
                        "producer",
                        "string",
                        "Producer",
                    ),
                    (
                        "m4a",
                        "m4a_----:com.apple.iTunes:LYRICIST",
                        "lyricist",
                        "string",
                        "Lyricist",
                    ),
                    (
                        "m4a",
                        "m4a_----:com.apple.iTunes:ARRANGER",
                        "arranger",
                        "string",
                        "Arranger",
                    ),
                    (
                        "m4a",
                        "m4a_----:com.apple.iTunes:publisher",
                        "label",
                        "string",
                        "Record label",
                    ),
                    # M4A musical analysis mappings
                    (
                        "m4a",
                        "m4a_----:com.apple.iTunes:ab:lo:rhythm:bpm",
                        "bpm",
                        "float",
                        "Tempo in BPM",
                    ),
                    (
                        "m4a",
                        "m4a_----:com.apple.iTunes:ab:lo:tonal:key_key",
                        "musical_key",
                        "string",
                        "Musical key",
                    ),
                    (
                        "m4a",
                        "m4a_----:com.apple.iTunes:ab:lo:tonal:key_scale",
                        "key_scale",
                        "string",
                        "Key scale",
                    ),
                    (
                        "m4a",
                        "m4a_----:com.apple.iTunes:ab:lo:tonal:chords_key",
                        "chord_key",
                        "string",
                        "Chord key",
                    ),
                    (
                        "m4a",
                        "m4a_----:com.apple.iTunes:ab:lo:tonal:chords_scale",
                        "chord_scale",
                        "string",
                        "Chord scale",
                    ),
                    (
                        "m4a",
                        "m4a_----:com.apple.iTunes:ab:lo:tonal:chords_changes_rate",
                        "chord_changes_rate",
                        "float",
                        "Chord changes rate",
                    ),
                    (
                        "m4a",
                        "m4a_----:com.apple.iTunes:ab:mood",
                        "mood",
                        "string",
                        "Mood classification",
                    ),
                    (
                        "m4a",
                        "m4a_----:com.apple.iTunes:ab:genre",
                        "analyzed_genre",
                        "string",
                        "Analyzed genre",
                    ),
                ]

                cursor.executemany(
                    """
                    INSERT INTO metadata_mappings 
                    (source_format, source_tag, normalized_tag, data_type, description)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    mappings,
                )

                conn.commit()
                logger.info(
                    f"✅ Created metadata_mappings table with {len(mappings)} mappings"
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
    import argparse

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
