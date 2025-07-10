#!/usr/bin/env python3
# mypy: disable-error-code="unreachable"
"""
Metadata reader for music files using mutagen library.
Supports MP3, FLAC, OGG, M4A, WAV, and AIFF formats.
"""

import logging
import os
import sqlite3
import sys
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Set

try:
    import mutagen.aiff
    import mutagen.easyid3
    import mutagen.flac
    import mutagen.id3
    import mutagen.mp3
    import mutagen.mp4
    import mutagen.oggvorbis
    import mutagen.wave
except ImportError:
    print("Error: mutagen is required. Install it with: pip install mutagen")
    sys.exit(1)

from mutagen.aiff import AIFF
from mutagen.easyid3 import EasyID3
from mutagen.flac import FLAC
from mutagen.id3 import ID3
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from mutagen.oggvorbis import OggVorbis
from mutagen.wave import WAVE

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MetadataReader:
    """Read comprehensive metadata from music files and populate database"""

    def __init__(self, db_path: str = "music_library.db"):
        """Initialize the metadata reader"""
        self.db_path = db_path
        self.supported_formats = {
            ".mp3": self._read_mp3_metadata,
            ".flac": self._read_flac_metadata,
            ".ogg": self._read_ogg_metadata,
            ".m4a": self._read_m4a_metadata,
            ".wav": self._read_wav_metadata,
            ".aiff": self._read_aiff_metadata,
        }

        # Load metadata mappings from database
        self.metadata_mappings = self._load_metadata_mappings()

    def get_supported_extensions(self) -> Set[str]:
        """Get list of supported file extensions"""
        return set(self.supported_formats.keys())

    def _load_metadata_mappings(self) -> Dict[str, Dict[str, str]]:
        """Load metadata mappings from the database"""
        mappings: Dict[str, Dict[str, str]] = {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT source_format, source_tag, normalized_tag, data_type
                    FROM metadata_mappings
                """
                )

                for row in cursor.fetchall():
                    source_format, source_tag, normalized_tag, data_type = row
                    if source_format not in mappings:
                        mappings[source_format] = {}
                    mappings[source_format][source_tag] = normalized_tag

        except sqlite3.Error as e:
            logger.warning(f"Could not load metadata mappings: {e}")

        return mappings

    def _decode_file_path(self, file_path: str) -> str:
        """Decode URL-encoded file path"""
        if file_path.startswith("file://"):
            # Remove file:// protocol and decode URL encoding
            decoded_path = urllib.parse.unquote(file_path[7:])
            return decoded_path
        else:
            # Decode URL encoding if present
            return urllib.parse.unquote(file_path)

    def read_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Read metadata from a music file

        Args:
            file_path: Path to the music file (should be decoded)

        Returns:
            Dictionary containing normalized metadata
        """
        # Check if path needs decoding (only if it starts with file://)
        if file_path.startswith("file://"):
            decoded_path = self._decode_file_path(file_path)
        else:
            decoded_path = file_path

        file_path_obj = Path(decoded_path)

        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {decoded_path}")

        extension = file_path_obj.suffix.lower()

        if extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {extension}")

        # Get raw metadata
        raw_metadata = self.supported_formats[extension](file_path_obj)

        # Normalize metadata using mappings
        normalized_metadata = self._normalize_metadata(raw_metadata, extension)

        return normalized_metadata

    def _normalize_metadata(
        self, raw_metadata: Dict[str, Any], format_type: str
    ) -> Dict[str, Any]:
        """Normalize metadata using the mappings table"""
        normalized = {}

        # Strip the dot from format_type to match database format
        format_key = format_type.lstrip(".")

        # Get format-specific mappings
        format_mappings = self.metadata_mappings.get(format_key, {})

        logger.debug(
            f"Normalizing {len(raw_metadata)} fields for format {format_type} (key: {format_key})"
        )
        logger.debug(
            f"Available mappings for {format_key}: {list(format_mappings.keys())}"
        )

        for source_tag, value in raw_metadata.items():
            if source_tag in format_mappings:
                normalized_tag = format_mappings[source_tag]
                normalized[normalized_tag] = value
                logger.debug(f"Mapped {source_tag} -> {normalized_tag}")
            else:
                # Keep original tag if no mapping found
                normalized[source_tag] = value
                logger.debug(f"No mapping for {source_tag}, keeping original")

        logger.debug(
            f"Normalized to {len(normalized)} fields: {list(normalized.keys())}"
        )
        return normalized

    def _read_mp3_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Read metadata from MP3 files"""
        metadata: Dict[str, Any] = {}

        try:
            # Try EasyID3 first (standard tags)
            easy_audio = EasyID3(file_path)
            if easy_audio:
                for key, value in easy_audio.items():
                    # Handle multiple values by joining them
                    if isinstance(value, list) and len(value) > 1:
                        metadata[f"easyid3_{key}"] = ", ".join(str(v) for v in value)
                    else:
                        metadata[f"easyid3_{key}"] = value[0] if value else ""

            # Try ID3 for extended tags
            id3_audio = ID3(file_path)
            if id3_audio:
                for key, frame in id3_audio.items():
                    if hasattr(frame, "text"):
                        # Handle multiple text values
                        if isinstance(frame.text, list) and len(frame.text) > 1:
                            metadata[f"id3_{key}"] = ", ".join(
                                str(t) for t in frame.text
                            )
                        else:
                            metadata[f"id3_{key}"] = frame.text[0] if frame.text else ""
                    elif hasattr(frame, "url"):
                        metadata[f"id3_{key}"] = frame.url
                    else:
                        metadata[f"id3_{key}"] = str(frame)

            # Get basic info
            mp3_audio = MP3(file_path)
            if mp3_audio:
                info = mp3_audio.info
                if info:
                    metadata["length"] = int(info.length)
                    metadata["bitrate"] = info.bitrate
                    metadata["sample_rate"] = info.sample_rate

        except Exception as e:
            metadata["error"] = str(e)

        return metadata

    def _read_flac_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Read metadata from FLAC files"""
        metadata: Dict[str, Any] = {}

        try:
            audio = FLAC(file_path)
            if audio:
                # Get tags
                if audio.tags:
                    for key, value in audio.tags.items():
                        # Handle multiple values by joining them
                        if isinstance(value, list) and len(value) > 1:
                            metadata[f"flac_{key}"] = ", ".join(str(v) for v in value)
                        else:
                            metadata[f"flac_{key}"] = value[0] if value else ""

                # Get basic info
                info = audio.info
                if info:
                    metadata["length"] = int(info.length)
                    metadata["sample_rate"] = info.sample_rate
                    metadata["channels"] = info.channels
                    metadata["bits_per_sample"] = info.bits_per_sample

        except Exception as e:
            metadata["error"] = str(e)

        return metadata

    def _read_ogg_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Read metadata from OGG files"""
        metadata: Dict[str, Any] = {}

        try:
            audio = OggVorbis(file_path)
            if audio:
                # Get tags
                if audio.tags:
                    for key, value in audio.tags.items():
                        # Handle multiple values by joining them
                        if isinstance(value, list) and len(value) > 1:
                            metadata[f"ogg_{key}"] = ", ".join(str(v) for v in value)
                        else:
                            metadata[f"ogg_{key}"] = value[0] if value else ""

                # Get basic info
                info = audio.info
                if info:
                    metadata["length"] = int(info.length)
                    metadata["bitrate"] = info.bitrate
                    metadata["sample_rate"] = info.sample_rate

        except Exception as e:
            metadata["error"] = str(e)

        return metadata

    def _read_m4a_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Read metadata from M4A files"""
        metadata: Dict[str, Any] = {}

        try:
            audio = MP4(file_path)
            if audio:
                # Get tags
                if audio.tags:
                    for key, value in audio.tags.items():
                        if isinstance(value, list) and value:
                            # Handle multiple values by joining them
                            if len(value) > 1:
                                metadata[f"m4a_{key}"] = ", ".join(
                                    str(v) for v in value
                                )
                            else:
                                metadata[f"m4a_{key}"] = value[0]
                        else:
                            metadata[f"m4a_{key}"] = str(value)

                # Get basic info
                info = audio.info
                if info:
                    metadata["length"] = int(info.length)
                    metadata["sample_rate"] = info.sample_rate
                    metadata["channels"] = info.channels

        except Exception as e:
            metadata["error"] = str(e)

        return metadata

    def _read_wav_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Read metadata from WAV files"""
        metadata: Dict[str, Any] = {}

        try:
            audio = WAVE(file_path)
            if audio:
                # Get basic info
                info = audio.info
                if info:
                    metadata["length"] = int(info.length)
                    metadata["sample_rate"] = info.sample_rate
                    metadata["channels"] = info.channels
                    metadata["bits_per_sample"] = info.bits_per_sample

        except Exception as e:
            metadata["error"] = str(e)

        return metadata

    def _read_aiff_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Read metadata from AIFF files"""
        metadata: Dict[str, Any] = {}

        try:
            audio = AIFF(file_path)
            if audio:
                # Get basic info
                info = audio.info
                if info:
                    metadata["length"] = int(info.length)
                    metadata["sample_rate"] = info.sample_rate
                    metadata["channels"] = info.channels
                    metadata["bits_per_sample"] = info.bits_per_sample

        except Exception as e:
            metadata["error"] = str(e)

        return metadata

    def update_track_metadata(self, track_id: int, file_path: str) -> bool:
        """
        Update a single track's metadata in the database

        Args:
            track_id: Database ID of the track
            file_path: Path to the music file (should be decoded)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Read metadata from file (file_path should already be decoded)
            metadata = self.read_metadata(file_path)

            if "error" in metadata:
                logger.warning(
                    f"Error reading metadata for track {track_id} ({file_path}): {metadata['error']}"
                )
                return False

            # Update database with new metadata
            return self._update_database_metadata(track_id, metadata, file_path)

        except Exception as e:
            logger.error(
                f"Error updating metadata for track {track_id} ({file_path}): {e}"
            )
            return False

    def _update_database_metadata(
        self, track_id: int, metadata: Dict[str, Any], file_path: str
    ) -> bool:
        """Update track metadata in the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get current schema to know which columns exist
                cursor.execute("PRAGMA table_info(tracks)")
                columns = [col[1] for col in cursor.fetchall()]

                # Debug: Log what metadata was found
                logger.debug(
                    f"Found {len(metadata)} metadata fields for {file_path}: {list(metadata.keys())}"
                )
                logger.debug(f"Available database columns: {columns}")

                # Prepare update data for existing columns
                update_data: Dict[str, Any] = {}
                for key, value in metadata.items():
                    if key in columns:
                        # Convert data types appropriately
                        if key in ["bpm", "chord_changes_rate"] and value:
                            try:
                                update_data[key] = (
                                    float(value) if value is not None else None
                                )
                            except (ValueError, TypeError):
                                continue
                        elif key in ["original_year"] and value:
                            try:
                                update_data[key] = (
                                    int(value) if value is not None else None
                                )
                            except (ValueError, TypeError):
                                continue
                        else:
                            update_data[key] = str(value) if value else None
                    else:
                        # Debug: Log unmapped fields
                        logger.debug(
                            f"Field '{key}' not in database columns for {file_path}"
                        )

                if update_data:
                    # Validate column names against database schema for security
                    valid_columns = set(columns)
                    safe_update_data = {}

                    for key, value in update_data.items():
                        if key in valid_columns:
                            safe_update_data[key] = value
                        else:
                            logger.warning(f"⚠️  Skipping invalid column: {key}")

                    if safe_update_data:
                        # Build dynamic UPDATE query with validated column names
                        set_clause = ", ".join(
                            [f"{key} = ?" for key in safe_update_data.keys()]
                        )
                        values = list(safe_update_data.values())
                        values.append(track_id)  # For WHERE clause

                        cursor.execute(
                            f"""
                            UPDATE tracks
                            SET {set_clause}
                            WHERE id = ?
                        """,
                            values,
                        )

                        conn.commit()
                        logger.info(
                            f"✅ Updated metadata for track {track_id} ({len(safe_update_data)} fields) - {file_path}"
                        )
                        return True
                    else:
                        logger.warning(
                            f"No valid metadata fields found for track {track_id} ({file_path})"
                        )
                        return False
                else:
                    logger.warning(
                        f"No valid metadata fields found for track {track_id} ({file_path})"
                    )
                    logger.debug(f"Raw metadata keys: {list(metadata.keys())}")
                    logger.debug(f"Database columns: {columns}")
                    return False

        except sqlite3.Error as e:
            logger.error(f"Database error updating track {track_id} ({file_path}): {e}")
            return False

    def update_all_tracks_metadata(self, batch_size: int = 100) -> int:
        """
        Update metadata for all tracks in the database

        Args:
            batch_size: Number of tracks to process in each batch

        Returns:
            Number of tracks successfully updated
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get all tracks with file locations
                cursor.execute(
                    """
                    SELECT id, location
                    FROM tracks
                    WHERE location IS NOT NULL AND location != ''
                """
                )

                tracks = cursor.fetchall()
                total_tracks = len(tracks)
                updated_count = 0

                logger.info(f"🔄 Starting metadata update for {total_tracks} tracks...")

                for i, (track_id, file_path) in enumerate(tracks, 1):
                    # Decode the file path
                    decoded_path = self._decode_file_path(file_path)

                    if os.path.exists(decoded_path):
                        if self.update_track_metadata(
                            track_id, decoded_path
                        ):  # Pass decoded path for consistency
                            updated_count += 1

                        # Progress logging
                        if i % batch_size == 0 or i == total_tracks:
                            logger.info(
                                f"📊 Progress: {i}/{total_tracks} tracks processed ({updated_count} updated)"
                            )
                    else:
                        logger.warning(
                            f"⚠️  File not found for track {track_id}: {decoded_path}"
                        )

                logger.info(
                    f"✅ Metadata update complete: {updated_count}/{total_tracks} tracks updated"
                )
                return updated_count

        except sqlite3.Error as e:
            logger.error(f"Database error during metadata update: {e}")
            return 0

    def get_metadata_stats(self) -> Dict[str, Any]:
        """Get statistics about metadata coverage in the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get total tracks
                cursor.execute("SELECT COUNT(*) FROM tracks")
                total_tracks = cursor.fetchone()[0]

                # Whitelist of valid metadata fields for security
                valid_metadata_fields = {
                    "bpm",
                    "musical_key",
                    "key_scale",
                    "mood",
                    "label",
                    "producer",
                    "arranger",
                    "lyricist",
                    "original_year",
                }

                stats = {"total_tracks": total_tracks}
                for field in valid_metadata_fields:
                    # Validate field name against whitelist
                    if field not in valid_metadata_fields:
                        logger.warning(f"⚠️  Skipping invalid field: {field}")
                        continue

                    # Use parameterized query with proper escaping
                    cursor.execute(
                        "SELECT COUNT(*) FROM tracks WHERE ? IS NOT NULL", (field,)
                    )
                    count = cursor.fetchone()[0]
                    stats[f"{field}_coverage"] = count
                    stats[f"{field}_percentage"] = (
                        round((count / total_tracks * 100), 2)
                        if total_tracks > 0
                        else 0
                    )

                return stats

        except sqlite3.Error as e:
            logger.error(f"Error getting metadata stats: {e}")
            return {}

    def debug_metadata_for_file(self, file_path: str) -> None:
        """Debug metadata reading for a specific file"""
        try:
            logger.info(f"🔍 Debugging metadata for: {file_path}")

            # Read raw metadata
            raw_metadata = self.read_metadata(file_path)
            logger.info(f"📊 Raw metadata ({len(raw_metadata)} fields):")
            for key, value in sorted(raw_metadata.items()):
                # Skip fingerprint fields to reduce noise
                if "fingerprint" in key.lower():
                    logger.info(f"  {key}: <fingerprint data omitted>")
                else:
                    logger.info(f"  {key}: {value}")

            # Check database schema
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(tracks)")
                columns = [col[1] for col in cursor.fetchall()]
                logger.info(f"📋 Database columns ({len(columns)}): {columns}")

                # Check which metadata fields would be saved
                update_data = {}
                for key, value in raw_metadata.items():
                    if key in columns:
                        update_data[key] = value

                logger.info(f"💾 Fields that would be saved ({len(update_data)}):")
                for key, value in update_data.items():
                    logger.info(f"  {key}: {value}")

                if not update_data:
                    logger.warning("❌ No fields would be saved to database!")

        except Exception as e:
            logger.error(f"❌ Error debugging metadata: {e}")


def main() -> None:
    """Main function for testing metadata reading"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Update music file metadata in database"
    )
    parser.add_argument(
        "--db-path", default="music_library.db", help="Path to SQLite database"
    )
    parser.add_argument(
        "--update-all", action="store_true", help="Update metadata for all tracks"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show metadata coverage statistics"
    )
    parser.add_argument("--test-file", help="Test metadata reading for a specific file")
    parser.add_argument(
        "--debug-file",
        help="Debug metadata reading for a specific file with detailed output",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose debug logging"
    )

    args = parser.parse_args()

    # Set debug logging if verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    reader = MetadataReader(args.db_path)

    if args.debug_file:
        reader.debug_metadata_for_file(args.debug_file)

    elif args.test_file:
        print(f"🔍 Testing metadata reading for: {args.test_file}")
        try:
            metadata = reader.read_metadata(args.test_file)
            print(f"📊 Found {len(metadata)} metadata fields:")
            for key, value in sorted(metadata.items()):
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"❌ Error: {e}")

    elif args.stats:
        stats = reader.get_metadata_stats()
        print("📊 METADATA COVERAGE STATISTICS:")
        print("=" * 50)
        print(f"Total tracks: {stats.get('total_tracks', 0)}")
        print()
        for field in ["bpm", "musical_key", "key_scale", "mood", "label", "producer"]:
            count = stats.get(f"{field}_coverage", 0)
            percentage = stats.get(f"{field}_percentage", 0)
            print(f"{field}: {count} tracks ({percentage}%)")

    elif args.update_all:
        updated = reader.update_all_tracks_metadata()
        print(f"✅ Updated metadata for {updated} tracks")

    else:
        print("Use --help to see available options")


if __name__ == "__main__":
    main()
