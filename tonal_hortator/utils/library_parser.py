#!/usr/bin/env python3
"""
Library parser for iTunes XML library files.
Parses iTunes XML library and stores track data in SQLite database.
"""

import sqlite3
import urllib.parse
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

from loguru import logger

from tonal_hortator.core.database import (
    CHECK_TRACK_EXISTS,
    CREATE_TRACKS_TABLE,
    INSERT_TRACK,
)

# Import metadata reader
from tonal_hortator.utils.metadata_reader import MetadataReader

# Use defusedxml for safe XML parsing
try:
    from defusedxml import ElementTree as ET
    from defusedxml.ElementTree import Element
except ImportError:
    # Fallback to regular ElementTree with warning
    import warnings
    import xml.etree.ElementTree as ET
    from xml.etree.ElementTree import Element

    warnings.warn(
        "defusedxml not available. Using regular ElementTree. "
        "Install defusedxml for better security: pip install defusedxml"
    )


class LibraryParser:
    """Parser for iTunes XML library files"""

    def __init__(self, db_path: str = "music_library.db"):
        self.db_path = db_path
        self._field_processors = self._get_field_processors()
        self._field_mapping = self._get_field_mapping()
        self._create_table()

        # Initialize metadata reader
        self.metadata_reader = MetadataReader(db_path)

    def _create_table(self) -> None:
        """Create the tracks table if it doesn't exist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                cursor = conn.cursor()
                cursor.execute(CREATE_TRACKS_TABLE)
                conn.commit()
                logger.info("✅ 'tracks' table created or already exists.")
        except sqlite3.Error as e:
            logger.error(f"❌ Error creating table: {e}")
            raise

    def parse_library(self, xml_path: str) -> int:
        """
        Parses the XML library and inserts tracks into the database.

        Args:
            xml_path: The path to the Apple Music XML library file.

        Returns:
            The number of tracks inserted into the database.
        """
        if not Path(xml_path).exists():
            logger.error(f"❌ XML file not found at: {xml_path}")
            return 0

        logger.info(f"Parsing library from: {xml_path}")
        tracks_generator = self._parse_tracks(xml_path)

        inserted_count = self._insert_tracks(tracks_generator)
        logger.info(
            f"✅ Library parsing complete. Inserted {inserted_count} new tracks."
        )
        return inserted_count

    def _parse_tracks(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """
        Parses the XML file and yields track dictionaries using an iterparse
        strategy to keep memory usage low.
        """
        logger.info(
            "Starting XML parsing. This may take a moment for large libraries..."
        )
        in_tracks_section = False
        try:
            # Use defusedxml's iterparse for safe XML parsing
            # This prevents XML attacks like billion laughs, quadratic blowup, etc.
            for event, elem in ET.iterparse(file_path, events=("start", "end")):
                if event == "end" and elem.tag == "key" and elem.text == "Tracks":
                    in_tracks_section = True
                    continue

                if in_tracks_section and event == "end" and elem.tag == "dict":
                    if elem.find("key[.='Track ID']") is not None:
                        track_data = self._extract_track_data(elem)
                        if track_data:
                            yield track_data
                        elem.clear()
                    else:
                        break
        except ET.ParseError as e:
            logger.error(f"❌ Failed to parse XML file: {e}")
            return
        except Exception as e:
            logger.error(f"❌ Unexpected error parsing XML file: {e}")
            return

    def _process_string_field(self, value_elem: ET.Element) -> Optional[str]:
        """Process a string field from XML"""
        return value_elem.text if value_elem.text is not None else None

    def _process_location_field(self, value_elem: ET.Element) -> Optional[str]:
        """Process location field with URL decoding"""
        if not value_elem.text:
            return None

        location = value_elem.text

        # Decode URL encoding if present
        if location.startswith("file://"):
            # Remove file:// protocol and decode URL encoding
            decoded_location = urllib.parse.unquote(location[7:])
            return decoded_location
        else:
            # Decode URL encoding if present
            return urllib.parse.unquote(location)

    def _process_int_field(self, value_elem: ET.Element) -> int:
        """Process an integer field from XML"""
        return int(value_elem.text) if value_elem.text else 0

    def _process_optional_int_field(self, value_elem: ET.Element) -> Optional[int]:
        """Process an optional integer field from XML"""
        return int(value_elem.text) if value_elem.text else None

    def _get_field_processors(self) -> Dict[str, Callable[[ET.Element], Any]]:
        """Get mapping of field names to their processing functions"""
        return {
            "Name": lambda elem: self._process_string_field(elem),
            "Artist": lambda elem: self._process_string_field(elem),
            "Album Artist": lambda elem: self._process_string_field(elem),
            "Composer": lambda elem: self._process_string_field(elem),
            "Album": lambda elem: self._process_string_field(elem),
            "Genre": lambda elem: self._process_string_field(elem),
            "Year": lambda elem: self._process_optional_int_field(elem),
            "Total Time": lambda elem: self._process_int_field(elem),
            "Track Number": lambda elem: self._process_int_field(elem),
            "Disc Number": lambda elem: self._process_int_field(elem),
            "Play Count": lambda elem: self._process_int_field(elem),
            "Date Added": lambda elem: self._process_string_field(elem),
            "Location": lambda elem: self._process_string_field(elem),
        }

    def _get_field_mapping(self) -> Dict[str, str]:
        """Get mapping of XML field names to data dictionary keys"""
        return {
            "Name": "name",
            "Artist": "artist",
            "Album Artist": "album_artist",
            "Composer": "composer",
            "Album": "album",
            "Genre": "genre",
            "Year": "year",
            "Total Time": "total_time",
            "Track Number": "track_number",
            "Disc Number": "disc_number",
            "Play Count": "play_count",
            "Date Added": "date_added",
            "Location": "location",
        }

    def _process_track_field(
        self, key_name: Optional[str], value_elem: ET.Element, data: Dict[str, Any]
    ) -> None:
        """Process a single track field based on its key name"""
        if key_name in self._field_processors and key_name in self._field_mapping:
            data[self._field_mapping[key_name]] = self._field_processors[key_name](
                value_elem
            )

    def _extract_track_data(self, track_dict: ET.Element) -> Optional[Dict[str, Any]]:
        """
        Extracts relevant data for a single track from its XML element.

        Args:
            track_dict: The XML element for a track.

        Returns:
            A dictionary with track data or None if essential data is missing.
        """
        data: Dict[str, Any] = {
            "name": None,
            "artist": None,
            "album_artist": None,
            "composer": None,
            "album": None,
            "genre": None,
            "year": None,
            "total_time": 0,
            "track_number": 0,
            "disc_number": 0,
            "play_count": 0,
            "date_added": None,
            "location": None,
        }

        it = iter(track_dict)
        for key_elem in it:
            key_name = key_elem.text
            value_elem = next(it)
            self._process_track_field(key_name, value_elem, data)

        return data if data["name"] else None

    def _insert_tracks(self, tracks: Iterator[Dict[str, Any]]) -> int:
        """
        Inserts tracks into the database, avoiding duplicates based on location.

        Args:
            tracks: An iterator of track dictionaries.

        Returns:
            The number of tracks newly inserted.
        """
        count = 0
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Enable WAL mode for better concurrent performance
                conn.execute("PRAGMA journal_mode=WAL;")
                cursor = conn.cursor()
                for track in tracks:
                    if (
                        track.get("location")
                        and cursor.execute(
                            CHECK_TRACK_EXISTS,
                            (track.get("location"),),
                        ).fetchone()
                        is None
                    ):
                        # Insert basic track data using centralized query
                        cursor.execute(
                            INSERT_TRACK,
                            (
                                track.get("name"),
                                track.get("artist"),
                                track.get("album_artist"),
                                track.get("composer"),
                                track.get("album"),
                                track.get("genre"),
                                track.get("year"),
                                track.get("total_time"),
                                track.get("track_number"),
                                track.get("disc_number"),
                                track.get("play_count"),
                                track.get("date_added"),
                                track.get("location"),
                            ),
                        )

                        # Get the inserted track ID
                        track_id = cursor.lastrowid

                        # Read and populate metadata if file exists
                        if (
                            track_id is not None
                            and track.get("location")
                            and Path(track["location"]).exists()
                        ):
                            try:
                                self.metadata_reader.update_track_metadata(
                                    track_id, track["location"]
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to read metadata for {track['location']}: {e}"
                                )

                        count += 1
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"❌ Error inserting tracks: {e}")
        return count


def main() -> None:
    """Main function to run the parser from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Parse Apple Music XML library.")
    parser.add_argument("xml_path", help="Path to the Apple Music XML library file.")
    parser.add_argument(
        "--db-path",
        default="music_library.db",
        help="Path to the SQLite database file.",
    )
    args = parser.parse_args()

    library_parser = LibraryParser(db_path=args.db_path)
    library_parser.parse_library(xml_path=args.xml_path)


if __name__ == "__main__":
    main()
