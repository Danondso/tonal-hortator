#!/usr/bin/env python3
"""
Parses an Apple Music XML library file and populates a SQLite database.
"""

import logging
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LibraryParser:
    """Parses an Apple Music XML library and stores tracks in a database."""

    def __init__(self, db_path: str = "music_library.db"):
        """
        Initializes the LibraryParser.

        Args:
            db_path: The path to the SQLite database file.
        """
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        """Creates the 'tracks' table in the database if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tracks (
                        id INTEGER PRIMARY KEY,
                        name TEXT,
                        artist TEXT,
                        album_artist TEXT,
                        composer TEXT,
                        album TEXT,
                        genre TEXT,
                        year INTEGER,
                        total_time INTEGER,
                        track_number INTEGER,
                        disc_number INTEGER,
                        play_count INTEGER,
                        date_added TEXT,
                        location TEXT UNIQUE
                    )
                """
                )
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

            if key_name == "Name":
                data["name"] = value_elem.text
            elif key_name == "Artist":
                data["artist"] = value_elem.text
            elif key_name == "Album Artist":
                data["album_artist"] = value_elem.text
            elif key_name == "Composer":
                data["composer"] = value_elem.text
            elif key_name == "Album":
                data["album"] = value_elem.text
            elif key_name == "Genre":
                data["genre"] = value_elem.text
            elif key_name == "Year":
                data["year"] = int(value_elem.text) if value_elem.text else None
            elif key_name == "Total Time":
                data["total_time"] = int(value_elem.text) if value_elem.text else 0
            elif key_name == "Track Number":
                data["track_number"] = int(value_elem.text) if value_elem.text else 0
            elif key_name == "Disc Number":
                data["disc_number"] = int(value_elem.text) if value_elem.text else 0
            elif key_name == "Play Count":
                data["play_count"] = int(value_elem.text) if value_elem.text else 0
            elif key_name == "Date Added":
                data["date_added"] = value_elem.text
            elif key_name == "Location":
                data["location"] = value_elem.text

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
                cursor = conn.cursor()
                for track in tracks:
                    if (
                        track.get("location")
                        and cursor.execute(
                            "SELECT id FROM tracks WHERE location = ?",
                            (track.get("location"),),
                        ).fetchone()
                        is None
                    ):
                        cursor.execute(
                            """
                            INSERT INTO tracks (
                                name, artist, album_artist, composer, album,
                                genre, year, total_time, track_number, disc_number,
                                play_count, date_added, location
                            )
                            VALUES (
                                :name, :artist, :album_artist, :composer, :album,
                                :genre, :year, :total_time, :track_number, :disc_number,
                                :play_count, :date_added, :location
                            )
                        """,
                            track,
                        )
                        count += 1
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"❌ Error inserting tracks: {e}")
        return count


def main():
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

    parser = LibraryParser(db_path=args.db_path)
    parser.parse_library(xml_path=args.xml_path)


if __name__ == "__main__":
    main()
