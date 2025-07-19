#!/usr/bin/env python3
"""
CSV ingestion module for importing music data from iPod exports.
Handles parsing music.csv and updating the tracks table.
"""

import csv
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from tonal_hortator.core.database import (
    CHECK_TABLE_EXISTS,
    CHECK_TRACK_BY_LOCATION,
    CREATE_TRACKS_TABLE,
)
from tonal_hortator.core.database.query_helpers import (
    build_insert_track_query,
    build_update_track_query,
)
from tonal_hortator.utils.loader import create_progress_spinner

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

CSV_FIELD_MAPPING = {
    "Song Title": "name",
    "Artist": "artist",
    "Album": "album",
    "Year released": "year",
    "Song duration (seconds)": "total_time",
    "Play count": "play_count",
    "Genre": "genre",
    "Composer": "composer",
    "Filename": "location",  # Use as unique identifier
    "Added to library on (timestamp)": "date_added",
}


class MusicCSVIngester:
    def __init__(self, db_path: str = "music_library.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._ensure_tracks_table()

    def _ensure_tracks_table(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute(CHECK_TABLE_EXISTS, ("tracks",))
        if not cursor.fetchone():
            logger.info("Creating tracks table...")
            cursor.execute(CREATE_TRACKS_TABLE)
            self.conn.commit()
            logger.info("✅ Created tracks table")
        else:
            logger.info("✅ Tracks table already exists")

    def _normalize_track_data(self, csv_row: Dict[str, str]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for csv_field, db_field in CSV_FIELD_MAPPING.items():
            value = csv_row.get(csv_field, None)
            if value is not None:
                value = value.strip()
                is_numeric: bool = (
                    db_field == "year"
                    or db_field == "play_count"
                    or db_field == "total_time"
                )

                if is_numeric:
                    try:
                        normalized[db_field] = int(value) if value else None
                    except Exception:
                        normalized[db_field] = None
                else:
                    normalized[db_field] = value if value else None
        if not normalized.get("name"):
            normalized["name"] = "Unknown Track"
        if not normalized.get("artist"):
            normalized["artist"] = "Unknown Artist"
        return normalized

    def _check_track_exists(self, location: str) -> Optional[int]:
        cursor = self.conn.cursor()
        cursor.execute(CHECK_TRACK_BY_LOCATION, (location,))
        result = cursor.fetchone()
        return result[0] if result else None

    def _update_track(self, track_id: int, track_data: Dict[str, Any]) -> bool:
        cursor = self.conn.cursor()
        fields = []
        values = []
        for field, value in track_data.items():
            if field != "location" and value is not None:
                fields.append(field)
                values.append(value)
        if not fields:
            return True
        values.append(track_id)
        query = build_update_track_query(fields)
        cursor.execute(query, values)
        self.conn.commit()
        return True

    def _insert_track(self, track_data: Dict[str, Any]) -> Optional[int]:
        cursor = self.conn.cursor()
        fields = []
        values = []
        for field, value in track_data.items():
            if value is not None:
                fields.append(field)
                values.append(value)
        if not fields:
            return None
        query = build_insert_track_query(fields)
        cursor.execute(query, values)
        self.conn.commit()
        return cursor.lastrowid

    def ingest_csv(
        self, csv_path: str, dry_run: bool = False, batch_size: int = 100
    ) -> Dict[str, Any]:
        csv_path_obj = Path(csv_path)
        if not csv_path_obj.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        logger.info(f"Starting CSV ingestion from: {csv_path}")
        if dry_run:
            logger.info("DRY RUN MODE - No database changes will be made")
        stats: Dict[str, Any] = {
            "total_rows": 0,
            "processed": 0,
            "inserted": 0,
            "updated": 0,
            "skipped": 0,
            "errors": 0,
            "tracks_to_update_embeddings": [],
        }
        with open(csv_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            required_fields = ["Song Title", "Artist", "Filename"]
            if reader.fieldnames is None:
                raise ValueError("CSV file has no headers")
            missing_fields = [
                field for field in required_fields if field not in reader.fieldnames
            ]
            if missing_fields:
                raise ValueError(f"Missing required CSV fields: {missing_fields}")
            rows = list(reader)
            stats["total_rows"] = len(rows)
            if stats["total_rows"] == 0:
                logger.warning("CSV file is empty")
                return stats
            spinner = create_progress_spinner(
                stats["total_rows"], "Ingesting CSV data", batch_size
            )
            spinner.start()
            for row in rows:
                try:
                    track_data = self._normalize_track_data(row)
                    if not track_data.get("location"):
                        logger.warning(
                            f"Skipping track without location: {track_data.get('name', 'Unknown')}"
                        )
                        stats["skipped"] += 1
                        spinner.update(1)
                        continue
                    existing_id = self._check_track_exists(track_data["location"])
                    if existing_id:
                        if not dry_run:
                            self._update_track(existing_id, track_data)
                        stats["updated"] += 1
                        stats["tracks_to_update_embeddings"].append(existing_id)
                    else:
                        if not dry_run:
                            new_id = self._insert_track(track_data)
                        else:
                            new_id = 0
                        stats["inserted"] += 1
                        if new_id:
                            stats["tracks_to_update_embeddings"].append(new_id)
                    stats["processed"] += 1
                except Exception as e:
                    logger.error(f"Error processing row: {e}")
                    stats["errors"] += 1
                spinner.update(1)
            spinner.stop()
        logger.info(f"CSV ingestion complete: {stats}")
        return stats

    def close(self) -> None:
        if self.conn:
            self.conn.close()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python csv_ingester.py <csv_file> [--dry-run]")
        sys.exit(1)
    csv_file = sys.argv[1]
    dry_run = "--dry-run" in sys.argv
    try:
        ingester = MusicCSVIngester()
        stats = ingester.ingest_csv(csv_file, dry_run=dry_run)
        if dry_run:
            print("DRY RUN COMPLETE - No changes made to database")
        else:
            print("CSV ingestion completed successfully!")
        print(f"Statistics: {stats}")
    except Exception as e:
        logger.error(f"CSV ingestion failed: {e}")
        sys.exit(1)
    finally:
        ingester.close()


if __name__ == "__main__":
    main()
