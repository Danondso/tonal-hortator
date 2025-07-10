import os
import unittest
from pathlib import Path

from tonal_hortator.utils.metadata_reader import MetadataReader

KENDRICK_FILE = Path(__file__).parent / "test_data" / "10 Sing About Me, I'm Dying of Thirst.mp3"


class TestMetadataReader(unittest.TestCase):
    def setUp(self) -> None:
        # Use the default database path or override if needed
        self.db_path = "music_library.db"
        self.reader = MetadataReader(self.db_path)

    def test_kendrick_lamar_metadata_mapping(self) -> None:
        if not os.path.exists(KENDRICK_FILE):
            self.skipTest(f"Test file not found: {KENDRICK_FILE}")

        # Read raw metadata
        raw_metadata = self.reader._read_mp3_metadata(Path(KENDRICK_FILE))
        print("\nRAW METADATA:")
        for k, v in raw_metadata.items():
            print(f"  {k}: {v}")

        # Normalize metadata
        normalized = self.reader._normalize_metadata(raw_metadata, "mp3")
        print("\nNORMALIZED METADATA:")
        for k, v in normalized.items():
            print(f"  {k}: {v}")

        # Get DB columns
        import sqlite3

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(tracks)")
            db_columns = {col[1] for col in cursor.fetchall()}
        print("\nDB COLUMNS:")
        for col in sorted(db_columns):
            print(f"  {col}")

        # Check that at least one normalized key matches a DB column
        matches = [k for k in normalized if k in db_columns]
        print(f"\nMAPPED FIELDS IN DB: {matches}")
        self.assertTrue(matches, "No normalized metadata fields match DB columns!")


if __name__ == "__main__":
    unittest.main()
