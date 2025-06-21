### tonal_hortator/parse_library.py

import plistlib
import sqlite3
import os

# CONFIG
xml_path = os.environ.get("ITUNES_LIBRARY_PATH", "Put your library path here")
db_path = "music_library.db"

# DATABASE INIT
conn = sqlite3.connect(db_path)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    artist TEXT,
    album TEXT,
    genre TEXT,
    year INTEGER,
    duration_ms INTEGER,
    bitrate INTEGER,
    bpm INTEGER,
    added TEXT,
    play_count INTEGER,
    album_artist TEXT,
    composer TEXT,
    location TEXT
)
""")
conn.commit()

# PARSE XML
with open(xml_path, 'rb') as f:
    plist = plistlib.load(f)

tracks = plist['Tracks']

# INSERT
for track_id, track in tracks.items():
    t = (
        track.get("Name"),
        track.get("Artist"),
        track.get("Album"),
        track.get("Genre"),
        track.get("Year"),
        track.get("Total Time"),
        track.get("Bit Rate"),
        track.get("BPM"),
        track.get("Date Added"),
        track.get("Play Count"),
        track.get("Album Artist"),
        track.get("Composer"),
        track.get("Location")
    )
    c.execute("""
    INSERT INTO tracks (title, artist, album, genre, year, duration_ms, bitrate, bpm, added, play_count, album_artist, composer, location)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, t)

conn.commit()
conn.close()
print("Library ingested into DB.")
