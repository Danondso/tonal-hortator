"""
Centralized SQL queries for Tonal Hortator database operations.

This module contains all SQL queries used throughout the application
to ensure consistency and maintainability.
"""

# Table Creation Queries
CREATE_TRACKS_TABLE = """
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

CREATE_TRACK_EMBEDDINGS_TABLE = """
CREATE TABLE track_embeddings (
    track_id INTEGER PRIMARY KEY,
    embedding BLOB,
    embedding_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (track_id) REFERENCES tracks (id)
)
"""

CREATE_METADATA_MAPPINGS_TABLE = """
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

# Common Queries
GET_TRACKS_WITHOUT_EMBEDDINGS = """
SELECT
    t.id, t.name, t.artist, t.album, t.genre, t.year,
    t.play_count, t.album_artist, t.composer, t.total_time,
    t.track_number, t.disc_number, t.date_added, t.location,
    t.bpm, t.musical_key, t.key_scale, t.mood, t.label,
    t.producer, t.arranger, t.lyricist, t.original_year,
    t.original_date, t.chord_changes_rate, t.script,
    t.replay_gain, t.release_country
FROM tracks t
LEFT JOIN track_embeddings te ON t.id = te.track_id
WHERE te.track_id IS NULL
"""

INSERT_TRACK = """
INSERT INTO tracks (
    name, artist, album_artist, composer, album, genre, year,
    total_time, track_number, disc_number, play_count, date_added, location
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

INSERT_TRACK_EMBEDDING = """
INSERT OR REPLACE INTO track_embeddings
(track_id, embedding, embedding_text)
VALUES (?, ?, ?)
"""

INSERT_METADATA_MAPPING = """
INSERT INTO metadata_mappings 
(source_format, source_tag, normalized_tag, data_type, description)
VALUES (?, ?, ?, ?, ?)
"""

# Statistics Queries
GET_EMBEDDING_STATS = {
    "total_tracks": "SELECT COUNT(*) FROM tracks",
    "total_embeddings": "SELECT COUNT(*) FROM track_embeddings",
    "bpm": "SELECT COUNT(*) FROM tracks WHERE bpm IS NOT NULL",
    "musical_key": "SELECT COUNT(*) FROM tracks WHERE musical_key IS NOT NULL",
    "key_scale": "SELECT COUNT(*) FROM tracks WHERE key_scale IS NOT NULL",
    "mood": "SELECT COUNT(*) FROM tracks WHERE mood IS NOT NULL",
    "label": "SELECT COUNT(*) FROM tracks WHERE label IS NOT NULL",
    "producer": "SELECT COUNT(*) FROM tracks WHERE producer IS NOT NULL",
    "arranger": "SELECT COUNT(*) FROM tracks WHERE arranger IS NOT NULL",
    "lyricist": "SELECT COUNT(*) FROM tracks WHERE lyricist IS NOT NULL",
    "original_year": "SELECT COUNT(*) FROM tracks WHERE original_year IS NOT NULL",
}

# Utility Queries
CHECK_TABLE_EXISTS = """
SELECT name FROM sqlite_master 
WHERE type='table' AND name=?
"""

CHECK_TRACK_EXISTS = """
SELECT id FROM tracks WHERE location = ?
"""

GET_SAMPLE_MAPPINGS = """
SELECT source_format, source_tag, normalized_tag FROM metadata_mappings LIMIT 10
"""

# Test Queries (for unit tests)
TEST_CREATE_TRACKS_TABLE = """
CREATE TABLE tracks (
    id INTEGER PRIMARY KEY,
    name TEXT,
    artist TEXT,
    album TEXT,
    genre TEXT,
    year INTEGER,
    play_count INTEGER,
    location TEXT
)
"""

TEST_INSERT_TRACK = """
INSERT INTO tracks (id, name, artist, album, genre, year, play_count)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

TEST_INSERT_TRACK_EMBEDDING = """
INSERT INTO track_embeddings (track_id, embedding, embedding_text)
VALUES (?, ?, ?)
"""

TEST_GET_TRACK = """
SELECT name, artist FROM tracks WHERE id = ?
"""

TEST_GET_TRACK_COUNT = """
SELECT COUNT(*) FROM tracks
"""

TEST_GET_EMBEDDING_COUNT = """
SELECT COUNT(*) FROM track_embeddings
"""
