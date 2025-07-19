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
CREATE TABLE IF NOT EXISTS track_embeddings (
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

CREATE_USER_FEEDBACK_TABLE = """
CREATE TABLE IF NOT EXISTS user_feedback (
    id INTEGER PRIMARY KEY,
    query TEXT NOT NULL,
    query_type TEXT NOT NULL,
    parsed_artist TEXT,
    parsed_reference_artist TEXT,
    parsed_genres TEXT,
    parsed_mood TEXT,
    generated_tracks TEXT,
    user_rating INTEGER,
    user_comments TEXT,
    user_actions TEXT,
    playlist_length INTEGER,
    requested_length INTEGER,
    similarity_threshold REAL,
    search_breadth INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_USER_PREFERENCES_TABLE = """
CREATE TABLE IF NOT EXISTS user_preferences (
    id INTEGER PRIMARY KEY,
    preference_key TEXT UNIQUE NOT NULL,
    preference_value TEXT NOT NULL,
    preference_type TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_TRACK_RATINGS_TABLE = """
CREATE TABLE IF NOT EXISTS track_ratings (
    id INTEGER PRIMARY KEY,
    track_id INTEGER NOT NULL,
    rating INTEGER NOT NULL,
    context TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (track_id) REFERENCES tracks (id)
)
"""

CREATE_QUERY_LEARNING_TABLE = """
CREATE TABLE IF NOT EXISTS query_learning (
    id INTEGER PRIMARY KEY,
    original_query TEXT NOT NULL,
    llm_parsed_result TEXT NOT NULL,
    user_correction TEXT,
    feedback_score REAL,
    learning_applied BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    t.replay_gain, t.release_country,
    COALESCE(AVG(tr.rating), 0) as avg_rating,
    COUNT(tr.id) as rating_count
FROM tracks t
LEFT JOIN track_embeddings te ON t.id = te.track_id
LEFT JOIN track_ratings tr ON t.id = tr.track_id
WHERE te.track_id IS NULL
GROUP BY t.id, t.name, t.artist, t.album, t.genre, t.year,
         t.play_count, t.album_artist, t.composer, t.total_time,
         t.track_number, t.disc_number, t.date_added, t.location,
         t.bpm, t.musical_key, t.key_scale, t.mood, t.label,
         t.producer, t.arranger, t.lyricist, t.original_year,
         t.original_date, t.chord_changes_rate, t.script,
         t.replay_gain, t.release_country
"""

GET_TRACKS_WITH_RATINGS = """
SELECT
    t.id, t.name, t.artist, t.album, t.genre, t.year,
    t.play_count, t.album_artist, t.composer, t.total_time,
    t.track_number, t.disc_number, t.date_added, t.location,
    t.bpm, t.musical_key, t.key_scale, t.mood, t.label,
    t.producer, t.arranger, t.lyricist, t.original_year,
    t.original_date, t.chord_changes_rate, t.script,
    t.replay_gain, t.release_country,
    COALESCE(AVG(tr.rating), 0) as avg_rating,
    COUNT(tr.id) as rating_count
FROM tracks t
LEFT JOIN track_ratings tr ON t.id = tr.track_id
GROUP BY t.id, t.name, t.artist, t.album, t.genre, t.year,
         t.play_count, t.album_artist, t.composer, t.total_time,
         t.track_number, t.disc_number, t.date_added, t.location,
         t.bpm, t.musical_key, t.key_scale, t.mood, t.label,
         t.producer, t.arranger, t.lyricist, t.original_year,
         t.original_date, t.chord_changes_rate, t.script,
         t.replay_gain, t.release_country
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

INSERT_USER_FEEDBACK = """
INSERT INTO user_feedback (
    query, query_type, parsed_artist, parsed_reference_artist,
    parsed_genres, parsed_mood, generated_tracks, user_rating,
    user_comments, user_actions, playlist_length, requested_length,
    similarity_threshold, search_breadth
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

INSERT_USER_PREFERENCE = """
INSERT OR REPLACE INTO user_preferences (
    preference_key, preference_value, preference_type, description, updated_at
) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
"""

INSERT_TRACK_RATING = """
INSERT INTO track_ratings (track_id, rating, context)
VALUES (?, ?, ?)
"""

INSERT_QUERY_LEARNING = """
INSERT INTO query_learning (
    original_query, llm_parsed_result, user_correction, feedback_score
) VALUES (?, ?, ?, ?)
"""

# Feedback and preferences queries
GET_USER_FEEDBACK_STATS = {
    "total_feedback": "SELECT COUNT(*) FROM user_feedback",
    "average_rating": "SELECT AVG(user_rating) FROM user_feedback WHERE user_rating IS NOT NULL",
    "feedback_by_type": "SELECT query_type, COUNT(*) FROM user_feedback GROUP BY query_type",
    "recent_feedback": "SELECT * FROM user_feedback ORDER BY created_at DESC LIMIT 10",
}

GET_USER_PREFERENCES = """
SELECT preference_key, preference_value, preference_type, description
FROM user_preferences
ORDER BY updated_at DESC
"""

GET_TRACK_RATINGS = """
SELECT tr.rating, tr.context, t.name, t.artist
FROM track_ratings tr
JOIN tracks t ON tr.track_id = t.id
ORDER BY tr.created_at DESC
"""

GET_QUERY_LEARNING_DATA = """
SELECT original_query, llm_parsed_result, user_correction, feedback_score
FROM query_learning
WHERE learning_applied = FALSE
ORDER BY created_at DESC
"""

UPDATE_QUERY_LEARNING_APPLIED = """
UPDATE query_learning
SET learning_applied = TRUE
WHERE id = ?
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

# CSV Ingester Queries
CHECK_TRACK_BY_LOCATION = """
SELECT id FROM tracks WHERE location = ?
"""

# Note: Dynamic queries are handled by helper functions that use templates
UPDATE_TRACK_TEMPLATE = """
UPDATE tracks SET {fields} WHERE id = ?
"""

INSERT_TRACK_TEMPLATE = """
INSERT INTO tracks ({fields}) VALUES ({placeholders})
"""

# Embedding Updater Queries
GET_TRACK_EMBEDDING = """
SELECT embedding FROM track_embeddings WHERE track_id = ?
"""

INSERT_OR_REPLACE_TRACK_EMBEDDING = """
INSERT OR REPLACE INTO track_embeddings (track_id, embedding, embedding_text)
VALUES (?, ?, ?)
"""

# Note: Dynamic queries with placeholders are handled by helper functions
GET_TRACKS_BY_IDS_TEMPLATE = """
SELECT * FROM tracks WHERE id IN ({placeholders}) ORDER BY id
"""

DELETE_TRACK_EMBEDDINGS_BY_IDS_TEMPLATE = """
DELETE FROM track_embeddings WHERE track_id IN ({placeholders})
"""

# Utility Queries
GET_TRACK_COUNT = """
SELECT COUNT(*) FROM tracks
"""

GET_EMBEDDING_COUNT = """
SELECT COUNT(*) FROM track_embeddings
"""
