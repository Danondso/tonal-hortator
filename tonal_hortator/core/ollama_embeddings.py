import pickle
import sqlite3

import config
import ollama

# CONFIG
model_name = "nomic-embed-text"

# INIT DB
conn = sqlite3.connect(config.DB_PATH)
c = conn.cursor()

# CREATE EMBEDDING TABLE
c.execute(
    """
CREATE TABLE IF NOT EXISTS track_embeddings (
    track_id INTEGER PRIMARY KEY,
    embedding BLOB
)
"""
)
conn.commit()

# FETCH TRACKS
tracks = c.execute(
    "SELECT id, title, artist, album, genre, year FROM tracks"
).fetchall()

# EMBED TRACKS
instruction = "Represent this music track for semantic search:"

for t in tracks:
    track_id, title, artist, album, genre, year = t
    text = f"{instruction} {title} by {artist}, from album '{album}' ({genre}, {year})"

    response = ollama.embeddings(model=model_name, prompt=text)
    embedding = response["embedding"]

    embedding_blob = pickle.dumps(embedding)

    c.execute(
        "INSERT OR REPLACE INTO track_embeddings (track_id, embedding) VALUES (?, ?)",
        (track_id, embedding_blob),
    )

conn.commit()
conn.close()

print("Ollama embeddings generated.")
