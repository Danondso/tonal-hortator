import sqlite3
import numpy as np
import sys
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'music_library.db')
EMBEDDING_DIM = 768  # For nomic-embed-text

def main():
    db_path = DB_PATH
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    print(f"🔍 Cleaning up embeddings in: {db_path}")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Count total embeddings
    cur.execute("SELECT COUNT(*) FROM track_embeddings")
    total = cur.fetchone()[0]

    # Find invalid embeddings (empty or wrong dimension)
    cur.execute("SELECT track_id, embedding FROM track_embeddings")
    to_delete = []
    for track_id, blob in cur.fetchall():
        try:
            arr = np.frombuffer(blob, dtype=np.float32)
            if arr.size != EMBEDDING_DIM:
                to_delete.append(track_id)
        except Exception:
            to_delete.append(track_id)

    # Delete invalid embeddings
    for track_id in to_delete:
        cur.execute("DELETE FROM track_embeddings WHERE track_id = ?", (track_id,))
    conn.commit()

    print(f"✅ Removed {len(to_delete)} invalid embeddings out of {total} total.")
    if to_delete:
        print(f"🗑️  Track IDs removed: {to_delete}")
    else:
        print("🎉 No invalid embeddings found!")
    conn.close()

if __name__ == "__main__":
    main() 