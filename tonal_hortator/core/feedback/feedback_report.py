def seed_feedback() -> None:
    import sqlite3
    from datetime import datetime

    db_path = DB_PATH
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            track_id TEXT,
            feedback TEXT,
            adjustment REAL,
            timestamp TEXT,
            query_context TEXT,
            source TEXT DEFAULT 'user'
        )
    """
    )

    now = datetime.now().isoformat()

    print("🌱 Seeding preferences...")

    fav_artists = input(
        "Enter comma-separated favorite artists (e.g., 'Aphex Twin, Nujabes'): "
    )
    for artist in [a.strip() for a in fav_artists.split(",") if a.strip()]:
        track_id = f"artist:{artist.lower().replace(' ', '_')}"
        cur.execute(
            "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?)",
            (track_id, "like", 0.25, now, "seed", "seed"),
        )

    blocked_genres = input(
        "Enter comma-separated disliked genres (e.g., 'country, reggaeton'): "
    )
    for genre in [g.strip() for g in blocked_genres.split(",") if g.strip()]:
        track_id = f"genre:{genre.lower().replace(' ', '_')}"
        cur.execute(
            "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?)",
            (track_id, "dislike", -0.3, now, "seed", "seed"),
        )

    hard_blocks = input("Enter comma-separated hard blocks (artist, song, etc.): ")
    for block in [b.strip() for b in hard_blocks.split(",") if b.strip()]:
        track_id = f"block:{block.lower().replace(' ', '_')}"
        cur.execute(
            "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?)",
            (track_id, "block", -1.0, now, "seed", "seed"),
        )

    notes = input(
        "Enter optional notes or tags for this seeding session (press enter to skip): "
    )
    if notes.strip():
        track_id = f"notes:{notes.lower().replace(' ', '_')}"
        cur.execute(
            "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?)",
            (track_id, "note", 0.0, now, "seed", "seed"),
        )

    conn.commit()
    conn.close()
    print("✅ Seeding complete.\n")


import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

DB_PATH = "feedback.db"


def load_feedback() -> list[tuple[str, str, float, str, str, str]]:
    if not Path(DB_PATH).exists():
        print("No feedback database found.")
        return []

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT track_id, feedback, adjustment, timestamp, query_context, source
        FROM feedback
    """
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def summarize_feedback(rows: list[tuple[str, str, float, str, str, str]]) -> None:
    print(f"🧾 Total feedback entries: {len(rows)}\n")

    likes = [r for r in rows if r[1] == "like"]
    dislikes = [r for r in rows if r[1] == "dislike"]
    blocks = [r for r in rows if r[1] == "block"]

    print(f"👍 Likes: {len(likes)}")
    print(f"👎 Dislikes: {len(dislikes)}")
    print(f"🚫 Blocks: {len(blocks)}\n")

    # Frequency per track
    track_counter = Counter([r[0] for r in rows])
    print("🎵 Top 5 most frequently adjusted tracks:")
    for tid, count in track_counter.most_common(5):
        print(f"  {tid}: {count} entries")
    print()

    # Frequency per query context
    context_counter = Counter([r[4] for r in rows if r[4]])
    print("🔍 Top 5 query contexts:")
    for ctx, count in context_counter.most_common(5):
        print(f"  '{ctx}': {count} entries")
    print()

    # Histogram of decay age
    now = datetime.now()
    decay_bins: defaultdict[int, int] = defaultdict(int)
    for _, _, _, ts, _, _ in rows:
        try:
            weeks = (now - datetime.fromisoformat(ts)).days // 7
            decay_bins[weeks] += 1
        except Exception:
            pass

    print("⏳ Feedback age (in weeks):")
    for wk in sorted(decay_bins):
        print(f"  {wk:>2} weeks ago: {decay_bins[wk]} entries")

    # Add notes display
    notes = [r for r in rows if r[1] == "note"]
    if notes:
        print()
        print("📝 Notes from seeding:")
        for note in notes:
            print(f"  - {note[0].replace('notes:', '').replace('_', ' ')}")
        print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "seed":
        seed_feedback()
    else:
        rows = load_feedback()
        if rows:
            summarize_feedback(rows)
