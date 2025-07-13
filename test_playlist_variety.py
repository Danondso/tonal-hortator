#!/usr/bin/env python3
"""
Test script to check playlist variety by running the same query multiple times
"""

import sys
import os
from typing import List, Set

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tonal_hortator.core.playlist.playlist_generator import LocalPlaylistGenerator
from tonal_hortator.core.embeddings.embeddings import OllamaEmbeddingService


def get_playlist_tracks(query: str, max_tracks: int = 5) -> List[str]:
    """Get playlist tracks for a given query"""
    try:
        generator = LocalPlaylistGenerator()
        playlist = generator.generate_playlist(query, max_tracks=max_tracks)

        # Extract track names in format "Artist - Song"
        tracks = []
        for track in playlist:
            artist = track.get("artist", "Unknown Artist")
            name = track.get("name", "Unknown Track")
            tracks.append(f"{artist} - {name}")

        return tracks
    except Exception as e:
        print(f"Error generating playlist: {e}")
        return []


def test_playlist_variety(query: str = "5 grunge songs", num_runs: int = 5):
    """Test variety by running the same query multiple times"""
    print(f"🎵 Testing playlist variety for: '{query}'")
    print(f"📊 Running {num_runs} times...\n")

    all_tracks = []
    unique_tracks = set()

    for i in range(num_runs):
        print(f"Run {i+1}:")
        tracks = get_playlist_tracks(query)
        all_tracks.append(tracks)
        unique_tracks.update(tracks)

        for j, track in enumerate(tracks, 1):
            print(f"  {j}. {track}")
        print()

    # Analyze results
    print("📈 Variety Analysis:")
    print(f"Total unique tracks found: {len(unique_tracks)}")
    print(f"Expected total tracks: {num_runs * 5}")
    print(f"Variety ratio: {len(unique_tracks) / (num_runs * 5):.2%}")

    # Check for duplicates
    print(f"\n🔍 Duplicate Analysis:")
    track_counts = {}
    for tracks in all_tracks:
        for track in tracks:
            track_counts[track] = track_counts.get(track, 0) + 1

    duplicates = {track: count for track, count in track_counts.items() if count > 1}
    if duplicates:
        print("Tracks that appeared multiple times:")
        for track, count in sorted(
            duplicates.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {track} (appeared {count} times)")
    else:
        print("✅ No duplicates found - perfect variety!")

    # Show all unique tracks
    print(f"\n📋 All unique tracks found:")
    for i, track in enumerate(sorted(unique_tracks), 1):
        print(f"  {i}. {track}")


if __name__ == "__main__":
    test_playlist_variety()
