#!/usr/bin/env python3
"""
Generate music playlists using local Ollama embeddings
This script creates playlists based on semantic search queries
"""

import logging
import os
import re
import sqlite3
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from tonal_hortator.core.embeddings import OllamaEmbeddingService
from tonal_hortator.core.track_embedder import LocalTrackEmbedder

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LocalPlaylistGenerator:
    """Generate playlists using local Ollama embeddings"""

    def __init__(
        self,
        db_path: str = "music_library.db",
        model_name: str = "nomic-embed-text:latest",
    ):
        """
        Initialize the local playlist generator

        Args:
            db_path: Path to SQLite database
            model_name: Name of the embedding model to use
        """
        self.db_path = db_path
        self.embedding_service = OllamaEmbeddingService(model_name=model_name)
        self.track_embedder = LocalTrackEmbedder(
            db_path, embedding_service=self.embedding_service
        )

    def generate_playlist(
        self, query: str, max_tracks: int = 20, min_similarity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Generate a playlist based on a semantic query

        Args:
            query: Search query (e.g., "upbeat rock songs")
            max_tracks: Maximum number of tracks in playlist
            min_similarity: Minimum similarity score threshold

        Returns:
            List of track dictionaries with similarity scores
        """
        try:
            logger.info(f"🎵 Generating playlist for query: '{query}'")
            start_time = time.time()

            # 1. Extract track count from query
            extracted_count = self._extract_track_count(query)
            if extracted_count:
                max_tracks = extracted_count
                logger.info(
                    f"🔢 Found track count in query, setting max_tracks to {max_tracks}"
                )

            # 2. Get embeddings from database
            embeddings, track_data = self.track_embedder.get_all_embeddings()

            if not embeddings or not track_data:
                logger.warning(
                    "No embeddings found. Running embedding process first..."
                )
                self.track_embedder.embed_all_tracks()
                embeddings, track_data = self.track_embedder.get_all_embeddings()

                if not embeddings:
                    logger.error(
                        "❌ Still no embeddings available after embedding process"
                    )
                    return []

            logger.info(f"📊 Using {len(embeddings)} track embeddings for search")

            # Perform similarity search
            results = self.embedding_service.similarity_search(
                query, embeddings, track_data, top_k=max_tracks * 5
            )

            # 3. Filter by artist if detected
            artist = self._extract_artist_from_query(query)
            if artist:
                logger.info(f"🎤 Found artist in query: '{artist}'. Filtering results.")
                results = [
                    track
                    for track in results
                    if artist.lower() in track.get("artist", "").lower()
                ]

            # 4. Apply genre filtering and boosting
            results = self._apply_genre_filtering(query, results)

            # 5. Filter by similarity threshold and deduplicate
            filtered_results = self._filter_and_deduplicate_results(
                results, min_similarity, max_tracks
            )

            generation_time = time.time() - start_time
            logger.info(
                f"✅ Generated playlist with {len(filtered_results)} tracks in {generation_time:.2f}s"
            )

            return filtered_results

        except Exception as e:
            logger.error(f"❌ Error generating playlist: {e}")
            raise

    def _apply_genre_filtering(
        self, query: str, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply genre-aware filtering and boosting to search results

        Args:
            query: Original search query
            results: List of track results from similarity search

        Returns:
            Filtered and boosted results
        """
        if not results:
            return results

        # Extract potential genres from query
        query_lower = query.lower()
        genre_keywords = self._extract_genre_keywords(query_lower)

        if not genre_keywords:
            logger.info(
                "🎵 No specific genre detected in query, using semantic similarity only"
            )
            return results

        logger.info(f"🎵 Detected genre keywords: {genre_keywords}")

        # Separate tracks by genre match
        genre_matches = []
        other_tracks = []

        for track in results:
            track_genre = track.get("genre", "").lower() if track.get("genre") else ""
            similarity_score = track.get("similarity_score", 0)

            # Check if track genre matches any of the detected genre keywords
            genre_match = any(keyword in track_genre for keyword in genre_keywords)

            if genre_match:
                # Boost similarity score for genre matches
                boosted_score = min(
                    1.0, similarity_score + 0.1
                )  # Boost by 0.1, cap at 1.0
                track["similarity_score"] = boosted_score
                track["genre_boosted"] = True
                genre_matches.append(track)
            else:
                track["genre_boosted"] = False
                other_tracks.append(track)

        # Combine results: genre matches first, then others
        # But only include a limited number of non-genre matches to maintain quality
        max_other_tracks = max(
            5, len(genre_matches) // 2
        )  # At most half as many non-genre tracks
        combined_results = genre_matches + other_tracks[:max_other_tracks]

        logger.info(
            f"🎵 Genre filtering: {len(genre_matches)} genre matches, {len(other_tracks[:max_other_tracks])} other tracks"
        )

        return combined_results

    def _extract_genre_keywords(self, query: str) -> List[str]:
        """
        Extract potential genre keywords from a query

        Args:
            query: Lowercase search query

        Returns:
            List of detected genre keywords
        """
        # Common music genres and their variations
        genre_mapping = {
            "jazz": ["jazz", "bebop", "swing", "fusion", "smooth jazz", "acid jazz"],
            "rock": [
                "rock",
                "hard rock",
                "soft rock",
                "classic rock",
                "punk rock",
                "indie rock",
            ],
            "pop": ["pop", "pop rock", "synth pop", "indie pop"],
            "hip hop": ["hip hop", "rap", "trap", "r&b", "soul"],
            "electronic": [
                "electronic",
                "edm",
                "techno",
                "house",
                "trance",
                "ambient",
                "dubstep",
            ],
            "country": ["country", "folk", "bluegrass", "americana"],
            "classical": ["classical", "orchestral", "symphony", "chamber"],
            "blues": ["blues", "delta blues", "electric blues"],
            "reggae": ["reggae", "dub", "ska"],
            "metal": ["metal", "heavy metal", "death metal", "black metal"],
            "funk": ["funk", "disco", "motown"],
            "latin": ["latin", "salsa", "bossa nova", "tango"],
            "world": ["world", "ethnic", "traditional"],
            "soundtrack": ["soundtrack", "score", "film music"],
            "christian": ["christian", "gospel", "worship"],
            "children": ["children", "kids", "nursery"],
            "comedy": ["comedy", "humor", "parody"],
            "holiday": ["holiday", "christmas", "xmas", "seasonal"],
        }

        detected_genres = []

        for genre, keywords in genre_mapping.items():
            if any(keyword in query for keyword in keywords):
                detected_genres.append(genre)

        return detected_genres

    def _extract_artist_from_query(self, query: str) -> Optional[str]:
        """
        Extract artist from a query using patterns like "by [artist]" or "[artist] songs"
        """
        # Patterns to detect artist in query
        patterns = [
            r"by\s+(.+)",  # "songs by [artist]"
            r"(.+?)\s+songs",  # "[artist] songs"
            r"(.+?)\s+radio",  # "[artist] radio"
        ]

        query_lower = query.lower()

        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                artist = match.group(1).strip()
                # Capitalize the artist name for better matching
                return artist.title()

        return None

    def _filter_and_deduplicate_results(
        self, results: List[Dict[str, Any]], min_similarity: float, max_tracks: int
    ) -> List[Dict[str, Any]]:
        """Filter results by similarity and remove duplicates using multiple strategies"""
        # Filter by similarity threshold
        filtered = [
            track
            for track in results
            if track.get("similarity_score", 0) >= min_similarity
        ]

        logger.info(
            f"🔍 Starting deduplication on {len(filtered)} tracks (similarity ≥ {min_similarity})"
        )

        # Strategy 1: Deduplicate by file location (normalized path)
        seen_locations = set()
        location_deduplicated = []

        for track in filtered:
            location = self._normalize_file_location(track.get("location", ""))
            if location and location not in seen_locations:
                seen_locations.add(location)
                location_deduplicated.append(track)
            elif not location:  # Include tracks without location
                location_deduplicated.append(track)

        logger.info(
            f"📍 Location deduplication: {len(filtered)} → {len(location_deduplicated)} tracks"
        )

        # Strategy 2: Deduplicate by title/artist combination
        seen_combinations = set()
        combination_deduplicated = []

        for track in location_deduplicated:
            title = track.get("name", "").strip().lower()
            artist = track.get("artist", "").strip().lower()
            combination = f"{title}|{artist}"

            if combination and combination not in seen_combinations:
                seen_combinations.add(combination)
                combination_deduplicated.append(track)
            elif not combination:  # Include tracks without title/artist
                combination_deduplicated.append(track)

        logger.info(
            f"🎵 Title/Artist deduplication: {len(location_deduplicated)} → {len(combination_deduplicated)} tracks"
        )

        # Strategy 3: Deduplicate by track ID (shouldn't happen, but just in case)
        seen_ids = set()
        final_deduplicated = []

        for track in combination_deduplicated:
            track_id = track.get("id")
            if track_id not in seen_ids:
                seen_ids.add(track_id)
                final_deduplicated.append(track)

        logger.info(
            f"🆔 Track ID deduplication: {len(combination_deduplicated)} → {len(final_deduplicated)} tracks"
        )

        # Strategy 4: Smart deduplication for similar titles with slight variations
        smart_deduplicated = self._smart_title_deduplication(final_deduplicated)

        logger.info(
            f"🧠 Smart title deduplication: {len(final_deduplicated)} → {len(smart_deduplicated)} tracks"
        )

        # Limit to max_tracks
        final_results = smart_deduplicated[:max_tracks]

        logger.info(
            f"✅ Final deduplication summary: {len(results)} → {len(final_results)} tracks"
        )

        return final_results

    def _smart_title_deduplication(
        self, tracks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Smart deduplication for titles with slight variations (e.g., 'Song (Remix)' vs 'Song')"""
        if len(tracks) <= 1:
            return tracks

        # Group tracks by artist
        artist_groups = {}
        for track in tracks:
            artist = track.get("artist", "").strip().lower()
            if artist:
                if artist not in artist_groups:
                    artist_groups[artist] = []
                artist_groups[artist].append(track)

        deduplicated = []

        for artist, artist_tracks in artist_groups.items():
            if len(artist_tracks) == 1:
                deduplicated.append(artist_tracks[0])
                continue

            # For multiple tracks by same artist, check for similar titles
            processed_titles = set()

            for track in artist_tracks:
                title = track.get("name", "").strip().lower()
                base_title = self._extract_base_title(title)

                if base_title and base_title not in processed_titles:
                    processed_titles.add(base_title)
                    # Keep the track with highest similarity score
                    best_track = track
                    for other_track in artist_tracks:
                        other_title = other_track.get("name", "").strip().lower()
                        other_base = self._extract_base_title(other_title)
                        if other_base == base_title:
                            if other_track.get("similarity_score", 0) > best_track.get(
                                "similarity_score", 0
                            ):
                                best_track = other_track
                    deduplicated.append(best_track)
                elif not base_title:
                    deduplicated.append(track)

        return deduplicated

    def _extract_base_title(self, title: str) -> str:
        """Extract base title by removing common suffixes and variations"""
        if not title:
            return ""

        # Remove common suffixes in parentheses
        import re

        base = re.sub(r"\s*\([^)]*\)\s*$", "", title)

        # Remove common suffixes
        suffixes_to_remove = [
            " (remix)",
            " (remastered)",
            " (live)",
            " (acoustic)",
            " (radio edit)",
            " (extended)",
            " (clean)",
            " (explicit)",
            " (original mix)",
            " (club mix)",
            " (dub mix)",
            " - remix",
            " - remastered",
            " - live",
            " - acoustic",
        ]

        for suffix in suffixes_to_remove:
            if base.lower().endswith(suffix.lower()):
                base = base[: -len(suffix)]
                break

        return base.strip()

    def _normalize_file_location(self, location: str) -> str:
        """Normalize file location for deduplication and Apple Music compatibility"""
        if not location:
            return ""

        # Convert to lowercase and normalize separators
        normalized = location.lower().replace("\\", "/")

        # Remove common OS-specific prefixes with username
        import re

        # Match /users/<username>/, /home/<username>/, c:/users/<username>/, etc.
        match = re.match(
            r"^(?:/users/|/home/|c:/users/|d:/users/|e:/users/)([^/]+)/(.+)$",
            normalized,
        )
        if match:
            return match.group(2)
        return normalized

    def save_playlist_m3u(
        self, tracks: List[Dict[str, Any]], query: str, output_dir: str = "playlists"
    ) -> str:
        """
        Save playlist as M3U file optimized for Apple Music

        Args:
            tracks: List of track dictionaries
            query: Original search query
            output_dir: Output directory for playlist files

        Returns:
            Path to saved playlist file
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Create filename with timestamp and sanitized query
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized_query = re.sub(r"[^\w\s-]", "", query).strip()
            sanitized_query = re.sub(r"[-\s]+", "-", sanitized_query)
            sanitized_query = sanitized_query[:50]  # Limit length

            filename = f"playlist_{timestamp}_{sanitized_query}.m3u"
            filepath = os.path.join(output_dir, filename)

            # Write M3U file optimized for Apple Music
            with open(filepath, "w", encoding="utf-8") as f:
                # Write M3U header
                f.write("#EXTM3U\n")

                # Write playlist metadata
                f.write(f"# Generated by Tonal Hortator (Local)\n")
                f.write(f"# Query: {query}\n")
                f.write(
                    f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"# Tracks: {len(tracks)}\n")
                f.write(f"# Format: M3U Extended\n")
                f.write(f"# Target: Apple Music\n\n")

                # Write tracks
                for i, track in enumerate(tracks, 1):
                    # Get track info
                    artist = track.get("artist", "Unknown")
                    title = track.get("name", "Unknown")
                    album = track.get("album", "Unknown")
                    duration = track.get("duration_ms", 0) // 1000  # Convert to seconds
                    similarity = track.get("similarity_score", 0)
                    location = track.get("location", "")

                    # Write extended info line
                    # Format: #EXTINF:duration,artist - title
                    f.write(f"#EXTINF:{duration},{artist} - {title}\n")

                    # Write file path
                    if location:
                        # Convert file:// URLs to local paths for better Apple Music compatibility
                        if location.startswith("file://"):
                            # Remove file:// prefix and decode URL encoding
                            import urllib.parse

                            local_path = urllib.parse.unquote(location[7:])
                            f.write(f"{local_path}\n")
                        else:
                            f.write(f"{location}\n")
                    else:
                        # If no location, write a comment
                        f.write(f"# Missing file location for: {artist} - {title}\n")

                    # Write additional metadata as comments
                    f.write(f"# Album: {album}\n")
                    f.write(f"# Similarity: {similarity:.3f}\n")
                    f.write(f"# Track: {i}/{len(tracks)}\n\n")

            logger.info(f"💾 Saved Apple Music playlist to: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"❌ Error saving playlist: {e}")
            raise

    def print_playlist_summary(self, tracks: List[Dict[str, Any]], query: str):
        """Print a summary of the generated playlist"""
        print(f"\n🎵 Playlist for: '{query}'")
        print(f"📊 {len(tracks)} tracks found")
        print("-" * 60)

        for i, track in enumerate(tracks, 1):
            artist = track.get("artist", "Unknown")
            name = track.get("name", "Unknown")
            album = track.get("album", "Unknown")
            similarity = track.get("similarity_score", 0)

            print(f"{i:2d}. {artist} - {name}")
            print(f"    Album: {album}")
            print(f"    Similarity: {similarity:.3f}")
            print()

        if tracks:
            avg_similarity = sum(t.get("similarity_score", 0) for t in tracks) / len(
                tracks
            )
            print(f"📈 Average similarity: {avg_similarity:.3f}")
        print("-" * 60)

    def _extract_track_count(self, query: str) -> Optional[int]:
        """Extract desired track count from query using regex"""
        # Patterns to look for a number followed by "tracks" or "songs"
        # e.g., "15 tracks", "a playlist of 10 songs", "top 20"
        patterns = [
            r"(\d+)\s+(?:tracks|songs)",
            r"top\s+(\d+)",
            r"playlist\s+of\s+(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None


def main():
    """Main function for interactive playlist generation"""
    try:
        logger.info("🚀 Starting local playlist generator")

        # Initialize generator
        generator = LocalPlaylistGenerator()

        # Get embedding stats
        stats = generator.track_embedder.get_embedding_stats()
        logger.info(f"📊 Database stats: {stats}")

        if stats["tracks_with_embeddings"] == 0:
            logger.warning(
                "No embeddings found. Please run embed_tracks_local.py first."
            )
            return

        # Interactive mode
        print("\n🎵 Local Playlist Generator")
        print("=" * 40)

        while True:
            query = input("\nEnter your playlist query (or 'quit' to exit): ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                break

            if not query:
                print("Please enter a query.")
                continue

            try:
                # Generate playlist
                tracks = generator.generate_playlist(query, max_tracks=20)

                if not tracks:
                    print(
                        "❌ No tracks found for your query. Try a different search term."
                    )
                    continue

                # Print summary
                generator.print_playlist_summary(tracks, query)

                # Ask if user wants to save
                save = input("\nSave playlist to file? (y/n): ").strip().lower()
                if save in ["y", "yes"]:
                    filepath = generator.save_playlist_m3u(tracks, query)
                    print(f"✅ Playlist saved to: {filepath}")

                    # Ask if user wants to open in Apple Music
                    open_music = input("Open in Apple Music? (y/n): ").strip().lower()
                    if open_music in ["y", "yes"]:
                        try:
                            import subprocess

                            subprocess.run(
                                ["open", "-a", "Music", filepath], check=True
                            )
                            print("🎵 Opened in Apple Music!")
                        except Exception as e:
                            print(f"❌ Could not open in Apple Music: {e}")
                            print(
                                "💡 You can open it manually with: python open_in_apple_music.py"
                            )

            except Exception as e:
                logger.error(f"❌ Error generating playlist: {e}")
                print(f"❌ Error: {e}")

        print("\n👋 Thanks for using the Local Playlist Generator!")

    except Exception as e:
        logger.error(f"❌ Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
