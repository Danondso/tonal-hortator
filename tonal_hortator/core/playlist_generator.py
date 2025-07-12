#!/usr/bin/env python3
"""
Generate music playlists using local Ollama embeddings
This script creates playlists based on semantic search queries
"""

import logging
import os
import random
import re
import secrets
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from tonal_hortator.core.embeddings import OllamaEmbeddingService
from tonal_hortator.core.track_embedder import LocalTrackEmbedder

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# --- LLMQueryParser for extracting structured intent from queries ---
class LLMQueryParser:
    def __init__(self, model_name: str = "llama3:8b"):
        from tonal_hortator.core.llm_client import (  # You'll need to implement this or wrap Ollama
            LocalLLMClient,
        )

        self.model_name = model_name
        self.client = LocalLLMClient(model_name)

    def parse(self, query: str) -> dict:
        prompt = self._build_prompt(query)
        response = self.client.generate(prompt)
        return self._extract_json(response)

    def _build_prompt(self, query: str) -> str:
        return f"""You are a playlist assistant. Extract the user's intent from the following query and output it as a JSON object.

Preserve full genre names, including compound genres like "bedroom pop", "progressive metal", "lo-fi hip hop", etc. Do not shorten them to a single word. If the user mentions a specific number of tracks, include it in `count`. If they refer to artists or moods, extract those as well. Return only valid JSON.

Query: "{query}"

Output fields:
- count: (int or null)
- artist: (string or null)
- genres: (list of strings)
- mood: (string or null)
- unplayed: (true/false)
- vague: (true/false)

Examples:
Query: "5 bedroom pop songs"
Output: {{
  "count": 5,
  "artist": null,
  "genres": ["bedroom pop"],
  "mood": null,
  "unplayed": false,
  "vague": false
}}

Query: "jazz and rain on a stormy evening"
Output: {{
  "count": null,
  "artist": null,
  "genres": ["jazz"],
  "mood": "rain on a stormy evening",
  "unplayed": false,
  "vague": false
}}

Now respond for the current query.
"""

    def _extract_json(self, response: str) -> dict[Any, Any]:
        import json
        import re

        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            return json.loads(match.group(0))  # type: ignore
        raise ValueError("No JSON found in LLM response")


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
        self.query_parser = LLMQueryParser()

    def _determine_max_tracks(
        self, parsed_count: Optional[int], max_tracks: Optional[int]
    ) -> int:
        """
        Determine the final max_tracks value based on parsed count and provided max_tracks

        Args:
            parsed_count: Count extracted from the query (can be None)
            max_tracks: User-provided max_tracks (can be None)

        Returns:
            Final max_tracks value to use
        """
        # If max_tracks is None, use parsed count or fallback to default (20)
        if max_tracks is None:
            return parsed_count if parsed_count is not None else 20
        else:
            return parsed_count if parsed_count is not None else max_tracks

    def generate_playlist(
        self, query: str, max_tracks: Optional[int] = 20, min_similarity: float = 0.3
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

            # Extract intent from query using LLM
            parsed = self.query_parser.parse(query)

            # Determine final max_tracks value
            parsed_count = parsed.get("count")
            max_tracks = self._determine_max_tracks(parsed_count, max_tracks)

            artist = parsed.get("artist")
            genres = parsed.get("genres", [])
            # Note: mood and unplayed are parsed but not currently used in filtering
            is_vague = parsed.get("vague", False)

            logger.info(f"🧠 Parsed intent: {parsed}")
            logger.info(f"🎯 Using max_tracks={max_tracks}")

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
            is_artist_specific = bool(artist)
            if artist:
                logger.info(f"🎤 Found artist in query: '{artist}'. Filtering results.")
                results = [
                    track
                    for track in results
                    if artist.lower() in track.get("artist", "").lower()
                ]

            # 4. Apply genre filtering and boosting
            results = self._apply_genre_filtering(genres, results)

            # 5. Filter by similarity threshold and deduplicate
            filtered_results = self._filter_and_deduplicate_results(
                results, min_similarity, max_tracks, is_artist_specific
            )

            # 6. Apply artist randomization
            filtered_results = self._apply_artist_randomization(
                filtered_results, max_tracks, is_vague
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
        self, genres: List[str], results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply genre-aware filtering and boosting to search results

        Args:
            genres: List of genres to filter/boost (from parsed query)
            results: List of track results from similarity search

        Returns:
            Filtered and boosted results
        """
        if not results:
            return results

        genre_keywords = [g.lower() for g in genres if isinstance(g, str)]

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
        # Patterns to detect artist in query - more specific to avoid false positives
        patterns = [
            r"by\s+([a-zA-Z\s]+?)(?:\s+(?:for|with|and|songs|music|tracks)|$)",  # "songs by [artist]" - capture full name
            r"^([a-zA-Z\s]+?)\s+songs(?:\s|$)",  # "[artist] songs" - start of query only
            r"([a-zA-Z\s]+?)\s+radio(?:\s|$)",  # "[artist] radio"
        ]

        query_lower = query.lower()

        # Common words that shouldn't be treated as artist names
        common_words = {
            "relaxed",
            "calm",
            "upbeat",
            "happy",
            "sad",
            "energetic",
            "mellow",
            "acoustic",
            "electronic",
            "classical",
            "jazz",
            "rock",
            "pop",
            "folk",
            "grunge",
            "metal",
            "punk",
            "indie",
            "alternative",
            "country",
            "blues",
            "r&b",
            "hip-hop",
            "rap",
            "edm",
            "house",
            "techno",
            "meditation",
            "driving",
            "workout",
            "party",
            "romantic",
            "melancholy",
            "atmospheric",
            "ambient",
            "chill",
            "loud",
            "quiet",
            "fast",
            "slow",
            "complex",
            "simple",
            "modern",
            "traditional",
            "vintage",
            "contemporary",
            "bedroom-pop",
        }

        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                artist = match.group(1).strip()

                # Validate that this looks like an actual artist name
                if self._is_valid_artist_name(artist, common_words):
                    # Capitalize the artist name for better matching
                    return artist.title()

        return None

    def _is_valid_artist_name(self, artist: str, common_words: set) -> bool:
        """
        Validate if a potential artist name is actually an artist name
        """
        if not artist or len(artist) < 2:
            return False

        # Check if it's just common words
        artist_words = artist.lower().split()
        if len(artist_words) <= 2 and all(
            word in common_words for word in artist_words
        ):
            return False

        # Check if it contains common descriptive words that aren't artist names
        if any(
            word in artist.lower()
            for word in ["relaxed", "calm", "upbeat", "acoustic", "electronic"]
        ):
            return False

        # Must contain at least one word that looks like a name (not just adjectives)
        name_words = [word for word in artist_words if word not in common_words]
        if not name_words:
            return False

        return True

    def _filter_and_deduplicate_results(  # noqa: C901
        self,
        results: List[Dict[str, Any]],
        min_similarity: float,
        max_tracks: int,
        is_artist_specific: bool,
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
        smart_deduplicated = self._smart_name_deduplication(final_deduplicated)

        logger.info(
            f"🧠 Smart name deduplication: {len(final_deduplicated)} → {len(smart_deduplicated)} tracks"
        )

        # Strategy 5: Enforce artist diversity (NEW)
        # Calculate max tracks per artist based on playlist size
        # For artist-specific queries, be more lenient but still respect user's track count
        if is_artist_specific:
            # For artist-specific queries, skip artist diversity enforcement entirely
            # since we want all tracks from the specified artist
            logger.info(
                "🎤 Artist-specific query detected, skipping artist diversity enforcement"
            )
            diverse_tracks = smart_deduplicated
        else:
            # For general queries, maintain diversity
            max_tracks_per_artist = max(
                1, min(5, max_tracks // 3)
            )  # 1-5 tracks per artist, more lenient
            logger.info(
                f"🎵 General query, allowing up to {max_tracks_per_artist} tracks per artist"
            )

            diverse_tracks = self._enforce_artist_diversity(
                smart_deduplicated, max_tracks, max_tracks_per_artist
            )

        logger.info(
            f"✅ Final deduplication summary: {len(results)} → {len(diverse_tracks)} tracks"
        )

        # After artist diversity, use weighted random sampling for variance
        # Ensure we return exactly max_tracks
        top_k = max_tracks * 10
        diverse_tracks.sort(key=lambda t: t.get("similarity_score", 0), reverse=True)
        candidates = diverse_tracks[:top_k]

        if len(candidates) < max_tracks:
            # If we don't have enough candidates, return what we have
            logger.warning(
                f"⚠️ Only {len(candidates)} tracks available, requested {max_tracks}"
            )
            return candidates[:max_tracks]

        # Assign weights proportional to similarity score (shifted to be >=0)
        min_similarity_score = min(
            (t.get("similarity_score", 0) for t in candidates), default=0
        )
        weights = [
            max(0.0, t.get("similarity_score", 0) - min_similarity_score + 1e-6)
            for t in candidates
        ]

        # Sample exactly max_tracks unique tracks
        selected = set()
        final_tracks: list[dict[str, Any]] = []
        attempts = 0
        max_attempts = top_k * 3  # Increase attempts to ensure we get enough tracks
        total_weight = sum(weights)
        rng = secrets.SystemRandom()

        while len(final_tracks) < max_tracks and attempts < max_attempts:

            if total_weight > 0:
                # Generate a random value between 0 and total weight
                rand_val = rng.uniform(0, total_weight)
                cumulative_weight = 0
                pick = None

                # Find the selected item based on cumulative weights
                for i, weight in enumerate(weights):
                    cumulative_weight += weight
                    if rand_val <= cumulative_weight:
                        pick = candidates[i]
                        break

                if pick is None:
                    pick = candidates[-1]  # Fallback to last item
            else:
                # If no weights, use uniform random selection
                pick = rng.choice(candidates)

            pick_id = (
                pick.get("id") or f"{pick.get('name', '')}-{pick.get('artist', '')}"
            )
            if pick_id not in selected:
                final_tracks.append(pick)
                selected.add(pick_id)
            attempts += 1

        # If weighted sampling didn't give us enough tracks, fill with highest-similarity tracks
        if len(final_tracks) < max_tracks:
            logger.info(
                f"🎲 Weighted sampling gave {len(final_tracks)} tracks, filling with top similarity tracks"
            )
            for t in candidates:
                t_id = t.get("id") or f"{t.get('name', '')}-{t.get('artist', '')}"
                if t_id not in selected:
                    final_tracks.append(t)
                    selected.add(t_id)
                if len(final_tracks) >= max_tracks:
                    break

        # Ensure we return exactly max_tracks (or fewer if not enough available)
        final_tracks = final_tracks[:max_tracks]

        logger.info(
            f"🎲 Final playlist: {len(final_tracks)} tracks (requested: {max_tracks})"
        )
        return final_tracks

    def _group_tracks_by_artist(self, tracks: List[Dict[str, Any]]) -> dict[str, list]:
        """Group tracks by artist for deduplication"""
        artist_groups: dict[str, list] = {}
        for track in tracks:
            artist = track.get("artist", "").strip().lower()
            if artist:
                if artist not in artist_groups:
                    artist_groups[artist] = []
                artist_groups[artist].append(track)
        return artist_groups

    def _find_best_track_for_base_name(
        self, artist_tracks: List[Dict[str, Any]], base_name: str
    ) -> Optional[Dict[str, Any]]:
        """Find the track with the highest similarity score for a given base name"""
        best_track = None
        best_score = -1
        for track in artist_tracks:
            name = track.get("name", "").strip().lower()
            track_base = self._extract_base_name(name)
            if track_base == base_name:
                score = track.get("similarity_score", 0)
                if score > best_score:
                    best_score = score
                    best_track = track
        return best_track

    def _process_artist_tracks(
        self, artist_tracks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process tracks for a single artist to remove duplicates"""
        if len(artist_tracks) == 1:
            return [artist_tracks[0]]

        deduplicated = []
        processed_names = set()

        for track in artist_tracks:
            name = track.get("name", "").strip().lower()
            base_name = self._extract_base_name(name)

            if base_name and base_name not in processed_names:
                processed_names.add(base_name)
                best_track = self._find_best_track_for_base_name(
                    artist_tracks, base_name
                )
                if best_track:
                    deduplicated.append(best_track)
            elif not base_name:
                deduplicated.append(track)

        return deduplicated

    def _smart_name_deduplication(
        self, tracks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate tracks based on name and artist similarity"""
        if not tracks:
            return tracks

        # Group tracks by base name and artist
        name_artist_groups: Dict[str, List[Dict[str, Any]]] = {}

        for track in tracks:
            base_name = self._extract_base_name(track.get("name", ""))
            artist = track.get("artist", "")
            key = f"{base_name}|{artist}"

            if key not in name_artist_groups:
                name_artist_groups[key] = []
            name_artist_groups[key].append(track)

        # Keep the track with highest similarity score from each group
        deduplicated = []
        for group in name_artist_groups.values():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Sort by similarity score (descending) and keep the best
                best_track = max(group, key=lambda t: t.get("similarity_score", 0))
                deduplicated.append(best_track)

        return deduplicated

    def _extract_base_name(self, name: str) -> str:
        """Extract base name by removing common suffixes in parentheses"""
        # Remove common suffixes like (Remix), (Live), (Acoustic), etc.
        import re

        return re.sub(r"\s*\([^)]*\)$", "", name).strip()

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
                f.write("# Generated by Tonal Hortator (Local)\n")
                f.write(f"# Query: {query}\n")
                f.write(
                    f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"# Tracks: {len(tracks)}\n")
                f.write("# Format: M3U Extended\n")
                f.write("# Target: Apple Music\n\n")

                # Write tracks
                for i, track in enumerate(tracks, 1):
                    # Get track info
                    artist = track.get("artist", "Unknown")
                    name = track.get("name", "Unknown")
                    album = track.get("album", "Unknown")
                    duration = track.get("duration_ms", 0) // 1000  # Convert to seconds
                    similarity = track.get("similarity_score", 0)
                    location = track.get("location", "")

                    # Write extended info line
                    # Format: #EXTINF:duration,artist - title
                    f.write(f"#EXTINF:{duration},{artist} - {name}\n")

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
                        f.write(f"# Missing file location for: {artist} - {name}\n")

                    # Write additional metadata as comments
                    f.write(f"# Album: {album}\n")
                    f.write(f"# Similarity: {similarity:.3f}\n")
                    f.write(f"# Track: {i}/{len(tracks)}\n\n")

            logger.info(f"💾 Saved Apple Music playlist to: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"❌ Error saving playlist: {e}")
            raise

    def print_playlist_summary(self, tracks: List[Dict[str, Any]], query: str) -> None:
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
        # e.g., "15 tracks", "5 grunge songs", "a playlist of 10 songs", "top 20"
        patterns = [
            r"(\d+)\s+(?:tracks|songs)",
            r"(\d+)\s+\w+\s+(?:tracks|songs)",  # e.g., "5 grunge songs"
            r"top\s+(\d+)",
            r"playlist\s+of\s+(\d+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    def _is_vague_query(self, query: str) -> bool:
        """
        Detect if a query is vague and would benefit from artist randomization

        Args:
            query: Search query

        Returns:
            True if query is considered vague
        """
        query_lower = query.lower().strip()

        # Vague query patterns that typically return many artists
        vague_patterns = [
            # Generic music types
            r"\b(video game|game)\s+music\b",
            r"\b(early|late|mid)\s+\d{2}s\s+\w+\b",  # e.g., "early 90s grunge"
            r"\b(upbeat|happy|sad|melancholy|energetic|chill|relaxing)\s+(music|songs?|tracks?)\b",
            r"\b(party|workout|study|sleep|morning|evening|night)\s+(music|songs?|playlist)\b",
            r"\b(classic|old|vintage|retro|nostalgic)\s+(music|songs?)\b",
            r"\b(modern|contemporary|new|recent)\s+(music|songs?)\b",
            r"\b(background|ambient|atmospheric)\s+(music|songs?)\b",
            r"\b(instrumental|acoustic|electronic|digital)\s+(music|songs?)\b",
            # Generic genres without specific artists
            r"\b(rock|pop|jazz|blues|country|folk|electronic|hip.?hop|rap|r&b|soul|funk|disco|punk|metal|indie|alternative)\s+(music|songs?)\b",
            # Very short queries
            r"^\w{1,3}$",  # Single words or very short phrases
        ]

        for pattern in vague_patterns:
            if re.search(pattern, query_lower):
                return True

        # Check if query is very short (likely vague)
        if len(query_lower.split()) <= 2:
            return True

        return False

    def _apply_artist_randomization(
        self, tracks: List[Dict[str, Any]], max_tracks: int, is_vague: bool
    ) -> List[Dict[str, Any]]:
        """
        Apply artist randomization to increase diversity for vague queries

        Args:
            tracks: List of tracks to process
            max_tracks: Maximum number of tracks to return
            is_vague: Whether the query is considered vague

        Returns:
            List of tracks with improved artist diversity
        """
        if not is_vague or len(tracks) <= max_tracks:
            return tracks[:max_tracks]

        # Group tracks by artist
        artist_groups: dict[str, list[dict[str, Any]]] = {}
        for track in tracks:
            artist = track.get("artist", "").strip()
            if artist:
                if artist not in artist_groups:
                    artist_groups[artist] = []
                artist_groups[artist].append(track)

        # If we have many artists, prioritize diversity
        if (
            len(artist_groups) > max_tracks // 2
        ):  # More artists than half the max tracks
            logger.info(
                f"🎤 Vague query detected. Randomizing {len(artist_groups)} artists for diversity"
            )

            # Sort artists by their best track's similarity score
            artist_scores = []
            for artist, artist_tracks in artist_groups.items():
                best_score = max(
                    track.get("similarity_score", 0) for track in artist_tracks
                )
                artist_scores.append((artist, best_score, artist_tracks))

            # Sort by score (descending)
            artist_scores.sort(key=lambda x: x[1], reverse=True)

            # Take top artists but randomize their order slightly
            top_artists = artist_scores[:max_tracks]

            # Shuffle the order of artists (but keep top ones)
            if len(top_artists) > 3:
                # Keep top 3 artists in order, shuffle the rest
                top_3 = top_artists[:3]
                rest = top_artists[3:]
                random.shuffle(rest)
                top_artists = top_3 + rest

            # Take best track from each artist
            randomized_tracks = []
            for artist, _, artist_tracks in top_artists:
                # Sort tracks by similarity score and take the best one
                best_track = max(
                    artist_tracks, key=lambda t: t.get("similarity_score", 0)
                )
                randomized_tracks.append(best_track)

                if len(randomized_tracks) >= max_tracks:
                    break

            return randomized_tracks
        else:
            # Not enough artists to randomize, return original order
            return tracks[:max_tracks]

    def _enforce_artist_diversity(
        self,
        tracks: List[Dict[str, Any]],
        max_tracks: int,
        max_tracks_per_artist: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Enforce artist diversity by limiting tracks per artist

        Args:
            tracks: List of tracks to process
            max_tracks: Maximum number of tracks to return
            max_tracks_per_artist: Maximum number of tracks allowed per artist

        Returns:
            List of tracks with enforced artist diversity
        """
        if not tracks:
            return tracks

        # Group tracks by artist
        artist_groups: dict[str, list[dict[str, Any]]] = {}
        for track in tracks:
            artist = track.get("artist", "").strip()
            if artist:
                if artist not in artist_groups:
                    artist_groups[artist] = []
                artist_groups[artist].append(track)
            else:
                # Handle tracks without artist info
                if "unknown" not in artist_groups:
                    artist_groups["unknown"] = []
                artist_groups["unknown"].append(track)

        logger.info(
            f"🎤 Enforcing artist diversity: {len(artist_groups)} artists, max {max_tracks_per_artist} tracks per artist"
        )

        # Process each artist group
        diverse_tracks = []
        for artist, artist_tracks in artist_groups.items():
            if len(artist_tracks) > max_tracks_per_artist:
                # Sort by similarity score and take the best tracks
                sorted_tracks = sorted(
                    artist_tracks,
                    key=lambda t: t.get("similarity_score", 0),
                    reverse=True,
                )
                # Take only the best tracks up to the limit
                limited_tracks = sorted_tracks[:max_tracks_per_artist]
                diverse_tracks.extend(limited_tracks)
                logger.info(
                    f"🎵 Limited {artist}: {len(artist_tracks)} → {len(limited_tracks)} tracks"
                )
            else:
                # Artist is within limit, keep all tracks
                diverse_tracks.extend(artist_tracks)

        # Sort by similarity score and limit to max_tracks
        diverse_tracks.sort(key=lambda t: t.get("similarity_score", 0), reverse=True)
        final_tracks = diverse_tracks[:max_tracks]

        # Distribute artists throughout the playlist to avoid grouping
        distributed_tracks = self._distribute_artists(final_tracks, max_tracks)

        logger.info(
            f"✅ Artist diversity enforcement: {len(tracks)} → {len(distributed_tracks)} tracks"
        )
        return distributed_tracks

    def _distribute_artists(
        self, tracks: List[Dict[str, Any]], max_tracks: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Distribute artists throughout the playlist to avoid grouping

        Args:
            tracks: List of tracks to distribute
            max_tracks: Maximum number of tracks to return (if None, return all tracks)

        Returns:
            List of tracks with distributed artists
        """
        if len(tracks) <= 1:
            return tracks[:max_tracks] if max_tracks else tracks

        # Group tracks by artist
        artist_groups = self._group_tracks_by_artist(tracks)

        # If we only have one artist or very few tracks, no need to distribute
        if len(artist_groups) <= 1 or len(tracks) <= 3:
            return tracks[:max_tracks] if max_tracks else tracks

        logger.info(
            f"🎯 Distributing {len(artist_groups)} artists across {len(tracks)} tracks"
        )

        # Create a distribution pattern to spread artists evenly
        distributed: List[Dict[str, Any]] = []
        artist_queues = self._create_artist_queues(artist_groups)

        # Distribute tracks using round-robin with randomization
        distributed = self._round_robin_distribute(artist_queues, tracks, distributed)

        # Add any remaining tracks
        distributed = self._add_remaining_tracks(artist_queues, tracks, distributed)

        # Limit to max_tracks if specified
        if max_tracks and len(distributed) > max_tracks:
            distributed = distributed[:max_tracks]

        logger.info(
            f"🎵 Artist distribution complete: {len(distributed)} tracks distributed"
        )
        return distributed

    def _create_artist_queues(
        self, artist_groups: dict[str, list[dict[str, Any]]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Create queues for each artist's tracks"""
        artist_queues: dict[str, list[dict[str, Any]]] = {}
        for artist, artist_tracks in artist_groups.items():
            artist_queues[artist] = artist_tracks.copy()
        return artist_queues

    def _round_robin_distribute(
        self,
        artist_queues: dict[str, list[dict[str, Any]]],
        tracks: List[Dict[str, Any]],
        distributed: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Distribute tracks using round-robin with randomization"""
        while artist_queues and len(distributed) < len(tracks):
            # Get all artists that still have tracks
            available_artists = list(artist_queues.keys())

            # Add some randomization to avoid predictable patterns
            if len(available_artists) > 1:
                # Shuffle the order occasionally to add variety
                if len(distributed) % 3 == 0:  # Every 3rd track
                    random.shuffle(available_artists)

            # Take one track from each available artist in round-robin fashion
            for artist in available_artists:
                if len(distributed) >= len(tracks):
                    break

                if artist_queues[artist]:
                    track = artist_queues[artist].pop(0)
                    distributed.append(track)

                    # Remove artist from queue if no more tracks
                    if not artist_queues[artist]:
                        del artist_queues[artist]

        return distributed

    def _add_remaining_tracks(
        self,
        artist_queues: dict[str, list[dict[str, Any]]],
        tracks: List[Dict[str, Any]],
        distributed: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Add any remaining tracks to the distributed list"""
        # If we still have space and tracks left, add remaining tracks
        remaining_tracks: List[Dict[str, Any]] = []
        for artist_tracks in artist_queues.values():
            remaining_tracks.extend(artist_tracks)

        # Add remaining tracks at the end, maintaining artist diversity
        if remaining_tracks and len(distributed) < len(tracks):
            # Sort remaining tracks by similarity and add them
            remaining_tracks.sort(
                key=lambda t: t.get("similarity_score", 0), reverse=True
            )
            space_left = len(tracks) - len(distributed)
            distributed.extend(remaining_tracks[:space_left])

        return distributed


def _check_embeddings_available(generator: LocalPlaylistGenerator) -> bool:
    """Check if embeddings are available in the database"""
    stats = generator.track_embedder.get_embedding_stats()
    logger.info(f"📊 Database stats: {stats}")

    if stats["tracks_with_embeddings"] == 0:
        logger.warning("No embeddings found. Please run embed_tracks_local.py first.")
        return False
    return True


def _process_playlist_request(generator: LocalPlaylistGenerator, query: str) -> bool:
    """Process a single playlist request"""
    try:
        # Pass max_tracks=None so generate_playlist uses the count from the query
        tracks = generator.generate_playlist(query, max_tracks=None)

        if not tracks:
            print("❌ No tracks found for your query. Try a different search term.")
            return False

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

                    subprocess.run(["open", "-a", "Music", filepath], check=True)
                    print("🎵 Opened in Apple Music!")
                except Exception as e:
                    print(f"❌ Could not open in Apple Music: {e}")
                    print(
                        "💡 You can open it manually with: python open_in_apple_music.py"
                    )

        return True

    except Exception as e:
        logger.error(f"❌ Error generating playlist: {e}")
        print(f"❌ Error: {e}")
        return False


def _run_interactive_loop(generator: LocalPlaylistGenerator) -> None:
    """Run the interactive playlist generation loop"""
    print("\n🎵 Local Playlist Generator")
    print("=" * 40)

    while True:
        query = input("\nEnter your playlist query (or 'quit' to exit): ").strip()

        if query.lower() in ["quit", "exit", "q"]:
            break

        if not query:
            print("Please enter a query.")
            continue

        _process_playlist_request(generator, query)

    print("\n👋 Thanks for using the Local Playlist Generator!")


def main() -> None:
    """Main function for interactive playlist generation"""
    try:
        logger.info("🚀 Starting local playlist generator")

        # Initialize generator
        generator = LocalPlaylistGenerator()

        # Check if embeddings are available
        if not _check_embeddings_available(generator):
            return

        # Run interactive loop
        _run_interactive_loop(generator)

    except Exception as e:
        logger.error(f"❌ Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
