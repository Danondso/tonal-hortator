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
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

from tonal_hortator.core.config import get_config
from tonal_hortator.core.embeddings.embeddings import OllamaEmbeddingService
from tonal_hortator.core.embeddings.track_embedder import LocalTrackEmbedder
from tonal_hortator.core.feedback import FeedbackManager
from tonal_hortator.core.models import Track

from .feedback_service import FeedbackService, PlaylistFeedbackService
from .llm_query_parser import LLMQueryParser
from .playlist_deduplicator import PlaylistDeduplicator
from .playlist_exporter import PlaylistExporter
from .playlist_filter import PlaylistFilter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class LocalPlaylistGenerator:
    """Generate playlists using local Ollama embeddings."""

    def __init__(
        self,
        db_path: str = "music_library.db",
        model_name: Optional[str] = None,
        feedback_service: Optional[FeedbackService] = None,
        deduplicator: Optional[PlaylistDeduplicator] = None,
        filter_: Optional[PlaylistFilter] = None,
        exporter: Optional[PlaylistExporter] = None,
    ):
        self.db_path = db_path
        self.config = get_config()

        # Use configured model name if not provided
        if model_name is None:
            model_name = self.config.get(
                "llm.embedding_model", "nomic-embed-text:latest"
            )

        self.embedding_service = OllamaEmbeddingService(model_name=model_name)
        self.track_embedder = LocalTrackEmbedder(
            db_path, embedding_service=self.embedding_service
        )
        self.query_parser = LLMQueryParser()
        self.feedback_manager = FeedbackManager()
        self.feedback_service = feedback_service or PlaylistFeedbackService()
        self.deduplicator = deduplicator or PlaylistDeduplicator()
        self.filter_ = filter_ or PlaylistFilter()
        self.exporter = exporter or PlaylistExporter()

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
        # Prioritize user-provided max_tracks over parsed count
        if max_tracks is not None:
            return max_tracks
        # If no user-provided max_tracks, use parsed count or fallback to config default
        default_max_tracks: int = self.config.get(
            "playlist_defaults.default_max_tracks", 20
        )
        return parsed_count if parsed_count is not None else default_max_tracks

    def generate_playlist(
        self,
        query: str,
        max_tracks: Optional[int] = None,
        min_similarity: Optional[float] = None,
        max_artist_ratio: Optional[float] = None,
        search_breadth_factor: Optional[int] = None,
    ) -> List[Track]:
        """
        Generate a music playlist based on a user query using local embeddings

        Args:
            query: Natural language query describing the desired music
            max_tracks: Maximum number of tracks to include (uses config default if None)
            min_similarity: Minimum similarity score (0.0-1.0) (uses config default if None)
            max_artist_ratio: Maximum ratio of tracks per artist (uses config default if None)
            search_breadth_factor: Factor to multiply max_tracks by for search breadth (uses config default if None)

        Returns:
            List of track dictionaries with metadata and similarity scores
        """
        try:
            # Use configuration defaults when parameters are None
            defaults = self.config.playlist_defaults
            if max_tracks is None:
                max_tracks = defaults["max_tracks"]
            if min_similarity is None:
                min_similarity = defaults["min_similarity"]
            if max_artist_ratio is None:
                max_artist_ratio = defaults["max_artist_ratio"]
            if search_breadth_factor is None:
                search_breadth_factor = defaults["search_breadth_factor"]

            logger.info(f"🎵 Generating playlist for query: '{query}'")
            logger.info(
                f"📊 Parameters: max_tracks={max_tracks}, min_similarity={min_similarity}"
            )

            # 1. Parse the query to understand intent
            query_lower = query.lower().strip()

            # Check for similarity queries (e.g., "like radiohead", "artists similar to...")
            is_similarity_query = any(
                [
                    "like " in query_lower,
                    "similar" in query_lower,
                    "reminds me of" in query_lower,
                    "sounds like" in query_lower,
                    "in the style of" in query_lower,
                ]
            )

            parsed: Dict[str, Any] = {}
            artist = ""
            genres: List[str] = []
            is_artist_specific = False
            is_vague = False

            if not is_similarity_query and self._check_artist_in_database(query_lower):
                # Query exactly matches an artist - treat as artist-specific without LLM parsing
                artist = query_lower.title()
                is_artist_specific = True
                genres = []  # No genres for artist-specific queries
                is_vague = False
                logger.info(
                    f"🎤 Query '{query_lower}' exactly matches artist '{artist}'. Skipping LLM parsing."
                )
            else:
                # Extract intent from query using LLM
                parsed_result = self.query_parser.parse(query)
                parsed_artist: Optional[str] = parsed_result.artist
                genres = parsed_result.genres or []
                is_vague = parsed_result.vague or False
                parsed = (
                    parsed_result.to_dict()
                    if hasattr(parsed_result, "to_dict")
                    else parsed_result.__dict__
                )
                logger.info(f"🧠 Parsed intent: {parsed}")

            # Determine final max_tracks value
            parsed_count = parsed.get("count") if "parsed" in locals() else None
            max_tracks = self._determine_max_tracks(parsed_count, max_tracks)

            logger.info(f"🎯 Using max_tracks={max_tracks}")

            # Use the new query_type field from improved LLM parser
            if "parsed" in locals():
                query_type = parsed.get("query_type", "general")
                reference_artist = parsed.get("reference_artist")
                is_artist_specific = query_type == "artist_specific"
                is_similarity_query = query_type == "similarity"
                # Use parsed artist for LLM results
                artist = parsed_artist or ""
            else:
                # Fallback for direct database matches
                query_type = "artist_specific"
                reference_artist = None
                is_artist_specific = True
                is_similarity_query = False

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
            start_time = time.time()
            # For artist-specific queries, get all tracks by that artist directly from database
            if is_artist_specific:
                logger.info(
                    f"🎤 Artist-specific query detected. Getting all tracks by '{artist}' from database."
                )
                results = self._get_tracks_by_artist(artist)
            else:
                # For general queries, use similarity search
                results = self.embedding_service.similarity_search(
                    query,
                    embeddings,
                    track_data,
                    top_k=max_tracks * search_breadth_factor,
                )

                # For similarity queries, exclude the reference artist from results
                if is_similarity_query and reference_artist:
                    logger.info(
                        f"🔄 Similarity query detected. Excluding reference artist '{reference_artist}' from results."
                    )
                    original_count = len(results)
                    results = [
                        track
                        for track in results
                        if (track.artist or "").strip().lower()
                        != reference_artist.lower()
                    ]
                    filtered_count = len(results)
                    logger.info(
                        f"🔄 Filtered out {original_count - filtered_count} tracks by reference artist '{reference_artist}'"
                    )

            search_time = time.time() - start_time
            logger.info(
                f"🔍 Similarity search took {search_time:.2f}s for {len(results)} results"
            )

            # 4. Apply genre filtering and boosting using PlaylistFilter
            if is_artist_specific:
                logger.info(
                    "🎤 Artist-specific query detected, skipping genre filtering"
                )
            else:
                results = self.filter_.apply_genre_filtering(genres, results)

            # 5. Apply feedback + iTunes score adjustments
            for track in results:
                track_id = track.id
                if track_id is not None:
                    track_dict = track.to_dict()
                    adjustment = self.feedback_manager.get_adjusted_score(
                        track_id, track_dict
                    )
                    original_score = track.similarity_score or 0
                    adjusted_score = max(0.0, min(1.0, original_score + adjustment))
                    # Update track in-place for better performance
                    track.similarity_score = adjusted_score
                    track.feedback_adjusted = adjustment
            # 🧮 Summarize adjustment impact
            adjustments = [
                track.feedback_adjusted
                for track in results
                if track.feedback_adjusted is not None
            ]
            logger.info(f"🔄 Adjustments: {adjustments}")
            if adjustments:
                avg_adj = sum(adjustments) / len(adjustments)
                pos_count = len([a for a in adjustments if a > 0])
                neg_count = len([a for a in adjustments if a < 0])
                max_adj = max(adjustments)
                min_adj = min(adjustments)
                logger.info(
                    f"📈 Feedback summary: avg_adj={avg_adj:.3f}, +{pos_count}, -{neg_count}, max={max_adj:.3f}, min={min_adj:.3f}"
                )
                top_positive = max(results, key=lambda t: t.feedback_adjusted or 0)
                top_negative = min(results, key=lambda t: t.feedback_adjusted or 0)
                logger.info(
                    f"🔺 Most boosted: {top_positive.name} by {top_positive.artist} (+{top_positive.feedback_adjusted or 0:.2f})"
                )
                logger.info(
                    f"🔻 Most penalized: {top_negative.name} by {top_negative.artist} ({top_negative.feedback_adjusted or 0:.2f})"
                )
            # 6. Filter by similarity threshold and deduplicate using PlaylistDeduplicator
            filtered_results = self.deduplicator.filter_and_deduplicate_results(
                results,
                min_similarity,
                max_tracks,
                is_artist_specific,
                max_artist_ratio,
                sample_with_randomization=self._sample_with_randomization,
                smart_name_deduplication=self._smart_name_deduplication,
                enforce_artist_diversity=self._enforce_artist_diversity,
                distribute_artists=self._distribute_artists,
                logger=logger,
            )

            # 7. Apply artist randomization (still in generator for now)
            filtered_results = self._apply_artist_randomization(
                filtered_results, max_tracks, is_vague
            )

            generation_time = time.time() - start_time
            logger.info(
                f"✅ Generated playlist with {len(filtered_results)} tracks in {generation_time:.2f}s"
            )

            # Record feedback data for learning
            self._record_playlist_generation_feedback(
                query=query,
                query_type=query_type,
                parsed_data=parsed if "parsed" in locals() else {},
                generated_tracks=filtered_results,
                playlist_length=len(filtered_results),
                requested_length=max_tracks,
                similarity_threshold=min_similarity,
                search_breadth=search_breadth_factor,
            )

            return filtered_results

        except Exception as e:
            logger.error(f"❌ Error generating playlist: {e}")
            raise

    @staticmethod
    def _normalize_file_location_static(location: str) -> str:
        """Static version for use in deduplicator."""
        if not location:
            return ""
        normalized = location.lower().replace("\\", "/")
        match = re.match(
            r"^(?:/users/|/home/|c:/users/|d:/users/|e:/users/)([^/]+)/(.+)$",
            normalized,
        )
        if match:
            return match.group(2)
        return normalized

    @staticmethod
    def _create_playlist_name_static(query: str) -> str:
        """Static version for use in exporter."""
        # Remove common prefixes and clean up the query
        query_lower = query.lower().strip()
        prefixes_to_remove = [
            "generate",
            "create",
            "make",
            "find",
            "get",
            "show",
            "give",
            "play",
        ]
        for prefix in prefixes_to_remove:
            if query_lower.startswith(f"{prefix} "):
                query_lower = query_lower[len(prefix) :].strip()
        query_lower = query_lower.strip("\"'")
        words = query_lower.split()
        filtered_words = []
        i = 0
        while i < len(words):
            word = words[i]
            if word == "me" and i + 1 < len(words) and words[i + 1] == "some":
                i += 2
                continue
            if word == "some" and i > 0 and words[i - 1] == "me":
                i += 1
                continue
            if word in ["me", "some"]:
                i += 1
                continue
            if word in ["a", "the"] and i + 1 < len(words):
                next_word = words[i + 1]
                if next_word in ["playlist", "mix", "collection", "set"]:
                    i += 1
                    continue
            filtered_words.append(word)
            i += 1
        query_lower = " ".join(filtered_words).strip()
        playlist_name = query_lower.title()
        if "for" in playlist_name.lower():
            playlist_name = re.sub(r"\bFor\b", "for", playlist_name)
        playlist_name = re.sub(r"\bMusic\b", "Music", playlist_name)
        playlist_name = re.sub(r"\bSongs\b", "Songs", playlist_name)
        playlist_name = re.sub(r"\bTracks\b", "Tracks", playlist_name)
        playlist_name = re.sub(r"\bPlaylist\b", "Playlist", playlist_name)
        if len(playlist_name.split()) <= 1:
            if playlist_name.strip() == "":
                playlist_name = "Mix"
            else:
                playlist_name = f"{playlist_name} Mix"
        return playlist_name.strip()

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
            r"^([a-zA-Z\s]+)$",  # Direct artist name (whole query, greedy)
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
            "music",  # Add "music" to common words
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

        # If all words are common words, it's not an artist name
        if all(word in common_words for word in artist_words):
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

    def _sample_with_randomization(
        self, tracks: List[Track], max_count: int, top_ratio: float = 0.3
    ) -> List[Track]:
        """
        Sample tracks with a mix of top similarity and random selection

        Args:
            tracks: List of tracks to sample from
            max_count: Maximum number of tracks to return
            top_ratio: Ratio of top similarity tracks (default 0.5 = 50%)

        Returns:
            List of sampled tracks
        """
        if len(tracks) <= max_count:
            return tracks

        # Sort by similarity first
        tracks.sort(key=lambda t: t.similarity_score or 0, reverse=True)

        # Take top tracks by similarity, then random from the rest
        top_count = int(max_count * top_ratio)
        remaining_count = max_count - top_count

        top_tracks = tracks[:top_count]
        remaining_tracks = tracks[top_count:]

        # Randomly sample from remaining tracks
        if remaining_tracks and remaining_count > 0:
            rng = secrets.SystemRandom()
            random_tracks = rng.sample(
                remaining_tracks, min(remaining_count, len(remaining_tracks))
            )
            return top_tracks + random_tracks
        else:
            return top_tracks[:max_count]

    def _filter_and_deduplicate_results(  # noqa: C901
        self,
        results: List[Track],
        min_similarity: float,
        max_tracks: int,
        is_artist_specific: bool,
        max_artist_ratio: float = 0.3,
    ) -> List[Track]:
        """Filter results by similarity and remove duplicates using multiple strategies"""
        # Filter by similarity threshold
        filtered = [
            track
            for track in results
            if (track.similarity_score or 0) >= min_similarity
        ]

        logger.info(
            f"🔍 Starting deduplication on {len(filtered)} tracks (similarity ≥ {min_similarity})"
        )

        # Strategy 1: Deduplicate by file location (normalized path)
        seen_locations = set()
        location_deduplicated = []

        for track in filtered:
            location = self._normalize_file_location(track.location or "")
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
            title = (track.name or "").strip().lower()
            artist = (track.artist or "").strip().lower()
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
            track_id = track.id
            if track_id not in seen_ids:
                seen_ids.add(track_id)
                final_deduplicated.append(track)

        logger.info(
            f"🆔 Track ID deduplication: {len(combination_deduplicated)} → {len(final_deduplicated)} tracks"
        )

        # Strategy 4: Smart deduplication for similar titles with slight variations
        if is_artist_specific:
            # For artist-specific queries, skip smart name deduplication
            smart_deduplicated = final_deduplicated
        else:
            # For general queries, apply smart name deduplication
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
                2, int(max_tracks * max_artist_ratio)
            )  # e.g., 70% of playlist per artist, minimum 2
            logger.info(
                f"🎵 General query, allowing up to {max_tracks_per_artist} tracks per artist (ratio {max_artist_ratio})"
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

        # Use shared sampling logic
        candidates = self._sample_with_randomization(diverse_tracks, top_k)

        if len(candidates) < max_tracks:
            # If we don't have enough candidates, return what we have
            logger.warning(
                f"⚠️ Only {len(candidates)} tracks available, requested {max_tracks}"
            )
            return candidates[:max_tracks]

        # Assign weights proportional to similarity score (shifted to be >=0)
        min_similarity_score = min(
            (t.similarity_score or 0 for t in candidates), default=0
        )
        weights = [
            max(0.0, (t.similarity_score or 0) - min_similarity_score + 1e-6)
            for t in candidates
        ]

        # Sample exactly max_tracks unique tracks
        selected = set()
        final_tracks: List[Track] = []
        attempts = 0
        max_attempts = top_k * 3  # Increase attempts to ensure we get enough tracks
        rng = secrets.SystemRandom()

        while len(final_tracks) < max_tracks and attempts < max_attempts:
            # Use secrets.SystemRandom().choices for efficient weighted random selection
            if weights and any(w > 0 for w in weights):
                # Use choices for weighted random selection (more efficient than manual loop)
                pick = rng.choices(candidates, weights=weights, k=1)[0]
            else:
                # If no weights or all weights are zero, use uniform random selection
                pick = rng.choice(candidates)

            pick_id = pick.id or f"{pick.name or ''}-{pick.artist or ''}"
            if pick_id not in selected:
                final_tracks.append(pick)
                selected.add(pick_id)
            attempts += 1

        # If weighted sampling didn't give us enough tracks, fill with randomized remaining tracks
        if len(final_tracks) < max_tracks:
            logger.info(
                f"🎲 Weighted sampling gave {len(final_tracks)} tracks, filling with randomized remaining tracks"
            )
            # Get remaining tracks that weren't selected
            remaining_candidates = [
                t
                for t in candidates
                if (t.id or f"{t.name or ''}-{t.artist or ''}") not in selected
            ]

            # Shuffle remaining candidates for variety
            rng.shuffle(remaining_candidates)

            for t in remaining_candidates:
                t_id = t.id or f"{t.name or ''}-{t.artist or ''}"
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

    def _group_tracks_by_artist(self, tracks: List[Track]) -> dict[str, list]:
        """Group tracks by artist for deduplication"""
        artist_groups: dict[str, list] = {}
        for track in tracks:
            artist = (track.artist or "").strip().lower()
            if artist:
                if artist not in artist_groups:
                    artist_groups[artist] = []
                artist_groups[artist].append(track)
        return artist_groups

    def _find_best_track_for_base_name(
        self, artist_tracks: List[Track], base_name: str
    ) -> Optional[Track]:
        """Find the track with the highest similarity score for a given base name"""
        best_track = None
        best_score = -1.0
        for track in artist_tracks:
            name = (track.name or "").strip().lower()
            track_base = self._extract_base_name(name)
            if track_base == base_name:
                score = track.similarity_score or 0
                if score > best_score:
                    best_score = score
                    best_track = track
        return best_track

    def _process_artist_tracks(self, artist_tracks: List[Track]) -> List[Track]:
        """Process tracks for a single artist to remove duplicates"""
        if len(artist_tracks) == 1:
            return [artist_tracks[0]]

        deduplicated = []
        processed_names = set()

        for track in artist_tracks:
            name = (track.name or "").strip().lower()
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

    def _smart_name_deduplication(self, tracks: List[Track]) -> List[Track]:
        """Remove duplicate tracks based on name and artist similarity"""
        if not tracks:
            return tracks

        # Group tracks by base name and artist
        name_artist_groups: Dict[str, List[Track]] = {}

        for track in tracks:
            base_name = self._extract_base_name(track.name or "")
            artist = track.artist or ""
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
                best_track = max(group, key=lambda t: t.similarity_score or 0)
                deduplicated.append(best_track)

        return deduplicated

    def _extract_base_name(self, name: str) -> str:
        """Extract base name by removing common suffixes in parentheses"""
        # Remove common suffixes like (Remix), (Live), (Acoustic), etc.
        return re.sub(r"\s*\([^)]*\)$", "", name).strip()

    def _normalize_file_location(self, location: str) -> str:
        """Normalize file location for deduplication and Apple Music compatibility"""
        if not location:
            return ""

        # Convert to lowercase and normalize separators
        normalized = location.lower().replace("\\", "/")

        # Remove common OS-specific prefixes with username

        # Match /users/<username>/, /home/<username>/, c:/users/<username>/, etc.
        match = re.match(
            r"^(?:/users/|/home/|c:/users/|d:/users/|e:/users/)([^/]+)/(.+)$",
            normalized,
        )
        if match:
            return match.group(2)
        return normalized

    def save_playlist_m3u(
        self, tracks: List[Track], query: str, output_dir: str = "playlists"
    ) -> str:
        """Delegate to PlaylistExporter."""
        return self.exporter.save_playlist_m3u(tracks, query, output_dir)

    def _create_playlist_name(self, query: str) -> str:
        """
        Create a user-friendly playlist name from the query

        Args:
            query: Original search query

        Returns:
            Clean, readable playlist name
        """
        # Remove common prefixes and clean up the query
        query_lower = query.lower().strip()

        # Remove common prefixes
        prefixes_to_remove = [
            "generate",
            "create",
            "make",
            "find",
            "get",
            "show",
            "give",
            "play",
        ]
        for prefix in prefixes_to_remove:
            if query_lower.startswith(f"{prefix} "):
                query_lower = query_lower[len(prefix) :].strip()

        # Remove quotes if present
        query_lower = query_lower.strip("\"'")

        # Remove common filler words that don't add meaning
        words = query_lower.split()
        filtered_words = []
        i = 0
        while i < len(words):
            word = words[i]

            # Skip "me" if followed by "some"
            if word == "me" and i + 1 < len(words) and words[i + 1] == "some":
                i += 2  # Skip both "me" and "some"
                continue

            # Skip "some" if preceded by "me"
            if word == "some" and i > 0 and words[i - 1] == "me":
                i += 1
                continue

            # Skip standalone "me" or "some"
            if word in ["me", "some"]:
                i += 1
                continue

            # Keep "a" and "the" only if they're followed by descriptive words
            if word in ["a", "the"] and i + 1 < len(words):
                next_word = words[i + 1]
                if next_word in ["playlist", "mix", "collection", "set"]:
                    i += 1  # Skip the article
                    continue

            filtered_words.append(word)
            i += 1

        query_lower = " ".join(filtered_words).strip()

        # Convert to title case for better readability
        playlist_name = query_lower.title()

        # Handle special cases for better naming
        if "for" in playlist_name.lower():
            # Keep "for" lowercase for better readability
            playlist_name = re.sub(r"\bFor\b", "for", playlist_name)

        # Handle common music-related terms
        playlist_name = re.sub(r"\bMusic\b", "Music", playlist_name)
        playlist_name = re.sub(r"\bSongs\b", "Songs", playlist_name)
        playlist_name = re.sub(r"\bTracks\b", "Tracks", playlist_name)
        playlist_name = re.sub(r"\bPlaylist\b", "Playlist", playlist_name)

        # If the name is too generic or single word, add some context
        if len(playlist_name.split()) <= 1:
            if playlist_name.strip() == "":
                playlist_name = "Mix"
            else:
                playlist_name = f"{playlist_name} Mix"

        return playlist_name.strip()

    def print_playlist_summary(self, tracks: List[Track], query: str) -> None:
        """Print a summary of the generated playlist"""
        print(f"\n🎵 Playlist for: '{query}'")
        print(f"📊 {len(tracks)} tracks found")
        print("-" * 60)

        for i, track in enumerate(tracks, 1):
            artist = track.artist or "Unknown"
            name = track.name or "Unknown"
            album = track.album or "Unknown"
            similarity = track.similarity_score or 0

            print(f"{i:2d}. {artist} - {name}")
            print(f"    Album: {album}")
            print(f"    Similarity: {similarity:.3f}")
            print()

        if tracks:
            avg_similarity = sum(t.similarity_score or 0 for t in tracks) / len(tracks)
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
        self, tracks: List[Track], max_tracks: int, is_vague: bool
    ) -> List[Track]:
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
        artist_groups: dict[str, list[Track]] = {}
        for track in tracks:
            artist = (track.artist or "").strip()
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
                best_score = max(track.similarity_score or 0 for track in artist_tracks)
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
                best_track = max(artist_tracks, key=lambda t: t.similarity_score or 0)
                randomized_tracks.append(best_track)

                if len(randomized_tracks) >= max_tracks:
                    break

            return randomized_tracks
        else:
            # Not enough artists to randomize, return original order
            return tracks[:max_tracks]

    def _enforce_artist_diversity(
        self,
        tracks: List[Track],
        max_tracks: int,
        max_tracks_per_artist: int = 3,
    ) -> List[Track]:
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
        artist_groups: dict[str, list[Track]] = {}
        for track in tracks:
            artist = (track.artist or "").strip()
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
                    key=lambda t: t.similarity_score or 0,
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

        # Instead of always taking top tracks by similarity, add randomization
        if len(diverse_tracks) > max_tracks:
            # Sort by similarity first
            diverse_tracks.sort(key=lambda t: t.similarity_score or 0, reverse=True)

            # Use shared sampling logic
            final_tracks = self._sample_with_randomization(diverse_tracks, max_tracks)
        else:
            # If we have fewer tracks than max_tracks, use all of them
            final_tracks = diverse_tracks

        # Distribute artists throughout the playlist to avoid grouping
        distributed_tracks = self._distribute_artists(final_tracks, max_tracks)

        logger.info(
            f"✅ Artist diversity enforcement: {len(tracks)} → {len(distributed_tracks)} tracks"
        )
        return distributed_tracks

    def _distribute_artists(
        self, tracks: List[Track], max_tracks: Optional[int] = None
    ) -> List[Track]:
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
        distributed: List[Track] = []
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
        self, artist_groups: dict[str, list[Track]]
    ) -> dict[str, list[Track]]:
        """Create queues for each artist's tracks"""
        artist_queues: dict[str, list[Track]] = {}
        for artist, artist_tracks in artist_groups.items():
            artist_queues[artist] = artist_tracks.copy()
        return artist_queues

    def _round_robin_distribute(
        self,
        artist_queues: dict[str, list[Track]],
        tracks: List[Track],
        distributed: List[Track],
    ) -> List[Track]:
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
        artist_queues: dict[str, list[Track]],
        tracks: List[Track],
        distributed: List[Track],
    ) -> List[Track]:
        """Add any remaining tracks to the distributed list"""
        # If we still have space and tracks left, add remaining tracks
        remaining_tracks: List[Track] = []
        for artist_tracks in artist_queues.values():
            remaining_tracks.extend(artist_tracks)

        # Add remaining tracks at the end, maintaining artist diversity
        if remaining_tracks and len(distributed) < len(tracks):
            # Sort remaining tracks by similarity and add them
            remaining_tracks.sort(key=lambda t: t.similarity_score or 0, reverse=True)
            space_left = len(tracks) - len(distributed)
            distributed.extend(remaining_tracks[:space_left])

        return distributed

    def _get_tracks_by_artist(self, artist_name: str) -> List[Track]:
        """
        Get all tracks from the database for a specific artist.
        This bypasses the embedding search and uses direct database queries.
        """
        logger.info(f"🔍 Querying database for all tracks by artist: {artist_name}")
        return self.track_embedder.get_all_tracks_by_artist(artist_name)

    def _check_artist_in_database(self, query: str) -> bool:
        """
        Check if a query string exactly matches an artist in the database.
        This is a fallback for artist-specific queries that don't have a direct artist name.
        """
        try:
            cursor = self.track_embedder.conn.cursor()

            # Check if the query exactly matches any artist in the database (case-insensitive)
            query_sql = """
                SELECT COUNT(*)
                FROM tracks
                WHERE LOWER(artist) = LOWER(?)
            """

            cursor.execute(query_sql, (query,))
            count = cursor.fetchone()[0]

            return bool(count > 0)

        except Exception as e:
            logger.error(f"❌ Error checking artist in database: {e}")
            return False

    def _record_playlist_generation_feedback(
        self,
        query: str,
        query_type: str,
        parsed_data: Dict[str, Any],
        generated_tracks: List[Track],
        playlist_length: int,
        requested_length: int,
        similarity_threshold: float,
        search_breadth: int,
    ) -> None:
        """
        Records feedback data for playlist generation.
        This method is called after a playlist is successfully generated.
        """
        try:
            # Record playlist feedback using the feedback manager
            # Convert Track objects to dicts for feedback manager
            tracks_as_dicts = [track.to_dict() for track in generated_tracks]
            self.feedback_manager.record_playlist_feedback(
                query=query,
                query_type=query_type,
                parsed_data=parsed_data,
                generated_tracks=tracks_as_dicts,
                playlist_length=playlist_length,
                requested_length=requested_length,
                similarity_threshold=similarity_threshold,
                search_breadth=search_breadth,
            )

            logger.info(f"✅ Feedback recorded for playlist generation: {query}")

        except Exception as e:
            logger.error(f"❌ Error recording playlist generation feedback: {e}")


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

        # Ask for feedback on each track
        for track in tracks:
            artist = track.artist or "Unknown"
            name = track.name or "Unknown"
            track_id = str(track.id) if track.id is not None else ""
            print(f"\nTrack: {artist} - {name}")
            fb = (
                input("Feedback? (l = like, d = dislike, b = block, enter to skip): ")
                .strip()
                .lower()
            )
            if fb in ["l", "d", "b"]:
                fb_map = {"l": "like", "d": "dislike", "b": "block"}
                generator.feedback_service.record_user_feedback(
                    track_id, fb_map[fb], query
                )

        # Ask if user wants to save
        save = input("\nSave playlist to file? (y/n): ").strip().lower()
        if save in ["y", "yes"]:
            filepath = generator.save_playlist_m3u(tracks, query)
            print(f"✅ Playlist saved to: {filepath}")

            # Ask if user wants to open in Apple Music
            open_music = input("Open in Apple Music? (y/n): ").strip().lower()
            if open_music in ["y", "yes"]:
                try:
                    open_path = shutil.which("open")
                    if open_path is None:
                        print("❌ 'open' command not found (macOS required)")
                    else:
                        # Only allow .mp3, .m4a, .aac, .wav files
                        allowed_exts = {".mp3", ".m4a", ".aac", ".wav"}
                        if os.path.splitext(filepath)[1].lower() not in allowed_exts:
                            print("❌ File type not allowed for Apple Music open.")
                        else:
                            subprocess.run(
                                [open_path, "-a", "Music", filepath],
                                check=True,
                                shell=False,
                                timeout=10,
                            )
                        print("🎵 Opened in Apple Music!")
                except subprocess.TimeoutExpired:
                    print("❌ Timeout opening Apple Music")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Could not open in Apple Music: {e}")
                except FileNotFoundError:
                    print("❌ Apple Music not found")
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
