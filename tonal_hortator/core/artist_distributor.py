#!/usr/bin/env python3
"""
Artist distribution utilities for playlist generation
Handles artist diversity and distribution algorithms
"""

import logging
import random
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ArtistDistributor:
    """Handle artist diversity and distribution in playlists"""

    def enforce_artist_diversity(
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
        artist_groups = self._group_tracks_by_artist(tracks)

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
        distributed_tracks = self._distribute_artists(final_tracks)

        logger.info(
            f"✅ Artist diversity enforcement: {len(tracks)} → {len(distributed_tracks)} tracks"
        )
        return distributed_tracks

    def apply_artist_randomization(
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
        artist_groups = self._group_tracks_by_artist(tracks)

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

    def _group_tracks_by_artist(
        self, tracks: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group tracks by artist"""
        artist_groups: Dict[str, List[Dict[str, Any]]] = {}
        for track in tracks:
            artist = track.get("artist", "")
            if artist is None:
                artist = ""
            artist = artist.strip()
            if artist:
                if artist not in artist_groups:
                    artist_groups[artist] = []
                artist_groups[artist].append(track)
            else:
                # Handle tracks without artist info
                if "unknown" not in artist_groups:
                    artist_groups["unknown"] = []
                artist_groups["unknown"].append(track)
        return artist_groups

    def _distribute_artists(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Distribute artists throughout the playlist to avoid grouping

        Args:
            tracks: List of tracks to distribute

        Returns:
            List of tracks with distributed artists
        """
        if len(tracks) <= 1:
            return tracks

        # Group tracks by artist
        artist_groups = self._group_tracks_by_artist(tracks)

        # If we only have one artist or very few tracks, no need to distribute
        if len(artist_groups) <= 1 or len(tracks) <= 3:
            return tracks

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

        logger.info(
            f"🎵 Artist distribution complete: {len(tracks)} tracks distributed"
        )
        return distributed

    def _create_artist_queues(
        self, artist_groups: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Create queues for each artist's tracks"""
        queues: Dict[str, List[Dict[str, Any]]] = {}
        for artist, tracks in artist_groups.items():
            queues[artist] = list(tracks)
        return queues

    def _round_robin_distribute(
        self,
        artist_queues: Dict[str, List[Dict[str, Any]]],
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
        artist_queues: Dict[str, List[Dict[str, Any]]],
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
