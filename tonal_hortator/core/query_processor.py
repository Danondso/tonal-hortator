#!/usr/bin/env python3
"""
Query processing utilities for playlist generation
Handles genre detection, artist extraction, and parameter parsing
"""

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Process and extract information from playlist queries"""

    def __init__(self) -> None:
        # Common music genres and their variations
        self.genre_mapping = {
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

    def extract_genre_keywords(self, query: str) -> List[str]:
        """
        Extract potential genre keywords from a query

        Args:
            query: Lowercase search query

        Returns:
            List of detected genre keywords
        """
        detected_genres = []

        for genre, keywords in self.genre_mapping.items():
            if any(keyword in query for keyword in keywords):
                detected_genres.append(genre)

        return detected_genres

    def extract_artist_from_query(self, query: str) -> Optional[str]:
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

    def extract_track_count(self, query: str) -> Optional[int]:
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

    def is_vague_query(self, query: str) -> bool:
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
