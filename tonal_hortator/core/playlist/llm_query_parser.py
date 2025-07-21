"""
LLM Query Parser for Tonal Hortator.

Handles parsing user queries using local LLM to extract structured intent.
"""

import json
import logging
import re
from typing import Any

from tonal_hortator.core.llm.llm_client import LocalLLMClient

logger = logging.getLogger(__name__)


class LLMQueryParser:
    """Extracts structured intent from user queries using local LLM."""

    def __init__(self, model_name: str = "llama3:8b"):
        self.model_name = model_name
        self.client = LocalLLMClient(model_name)

    def parse(self, query: str) -> dict:
        prompt = self._build_prompt(query)
        response = self.client.generate(prompt)
        return self._extract_json(response)

    def _build_prompt(self, query: str) -> str:
        return f"""You are a music playlist assistant. Analyze the user's query and determine the query type and intent.

IMPORTANT: Distinguish between these query types:

1. ARTIST_SPECIFIC: User wants tracks by a specific artist
    - "oso oso" → artist: "oso oso", query_type: "artist_specific"
    - "The Beatles" → artist: "The Beatles", query_type: "artist_specific"
    - "Taylor Swift songs" → artist: "Taylor Swift", query_type: "artist_specific"

2. SIMILARITY: User wants artists/tracks similar to a reference artist
    - "artists similar to oso oso" → artist: null, query_type: "similarity", reference_artist: "oso oso"
    - "like oso oso" → artist: null, query_type: "similarity", reference_artist: "oso oso"
    - "sounds like The Beatles" → artist: null, query_type: "similarity", reference_artist: "The Beatles"
    - "recommendations for Taylor Swift" → artist: null, query_type: "similarity", reference_artist: "Taylor Swift"

3. GENERAL: User wants music by genre, mood, or other criteria
    - "rock music" → artist: null, query_type: "general", genres: ["rock"]
    - "upbeat rock" → artist: null, query_type: "general", genres: ["rock"], mood: "upbeat"
    - "jazz for studying" → artist: null, query_type: "general", genres: ["jazz"], mood: "studying"
    - "party music" → artist: null, query_type: "general", mood: "party"
    - "falling asleep in a trailer by the river" → artist: null, query_type: "general", mood: "melancholy", genres: ["folk", "country"]

CRITICAL RULES:
- For SIMILARITY queries, NEVER set artist to the reference artist
- For SIMILARITY queries, set artist to null and use reference_artist field
- For GENERAL queries, NEVER set artist - only set genres and mood
- Preserve full artist names, don't shorten them
- For genre detection, preserve compound genres like "bedroom pop", "progressive metal"
- Context clues like "falling asleep", "trailer", "river" suggest mood/genre, not artist names
- Common music terms like "rock music", "jazz", "hip hop" are genres, not artists

Query: "{query}"

Output JSON with these fields:
- query_type: "artist_specific" | "similarity" | "general"
- artist: (string or null) - ONLY for artist_specific queries
- reference_artist: (string or null) - ONLY for similarity queries
- genres: (list of strings) - detected genres
- mood: (string or null) - detected mood
- count: (int or null) - requested track count
- unplayed: (boolean) - whether user wants unplayed tracks
- vague: (boolean) - whether query is vague/general

Examples:

Query: "oso oso"
{{
  "query_type": "artist_specific",
  "artist": "oso oso",
  "reference_artist": null,
  "genres": [],
  "mood": null,
  "count": null,
  "unplayed": false,
  "vague": false
}}

Query: "artists similar to oso oso"
{{
  "query_type": "similarity",
  "artist": null,
  "reference_artist": "oso oso",
  "genres": ["indie rock", "emo"],
  "mood": null,
  "count": null,
  "unplayed": false,
  "vague": false
}}

Query: "rock music"
{{
  "query_type": "general",
  "artist": null,
  "reference_artist": null,
  "genres": ["rock"],
  "mood": null,
  "count": null,
  "unplayed": false,
  "vague": true
}}

Query: "falling asleep in a trailer by the river"
{{
  "query_type": "general",
  "artist": null,
  "reference_artist": null,
  "genres": ["folk", "country"],
  "mood": "melancholy",
  "count": null,
  "unplayed": false,
  "vague": true
}}

Now analyze the current query and respond with valid JSON:"""

    def _extract_json(self, response: str) -> dict[Any, Any]:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            return json.loads(match.group(0))  # type: ignore
        raise ValueError("No JSON found in LLM response")
