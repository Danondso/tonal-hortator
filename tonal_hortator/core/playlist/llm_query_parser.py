"""
LLM Query Parser for Tonal Hortator.

Handles parsing user queries using local LLM to extract structured intent.
"""

import json
import logging
import re
from typing import Any, Dict, Optional

from tonal_hortator.core.config import get_config
from tonal_hortator.core.llm.llm_client import LocalLLMClient

logger = logging.getLogger(__name__)


class LLMQueryParser:
    """Extracts structured intent from user queries using local LLM."""

    def __init__(self, model_name: Optional[str] = None):
        self.config = get_config()

        # Use configured model if not provided
        if model_name is None:
            model_name = self.config.llm_config["query_parser_model"]

        self.model_name = model_name
        self.client = LocalLLMClient(model_name)

    def parse(self, query: str) -> dict:
        prompt = self._build_prompt(query)

        # Get max tokens from configuration
        max_tokens = self.config.llm_config["max_tokens"]
        response = self.client.generate(prompt, max_tokens=max_tokens)
        return self._extract_json(response)

    def _build_prompt(self, query: str) -> str:
        # Get query parsing configuration
        parsing_config = self.config.get_section("llm").get("query_parsing", {})
        min_artist_name_length = parsing_config.get("min_artist_name_length", 2)

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
- Artist names must be at least {min_artist_name_length} characters long

Query: "{query}"

Output JSON with these fields:
- query_type: "artist_specific" | "similarity" | "general"
- artist: string | null (only for artist_specific queries)
- reference_artist: string | null (only for similarity queries)
- genres: array of strings
- mood: string | null
- count: number | null (if user specified track count like "10 songs")
- unplayed: boolean (if user wants only unplayed tracks)
- vague: boolean (true if query is very general/vague)

Output pure JSON only, no additional text:"""

    def _extract_json(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response"""
        try:
            # Try to find JSON in the response
            json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
            matches = re.findall(json_pattern, response, re.DOTALL)

            if matches:
                result = json.loads(matches[0])
                return result if isinstance(result, dict) else {}
            else:
                raise ValueError("No JSON found in LLM response")

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Response was: {response}")
            raise ValueError(f"Failed to parse LLM response: {e}")
