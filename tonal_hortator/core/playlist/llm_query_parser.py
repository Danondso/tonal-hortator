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
from tonal_hortator.core.models import LLMQueryResult

logger = logging.getLogger(__name__)


class LLMQueryParser:
    """Extracts structured intent from user queries using local LLM."""

    def __init__(self, model_name: Optional[str] = None):
        self.config = get_config()

        # Use configured model if not provided
        if model_name is None:
            model_name = self.config.get("llm.query_parser_model", "llama3:8b")

        self.model_name = model_name
        self.client = LocalLLMClient(model_name)

    def parse(self, query: str) -> LLMQueryResult:
        prompt = self._build_prompt(query)

        # Get max tokens from configuration
        max_tokens = self.config.get("llm.max_tokens", 1000)
        response = self.client.generate(prompt, max_tokens=max_tokens)
        parsed_dict = self._extract_json(response)
        return LLMQueryResult.from_dict(parsed_dict)

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
        """Extract JSON from LLM response using robust bracket counting"""
        try:
            # First try: parse the entire response as JSON
            try:
                result = json.loads(response.strip())
                return result if isinstance(result, dict) else {}
            except json.JSONDecodeError:
                pass

            # Second try: find JSON using bracket counting for nested structures
            json_str = self._find_json_with_bracket_counting(response)
            if json_str:
                result = json.loads(json_str)
                return result if isinstance(result, dict) else {}

            # Third try: fallback to regex for simple JSON
            json_pattern = r"\{[^{}]+\}"
            matches = re.findall(json_pattern, response, re.DOTALL)
            for match in matches:
                try:
                    result = json.loads(match)
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError as e:
                    logger.error(f"Regex JSON parsing failed for match '{match}': {e}")

            raise ValueError(
                "No valid JSON found in LLM response after all parsing attempts"
            )

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response after multiple attempts: {e}")
            logger.error(f"Response was: {response}")
            raise ValueError(
                f"Failed to parse LLM response after multiple attempts: {e}"
            )

    def _find_json_with_bracket_counting(self, text: str) -> Optional[str]:
        """Find JSON object using bracket counting to handle nested structures"""
        # Find the first opening brace
        start_idx = text.find("{")
        if start_idx == -1:
            return None

        # Count brackets to find the matching closing brace
        bracket_count = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start_idx:], start_idx):
            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == "{":
                    bracket_count += 1
                elif char == "}":
                    bracket_count -= 1
                    if bracket_count == 0:
                        # Found the matching closing brace
                        return text[start_idx : i + 1]

        return None
