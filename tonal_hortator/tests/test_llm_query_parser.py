#!/usr/bin/env python3
"""
Tests for tonal_hortator.core.playlist.llm_query_parser
"""

import json
import unittest
from typing import Any
from unittest.mock import Mock, patch

from tonal_hortator.core.playlist.llm_query_parser import LLMQueryParser


class TestLLMQueryParser(unittest.TestCase):
    """Test LLMQueryParser"""

    @patch("tonal_hortator.core.playlist.llm_query_parser.LocalLLMClient")
    def test_init(self, mock_llm_client_class: Mock) -> None:
        """Test initialization"""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client

        parser = LLMQueryParser()

        self.assertEqual(parser.model_name, "llama3:8b")
        self.assertEqual(parser.client, mock_client)
        mock_llm_client_class.assert_called_once_with("llama3:8b")

    @patch("tonal_hortator.core.playlist.llm_query_parser.LocalLLMClient")
    def test_init_custom_model(self, mock_llm_client_class: Mock) -> None:
        """Test initialization with custom model"""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client

        parser = LLMQueryParser("custom-model")

        self.assertEqual(parser.model_name, "custom-model")
        mock_llm_client_class.assert_called_once_with("custom-model")

    @patch("tonal_hortator.core.playlist.llm_query_parser.LocalLLMClient")
    def test_parse_artist_specific_query(self, mock_llm_client_class: Mock) -> None:
        """Test parsing artist-specific query"""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client

        # Mock response for artist-specific query
        mock_response: dict[str, Any] = {
            "query_type": "artist_specific",
            "artist": "The Beatles",
            "reference_artist": None,
            "genres": [],
            "mood": None,
            "count": None,
            "unplayed": False,
            "vague": False,
        }
        mock_client.generate.return_value = json.dumps(mock_response)

        parser = LLMQueryParser()
        result = parser.parse("The Beatles")

        self.assertEqual(result.query_type, "artist_specific")
        self.assertEqual(result.artist, "The Beatles")
        self.assertIsNone(result.reference_artist)

    @patch("tonal_hortator.core.playlist.llm_query_parser.LocalLLMClient")
    def test_parse_similarity_query(self, mock_llm_client_class: Mock) -> None:
        """Test parsing similarity query"""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client

        # Mock response for similarity query
        mock_response = {
            "query_type": "similarity",
            "artist": None,
            "reference_artist": "Radiohead",
            "genres": ["alternative", "rock"],
            "mood": None,
            "count": None,
            "unplayed": False,
            "vague": False,
        }
        mock_client.generate.return_value = json.dumps(mock_response)

        parser = LLMQueryParser()
        result = parser.parse("artists similar to Radiohead")

        self.assertEqual(result.query_type, "similarity")
        self.assertIsNone(result.artist)
        self.assertEqual(result.reference_artist, "Radiohead")

    @patch("tonal_hortator.core.playlist.llm_query_parser.LocalLLMClient")
    def test_parse_general_query(self, mock_llm_client_class: Mock) -> None:
        """Test parsing general query"""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client

        # Mock response for general query
        mock_response = {
            "query_type": "general",
            "artist": None,
            "reference_artist": None,
            "genres": ["rock"],
            "mood": "upbeat",
            "count": None,
            "unplayed": False,
            "vague": True,
        }
        mock_client.generate.return_value = json.dumps(mock_response)

        parser = LLMQueryParser()
        result = parser.parse("upbeat rock music")

        self.assertEqual(result.query_type, "general")
        self.assertIsNone(result.artist)
        self.assertIsNone(result.reference_artist)
        self.assertEqual(result.genres, ["rock"])
        self.assertEqual(result.mood, "upbeat")

    @patch("tonal_hortator.core.playlist.llm_query_parser.LocalLLMClient")
    def test_parse_with_track_count(self, mock_llm_client_class: Mock) -> None:
        """Test parsing query with track count"""
        mock_client = Mock()
        mock_llm_client_class.return_value = mock_client

        # Mock response with count
        mock_response = {
            "query_type": "general",
            "genres": ["jazz"],
            "artist": None,
            "reference_artist": None,
            "mood": None,
            "track_count": 15,
            "unplayed": False,
            "vague": False,
        }
        mock_client.generate.return_value = json.dumps(mock_response)

        parser = LLMQueryParser()
        result = parser.parse("15 jazz tracks")

        self.assertEqual(result.track_count, 15)
        self.assertEqual(result.genres, ["jazz"])

    @patch("tonal_hortator.core.playlist.llm_query_parser.LocalLLMClient")
    def test_extract_json_valid(self, mock_llm_client_class: Mock) -> None:
        """Test JSON extraction from valid response"""
        parser = LLMQueryParser()

        response = 'Here is the analysis: {"query_type": "general", "artist": null}'
        result = parser._extract_json(response)

        self.assertEqual(result["query_type"], "general")
        self.assertIsNone(result["artist"])

    @patch("tonal_hortator.core.playlist.llm_query_parser.LocalLLMClient")
    def test_extract_json_invalid(self, mock_llm_client_class: Mock) -> None:
        """Test JSON extraction from invalid response"""
        parser = LLMQueryParser()

        response = "This response has no JSON"

        with self.assertRaises(ValueError) as context:
            parser._extract_json(response)

        self.assertIn("No valid JSON found in LLM response", str(context.exception))

    def test_extract_json_nested_structures(self) -> None:
        """Test that _extract_json can handle complex nested JSON structures"""
        parser = LLMQueryParser()

        # Test 1: Nested object with multiple levels
        nested_json = {
            "query_type": "general",
            "artist": None,
            "metadata": {
                "genres": ["rock", "indie"],
                "details": {"year": 2021, "rating": 4.5},
            },
            "vague": True,
        }
        response1 = f"Here's the analysis: {json.dumps(nested_json)} That's the result."
        result1 = parser._extract_json(response1)
        self.assertEqual(result1, nested_json)

        # Test 2: Nested arrays and objects
        complex_json = {
            "query_type": "similarity",
            "reference_artist": "test band",
            "genres": ["rock", "alternative"],
            "preferences": {
                "similar_artists": ["band1", "band2"],
                "exclude": {"genres": ["pop"], "years": [2020, 2021]},
            },
        }
        response2 = f"Analysis result:\n{json.dumps(complex_json)}\nEnd of analysis."
        result2 = parser._extract_json(response2)
        self.assertEqual(result2, complex_json)

        # Test 3: JSON with escaped quotes
        json_with_escapes = {
            "artist": 'Band "The Great" Smith',
            "description": 'Music with "quotes" and backslashes\\',
            "details": {"note": "Contains special chars: {}[]"},
        }
        response3 = f"Result: {json.dumps(json_with_escapes)}"
        result3 = parser._extract_json(response3)
        self.assertEqual(result3, json_with_escapes)

        # Test 4: Clean JSON without surrounding text
        clean_json = {"query_type": "artist_specific", "artist": "test"}
        response4 = json.dumps(clean_json)
        result4 = parser._extract_json(response4)
        self.assertEqual(result4, clean_json)

    @patch("tonal_hortator.core.playlist.llm_query_parser.LocalLLMClient")
    def test_build_prompt_contains_query(self, mock_llm_client_class: Mock) -> None:
        """Test that build_prompt includes the query"""
        parser = LLMQueryParser()

        query = "test query"
        prompt = parser._build_prompt(query)

        self.assertIn(query, prompt)
        self.assertIn("Query:", prompt)
        self.assertIn("Output JSON", prompt)


if __name__ == "__main__":
    unittest.main()
