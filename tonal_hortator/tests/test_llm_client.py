#!/usr/bin/env python3
"""
Tests for LocalLLMClient

Tests the LLM client functionality including prompt loading and error handling.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from tonal_hortator.core.llm.llm_client import LocalLLMClient


class TestLocalLLMClient(unittest.TestCase):
    """Test cases for LocalLLMClient class."""

    def test_init_with_existing_prompt_file(self) -> None:
        """Test initialization when prompt file exists."""
        # Create a temporary prompt file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test prompt content")
            temp_prompt_path = f.name

        try:
            # Initialize client with existing prompt file
            client = LocalLLMClient(prompt_path=temp_prompt_path)

            # Verify the prompt was loaded
            self.assertEqual(client.prompt, "Test prompt content")
            self.assertEqual(client.prompt_path, temp_prompt_path)
            self.assertEqual(client.model_name, "llama3:8b")  # default
        finally:
            # Clean up temporary file
            os.unlink(temp_prompt_path)

    def test_init_with_missing_prompt_file(self) -> None:
        """Test initialization when prompt file doesn't exist."""
        # Use a non-existent file path
        nonexistent_path = "/tmp/nonexistent_prompt_file.txt"

        # Ensure the file doesn't exist
        if os.path.exists(nonexistent_path):
            os.unlink(nonexistent_path)

        # Should not raise an exception
        with patch("tonal_hortator.core.llm.llm_client.logger") as mock_logger:
            client = LocalLLMClient(prompt_path=nonexistent_path)

            # Verify the prompt is empty and warning was logged
            self.assertEqual(client.prompt, "")
            self.assertEqual(client.prompt_path, nonexistent_path)
            mock_logger.warning.assert_called_once_with(
                f"Prompt file not found: {nonexistent_path}. Using empty prompt."
            )

    def test_init_with_custom_model_name(self) -> None:
        """Test initialization with custom model name."""
        nonexistent_path = "/tmp/nonexistent_prompt_file.txt"

        client = LocalLLMClient(model_name="custom_model", prompt_path=nonexistent_path)

        self.assertEqual(client.model_name, "custom_model")
        self.assertEqual(client.prompt, "")  # File doesn't exist

    def test_load_prompt_with_existing_file(self) -> None:
        """Test load_prompt method with existing file."""
        # Create a temporary prompt file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("New prompt content")
            temp_prompt_path = f.name

        try:
            # Create client with non-existent file first
            client = LocalLLMClient(prompt_path="/tmp/nonexistent.txt")
            self.assertEqual(client.prompt, "")

            # Change path and reload
            client.prompt_path = temp_prompt_path
            with patch("tonal_hortator.core.llm.llm_client.logger") as mock_logger:
                client.load_prompt()

                # Verify the prompt was loaded and debug message logged
                self.assertEqual(client.prompt, "New prompt content")
                mock_logger.debug.assert_called_once_with(
                    f"Loaded prompt from {temp_prompt_path}"
                )
        finally:
            # Clean up temporary file
            os.unlink(temp_prompt_path)

    def test_load_prompt_with_missing_file(self) -> None:
        """Test load_prompt method when file doesn't exist."""
        nonexistent_path = "/tmp/definitely_nonexistent_prompt.txt"

        # Ensure the file doesn't exist
        if os.path.exists(nonexistent_path):
            os.unlink(nonexistent_path)

        client = LocalLLMClient(prompt_path="/tmp/some_other_file.txt")
        client.prompt = "existing content"  # Set some existing content
        client.prompt_path = nonexistent_path

        with patch("tonal_hortator.core.llm.llm_client.logger") as mock_logger:
            client.load_prompt()

            # Verify the prompt was cleared and warning logged
            self.assertEqual(client.prompt, "")
            mock_logger.warning.assert_called_once_with(
                f"Prompt file not found: {nonexistent_path}. Using empty prompt."
            )

    def test_reload_prompt_functionality(self) -> None:
        """Test reload_prompt method."""
        # Create a temporary prompt file with initial content
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Initial content")
            temp_prompt_path = f.name

        try:
            # Initialize client
            client = LocalLLMClient(prompt_path=temp_prompt_path)
            self.assertEqual(client.prompt, "Initial content")

            # Modify the file content
            with open(temp_prompt_path, "w") as f:
                f.write("Updated content")

            # Reload and verify
            with patch("builtins.print") as mock_print:
                client.reload_prompt()
                mock_print.assert_called_once_with("Prompt reloaded!")

            self.assertEqual(client.prompt, "Updated content")
        finally:
            # Clean up temporary file
            os.unlink(temp_prompt_path)

    def test_reload_prompt_with_missing_file(self) -> None:
        """Test reload_prompt when file goes missing."""
        # Create a temporary prompt file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Initial content")
            temp_prompt_path = f.name

        try:
            # Initialize client
            client = LocalLLMClient(prompt_path=temp_prompt_path)
            self.assertEqual(client.prompt, "Initial content")

            # Remove the file
            os.unlink(temp_prompt_path)

            # Reload should handle missing file gracefully
            with patch("tonal_hortator.core.llm.llm_client.logger") as mock_logger:
                with patch("builtins.print") as mock_print:
                    client.reload_prompt()

                    # Verify warning was logged and print was called
                    mock_logger.warning.assert_called_once_with(
                        f"Prompt file not found: {temp_prompt_path}. Using empty prompt."
                    )
                    mock_print.assert_called_once_with("Prompt reloaded!")

            # Prompt should be empty now
            self.assertEqual(client.prompt, "")
        except FileNotFoundError:
            # File was already removed, that's fine
            pass


if __name__ == "__main__":
    unittest.main()
