#!/usr/bin/env python3
"""
Tests for CLI modules
"""

import io
import unittest
from unittest.mock import Mock, patch

from tonal_hortator.cli.playlist_cli import PlaylistCLI, main


class TestCLI(unittest.TestCase):
    """Test CLI functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_generator = Mock()
        self.mock_output = Mock()
        self.cli = PlaylistCLI(self.mock_generator)

    @patch("tonal_hortator.cli.playlist_cli.PlaylistCLI")
    @patch("tonal_hortator.cli.playlist_cli.LocalPlaylistGenerator")
    def test_main_function(self, mock_generator_class, mock_cli_class):
        """Test the main function"""
        # Mock the generator
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        # Mock the CLI
        mock_cli = Mock()
        mock_cli_class.return_value = mock_cli
        mock_cli.check_embeddings_available.return_value = True

        # Test main function
        with patch("builtins.input", return_value="quit"):
            main()

        # Verify calls
        mock_generator_class.assert_called_once()
        mock_cli_class.assert_called_once_with(mock_generator)
        mock_cli.check_embeddings_available.assert_called_once()
        mock_cli.run_interactive_loop.assert_called_once()

    @patch("tonal_hortator.cli.playlist_cli.PlaylistCLI")
    @patch("tonal_hortator.cli.playlist_cli.LocalPlaylistGenerator")
    def test_main_function_no_embeddings(self, mock_generator_class, mock_cli_class):
        """Test main function when no embeddings are available"""
        # Mock the generator
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        # Mock the CLI
        mock_cli = Mock()
        mock_cli_class.return_value = mock_cli
        mock_cli.check_embeddings_available.return_value = False

        # Test main function
        main()

        # Verify calls
        mock_generator_class.assert_called_once()
        mock_cli_class.assert_called_once_with(mock_generator)
        mock_cli.check_embeddings_available.assert_called_once()
        mock_cli.run_interactive_loop.assert_not_called()

    def test_check_embeddings_available_true(self):
        """Test check_embeddings_available when embeddings exist"""
        # Mock the track embedder
        mock_track_embedder = Mock()
        mock_track_embedder.get_embedding_stats.return_value = {
            "tracks_with_embeddings": 10
        }
        self.mock_generator.track_embedder = mock_track_embedder

        result = self.cli.check_embeddings_available()

        self.assertTrue(result)
        mock_track_embedder.get_embedding_stats.assert_called_once()

    def test_check_embeddings_available_false(self):
        """Test check_embeddings_available when no embeddings exist"""
        # Mock the track embedder
        mock_track_embedder = Mock()
        mock_track_embedder.get_embedding_stats.return_value = {
            "tracks_with_embeddings": 0
        }
        self.mock_generator.track_embedder = mock_track_embedder

        result = self.cli.check_embeddings_available()

        self.assertFalse(result)
        mock_track_embedder.get_embedding_stats.assert_called_once()

    def test_process_playlist_request_success(self):
        """Test successful playlist request processing"""
        # Mock the generator
        mock_tracks = [{"name": "Test Song", "artist": "Test Artist"}]
        self.mock_generator.generate_playlist.return_value = mock_tracks

        # Mock input to return 'n' for save
        with patch("builtins.input", return_value="n"):
            result = self.cli.process_playlist_request("test query")

        self.assertTrue(result)
        self.mock_generator.generate_playlist.assert_called_once_with(
            "test query", max_tracks=20
        )

    def test_process_playlist_request_no_tracks(self):
        """Test playlist request when no tracks are found"""
        # Mock the generator to return no tracks
        self.mock_generator.generate_playlist.return_value = []

        # Capture stdout to check the message
        with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
            result = self.cli.process_playlist_request("test query")

        self.assertFalse(result)
        self.assertIn("No tracks found", mock_stdout.getvalue())

    def test_process_playlist_request_save_playlist(self):
        """Test playlist request with saving"""
        # Mock the generator
        mock_tracks = [{"name": "Test Song", "artist": "Test Artist"}]
        self.mock_generator.generate_playlist.return_value = mock_tracks

        # Mock the output
        mock_output = Mock()
        mock_output.save_playlist_m3u.return_value = "/path/to/playlist.m3u"
        self.cli.output = mock_output

        # Mock input to return 'y' for save, 'n' for open music
        with patch("builtins.input", side_effect=["y", "n"]):
            result = self.cli.process_playlist_request("test query")

        self.assertTrue(result)
        mock_output.save_playlist_m3u.assert_called_once_with(mock_tracks, "test query")

    @patch("subprocess.run")
    def test_process_playlist_request_open_music(self, mock_subprocess):
        """Test playlist request with opening in Apple Music"""
        # Mock the generator
        mock_tracks = [{"name": "Test Song", "artist": "Test Artist"}]
        self.mock_generator.generate_playlist.return_value = mock_tracks

        # Mock the output
        mock_output = Mock()
        mock_output.save_playlist_m3u.return_value = "/path/to/playlist.m3u"
        self.cli.output = mock_output

        # Mock subprocess to succeed
        mock_subprocess.return_value = Mock()

        # Mock input to return 'y' for both save and open music
        with patch("builtins.input", side_effect=["y", "y"]):
            with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
                result = self.cli.process_playlist_request("test query")

        self.assertTrue(result)
        mock_subprocess.assert_called_once_with(
            ["open", "-a", "Music", "/path/to/playlist.m3u"], check=True
        )
        self.assertIn("Opened in Apple Music", mock_stdout.getvalue())

    @patch("subprocess.run")
    def test_process_playlist_request_open_music_failure(self, mock_subprocess):
        """Test playlist request when opening Apple Music fails"""
        # Mock the generator
        mock_tracks = [{"name": "Test Song", "artist": "Test Artist"}]
        self.mock_generator.generate_playlist.return_value = mock_tracks

        # Mock the output
        mock_output = Mock()
        mock_output.save_playlist_m3u.return_value = "/path/to/playlist.m3u"
        self.cli.output = mock_output

        # Mock subprocess to fail
        mock_subprocess.side_effect = Exception("Music app not found")

        # Mock input to return 'y' for both save and open music
        with patch("builtins.input", side_effect=["y", "y"]):
            with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
                result = self.cli.process_playlist_request("test query")

        self.assertTrue(result)
        self.assertIn("Could not open in Apple Music", mock_stdout.getvalue())

    def test_process_playlist_request_exception(self):
        """Test playlist request when an exception occurs"""
        # Mock the generator to raise an exception
        self.mock_generator.generate_playlist.side_effect = Exception("Test error")

        # Capture stdout to check the error message
        with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
            result = self.cli.process_playlist_request("test query")

        self.assertFalse(result)
        self.assertIn("Error:", mock_stdout.getvalue())

    def test_run_interactive_loop(self):
        """Test the interactive loop"""
        # Mock input to return a query, then quit
        with patch("builtins.input", side_effect=["test query", "quit"]):
            with patch.object(self.cli, "process_playlist_request", return_value=True):
                with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
                    self.cli.run_interactive_loop()

        # Check that the welcome message was printed
        self.assertIn("Local Playlist Generator", mock_stdout.getvalue())
        self.assertIn("Thanks for using", mock_stdout.getvalue())

    def test_run_interactive_loop_empty_query(self):
        """Test interactive loop with empty query"""
        # Mock input to return empty query, then quit
        with patch("builtins.input", side_effect=["", "quit"]):
            with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
                self.cli.run_interactive_loop()

        # Check that the empty query message was printed
        self.assertIn("Please enter a query", mock_stdout.getvalue())

    def test_run_interactive_loop_exit_commands(self):
        """Test interactive loop with various exit commands"""
        exit_commands = ["quit", "exit", "q"]

        for exit_cmd in exit_commands:
            with patch("builtins.input", return_value=exit_cmd):
                with patch("sys.stdout", new=io.StringIO()) as mock_stdout:
                    self.cli.run_interactive_loop()

            # Check that the goodbye message was printed
            self.assertIn("Thanks for using", mock_stdout.getvalue())


class TestCLIImports(unittest.TestCase):
    """Test CLI module imports"""

    def test_cli_init_imports(self):
        """Test that CLI __init__.py imports work correctly"""
        from tonal_hortator.cli import playlist_main

        self.assertIsNotNone(playlist_main)

    def test_main_module_imports(self):
        """Test that main.py imports work correctly"""
        from tonal_hortator.cli.main import playlist_main

        self.assertIsNotNone(playlist_main)


if __name__ == "__main__":
    unittest.main()
