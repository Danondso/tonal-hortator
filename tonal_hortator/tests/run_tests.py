#!/usr/bin/env python3
"""
Test runner for tonal_hortator
"""

import os
import sys
import unittest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tonal_hortator.tests.test_artist_distributor import TestArtistDistributor

# Import test modules
from tonal_hortator.tests.test_cli import TestCLI, TestCLIImports
from tonal_hortator.tests.test_playlist_generator import TestLocalPlaylistGenerator
from tonal_hortator.tests.test_playlist_generator_comprehensive import (
    TestLocalPlaylistGeneratorComprehensive,
)
from tonal_hortator.tests.test_playlist_output import TestPlaylistOutput
from tonal_hortator.tests.test_track_embedder import TestLocalTrackEmbedder
from tonal_hortator.tests.test_utils import TestAppleMusicUtils, TestLibraryParser


def run_tests() -> None:
    """Run all tests"""
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with appropriate code
    sys.exit(not result.wasSuccessful())


if __name__ == "__main__":
    run_tests()
