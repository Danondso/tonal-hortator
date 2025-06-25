#!/usr/bin/env python3
"""
Test runner for tonal-hortator
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
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestCLI,
        TestCLIImports,
        TestAppleMusicUtils,
        TestLibraryParser,
        TestPlaylistOutput,
        TestArtistDistributor,
        TestLocalTrackEmbedder,
        TestLocalPlaylistGenerator,
        TestLocalPlaylistGeneratorComprehensive,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Return exit code
    sys.exit(not result.wasSuccessful())


if __name__ == "__main__":
    run_tests()
