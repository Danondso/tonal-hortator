#!/usr/bin/env python3
"""
Test runner for Tonal Hortator
"""

import os
import sys
import unittest

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def create_test_suite() -> unittest.TestSuite:
    """Create a test suite for all tests"""
    test_suite = unittest.TestSuite()

    # Add test cases
    loader = unittest.TestLoader()

    # Import test cases
    from tonal_hortator.tests.test_core_functionality import TestCoreFunctionality

    # Add tests to suite
    test_suite.addTest(loader.loadTestsFromTestCase(TestCoreFunctionality))

    return test_suite


def run_tests() -> int:
    """Run all tests"""
    # Create test suite
    test_suite = create_test_suite()

    # Create test runner
    runner = unittest.TextTestRunner(verbosity=2)

    # Run tests
    result = runner.run(test_suite)

    # Exit with status code based on test results
    if result.wasSuccessful():
        print("\nOverall: ✅ ALL TESTS PASSED")
        return 0
    else:
        print(f"\nOverall: ❌ SOME TESTS FAILED")
        return 1


def main() -> None:
    # Run tests
    exit_code = run_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
