"""Tests for the loader module's adaptive batch sizing functionality."""

import unittest
from unittest.mock import patch

from tonal_hortator.utils.loader import (
    get_batch_size_with_fallback,
    get_optimal_batch_size,
)


class TestAdaptiveBatchSizing(unittest.TestCase):
    """Test adaptive batch sizing functionality."""

    def test_get_optimal_batch_size_with_mock_system(self) -> None:
        """Test optimal batch size calculation with mocked system resources."""
        with (
            patch("psutil.virtual_memory") as mock_memory,
            patch("psutil.cpu_count") as mock_cpu,
        ):
            # Mock system with 8GB available memory and 4 CPU cores
            mock_memory.return_value.available = 8 * 1024**3  # 8GB
            mock_cpu.return_value = 4

            result = get_optimal_batch_size(
                base_size=500, memory_factor=0.1, cpu_factor=0.5
            )

            # Should be limited by CPU factor: 500 * 4 * 0.5 = 1000
            # But capped at max_size=1000
            self.assertEqual(result, 1000)

    def test_get_optimal_batch_size_memory_limited(self) -> None:
        """Test when memory is the limiting factor."""
        with (
            patch("psutil.virtual_memory") as mock_memory,
            patch("psutil.cpu_count") as mock_cpu,
        ):
            # Mock system with 1GB available memory and 8 CPU cores
            mock_memory.return_value.available = 1 * 1024**3  # 1GB
            mock_cpu.return_value = 8

            result = get_optimal_batch_size(
                base_size=500, memory_factor=0.1, cpu_factor=0.5
            )

            # Memory-based: 1GB * 1024 * 1024 * 0.1 = ~100,000
            # CPU-based: 500 * 8 * 0.5 = 2000
            # Should take minimum: 2000, but capped at max_size=1000
            self.assertEqual(result, 1000)

    def test_get_optimal_batch_size_fallback(self) -> None:
        """Test fallback when system info is unavailable."""
        with patch("psutil.virtual_memory", side_effect=Exception("No psutil")):
            result = get_optimal_batch_size(base_size=500)
            self.assertEqual(result, 500)

    def test_get_batch_size_with_fallback_user_specified(self) -> None:
        """Test when user specifies a batch size."""
        result = get_batch_size_with_fallback(user_specified=200, base_size=500)
        self.assertEqual(result, 200)

    def test_get_batch_size_with_fallback_auto_detection(self) -> None:
        """Test auto-detection when user doesn't specify."""
        with patch(
            "tonal_hortator.utils.loader.get_optimal_batch_size"
        ) as mock_optimal:
            mock_optimal.return_value = 750
            result = get_batch_size_with_fallback(user_specified=None, base_size=500)
            self.assertEqual(result, 750)

    def test_get_batch_size_with_fallback_validation(self) -> None:
        """Test that user-specified values are validated against min/max bounds."""
        # Test below minimum
        result = get_batch_size_with_fallback(
            user_specified=25, min_size=50, max_size=1000
        )
        self.assertEqual(result, 50)

        # Test above maximum
        result = get_batch_size_with_fallback(
            user_specified=1500, min_size=50, max_size=1000
        )
        self.assertEqual(result, 1000)

        # Test within bounds
        result = get_batch_size_with_fallback(
            user_specified=500, min_size=50, max_size=1000
        )
        self.assertEqual(result, 500)


if __name__ == "__main__":
    unittest.main()
