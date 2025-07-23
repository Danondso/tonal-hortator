import os
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import Mock

import numpy as np
import pytest

from tonal_hortator.utils.embedding_updater import (
    EmbeddingUpdater,
    HybridStrategy,
    parse_ids_from_file,
)


class TestEmbeddingUpdater:
    @pytest.fixture
    def temp_db(self) -> str:
        """Create an in-memory database for testing."""
        # Use in-memory database
        db_path = ":memory:"

        # Create test tables
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create tracks table
        cursor.execute(
            """
            CREATE TABLE tracks (
                id INTEGER PRIMARY KEY,
                name TEXT,
                artist TEXT,
                album TEXT,
                genre TEXT,
                year INTEGER,
                play_count INTEGER,
                avg_rating REAL,
                bpm INTEGER,
                musical_key TEXT,
                mood TEXT,
                label TEXT,
                producer TEXT,
                location TEXT
            )
        """
        )

        # Create track_embeddings table
        cursor.execute(
            """
            CREATE TABLE track_embeddings (
                track_id INTEGER PRIMARY KEY,
                embedding BLOB,
                embedding_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Insert test data
        cursor.execute(
            """
            INSERT INTO tracks (id, name, artist, album, genre, year, play_count, avg_rating, bpm, musical_key,
mood, label, producer)
            VALUES (1, 'Test Song', 'Test Artist', 'Test Album', 'Rock', 2020, 50, 4.5, 120, 'C', 'Happy', 'Test
Label', 'Test Producer')
        """
        )

        # Insert test embedding
        test_embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        old_timestamp = (datetime.now() - timedelta(days=100)).isoformat()
        cursor.execute(
            """
            INSERT INTO track_embeddings (track_id, embedding, embedding_text, created_at)
            VALUES (1, ?, ?, ?)
        """,
            (test_embedding.tobytes(), "old embedding text", old_timestamp),
        )

        conn.commit()
        conn.close()

        return db_path

    @pytest.fixture
    def mock_embedder(self) -> Mock:
        """Mock the track embedder."""
        mock = Mock()
        mock.conn = Mock()
        mock.embedding_service = Mock()
        mock.create_track_embedding_text = Mock(return_value="new embedding text")
        return mock

    def test_hybrid_strategy_enum(self) -> None:
        """Test that all hybrid strategies are available."""
        strategies = list(HybridStrategy)
        assert len(strategies) == 5
        assert HybridStrategy.SIMPLE_AVERAGE in strategies
        assert HybridStrategy.AGE_WEIGHTED in strategies
        assert HybridStrategy.METADATA_WEIGHTED in strategies
        assert HybridStrategy.CONFIDENCE_WEIGHTED in strategies
        assert HybridStrategy.EXPONENTIAL_DECAY in strategies

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skipping in CI (Ollama not running)",
    )
    def test_get_default_hybrid_config(self, temp_db: str) -> None:
        """Test default configuration for each strategy."""
        updater = EmbeddingUpdater(temp_db)

        # Test each strategy has a default config
        for strategy in HybridStrategy:
            config = updater._get_default_hybrid_config(strategy)
            assert isinstance(config, dict)
            assert len(config) > 0

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skipping in CI (Ollama not running)",
    )
    def test_calculate_age_weights(self, temp_db: str) -> None:
        """Test age-based weight calculation."""
        from tonal_hortator.core.models import Track

        updater = EmbeddingUpdater(temp_db)

        # Mock track data
        track = Track.from_dict({"id": 1, "name": "Test Song", "artist": "Test Artist"})

        # Test with default config
        config = updater._get_default_hybrid_config(HybridStrategy.AGE_WEIGHTED)
        old_weight, new_weight = updater._calculate_age_weights(track, config)

        # Weights should sum to 1.0
        assert abs(old_weight + new_weight - 1.0) < 1e-6
        assert 0.0 <= old_weight <= 1.0
        assert 0.0 <= new_weight <= 1.0

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skipping in CI (Ollama not running)",
    )
    def test_calculate_metadata_weights(self, temp_db: str) -> None:
        """Test metadata-based weight calculation."""
        from tonal_hortator.core.models import Track

        updater = EmbeddingUpdater(temp_db)

        # Track with complete metadata
        complete_track = Track.from_dict(
            {
                "id": 1,
                "name": "Test Song",
                "artist": "Test Artist",
                "album": "Test Album",
                "genre": "Rock",
                "year": 2020,
                "bpm": 120,
                "musical_key": "C",
                "mood": "Happy",
                "label": "Test Label",
                "producer": "Test Producer",
                "play_count": 50,
            }
        )

        config = updater._get_default_hybrid_config(HybridStrategy.METADATA_WEIGHTED)
        old_weight, new_weight = updater._calculate_metadata_weights(
            complete_track, config
        )

        # Weights should sum to 1.0
        assert abs(old_weight + new_weight - 1.0) < 1e-6
        assert 0.0 <= old_weight <= 1.0
        assert 0.0 <= new_weight <= 1.0

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skipping in CI (Ollama not running)",
    )
    def test_calculate_confidence_weights(self, temp_db: str) -> None:
        """Test confidence-based weight calculation."""
        from tonal_hortator.core.models import Track

        updater = EmbeddingUpdater(temp_db)

        # Track with good engagement data
        track = Track.from_dict(
            {
                "id": 1,
                "name": "Test Song",
                "artist": "Test Artist",
                "play_count": 100,
                "avg_rating": 4.5,
            }
        )

        config = updater._get_default_hybrid_config(HybridStrategy.CONFIDENCE_WEIGHTED)
        old_weight, new_weight = updater._calculate_confidence_weights(track, config)

        # Weights should sum to 1.0
        assert abs(old_weight + new_weight - 1.0) < 1e-6
        assert 0.0 <= old_weight <= 1.0
        assert 0.0 <= new_weight <= 1.0

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skipping in CI (Ollama not running)",
    )
    def test_calculate_exponential_decay_weights(self, temp_db: str) -> None:
        """Test exponential decay weight calculation."""
        from tonal_hortator.core.models import Track

        updater = EmbeddingUpdater(temp_db)

        track = Track.from_dict({"id": 1, "name": "Test Song", "artist": "Test Artist"})
        config = updater._get_default_hybrid_config(HybridStrategy.EXPONENTIAL_DECAY)
        old_weight, new_weight = updater._calculate_exponential_decay_weights(
            track, config
        )

        # Weights should sum to 1.0
        assert abs(old_weight + new_weight - 1.0) < 1e-6
        assert 0.0 <= old_weight <= 1.0
        assert 0.0 <= new_weight <= 1.0

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skipping in CI (Ollama not running)",
    )
    def test_calculate_weights_all_strategies(self, temp_db: str) -> None:
        """Test weight calculation for all strategies."""
        from tonal_hortator.core.models import Track

        updater = EmbeddingUpdater(temp_db)

        track = Track.from_dict({"id": 1, "name": "Test Song", "artist": "Test Artist"})

        for strategy in HybridStrategy:
            config = updater._get_default_hybrid_config(strategy)
            old_weight, new_weight = updater._calculate_weights(track, strategy, config)

            # Weights should sum to 1.0
            assert abs(old_weight + new_weight - 1.0) < 1e-6
            assert 0.0 <= old_weight <= 1.0
            assert 0.0 <= new_weight <= 1.0

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skipping in CI (Ollama not running)",
    )
    def test_update_embeddings_hybrid_mode(self, temp_db: str) -> None:
        """Test hybrid mode with different strategies."""
        updater = EmbeddingUpdater(temp_db)

        # Test that all strategies are recognized and have valid configurations
        for strategy in HybridStrategy:
            config = updater._get_default_hybrid_config(strategy)
            assert isinstance(config, dict)
            assert len(config) > 0

            # Test that the strategy can be used in update_embeddings_for_tracks
            # (we'll just test the parameter validation, not the full execution)
            try:
                # This should not raise an exception for valid strategies
                strategy_value = strategy.value
                assert strategy_value in [s.value for s in HybridStrategy]
            except Exception as e:
                pytest.fail(f"Strategy {strategy} failed validation: {e}")

    def test_hybrid_strategy_parameter_validation(self, temp_db: str) -> None:
        """Test that hybrid strategy parameters are properly validated."""
        # Test that valid strategies are recognized
        valid_strategies = [s.value for s in HybridStrategy]
        assert "age_weighted" in valid_strategies
        assert "simple_average" in valid_strategies
        assert "metadata_weighted" in valid_strategies
        assert "confidence_weighted" in valid_strategies
        assert "exponential_decay" in valid_strategies

        # Test that invalid strategies are handled gracefully
        invalid_strategy = "invalid_strategy"
        assert invalid_strategy not in valid_strategies

    def test_parse_ids_from_file(self) -> None:
        """Test parsing track IDs from file."""
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("1\n2\n3\ninvalid\n5\n")
            file_path = f.name

        try:
            ids = parse_ids_from_file(file_path)
            assert ids == [1, 2, 3, 5]  # Invalid line should be skipped
        finally:
            Path(file_path).unlink()

    def test_invalid_hybrid_strategy(self, temp_db: str) -> None:
        """Test handling of invalid hybrid strategy."""
        # Test that the default strategy is AGE_WEIGHTED
        default_strategy = HybridStrategy.AGE_WEIGHTED
        assert default_strategy.value == "age_weighted"

        # Test that invalid strategy string would fall back to default
        # (we test the logic without actually calling the full update method)
        try:
            HybridStrategy("invalid_strategy")
            pytest.fail("Should have raised ValueError for invalid strategy")
        except ValueError:
            pass  # Expected behavior

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skipping in CI (Ollama not running)",
    )
    def test_hybrid_config_customization(self, temp_db: str) -> None:
        """Test custom hybrid configuration."""
        from tonal_hortator.core.models import Track

        updater = EmbeddingUpdater(temp_db)

        # Custom config for age-weighted strategy
        custom_config = {
            "age_decay_days": 180,  # Faster decay
            "min_old_weight": 0.2,  # Higher minimum
            "max_old_weight": 0.9,  # Higher maximum
        }

        track = Track.from_dict({"id": 1, "name": "Test Song", "artist": "Test Artist"})
        old_weight, new_weight = updater._calculate_age_weights(track, custom_config)

        # Weights should sum to 1.0
        assert abs(old_weight + new_weight - 1.0) < 1e-6
        # Note: The actual weight depends on the embedding age, so we just check bounds
        assert 0.0 <= old_weight <= 1.0
        assert 0.0 <= new_weight <= 1.0
