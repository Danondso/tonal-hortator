"""
Configuration management for Tonal Hortator.

This module provides centralized configuration loading and access,
supporting YAML files, environment variable overrides, and A/B testing.
"""

import copy
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

import yaml

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ConfigPath:
    """Helper class for type-safe config path access."""

    path: str
    default: Any = None
    type_hint: Optional[Type] = None


class ConfigurationManager:
    """
    Centralized configuration management for Tonal Hortator.

    Loads configuration from YAML files, supports environment variable
    overrides, A/B testing variants, and provides type-safe access.
    """

    def __init__(
        self, config_path: Optional[str] = None, variant: Optional[str] = None
    ):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to the config YAML file (defaults to config.yml in project root)
            variant: A/B testing variant to use (optional)
        """
        self.config_path = config_path or self._find_config_file()
        self.variant = variant
        self._base_config: Dict[str, Any] = {}  # Original configuration from file
        self._config: Dict[str, Any] = (
            {}
        )  # Current configuration (with variants applied)
        self._load_config()

    def _find_config_file(self) -> str:
        """Find the config.yml file in the project root."""
        # Start from this file's directory and work up to find config.yml
        current_dir = Path(__file__).parent
        for _ in range(5):  # Limit search depth
            config_file = current_dir / "config.yml"
            if config_file.exists():
                return str(config_file)

            # Also check parent directories
            parent_config = current_dir.parent / "config.yml"
            if parent_config.exists():
                return str(parent_config)

            # Check project root (two levels up from core/)
            root_config = current_dir.parent.parent / "config.yml"
            if root_config.exists():
                return str(root_config)

            current_dir = current_dir.parent

        # Fallback: assume config.yml is in project root
        return "config.yml"

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._base_config = yaml.safe_load(f) or {}

            # Create a deep copy for working configuration
            self._config = copy.deepcopy(self._base_config)

            # Apply A/B testing variant if specified
            if self.variant and self._base_config.get("ab_testing", {}).get("enabled"):
                self._apply_variant(self.variant)

            # Apply environment variable overrides
            if self._base_config.get("environment_overrides", {}).get("enabled"):
                self._apply_env_overrides()

            logger.info(f"✅ Loaded configuration from {self.config_path}")

        except FileNotFoundError:
            logger.warning(
                f"⚠️  Config file not found: {self.config_path}. "
                f"Using empty configuration - this may cause errors when accessing config values."
            )
            self._base_config = {}
            self._config = {}
        except yaml.YAMLError as e:
            logger.error(
                f"❌ Error parsing config YAML: {e}. "
                f"Using empty configuration - this may cause errors when accessing config values."
            )
            self._base_config = {}
            self._config = {}

    def _apply_variant(self, variant: str) -> None:
        """Apply A/B testing variant configuration to a copy of base config."""
        variants = self._base_config.get("ab_testing", {}).get("variants", {})
        if variant in variants:
            variant_config = variants[variant]
            logger.info(f"🧪 Applying A/B testing variant: {variant}")

            # Reset to base config and apply variant to ensure clean state
            self._config = copy.deepcopy(self._base_config)
            self._deep_update(self._config, variant_config)
        else:
            available = list(variants.keys())
            logger.warning(f"⚠️  Unknown variant '{variant}'. Available: {available}")

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        env_config = self._base_config.get("environment_overrides", {})
        prefix = env_config.get("prefix", "TH_")
        mappings = env_config.get("mappings", {})

        overrides_applied = 0
        for config_path, env_suffix in mappings.items():
            env_var = f"{prefix}{env_suffix}"
            env_value = os.getenv(env_var)

            if env_value is not None:
                # Convert string value to appropriate type
                converted_value = self._convert_env_value(env_value)
                self._set_nested_value(config_path, converted_value)
                overrides_applied += 1
                logger.info(
                    f"🔧 Environment override: {config_path} = {converted_value}"
                )

        if overrides_applied:
            logger.info(f"✅ Applied {overrides_applied} environment overrides")

    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type."""
        # Try boolean
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        elif value.lower() in ("false", "no", "0", "off"):
            return False

        # Try integer
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    def _set_nested_value(self, path: str, value: Any) -> None:
        """Set a nested configuration value using dot notation."""
        keys = path.split(".")
        current = self._config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _deep_update(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Deep update one dictionary with another."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def get(
        self, path: str, default: Any = None, type_hint: Optional[Type[T]] = None
    ) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            path: Dot-separated path to the config value (e.g., 'playlist.default_max_tracks')
            default: Default value if path doesn't exist
            type_hint: Optional type hint for return value

        Returns:
            Configuration value with optional type casting
        """
        keys = path.split(".")
        current = self._config

        try:
            for key in keys:
                current = current[key]

            # Apply type conversion if hint provided
            if type_hint and current is not None:
                try:
                    if type_hint == bool:
                        return bool(current)
                    elif type_hint == int:
                        return int(current)  # type: ignore[call-overload]
                    elif type_hint == float:
                        return float(current)  # type: ignore[arg-type]
                    elif type_hint == str:
                        return str(current)
                except (ValueError, TypeError):
                    # If conversion fails, return as-is
                    pass

            return current

        except (KeyError, TypeError):
            return default

    def get_section(self, section: str) -> Any:
        """Get an entire configuration section."""
        return self._config.get(section, {})

    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()

    def set_variant(self, variant: str) -> None:
        """Switch to a different A/B testing variant efficiently."""
        self.variant = variant

        # Reset to base config
        self._config = copy.deepcopy(self._base_config)

        # Apply new variant if A/B testing is enabled
        if self.variant and self._base_config.get("ab_testing", {}).get("enabled"):
            self._apply_variant(self.variant)

        # Reapply environment overrides
        if self._base_config.get("environment_overrides", {}).get("enabled"):
            self._apply_env_overrides()

    def clear_variant(self) -> None:
        """Clear the current variant and return to base configuration."""
        self.variant = None
        self._config = copy.deepcopy(self._base_config)

        # Reapply environment overrides to base config
        if self._base_config.get("environment_overrides", {}).get("enabled"):
            self._apply_env_overrides()

    # Convenience methods for common configuration paths

    @property
    def playlist_defaults(self) -> Dict[str, Any]:
        """Get playlist generation defaults."""
        return {
            "max_tracks": self.get("playlist.default_max_tracks", 20),
            "min_similarity": self.get("playlist.default_min_similarity", 0.2),
            "max_artist_ratio": self.get("playlist.default_max_artist_ratio", 0.5),
            "search_breadth_factor": self.get(
                "playlist.default_search_breadth_factor", 15
            ),
        }

    @property
    def feedback_adjustments(self) -> Dict[str, float]:
        """Get feedback adjustment values."""
        return {
            "like": self.get("feedback.adjustments.like", 0.2),
            "dislike": self.get("feedback.adjustments.dislike", -0.2),
            "block": self.get("feedback.adjustments.block", -1.0),
            "note": self.get("feedback.adjustments.note", 0.0),
        }

    @property
    def validation_ranges(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """Get validation ranges for various parameters."""
        return {
            "user_rating": {
                "min": self.get("validation.user_rating.min", 0),
                "max": self.get("validation.user_rating.max", 5),
            },
            "similarity_threshold": {
                "min": self.get("validation.similarity_threshold.min", 0.0),
                "max": self.get("validation.similarity_threshold.max", 1.0),
            },
            "search_breadth": {
                "min": self.get("validation.search_breadth.min", 1),
            },
        }

    @property
    def llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return {
            "embedding_model": self.get(
                "llm.models.embedding", "nomic-embed-text:latest"
            ),
            "query_parser_model": self.get("llm.models.query_parser", "llama3:8b"),
            "max_tokens": self.get("llm.max_tokens", 512),
        }


# Global configuration instance with thread-safe initialization
_config_instance: Optional[ConfigurationManager] = None
_default_config_instance: Optional[ConfigurationManager] = None
_config_lock = threading.Lock()


def get_config(
    config_path: Optional[str] = None, variant: Optional[str] = None
) -> ConfigurationManager:
    """
    Get the global configuration instance using thread-safe double-checked locking.

    Args:
        config_path: Path to config file (only used on first call)
        variant: A/B testing variant (only used on first call)

    Returns:
        ConfigurationManager instance
    """
    global _config_instance

    # First check (unlocked)
    if _config_instance is None:
        # Acquire lock for thread-safe initialization
        with _config_lock:
            # Second check (locked) - another thread might have created instance
            if _config_instance is None:
                _config_instance = ConfigurationManager(config_path, variant)

    return _config_instance


def _get_default_config() -> ConfigurationManager:
    """
    Thread-safe default configuration instance using double-checked locking.

    This provides an alternative singleton pattern for cases where
    no custom parameters are needed.
    """
    global _default_config_instance

    # First check (unlocked)
    if _default_config_instance is None:
        # Acquire lock for thread-safe initialization
        with _config_lock:
            # Second check (locked) - another thread might have created instance
            if _default_config_instance is None:
                _default_config_instance = ConfigurationManager()

    return _default_config_instance


def reload_config() -> None:
    """Reload the global configuration in a thread-safe manner."""
    global _config_instance

    with _config_lock:
        if _config_instance:
            _config_instance.reload()
        else:
            # If no instance exists, create a new one
            _config_instance = ConfigurationManager()

        # Also reload the default config instance if it exists
        if _default_config_instance:
            _default_config_instance.reload()


def reset_config() -> None:
    """
    Reset the global configuration instances for testing purposes.

    This is primarily intended for unit tests to ensure clean state.
    """
    global _config_instance, _default_config_instance

    with _config_lock:
        _config_instance = None
        _default_config_instance = None


def get_default_config() -> ConfigurationManager:
    """
    Get a default configuration instance using thread-safe double-checked locking.

    This is an alternative to get_config() when you don't need custom
    config_path or variant parameters. Uses the same thread-safe pattern
    but maintains a separate instance for default configurations.

    Returns:
        Thread-safe ConfigurationManager instance with default settings
    """
    return _get_default_config()


# Common configuration constants as module-level functions
def get_playlist_defaults() -> Dict[str, Any]:
    """Get playlist generation defaults."""
    return get_config().playlist_defaults


def get_feedback_adjustments() -> Dict[str, float]:
    """Get feedback adjustment values."""
    return get_config().feedback_adjustments


def get_validation_ranges() -> Dict[str, Dict[str, Union[int, float]]]:
    """Get validation ranges."""
    return get_config().validation_ranges


def get_llm_config() -> Dict[str, Any]:
    """Get LLM configuration."""
    return get_config().llm_config
